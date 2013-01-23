"""
Data structures for Athena.

Author: Samuel W. Skillman <samskillman@gmail.com>
Affiliation: University of Colorado at Boulder
Author: Matthew Turk <matthewturk@gmail.com>
Author: J. S. Oishi <jsoishi@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
Homepage: http://yt-project.org/
License:
  Copyright (C) 2008-2011 Samuel W. Skillman, Matthew Turk, J. S. Oishi.  
  All Rights Reserved.

  This file is part of yt.

  yt is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import h5py
import numpy as np
import weakref
import glob #ST 9/12
from yt.funcs import *
from yt.data_objects.grid_patch import \
           AMRGridPatch
from yt.geometry.grid_geometry_handler import \
    GridGeometryHandler
from yt.data_objects.static_output import \
           StaticOutput
from yt.utilities.definitions import \
    mpc_conversion, sec_conversion

from .fields import AthenaFieldInfo, KnownAthenaFields
from yt.data_objects.field_info_container import \
    FieldInfoContainer, NullFunc

def _get_convert(fname):
    def _conv(data):
        return data.convert(fname)
    return _conv

class AthenaGrid(AMRGridPatch):
    _id_offset = 0
    def __init__(self, id, hierarchy, level, start, dimensions):
        df = hierarchy.storage_filename
        if 'id0' not in hierarchy.parameter_file.filename:
            gname = hierarchy.parameter_file.filename
        else:
            if id == 0:
                gname = 'id0/%s.vtk' % df
            else:
                gname = 'id%i/%s-id%i%s.vtk' % (id, df[:-5], id, df[-5:] )
        AMRGridPatch.__init__(self, id, filename = gname,
                              hierarchy = hierarchy)
        self.filename = gname
        self.Parent = []
        self.Children = []
        self.Level = level
        self.start_index = start.copy()
        self.stop_index = self.start_index + dimensions
        self.ActiveDimensions = dimensions.copy()

    def _setup_dx(self):
        # So first we figure out what the index is.  We don't assume
        # that dx=dy=dz , at least here.  We probably do elsewhere.
        id = self.id - self._id_offset
        if len(self.Parent) > 0:
            self.dds = self.Parent[0].dds / self.pf.refine_by
        else:
            LE, RE = self.hierarchy.grid_left_edge[id,:], \
                     self.hierarchy.grid_right_edge[id,:]
            self.dds = np.array((RE-LE)/self.ActiveDimensions)
        if self.pf.dimensionality < 2: self.dds[1] = 1.0
        if self.pf.dimensionality < 3: self.dds[2] = 1.0
        self.field_data['dx'], self.field_data['dy'], self.field_data['dz'] = self.dds

def parse_line(line, grid):
    # grid is a dictionary
    splitup = line.strip().split()
    if "vtk" in splitup:
        grid['vtk_version'] = splitup[-1]
    elif "Really" in splitup:
        grid['time'] = splitup[-1]
    elif any(x in ['PRIMITIVE','CONSERVED'] for x in splitup):
        grid['time'] = float(splitup[4].rstrip(','))
        grid['level'] = int(splitup[6].rstrip(','))
        grid['domain'] = int(splitup[8].rstrip(','))
    elif "DIMENSIONS" in splitup:
        grid['dimensions'] = np.array(splitup[-3:]).astype('int')
    elif "ORIGIN" in splitup:
        grid['left_edge'] = np.array(splitup[-3:]).astype('float64')
    elif "SPACING" in splitup:
        grid['dds'] = np.array(splitup[-3:]).astype('float64')
    elif "CELL_DATA" in splitup:
        grid["ncells"] = int(splitup[-1])
    elif "SCALARS" in splitup:
        field = splitup[1]
        grid['read_field'] = field
        grid['read_type'] = 'scalar'
    elif "VECTORS" in splitup:
        field = splitup[1]
        grid['read_field'] = field
        grid['read_type'] = 'vector'



class AthenaHierarchy(GridGeometryHandler):

    grid = AthenaGrid
    _data_style='athena'
    
    def __init__(self, pf, data_style='athena'):
        self.parameter_file = weakref.proxy(pf)
        self.data_style = data_style
        # for now, the hierarchy file is the parameter file!
        self.storage_filename = self.parameter_file.storage_filename
        self.hierarchy_filename = self.parameter_file.filename
        #self.directory = os.path.dirname(self.hierarchy_filename)
        self._fhandle = file(self.hierarchy_filename,'rb')
        AMRHierarchy.__init__(self, pf, data_style)

        self._fhandle.close()

    def _initialize_data_storage(self):
        pass

    def _detect_fields(self):
        field_map = {}
        f = open(self.hierarchy_filename,'rb')
        line = f.readline()
        while line != '':
            splitup = line.strip().split()
            if "DIMENSIONS" in splitup:
                grid_dims = np.array(splitup[-3:]).astype('int')
                line = f.readline()
            elif "CELL_DATA" in splitup:
                grid_ncells = int(splitup[-1])
                line = f.readline()
                if np.prod(grid_dims) != grid_ncells:
                    grid_dims -= 1
                    grid_dims[grid_dims==0]=1
                if np.prod(grid_dims) != grid_ncells:
                    mylog.error('product of dimensions %i not equal to number of cells %i' %
                          (np.prod(grid_dims), grid_ncells))
                    raise TypeError
                break
            else:
                line = f.readline()
        read_table = False
        read_table_offset = f.tell()
        while line != '':
            splitup = line.strip().split()
            if 'SCALARS' in splitup:
                field = splitup[1]
                if not read_table:
                    line = f.readline() # Read the lookup table line
                    read_table = True
                field_map[field] = ('scalar', f.tell() - read_table_offset)
                read_table=False

            elif 'VECTORS' in splitup:
                field = splitup[1]
                for ax in 'xyz':
                    field_map["%s_%s" % (field, ax)] =\
                            ('vector', f.tell() - read_table_offset)
            line = f.readline()

        f.close()

        self.field_list = field_map.keys()
        self._field_map = field_map

    def _setup_classes(self):
        dd = self._get_data_reader_dict()
        AMRHierarchy._setup_classes(self, dd)
        self.object_types.sort()

    def _count_grids(self):
        self.num_grids = self.parameter_file.nvtk

    def _parse_hierarchy(self):
        f = open(self.hierarchy_filename,'rb')
        grid = {}
        grid['read_field'] = None
        grid['read_type'] = None
        table_read=False
        line = f.readline()
        while grid['read_field'] is None:
            parse_line(line, grid)
            if "SCALAR" in line.strip().split():
                break
            if "VECTOR" in line.strip().split():
                break
            if 'TABLE' in line.strip().split():
                break
            if len(line) == 0: break
            line = f.readline()
        f.close()

        # It seems some datasets have a mismatch between ncells and 
        # the actual grid dimensions.
        if np.prod(grid['dimensions']) != grid['ncells']:
            grid['dimensions'] -= 1
            grid['dimensions'][grid['dimensions']==0]=1
        if np.prod(grid['dimensions']) != grid['ncells']:
            mylog.error('product of dimensions %i not equal to number of cells %i' % 
                  (np.prod(grid['dimensions']), grid['ncells']))
            raise TypeError

        # Need to determine how many grids: self.num_grids
        dname = self.hierarchy_filename
        gridlistread = glob.glob('id*/%s-id*%s' % (dname[4:-9],dname[-9:] ))
        gridlistread.insert(0,self.hierarchy_filename)
        self.num_grids = len(gridlistread)
        dxs=[]
        self.grids = np.empty(self.num_grids, dtype='object')
        levels = np.zeros(self.num_grids, dtype='int32')
        glis = np.empty((self.num_grids,3), dtype='float64')
        gdds = np.empty((self.num_grids,3), dtype='float64')
        gdims = np.ones_like(glis)
        j = 0
        while j < (self.num_grids):
            f = open(gridlistread[j],'rb')
            f.close()
            if j == 0:
                f = open(dname,'rb')
            if j != 0:
                f = open('id%i/%s-id%i%s' % (j, dname[4:-9],j, dname[-9:]),'rb')
            gridread = {}
            gridread['read_field'] = None
            gridread['read_type'] = None
            table_read=False
            line = f.readline()
            while gridread['read_field'] is None:
                parse_line(line, gridread)
                if "SCALAR" in line.strip().split():
                    break
                if "VECTOR" in line.strip().split():
                    break 
                if 'TABLE' in line.strip().split():
                    break
                if len(line) == 0: break
                line = f.readline()
            f.close()
            glis[j,0] = gridread['left_edge'][0]
            glis[j,1] = gridread['left_edge'][1]
            glis[j,2] = gridread['left_edge'][2]
            # It seems some datasets have a mismatch between ncells and 
            # the actual grid dimensions.
            if np.prod(gridread['dimensions']) != gridread['ncells']:
                gridread['dimensions'] -= 1
                gridread['dimensions'][gridread['dimensions']==0]=1
            if np.prod(gridread['dimensions']) != gridread['ncells']:
                mylog.error('product of dimensions %i not equal to number of cells %i' % 
                      (np.prod(gridread['dimensions']), gridread['ncells']))
                raise TypeError
            gdims[j,0] = gridread['dimensions'][0]
            gdims[j,1] = gridread['dimensions'][1]
            gdims[j,2] = gridread['dimensions'][2]
            # Setting dds=1 for non-active dimensions in 1D/2D datasets
            gridread['dds'][gridread['dimensions']==1] = 1.
            gdds[j,:] = gridread['dds']
            
            j=j+1

        gres = glis + gdims*gdds
        # Now we convert the glis, which were left edges (floats), to indices 
        # from the domain left edge.  Then we do a bunch of fixing now that we
        # know the extent of all the grids. 
        glis = np.round((glis - self.parameter_file.domain_left_edge)/gdds).astype('int')
        new_dre = np.max(gres,axis=0)
        self.parameter_file.domain_right_edge = np.round(new_dre, decimals=6)
        self.parameter_file.domain_width = \
                (self.parameter_file.domain_right_edge - 
                 self.parameter_file.domain_left_edge)
        self.parameter_file.domain_center = \
                0.5*(self.parameter_file.domain_left_edge + 
                     self.parameter_file.domain_right_edge)
        self.parameter_file.domain_dimensions = \
                np.round(self.parameter_file.domain_width/gdds[0]).astype('int')
        if self.parameter_file.dimensionality <= 2 :
            self.parameter_file.domain_dimensions[2] = np.int(1)
        if self.parameter_file.dimensionality == 1 :
            self.parameter_file.domain_dimensions[1] = np.int(1)
        for i in range(levels.shape[0]):
            self.grids[i] = self.grid(i,self,levels[i],
                                      glis[i],
                                      gdims[i])
            dx = (self.parameter_file.domain_right_edge-
                  self.parameter_file.domain_left_edge)/self.parameter_file.domain_dimensions
            dx = dx/self.parameter_file.refine_by**(levels[i])
            dxs.append(dx)
        
        dx = np.array(dxs)
        self.grid_left_edge = np.round(self.parameter_file.domain_left_edge + dx*glis, decimals=6)
        self.grid_dimensions = gdims.astype("int32")
        self.grid_right_edge = np.round(self.grid_left_edge + dx*self.grid_dimensions, decimals=6)
        self.grid_particle_count = np.zeros([self.num_grids, 1], dtype='int64')

    def _populate_grid_objects(self):
        for g in self.grids:
            g._prepare_grid()
            g._setup_dx()

        for g in self.grids:
            g.Children = self._get_grid_children(g)
            for g1 in g.Children:
                g1.Parent.append(g)
        self.max_level = self.grid_levels.max()

    def _get_grid_children(self, grid):
        mask = np.zeros(self.num_grids, dtype='bool')
        grids, grid_ind = self.get_box_grids(grid.LeftEdge, grid.RightEdge)
        mask[grid_ind] = True
        return [g for g in self.grids[mask] if g.Level == grid.Level + 1]

class AthenaStaticOutput(StaticOutput):
    _hierarchy_class = AthenaHierarchy
    _fieldinfo_fallback = AthenaFieldInfo
    _fieldinfo_known = KnownAthenaFields
    _data_style = "athena"

    def __init__(self, filename, data_style='athena',
                 storage_filename = None, parameters = {}):
        self.specified_parameters = parameters
        StaticOutput.__init__(self, filename, data_style)
        self.filename = filename
        self.storage_filename = filename[4:-4]
        
        # Unfortunately we now have to mandate that the hierarchy gets 
        # instantiated so that we can make sure we have the correct left 
        # and right domain edges.
        self.h

    def _set_units(self):
        """
        Generates the conversion to various physical _units based on the parameter file
        """
        self.units = {}
        self.time_units = {}
        if len(self.parameters) == 0:
            self._parse_parameter_file()
        self._setup_nounits_units()
        self.conversion_factors = defaultdict(lambda: 1.0)
        self.time_units['1'] = 1
        self.units['1'] = 1.0
        self.units['unitary'] = 1.0 / (self.domain_right_edge - self.domain_left_edge).max()

    def _setup_nounits_units(self):
        self.conversion_factors["Time"] = 1.0
        for unit in mpc_conversion.keys():
            self.units[unit] = mpc_conversion[unit] / mpc_conversion["cm"]

    def _parse_parameter_file(self):
        self._handle = open(self.parameter_filename, "rb")
        # Read the start of a grid to get simulation parameters.
        grid = {}
        grid['read_field'] = None
        line = self._handle.readline()
        while grid['read_field'] is None:
            parse_line(line, grid)
            if "SCALAR" in line.strip().split():
                break
            if "VECTOR" in line.strip().split():
                break
            if 'TABLE' in line.strip().split():
                break
            if len(line) == 0: break
            line = self._handle.readline()

        self.domain_left_edge = grid['left_edge']
        mylog.info("Temporarily setting domain_right_edge = -domain_left_edge."+
                  " This will be corrected automatically if it is not the case.")
        self.domain_right_edge = -self.domain_left_edge
        self.domain_width = self.domain_right_edge-self.domain_left_edge
        self.domain_dimensions = np.round(self.domain_width/grid['dds']).astype('int32')
        refine_by = None
        if refine_by is None: refine_by = 2
        self.refine_by = refine_by
        dimensionality = 3
        if grid['dimensions'][2] == 1 :
            dimensionality = 2
        if grid['dimensions'][1] == 1 :
            dimensionality = 1
        if dimensionality <= 2 : self.domain_dimensions[2] = np.int32(1)
        if dimensionality == 1 : self.domain_dimensions[1] = np.int32(1)
        self.dimensionality = dimensionality
        self.current_time = grid["time"]
        self.unique_identifier = self._handle.__hash__()
        self.cosmological_simulation = False
        self.num_ghost_zones = 0
        self.field_ordering = 'fortran'
        self.boundary_conditions = [1]*6

        dname = self.parameter_filename
        gridlistread = glob.glob('id*/%s-id*%s' % (dname[4:-9],dname[-9:] ))
        self.nvtk = len(gridlistread)+1 

        self.current_redshift = self.omega_lambda = self.omega_matter = \
            self.hubble_constant = self.cosmological_simulation = 0.0
        self.parameters['Time'] = self.current_time # Hardcode time conversion for now.
        self.parameters["HydroMethod"] = 0 # Hardcode for now until field staggering is supported.
        self._handle.close()


    @classmethod
    def _is_valid(self, *args, **kwargs):
        try:
            if 'vtk' in args[0]:
                return True
        except:
            pass
        return False

    def __repr__(self):
        return self.basename.rsplit(".", 1)[0]

