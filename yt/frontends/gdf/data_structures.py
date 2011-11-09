"""
Data structures for Chombo.

Author: Matthew Turk <matthewturk@gmail.com>
Author: J. S. Oishi <jsoishi@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
Homepage: http://yt-project.org/
License:
  Copyright (C) 2008-2011 Matthew Turk, J. S. Oishi.  All Rights Reserved.

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
import numpy as na
import weakref
from yt.funcs import *
from yt.data_objects.grid_patch import \
           AMRGridPatch
from yt.data_objects.hierarchy import \
           AMRHierarchy
from yt.data_objects.static_output import \
           StaticOutput

from .fields import GDFFieldContainer
import pdb

class GDFGrid(AMRGridPatch):
    _id_offset = 0
    def __init__(self, id, hierarchy, level, start, dimensions):
        AMRGridPatch.__init__(self, id, filename = hierarchy.hierarchy_filename,
                              hierarchy = hierarchy)
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
            self.dds = na.array((RE-LE)/self.ActiveDimensions)
        if self.pf.dimensionality < 2: self.dds[1] = 1.0
        if self.pf.dimensionality < 3: self.dds[2] = 1.0
        self.field_data['dx'], self.field_data['dy'], self.field_data['dz'] = self.dds

class GDFHierarchy(AMRHierarchy):

    grid = GDFGrid
    
    def __init__(self, pf, data_style='grid_data_format'):
        self.parameter_file = weakref.proxy(pf)
        self.data_style = data_style
        # for now, the hierarchy file is the parameter file!
        self.hierarchy_filename = self.parameter_file.parameter_filename
        self.directory = os.path.dirname(self.hierarchy_filename)
        self._fhandle = h5py.File(self.hierarchy_filename)
        AMRHierarchy.__init__(self,pf,data_style)

        self._fhandle.close()

    def _initialize_data_storage(self):
        pass

    def _detect_fields(self):
        self.field_list = self._fhandle['field_types'].keys()
    
    def _setup_classes(self):
        dd = self._get_data_reader_dict()
        AMRHierarchy._setup_classes(self, dd)
        self.object_types.sort()

    def _count_grids(self):
        self.num_grids = self._fhandle['/grid_parent_id'].shape[0]
        
    def _parse_hierarchy(self):
        f = self._fhandle 
        
        # this relies on the first Group in the H5 file being
        # 'Chombo_global'
        levels = f.listnames()[1:]
        dxs=[]
        self.grids = na.empty(self.num_grids, dtype='object')
        for i, grid in enumerate(f['data'].keys()):
            self.grids[i] = self.grid(i, self, f['grid_level'][i],
                                      f['grid_left_index'][i],
                                      f['grid_dimensions'][i])
            self.grids[i]._level_id = f['grid_level'][i]

            dx = (self.parameter_file.domain_right_edge-
                  self.parameter_file.domain_left_edge)/self.parameter_file.domain_dimensions
            dx = dx/self.parameter_file.refine_by**(f['grid_level'][i])
            dxs.append(dx)
        dx = na.array(dxs)
        self.grid_left_edge = self.parameter_file.domain_left_edge + dx*f['grid_left_index'][:]
        self.grid_dimensions = f['grid_dimensions'][:].astype("int32")
        self.grid_right_edge = self.grid_left_edge + dx*self.grid_dimensions
        self.grid_particle_count = f['grid_particle_count'][:]

    def _populate_grid_objects(self):
        for g in self.grids:
            g._prepare_grid()
            g._setup_dx()

        for g in self.grids:
            g.Children = self._get_grid_children(g)
            for g1 in g.Children:
                g1.Parent.append(g)
        self.max_level = self.grid_levels.max()

    def _setup_unknown_fields(self):
        pass

    def _setup_derived_fields(self):
        self.derived_field_list = []

    def _get_grid_children(self, grid):
        mask = na.zeros(self.num_grids, dtype='bool')
        grids, grid_ind = self.get_box_grids(grid.LeftEdge, grid.RightEdge)
        mask[grid_ind] = True
        return [g for g in self.grids[mask] if g.Level == grid.Level + 1]

class GDFStaticOutput(StaticOutput):
    _hierarchy_class = GDFHierarchy
    _fieldinfo_class = GDFFieldContainer
    
    def __init__(self, filename, data_style='grid_data_format',
                 storage_filename = None):
        StaticOutput.__init__(self, filename, data_style)
        self.storage_filename = storage_filename
        self.filename = filename
        self.field_info = self._fieldinfo_class()        
        
    def _set_units(self):
        """
        Generates the conversion to various physical _units based on the parameter file
        """
        self.units = {}
        self.time_units = {}
        if len(self.parameters) == 0:
            self._parse_parameter_file()
        self.time_units['1'] = 1
        self.units['1'] = 1.0
        self.units['cm'] = 1.0
        self.units['unitary'] = 1.0 / (self.domain_right_edge - self.domain_left_edge).max()
        seconds = 1
        self.time_units['years'] = seconds / (365*3600*24.0)
        self.time_units['days']  = seconds / (3600*24.0)
        # This should be improved.
        self._handle = h5py.File(self.parameter_filename, "r")
        for field_name in self._handle["/field_types"]:
            self.units[field_name] = self._handle["/field_types/%s" % field_name].attrs['field_to_cgs']
        del self._handle
        
    def _parse_parameter_file(self):
        self._handle = h5py.File(self.parameter_filename, "r")
        sp = self._handle["/simulation_parameters"].attrs
        self.domain_left_edge = sp["domain_left_edge"][:]
        self.domain_right_edge = sp["domain_right_edge"][:]
        self.domain_dimensions = sp["domain_dimensions"][:]
        self.refine_by = sp["refine_by"]
        self.dimensionality = sp["dimensionality"]
        self.current_time = sp["current_time"]
        self.unique_identifier = sp["unique_identifier"]
        self.cosmological_simulation = sp["cosmological_simulation"]
        if sp["num_ghost_zones"] != 0: raise RuntimeError
        self.num_ghost_zones = sp["num_ghost_zones"]
        self.field_ordering = sp["field_ordering"]
        self.boundary_conditions = sp["boundary_conditions"][:]
        if self.cosmological_simulation:
            self.current_redshift = sp["current_redshift"]
            self.omega_lambda = sp["omega_lambda"]
            self.omega_matter = sp["omega_matter"]
            self.hubble_constant = sp["hubble_constant"]
        else:
            self.current_redshift = self.omega_lambda = self.omega_matter = \
                self.hubble_constant = self.cosmological_simulation = 0.0
        self.parameters["HydroMethod"] = 0 # Hardcode for now until field staggering is supported.
        
        del self._handle
            
    @classmethod
    def _is_valid(self, *args, **kwargs):
        try:
            fileh = h5py.File(args[0],'r')
            if "gridded_data_format" in fileh:
                return True
        except:
            pass
        return False

    def __repr__(self):
        return self.basename.rsplit(".", 1)[0]
        
