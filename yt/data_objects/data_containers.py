"""
Various non-grid data containers.

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
Author: Britton Smith <Britton.Smith@colorado.edu>
Affiliation: University of Colorado at Boulder
Homepage: http://yt-project.org/
License:
  Copyright (C) 2007-2011 Matthew Turk.  All Rights Reserved.

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
from yt.data_objects.selection_data_containers import YTSelectedIndicesBase, YTValueCutExtractionBase, YTBooleanRegionBase

data_object_registry = {}

import numpy as na
import weakref
import shelve

from yt.funcs import *

from yt.data_objects.particle_io import particle_handler_registry
from yt.utilities.amr_utils import\
\
    march_cubes_grid, march_cubes_grid_flux\
from yt.utilities.definitions import  x_dict, y_dict
from yt.utilities.parallel_tools.parallel_analysis_interface import \
    ParallelAnalysisInterface
from yt.utilities.parameter_file_storage import \
    ParameterFileStore

from .derived_quantities import DerivedQuantityCollection
from .field_info_container import \
    NeedsGridType

def force_array(item, shape):
    try:
        sh = item.shape
        return item
    except AttributeError:
        if item:
            return na.ones(shape, dtype='bool')
        else:
            return na.zeros(shape, dtype='bool')

def restore_grid_state(func):
    """
    A decorator that takes a function with the API of (self, grid, field)
    and ensures that after the function is called, the field_parameters will
    be returned to normal.
    """
    def save_state(self, grid, field=None, *args, **kwargs):
        old_params = grid.field_parameters
        old_keys = grid.field_data.keys()
        grid.field_parameters = self.field_parameters
        tr = func(self, grid, field, *args, **kwargs)
        grid.field_parameters = old_params
        grid.field_data = YTFieldData( [(k, grid.field_data[k]) for k in old_keys] )
        return tr
    return save_state

def restore_field_information_state(func):
    """
    A decorator that takes a function with the API of (self, grid, field)
    and ensures that after the function is called, the field_parameters will
    be returned to normal.
    """
    def save_state(self, grid, field=None, *args, **kwargs):
        old_params = grid.field_parameters
        grid.field_parameters = self.field_parameters
        tr = func(self, grid, field, *args, **kwargs)
        grid.field_parameters = old_params
        return tr
    return save_state

def cache_mask(func):
    """
    For computationally intensive indexing operations, we can cache
    between calls.
    """
    def check_cache(self, grid):
        if isinstance(grid, FakeGridForParticles):
            return func(self, grid)
        elif grid.id not in self._cut_masks or \
                hasattr(self, "_boolean_touched"):
            cm = func(self, grid)
            self._cut_masks[grid.id] = cm
        return self._cut_masks[grid.id]
    return check_cache

def cache_point_indices(func):
    """
    For computationally intensive indexing operations, we can cache
    between calls.
    """
    def check_cache(self, grid, use_child_mask=True):
        if isinstance(grid, FakeGridForParticles):
            return func(self, grid, use_child_mask)
        elif grid.id not in self._point_indices:
            cm = func(self, grid, use_child_mask)
            self._point_indices[grid.id] = cm
        return self._point_indices[grid.id]
    return check_cache

def cache_vc_data(func):
    """
    For computationally intensive operations, we can cache between
    calls.
    """
    def check_cache(self, grid, field):
        if isinstance(grid, FakeGridForParticles):
            return func(self, grid, field)
        elif grid.id not in self._vc_data[field]:
            vc = func(self, grid, field)
            self._vc_data[field][grid.id] = vc
        return self._vc_data[field][grid.id]
    return check_cache

class YTFieldData(dict):
    """
    A Container object for field data, instead of just having it be a dict.
    """
    pass

class FakeGridForParticles(object):
    """
    Mock up a grid to insert particle positions and radii
    into for purposes of confinement in an :class:`YTSelectionContainer3D`.
    """
    def __init__(self, grid):
        self._corners = grid._corners
        self.field_parameters = {}
        self.field_data = YTFieldData({'x':grid['particle_position_x'],
                                       'y':grid['particle_position_y'],
                                       'z':grid['particle_position_z'],
                                       'dx':grid['dx'],
                                       'dy':grid['dy'],
                                       'dz':grid['dz']})
        self.dds = grid.dds.copy()
        self.real_grid = grid
        self.child_mask = 1
        self.ActiveDimensions = self.field_data['x'].shape
        self.DW = grid.pf.domain_right_edge - grid.pf.domain_left_edge
        
    def __getitem__(self, field):
        if field not in self.field_data.keys():
            if field == "RadiusCode":
                center = self.field_parameters['center']
                tempx = na.abs(self['x'] - center[0])
                tempx = na.minimum(tempx, self.DW[0] - tempx)
                tempy = na.abs(self['y'] - center[1])
                tempy = na.minimum(tempy, self.DW[1] - tempy)
                tempz = na.abs(self['z'] - center[2])
                tempz = na.minimum(tempz, self.DW[2] - tempz)
                tr = na.sqrt( tempx**2.0 + tempy**2.0 + tempz**2.0 )
            else:
                raise KeyError(field)
        else: tr = self.field_data[field]
        return tr

class YTDataContainer(object):
    """
    Generic YTDataContainer container.  By itself, will attempt to
    generate field, read fields (method defined by derived classes)
    and deal with passing back and forth field parameters.
    """
    _grids = None
    _num_ghost_zones = 0
    _con_args = ()
    _skip_add = False

    class __metaclass__(type):
        def __init__(cls, name, b, d):
            type.__init__(cls, name, b, d)
            if hasattr(cls, "_type_name") and not cls._skip_add:
                data_object_registry[cls._type_name] = cls

    def __init__(self, pf, fields, **kwargs):
        """
        Typically this is never called directly, but only due to inheritance.
        It associates a :class:`~yt.data_objects.api.StaticOutput` with the class,
        sets its initial set of fields, and the remainder of the arguments
        are passed as field_parameters.
        """
        if pf != None:
            self.pf = pf
            self.hierarchy = pf.hierarchy
        self.hierarchy.objects.append(weakref.proxy(self))
        mylog.debug("Appending object to %s (type: %s)", self.pf, type(self))
        if fields == None: fields = []
        self.fields = ensure_list(fields)[:]
        self.field_data = YTFieldData()
        self.field_parameters = {}
        self.__set_default_field_parameters()
        self._cut_masks = {}
        self._point_indices = {}
        self._vc_data = {}
        for key, val in kwargs.items():
            mylog.debug("Setting %s to %s", key, val)
            self.set_field_parameter(key, val)

    def __set_default_field_parameters(self):
        self.set_field_parameter("center",na.zeros(3,dtype='float64'))
        self.set_field_parameter("bulk_velocity",na.zeros(3,dtype='float64'))

    def _set_center(self, center):
        if center is None:
            pass
        elif isinstance(center, (types.ListType, types.TupleType, na.ndarray)):
            center = na.array(center)
        elif center == ("max"): # is this dangerous for race conditions?
            center = self.pf.h.find_max("Density")[1]
        elif center.startswith("max_"):
            center = self.pf.h.find_max(center[4:])[1]
        else:
            center = na.array(center, dtype='float64')
        self.center = center
        self.set_field_parameter('center', center)

    def get_field_parameter(self, name, default=None):
        """
        This is typically only used by derived field functions, but
        it returns parameters used to generate fields.
        """
        if self.field_parameters.has_key(name):
            return self.field_parameters[name]
        else:
            return default

    def set_field_parameter(self, name, val):
        """
        Here we set up dictionaries that get passed up and down and ultimately
        to derived fields.
        """
        self.field_parameters[name] = val

    def has_field_parameter(self, name):
        """
        Checks if a field parameter is set.
        """
        return self.field_parameters.has_key(name)

    def convert(self, datatype):
        """
        This will attempt to convert a given unit to cgs from code units.
        It either returns the multiplicative factor or throws a KeyError.
        """
        return self.pf[datatype]

    def clear_data(self):
        """
        Clears out all data from the YTDataContainer instance, freeing memory.
        """
        self.field_data.clear()
        if self._grids is not None:
            for grid in self._grids: grid.clear_data()

    def clear_cache(self):
        """
        Clears out all cache, freeing memory.
        """
        for _cm in self._cut_masks: del _cm
        for _pi in self._point_indices: del _pi
        for _field in self._vc_data:
            for _vc in _field: del _vc

    def has_key(self, key):
        """
        Checks if a data field already exists.
        """
        return self.field_data.has_key(key)

    def _refresh_data(self):
        """
        Wipes data and rereads/regenerates it from the self.fields.
        """
        self.clear_data()
        self.get_data()

    def keys(self):
        return self.field_data.keys()

    def __getitem__(self, key):
        """
        Returns a single field.  Will add if necessary.
        """
        if not self.field_data.has_key(key):
            if key not in self.fields:
                self.fields.append(key)
            self.get_data(key)
        return self.field_data[key]

    def __setitem__(self, key, val):
        """
        Sets a field to be some other value.
        """
        if key not in self.fields: self.fields.append(key)
        self.field_data[key] = val

    def __delitem__(self, key):
        """
        Deletes a field
        """
        try:
            del self.fields[self.fields.index(key)]
        except ValueError:
            pass
        del self.field_data[key]

    def _generate_field(self, field):
        if self.pf.field_info.has_key(field):
            # First we check the validator
            try:
                self.pf.field_info[field].check_available(self)
            except NeedsGridType, ngt_exception:
                # We leave this to be implementation-specific
                self._generate_field_in_grids(field, ngt_exception.ghost_zones)
                return False
            else:
                self[field] = self.pf.field_info[field](self)
                return True
        else: # Can't find the field, try as it might
            raise KeyError(field)

    def _generate_field_in_grids(self, field, num_ghost_zones=0):
        for grid in self._grids:
            grid[field] = self.__touch_grid_field(grid, field)

    @restore_grid_state
    def __touch_grid_field(self, grid, field):
        return grid[field]

    _key_fields = None
    def write_out(self, filename, fields=None, format="%0.16e"):
        if fields is None: fields=sorted(self.field_data.keys())
        if self._key_fields is None: raise ValueError
        field_order = self._key_fields[:]
        for field in field_order: self[field]
        field_order += [field for field in fields if field not in field_order]
        fid = open(filename,"w")
        fid.write("\t".join(["#"] + field_order + ["\n"]))
        field_data = na.array([self.field_data[field] for field in field_order])
        for line in range(field_data.shape[1]):
            field_data[:,line].tofile(fid, sep="\t", format=format)
            fid.write("\n")
        fid.close()

    def save_object(self, name, filename = None):
        """
        Save an object.  If *filename* is supplied, it will be stored in
        a :mod:`shelve` file of that name.  Otherwise, it will be stored via
        :meth:`yt.data_objects.api.AMRHierarchy.save_object`.
        """
        if filename is not None:
            ds = shelve.open(filename, protocol=-1)
            if name in ds:
                mylog.info("Overwriting %s in %s", name, filename)
            ds[name] = self
            ds.close()
        else:
            self.hierarchy.save_object(self, name)

    def __reduce__(self):
        args = tuple([self.pf._hash(), self._type_name] +
                     [getattr(self, n) for n in self._con_args] +
                     [self.field_parameters])
        return (_reconstruct_object, args)

    def __repr__(self, clean = False):
        # We'll do this the slow way to be clear what's going on
        if clean: s = "%s: " % (self.__class__.__name__)
        else: s = "%s (%s): " % (self.__class__.__name__, self.pf)
        s += ", ".join(["%s=%s" % (i, getattr(self,i))
                       for i in self._con_args])
        return s

class GridPropertiesMixin(object):

    def select_grids(self, level):
        """
        Return all grids on a given level.
        """
        grids = [g for g in self._grids if g.Level == level]
        return grids

    def select_grid_indices(self, level):
        return na.where(self.grid_levels == level)

    def __get_grid_left_edge(self):
        if self.__grid_left_edge == None:
            self.__grid_left_edge = na.array([g.LeftEdge for g in self._grids])
        return self.__grid_left_edge

    def __del_grid_left_edge(self):
        del self.__grid_left_edge
        self.__grid_left_edge = None

    def __set_grid_left_edge(self, val):
        self.__grid_left_edge = val

    __grid_left_edge = None
    grid_left_edge = property(__get_grid_left_edge, __set_grid_left_edge,
                              __del_grid_left_edge)

    def __get_grid_right_edge(self):
        if self.__grid_right_edge == None:
            self.__grid_right_edge = na.array([g.RightEdge for g in self._grids])
        return self.__grid_right_edge

    def __del_grid_right_edge(self):
        del self.__grid_right_edge
        self.__grid_right_edge = None

    def __set_grid_right_edge(self, val):
        self.__grid_right_edge = val

    __grid_right_edge = None
    grid_right_edge = property(__get_grid_right_edge, __set_grid_right_edge,
                             __del_grid_right_edge)

    def __get_grid_levels(self):
        if self.__grid_levels == None:
            self.__grid_levels = na.array([g.Level for g in self._grids])
        return self.__grid_levels

    def __del_grid_levels(self):
        del self.__grid_levels
        self.__grid_levels = None

    def __set_grid_levels(self, val):
        self.__grid_levels = val

    __grid_levels = None
    grid_levels = property(__get_grid_levels, __set_grid_levels,
                             __del_grid_levels)


    def __get_grid_dimensions(self):
        if self.__grid_dimensions == None:
            self.__grid_dimensions = na.array([g.ActiveDimensions for g in self._grids])
        return self.__grid_dimensions

    def __del_grid_dimensions(self):
        del self.__grid_dimensions
        self.__grid_dimensions = None

    def __set_grid_dimensions(self, val):
        self.__grid_dimensions = val

    __grid_dimensions = None
    grid_dimensions = property(__get_grid_dimensions, __set_grid_dimensions,
                             __del_grid_dimensions)

class YTSelectionContainer1D(YTDataContainer, GridPropertiesMixin):
    _spatial = False
    def __init__(self, pf, fields, **kwargs):
        YTDataContainer.__init__(self, pf, fields, **kwargs)
        self._grids = None
        self._sortkey = None
        self._sorted = {}

    def get_data(self, fields=None, in_grids=False):
        if self._grids == None:
            self._get_list_of_grids()
        points = []
        if not fields:
            fields_to_get = self.fields[:]
        else:
            fields_to_get = ensure_list(fields)
        if not self.sort_by in fields_to_get and \
            self.sort_by not in self.field_data:
            fields_to_get.insert(0, self.sort_by)
        mylog.debug("Going to obtain %s", fields_to_get)
        for field in fields_to_get:
            if self.field_data.has_key(field):
                continue
            mylog.info("Getting field %s from %s", field, len(self._grids))
            if field not in self.hierarchy.field_list and not in_grids:
                if field not in ("dts", "t") and self._generate_field(field):
                    continue # True means we already assigned it
            self[field] = na.concatenate(
                [self._get_data_from_grid(grid, field)
                 for grid in self._grids])
            if not self.field_data.has_key(field):
                continue
            if self._sortkey is None:
                self._sortkey = na.argsort(self[self.sort_by])
            # We *always* sort the field here if we have not successfully
            # generated it above.  This way, fields that are grabbed from the
            # grids are sorted properly.
            self[field] = self[field][self._sortkey]


class YTSelectionContainer2D(YTDataContainer, GridPropertiesMixin, ParallelAnalysisInterface):
    _key_fields = ['px','py','pdx','pdy']
    """
    Class to represent a set of :class:`YTDataContainer` that's 2-D in nature, and
    thus does not have as many actions as the 3-D data types.
    """
    _spatial = False
    def __init__(self, axis, fields, pf=None, **kwargs):
        """
        Prepares the YTSelectionContainer2D, normal to *axis*.  If *axis* is 4, we are not
        aligned with any axis.
        """
        ParallelAnalysisInterface.__init__(self)
        self.axis = axis
        YTDataContainer.__init__(self, pf, fields, **kwargs)
        self.field = ensure_list(fields)[0]
        self.set_field_parameter("axis",axis)
        
    def _convert_field_name(self, field):
        return field

    #@time_execution
    def get_data(self, fields = None):
        """
        Iterates over the list of fields and generates/reads them all.
        """
        # We get it for the values in fields and coords
        # We take a 3-tuple of the coordinate we want to slice through, as well
        # as the axis we're slicing along
        self._get_list_of_grids()
        if not self.has_key('pdx'):
            self._generate_coords()
        if fields == None:
            fields_to_get = self.fields[:]
        else:
            fields_to_get = ensure_list(fields)
        temp_data = {}
        for field in fields_to_get:
            if self.field_data.has_key(field): continue
            if field not in self.hierarchy.field_list:
                if self._generate_field(field):
                    continue # A "True" return means we did it
            # To ensure that we use data from this object as much as possible,
            # we're going to have to set the same thing several times
            data = [self._get_data_from_grid(grid, field)
                    for grid in self._get_grids()]
            if len(data) == 0: data = na.array([])
            else: data = na.concatenate(data)
            temp_data[field] = data
            # Now the next field can use this field
            self[field] = temp_data[field] 
        # We finalize
        if temp_data != {}:
            temp_data = self.comm.par_combine_object(temp_data,
                    datatype='dict', op='cat')
        # And set, for the next group
        for field in temp_data.keys():
            self[field] = temp_data[field]

    def to_frb(self, width, resolution, center = None):
        r"""This function returns a FixedResolutionBuffer generated from this
        object.

        A FixedResolutionBuffer is an object that accepts a variable-resolution
        2D object and transforms it into an NxM bitmap that can be plotted,
        examined or processed.  This is a convenience function to return an FRB
        directly from an existing 2D data object.

        Parameters
        ----------
        width : width specifier
            This can either be a floating point value, in the native domain
            units of the simulation, or a tuple of the (value, unit) style.
            This will be the width of the FRB.
        resolution : int or tuple of ints
            The number of pixels on a side of the final FRB.
        center : array-like of floats, optional
            The center of the FRB.  If not specified, defaults to the center of
            the current object.

        Returns
        -------
        frb : :class:`~yt.visualization.fixed_resolution.FixedResolutionBuffer`
            A fixed resolution buffer, which can be queried for fields.

        Examples
        --------

        >>> proj = pf.h.proj(0, "Density")
        >>> frb = proj.to_frb( (100.0, 'kpc'), 1024)
        >>> write_image(na.log10(frb["Density"]), 'density_100kpc.png')
        """
        if center is None:
            center = self.get_field_parameter("center")
            if center is None:
                center = (self.pf.domain_right_edge
                        + self.pf.domain_left_edge)/2.0
        if iterable(width):
            w, u = width
            width = w/self.pf[u]
        if not iterable(resolution):
            resolution = (resolution, resolution)
        from yt.visualization.fixed_resolution import FixedResolutionBuffer
        xax = x_dict[self.axis]
        yax = y_dict[self.axis]
        bounds = (center[xax] - width/2.0, center[xax] + width/2.0,
                  center[yax] - width/2.0, center[yax] + width/2.0)
        frb = FixedResolutionBuffer(self, bounds, resolution)
        return frb

    def interpolate_discretize(self, LE, RE, field, side, log_spacing=True):
        """
        This returns a uniform grid of points between *LE* and *RE*,
        interpolated using the nearest neighbor method, with *side* points on a
        side.
        """
        import yt.utilities.delaunay as de
        if log_spacing:
            zz = na.log10(self[field])
        else:
            zz = self[field]
        xi, yi = na.array( \
                 na.mgrid[LE[0]:RE[0]:side*1j, \
                          LE[1]:RE[1]:side*1j], 'float64')
        zi = de.Triangulation(self['px'],self['py']).nn_interpolator(zz)\
                 [LE[0]:RE[0]:side*1j, \
                  LE[1]:RE[1]:side*1j]
        if log_spacing:
            zi = 10**(zi)
        return [xi,yi,zi]

    _okay_to_serialize = True

    def _store_fields(self, fields, node_name = None, force = False):
        fields = ensure_list(fields)
        if node_name is None: node_name = self._gen_node_name()
        for field in fields:
            #mylog.debug("Storing %s in node %s",
                #self._convert_field_name(field), node_name)
            self.hierarchy.save_data(self[field], node_name,
                self._convert_field_name(field), force = force,
                passthrough = True)

    def _obtain_fields(self, fields, node_name = None):
        if not self._okay_to_serialize: return
        fields = ensure_list(fields)
        if node_name is None: node_name = self._gen_node_name()
        for field in fields:
            #mylog.debug("Trying to obtain %s from node %s",
                #self._convert_field_name(field), node_name)
            fdata=self.hierarchy.get_data(node_name, 
                self._convert_field_name(field))
            if fdata is not None:
                #mylog.debug("Got %s from node %s", field, node_name)
                self[field] = fdata[:]
        return True

    def _deserialize(self, node_name = None):
        if not self._okay_to_serialize: return
        self._obtain_fields(self._key_fields, node_name)
        self._obtain_fields(self.fields, node_name)

    def _serialize(self, node_name = None, force = False):
        if not self._okay_to_serialize: return
        self._store_fields(self._key_fields, node_name, force)
        self._store_fields(self.fields, node_name, force)


class YTSelectionContainer3D(YTDataContainer, GridPropertiesMixin, ParallelAnalysisInterface):
    _key_fields = ['x','y','z','dx','dy','dz']
    """
    Class describing a cluster of data points, not necessarily sharing any
    particular attribute.
    """
    _spatial = False
    _num_ghost_zones = 0
    def __init__(self, center, fields, pf = None, **kwargs):
        """
        Returns an instance of YTSelectionContainer3D, or prepares one.  Usually only
        used as a base class.  Note that *center* is supplied, but only used
        for fields and quantities that require it.
        """
        ParallelAnalysisInterface.__init__(self)
        YTDataContainer.__init__(self, pf, fields, **kwargs)
        self._set_center(center)
        self.coords = None
        self._grids = None

    def _generate_coords(self):
        mylog.info("Generating coords for %s grids", len(self._grids))
        points = []
        for i,grid in enumerate(self._grids):
            #grid._generate_coords()
            if ( (i%100) == 0):
                mylog.info("Working on % 7i / % 7i", i, len(self._grids))
            grid.set_field_parameter("center", self.center)
            points.append((na.ones(
                grid.ActiveDimensions,dtype='float64')*grid['dx'])\
                    [self._get_point_indices(grid)])
            t = na.concatenate([t,points])
            del points
        self['dx'] = t
        #self['dy'] = t
        #self['dz'] = t
        mylog.info("Done with coordinates")

    @restore_grid_state
    def _generate_grid_coords(self, grid, field=None):
        pointI = self._get_point_indices(grid)
        dx = na.ones(pointI[0].shape[0], 'float64') * grid.dds[0]
        tr = na.array([grid['x'][pointI].ravel(), \
                grid['y'][pointI].ravel(), \
                grid['z'][pointI].ravel(), \
                grid["RadiusCode"][pointI].ravel(),
                dx, grid["GridIndices"][pointI].ravel()], 'float64').swapaxes(0,1)
        return tr

    def get_data(self, fields=None, in_grids=False, force_particle_read = False):
        if self._grids == None:
            self._get_list_of_grids()
        points = []
        if not fields:
            fields_to_get = self.fields[:]
        else:
            fields_to_get = ensure_list(fields)
        mylog.debug("Going to obtain %s", fields_to_get)
        for field in fields_to_get:
            if self.field_data.has_key(field):
                continue
            if field not in self.hierarchy.field_list and not in_grids:
                if self._generate_field(field):
                    continue # True means we already assigned it
            # There are a lot of 'ands' here, but I think they are all
            # necessary.
            if force_particle_read == False and \
               self.pf.field_info.has_key(field) and \
               self.pf.field_info[field].particle_type and \
               self.pf.h.io._particle_reader and \
               not isinstance(self, YTBooleanRegionBase):
                self.particles.get_data(field)
                if field not in self.field_data:
                    if self._generate_field(field): continue
            mylog.info("Getting field %s from %s", field, len(self._grids))
            self[field] = na.concatenate(
                [self._get_data_from_grid(grid, field)
                 for grid in self._grids])
        for field in fields_to_get:
            if not self.field_data.has_key(field):
                continue
            self[field] = self[field]

    @restore_grid_state
    def _get_data_from_grid(self, grid, field):
        if field in self.pf.field_info and self.pf.field_info[field].particle_type:
            # int64 -> float64 with the first real set of data
            if grid.NumberOfParticles == 0: return na.array([], dtype='int64')
            pointI = self._get_particle_indices(grid)
            if self.pf.field_info[field].vector_field:
                f = grid[field]
                return na.array([f[i,:][pointI] for i in range(3)])
            if self._is_fully_enclosed(grid): return grid[field].ravel()
            return grid[field][pointI].ravel()
        if field in self.pf.field_info and self.pf.field_info[field].vector_field:
            pointI = self._get_point_indices(grid)
            f = grid[field]
            return na.array([f[i,:][pointI] for i in range(3)])
        else:
            tr = grid[field]
            if tr.size == 1: # dx, dy, dz, cellvolume
                tr = tr * na.ones(grid.ActiveDimensions, dtype='float64')
            if len(grid.Children) == 0 and grid.OverlappingSiblings is None \
                and self._is_fully_enclosed(grid):
                return tr.ravel()
            pointI = self._get_point_indices(grid)
            return tr[pointI].ravel()

    def _flush_data_to_grids(self, field, default_val, dtype='float32'):
        """
        A dangerous, thusly underscored, thing to do to a data object,
        we can flush back any changes in a given field that have been made
        with a default value for the rest of the grid.
        """
        i = 0
        for grid in self._grids:
            pointI = self._get_point_indices(grid)
            np = pointI[0].ravel().size
            if grid.has_key(field):
                new_field = grid[field]
            else:
                new_field = na.ones(grid.ActiveDimensions, dtype=dtype) * default_val
            new_field[pointI] = self[field][i:i+np]
            grid[field] = new_field
            i += np

    def _is_fully_enclosed(self, grid):
        return na.all(self._get_cut_mask)

    def _get_point_indices(self, grid, use_child_mask=True):
        k = na.zeros(grid.ActiveDimensions, dtype='bool')
        k = (k | self._get_cut_mask(grid))
        if use_child_mask: k = (k & grid.child_mask)
        return na.where(k)

    def _get_cut_particle_mask(self, grid):
        if self._is_fully_enclosed(grid):
            return True
        fake_grid = FakeGridForParticles(grid)
        return self._get_cut_mask(fake_grid)

    def _get_particle_indices(self, grid):
        k = na.zeros(grid.NumberOfParticles, dtype='bool')
        k = (k | self._get_cut_particle_mask(grid))
        return na.where(k)

    def cut_region(self, field_cuts):
        """
        Return an InLineExtractedRegion, where the grid cells are cut on the
        fly with a set of field_cuts.
        """
        return YTValueCutExtractionBase(self, field_cuts)

    def extract_region(self, indices):
        """
        Return an ExtractedRegion where the points contained in it are defined
        as the points in `this` data object with the given *indices*.
        """
        fp = self.field_parameters.copy()
        return YTSelectedIndicesBase(self, indices, **fp)

    def __get_quantities(self):
        if self.__quantities is None:
            self.__quantities = DerivedQuantityCollection(self)
        return self.__quantities
    __quantities = None
    quantities = property(__get_quantities)

    def extract_isocontours(self, field, value, filename = None,
                            rescale = False, sample_values = None):
        r"""This identifies isocontours on a cell-by-cell basis, with no
        consideration of global connectedness, and returns the vertices of the
        Triangles in that isocontour.

        This function simply returns the vertices of all the triangles
        calculated by the marching cubes algorithm; for more complex
        operations, such as identifying connected sets of cells above a given
        threshold, see the extract_connected_sets function.  This is more
        useful for calculating, for instance, total isocontour area, or
        visualizing in an external program (such as `MeshLab
        <http://meshlab.sf.net>`_.)
        
        Parameters
        ----------
        field : string
            Any field that can be obtained in a data object.  This is the field
            which will be isocontoured.
        value : float
            The value at which the isocontour should be calculated.
        filename : string, optional
            If supplied, this file will be filled with the vertices in .obj
            format.  Suitable for loading into meshlab.
        rescale : bool, optional
            If true, the vertices will be rescaled within their min/max.
        sample_values : string, optional
            Any field whose value should be extracted at the center of each
            triangle.

        Returns
        -------
        verts : array of floats
            The array of vertices, x,y,z.  Taken in threes, these are the
            triangle vertices.
        samples : array of floats
            If `sample_values` is specified, this will be returned and will
            contain the values of the field specified at the center of each
            triangle.

        References
        ----------

        .. [1] Marching Cubes: http://en.wikipedia.org/wiki/Marching_cubes

        Examples
        --------
        This will create a data object, find a nice value in the center, and
        output the vertices to "triangles.obj" after rescaling them.

        >>> dd = pf.h.all_data()
        >>> rho = dd.quantities["WeightedAverageQuantity"](
        ...     "Density", weight="CellMassMsun")
        >>> verts = dd.extract_isocontours("Density", rho,
        ...             "triangles.obj", True)
        """
        verts = []
        samples = []
        for i, g in enumerate(self._get_grid_objs()):
            my_verts = self._extract_isocontours_from_grid(
                            g, field, value, sample_values)
            if sample_values is not None:
                my_verts, svals = my_verts
                samples.append(svals)
            verts.append(my_verts)
        verts = na.concatenate(verts).transpose()
        verts = self.comm.par_combine_object(verts, op='cat', datatype='array')
        verts = verts.transpose()
        if sample_values is not None:
            samples = na.concatenate(samples)
            samples = self.comm.par_combine_object(samples, op='cat',
                                datatype='array')
        if rescale:
            mi = na.min(verts, axis=0)
            ma = na.max(verts, axis=0)
            verts = (verts - mi) / (ma - mi).max()
        if filename is not None and self.comm.rank == 0:
            f = open(filename, "w")
            for v1 in verts:
                f.write("v %0.16e %0.16e %0.16e\n" % (v1[0], v1[1], v1[2]))
            for i in range(len(verts)/3):
                f.write("f %s %s %s\n" % (i*3+1, i*3+2, i*3+3))
            f.close()
        if sample_values is not None:
            return verts, samples
        return verts


    @restore_grid_state
    def _extract_isocontours_from_grid(self, grid, field, value,
                                       sample_values = None):
        mask = self._get_cut_mask(grid) * grid.child_mask
        vals = grid.get_vertex_centered_data(field)
        if sample_values is not None:
            svals = grid.get_vertex_centered_data(sample_values)
        else:
            svals = None
        my_verts = march_cubes_grid(value, vals, mask, grid.LeftEdge,
                                    grid.dds, svals)
        return my_verts

    def calculate_isocontour_flux(self, field, value,
                    field_x, field_y, field_z, fluxing_field = None):
        r"""This identifies isocontours on a cell-by-cell basis, with no
        consideration of global connectedness, and calculates the flux over
        those contours.

        This function will conduct marching cubes on all the cells in a given
        data container (grid-by-grid), and then for each identified triangular
        segment of an isocontour in a given cell, calculate the gradient (i.e.,
        normal) in the isocontoured field, interpolate the local value of the
        "fluxing" field, the area of the triangle, and then return:

        area * local_flux_value * (n dot v)

        Where area, local_value, and the vector v are interpolated at the barycenter
        (weighted by the vertex values) of the triangle.  Note that this
        specifically allows for the field fluxing across the surface to be
        *different* from the field being contoured.  If the fluxing_field is
        not specified, it is assumed to be 1.0 everywhere, and the raw flux
        with no local-weighting is returned.

        Additionally, the returned flux is defined as flux *into* the surface,
        not flux *out of* the surface.
        
        Parameters
        ----------
        field : string
            Any field that can be obtained in a data object.  This is the field
            which will be isocontoured and used as the "local_value" in the
            flux equation.
        value : float
            The value at which the isocontour should be calculated.
        field_x : string
            The x-component field
        field_y : string
            The y-component field
        field_z : string
            The z-component field
        fluxing_field : string, optional
            The field whose passage over the surface is of interest.  If not
            specified, assumed to be 1.0 everywhere.

        Returns
        -------
        flux : float
            The summed flux.  Note that it is not currently scaled; this is
            simply the code-unit area times the fields.

        References
        ----------

        .. [1] Marching Cubes: http://en.wikipedia.org/wiki/Marching_cubes

        Examples
        --------
        This will create a data object, find a nice value in the center, and
        calculate the metal flux over it.

        >>> dd = pf.h.all_data()
        >>> rho = dd.quantities["WeightedAverageQuantity"](
        ...     "Density", weight="CellMassMsun")
        >>> flux = dd.calculate_isocontour_flux("Density", rho,
        ...     "x-velocity", "y-velocity", "z-velocity", "Metal_Density")
        """
        flux = 0.0
        for g in self._get_grid_objs():
            flux += self._calculate_flux_in_grid(g, field, value,
                    field_x, field_y, field_z, fluxing_field)
        flux = self.comm.mpi_allreduce(flux, op="sum")
        return flux

    @restore_grid_state
    def _calculate_flux_in_grid(self, grid, field, value,
                    field_x, field_y, field_z, fluxing_field = None):
        mask = self._get_cut_mask(grid) * grid.child_mask
        vals = grid.get_vertex_centered_data(field)
        if fluxing_field is None:
            ff = na.ones(vals.shape, dtype="float64")
        else:
            ff = grid.get_vertex_centered_data(fluxing_field)
        xv, yv, zv = [grid.get_vertex_centered_data(f) for f in 
                     [field_x, field_y, field_z]]
        return march_cubes_grid_flux(value, vals, xv, yv, zv,
                    ff, mask, grid.LeftEdge, grid.dds)

    def extract_connected_sets(self, field, num_levels, min_val, max_val,
                                log_space=True, cumulative=True, cache=False):
        """
        This function will create a set of contour objects, defined
        by having connected cell structures, which can then be
        studied and used to 'paint' their source grids, thus enabling
        them to be plotted.
        """
        if log_space:
            cons = na.logspace(na.log10(min_val),na.log10(max_val),
                               num_levels+1)
        else:
            cons = na.linspace(min_val, max_val, num_levels+1)
        contours = {}
        if cache: cached_fields = defaultdict(lambda: dict())
        else: cached_fields = None
        for level in range(num_levels):
            contours[level] = {}
            if cumulative:
                mv = max_val
            else:
                mv = cons[level+1]
            from yt.analysis_modules.level_sets.api import identify_contours
            cids = identify_contours(self, field, cons[level], mv,
                                     cached_fields)
            for cid, cid_ind in cids.items():
                contours[level][cid] = self.extract_region(cid_ind)
        return cons, contours

    def paint_grids(self, field, value, default_value=None):
        """
        This function paints every cell in our dataset with a given *value*.
        If default_value is given, the other values for the given in every grid
        are discarded and replaced with *default_value*.  Otherwise, the field is
        mandated to 'know how to exist' in the grid.

        Note that this only paints the cells *in the dataset*, so cells in grids
        with child cells are left untouched.
        """
        for grid in self._grids:
            if default_value != None:
                grid[field] = na.ones(grid.ActiveDimensions)*default_value
            grid[field][self._get_point_indices(grid)] = value

    _particle_handler = None

    @property
    def particles(self):
        if self._particle_handler is None:
            self._particle_handler = \
                particle_handler_registry[self._type_name](self.pf, self)
        return self._particle_handler


    def volume(self, unit = "unitary"):
        """
        Return the volume of the data container in units *unit*.
        This is found by adding up the volume of the cells with centers
        in the container, rather than using the geometric shape of
        the container, so this may vary very slightly
        from what might be expected from the geometric volume.
        """
        return self.quantities["TotalQuantity"]("CellVolume")[0] * \
            (self.pf[unit] / self.pf['cm']) ** 3.0

def _reconstruct_object(*args, **kwargs):
    pfid = args[0]
    dtype = args[1]
    field_parameters = args[-1]
    # will be much nicer when we can do pfid, *a, fp = args
    args, new_args = args[2:-1], []
    for arg in args:
        if iterable(arg) and len(arg) == 2 \
           and not isinstance(arg, types.DictType) \
           and isinstance(arg[1], YTDataContainer):
            new_args.append(arg[1])
        else: new_args.append(arg)
    pfs = ParameterFileStore()
    pf = pfs.get_pf_hash(pfid)
    cls = getattr(pf.h, dtype)
    obj = cls(*new_args)
    obj.field_parameters.update(field_parameters)
    return pf, obj
