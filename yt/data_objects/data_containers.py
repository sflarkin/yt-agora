"""
Various non-grid data containers.



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

data_object_registry = {}

import numpy as np
import math
import weakref
import exceptions
import itertools
import shelve
import cStringIO
import fileinput
from re import finditer

from yt.funcs import *
from yt.config import ytcfg

from yt.data_objects.derived_quantities import GridChildMaskWrapper
from yt.data_objects.particle_io import particle_handler_registry
from yt.utilities.lib import find_grids_in_inclined_box, \
    grid_points_in_volume, planar_points_in_volume, VoxelTraversal, \
    QuadTree, get_box_grids_below_level, ghost_zone_interpolate, \
    march_cubes_grid, march_cubes_grid_flux
from yt.utilities.data_point_utilities import CombineGrids, \
    DataCubeRefine, DataCubeReplace, FillRegion, FillBuffer
from yt.utilities.definitions import axis_names, x_dict, y_dict
from yt.utilities.parallel_tools.parallel_analysis_interface import \
    ParallelAnalysisInterface, parallel_root_only
from yt.utilities.linear_interpolators import \
    UnilinearFieldInterpolator, \
    BilinearFieldInterpolator, \
    TrilinearFieldInterpolator
from yt.utilities.parameter_file_storage import \
    ParameterFileStore
from yt.utilities.minimal_representation import \
    MinimalProjectionData, MinimalSliceData
from yt.utilities.orientation import Orientation
from yt.utilities.math_utils import get_rotation_matrix

from .derived_quantities import DerivedQuantityCollection
from .field_info_container import \
    NeedsGridType, \
    NeedsOriginalGrid, \
    NeedsDataField, \
    NeedsProperty, \
    NeedsParameter

def force_array(item, shape):
    try:
        sh = item.shape
        return item.copy()
    except AttributeError:
        if item:
            return np.ones(shape, dtype='bool')
        else:
            return np.zeros(shape, dtype='bool')

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
    into for purposes of confinement in an :class:`AMR3DData`.
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
                tempx = np.abs(self['x'] - center[0])
                tempx = np.minimum(tempx, self.DW[0] - tempx)
                tempy = np.abs(self['y'] - center[1])
                tempy = np.minimum(tempy, self.DW[1] - tempy)
                tempz = np.abs(self['z'] - center[2])
                tempz = np.minimum(tempz, self.DW[2] - tempz)
                tr = np.sqrt( tempx**2.0 + tempy**2.0 + tempz**2.0 )
            else:
                raise KeyError(field)
        else: tr = self.field_data[field]
        return tr

class AMRData(object):
    """
    Generic AMRData container.  By itself, will attempt to
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
        self.set_field_parameter("center",np.zeros(3,dtype='float64'))
        self.set_field_parameter("bulk_velocity",np.zeros(3,dtype='float64'))
        self.set_field_parameter("normal",np.array([0,0,1],dtype='float64'))

    def _set_center(self, center):
        if center is None:
            pass
        elif isinstance(center, (types.ListType, types.TupleType, np.ndarray)):
            center = np.array(center)
        elif center in ("c", "center"):
            center = self.pf.domain_center
        elif center == ("max"): # is this dangerous for race conditions?
            center = self.pf.h.find_max("Density")[1]
        elif center.startswith("max_"):
            center = self.pf.h.find_max(center[4:])[1]
        else:
            center = np.array(center, dtype='float64')
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
        Clears out all data from the AMRData instance, freeing memory.
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
        field_data = np.array([self.field_data[field] for field in field_order])
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
        return np.where(self.grid_levels[:,0] == level)

    def __get_grid_left_edge(self):
        if self.__grid_left_edge == None:
            self.__grid_left_edge = np.array([g.LeftEdge for g in self._grids])
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
            self.__grid_right_edge = np.array([g.RightEdge for g in self._grids])
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
            self.__grid_levels = np.array([g.Level for g in self._grids])
            self.__grid_levels.shape = (self.__grid_levels.size, 1)
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
            self.__grid_dimensions = np.array([g.ActiveDimensions for g in self._grids])
        return self.__grid_dimensions

    def __del_grid_dimensions(self):
        del self.__grid_dimensions
        self.__grid_dimensions = None

    def __set_grid_dimensions(self, val):
        self.__grid_dimensions = val

    __grid_dimensions = None
    grid_dimensions = property(__get_grid_dimensions, __set_grid_dimensions,
                             __del_grid_dimensions)

    @property
    def grid_corners(self):
        return np.array([
          [self.grid_left_edge[:,0], self.grid_left_edge[:,1], self.grid_left_edge[:,2]],
          [self.grid_right_edge[:,0], self.grid_left_edge[:,1], self.grid_left_edge[:,2]],
          [self.grid_right_edge[:,0], self.grid_right_edge[:,1], self.grid_left_edge[:,2]],
          [self.grid_left_edge[:,0], self.grid_right_edge[:,1], self.grid_left_edge[:,2]],
          [self.grid_left_edge[:,0], self.grid_left_edge[:,1], self.grid_right_edge[:,2]],
          [self.grid_right_edge[:,0], self.grid_left_edge[:,1], self.grid_right_edge[:,2]],
          [self.grid_right_edge[:,0], self.grid_right_edge[:,1], self.grid_right_edge[:,2]],
          [self.grid_left_edge[:,0], self.grid_right_edge[:,1], self.grid_right_edge[:,2]],
        ], dtype='float64')


class AMR1DData(AMRData, GridPropertiesMixin):
    _spatial = False
    def __init__(self, pf, fields, **kwargs):
        AMRData.__init__(self, pf, fields, **kwargs)
        self._grids = None
        self._sortkey = None
        self._sorted = {}

    def get_data(self, fields=None, in_grids=False):
        if self._grids is None:
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
            self[field] = np.concatenate(
                [self._get_data_from_grid(grid, field)
                 for grid in self._grids])
            if not self.field_data.has_key(field):
                continue
            if self._sortkey is None:
                self._sortkey = np.argsort(self[self.sort_by])
            # We *always* sort the field here if we have not successfully
            # generated it above.  This way, fields that are grabbed from the
            # grids are sorted properly.
            self[field] = self[field][self._sortkey]

class AMROrthoRayBase(AMR1DData):
    """
    This is an orthogonal ray cast through the entire domain, at a specific
    coordinate.

    This object is typically accessed through the `ortho_ray` object that
    hangs off of hierarchy objects.  The resulting arrays have their
    dimensionality reduced to one, and an ordered list of points at an
    (x,y) tuple along `axis` are available.

    Parameters
    ----------
    axis : int
        The axis along which to cast the ray.  Can be 0, 1, or 2 for x, y, z.
    coords : tuple of floats
        The (plane_x, plane_y) coordinates at which to cast the ray.  Note
        that this is in the plane coordinates: so if you are casting along
        x, this will be (y,z).  If you are casting along y, this will be
        (x,z).  If you are casting along z, this will be (x,y).
    fields : list of strings, optional
        If you want the object to pre-retrieve a set of fields, supply them
        here.  This is not necessary.
    kwargs : dict of items
        Any additional values are passed as field parameters that can be
        accessed by generated fields.

    Examples
    --------

    >>> pf = load("RedshiftOutput0005")
    >>> oray = pf.h.ortho_ray(0, (0.2, 0.74))
    >>> print oray["Density"]
    """
    _key_fields = ['x','y','z','dx','dy','dz']
    _type_name = "ortho_ray"
    _con_args = ('axis', 'coords')
    def __init__(self, axis, coords, fields=None, pf=None, **kwargs):
        AMR1DData.__init__(self, pf, fields, **kwargs)
        self.axis = axis
        self.px_ax = x_dict[self.axis]
        self.py_ax = y_dict[self.axis]
        self.px_dx = 'd%s'%(axis_names[self.px_ax])
        self.py_dx = 'd%s'%(axis_names[self.py_ax])
        self.px, self.py = coords
        self.sort_by = axis_names[self.axis]
        self._refresh_data()

    @property
    def coords(self):
        return (self.px, self.py)

    def _get_list_of_grids(self):
        # This bugs me, but we will give the tie to the LeftEdge
        y = np.where( (self.px >=  self.pf.hierarchy.grid_left_edge[:,self.px_ax])
                    & (self.px < self.pf.hierarchy.grid_right_edge[:,self.px_ax])
                    & (self.py >=  self.pf.hierarchy.grid_left_edge[:,self.py_ax])
                    & (self.py < self.pf.hierarchy.grid_right_edge[:,self.py_ax]))
        self._grids = self.hierarchy.grids[y]

    @restore_grid_state
    def _get_data_from_grid(self, grid, field):
        # We are orthogonal, so we can feel free to make assumptions
        # for the sake of speed.
        if grid.id not in self._cut_masks:
            gdx = just_one(grid[self.px_dx])
            gdy = just_one(grid[self.py_dx])
            x_coord = int((self.px - grid.LeftEdge[self.px_ax])/gdx)
            y_coord = int((self.py - grid.LeftEdge[self.py_ax])/gdy)
            sl = [None,None,None]
            sl[self.px_ax] = slice(x_coord,x_coord+1,None)
            sl[self.py_ax] = slice(y_coord,y_coord+1,None)
            sl[self.axis] = slice(None)
            self._cut_masks[grid.id] = sl
        else:
            sl = self._cut_masks[grid.id]
        if not iterable(grid[field]):
            gf = grid[field] * np.ones(grid.child_mask[sl].shape)
        else:
            gf = grid[field][sl]
        return gf[np.where(grid.child_mask[sl])]

class AMRRayBase(AMR1DData):
    """
    This is an arbitrarily-aligned ray cast through the entire domain, at a
    specific coordinate.

    This object is typically accessed through the `ray` object that hangs
    off of hierarchy objects.  The resulting arrays have their
    dimensionality reduced to one, and an ordered list of points at an
    (x,y) tuple along `axis` are available, as is the `t` field, which
    corresponds to a unitless measurement along the ray from start to
    end.

    Parameters
    ----------
    start_point : array-like set of 3 floats
        The place where the ray starts.
    end_point : array-like set of 3 floats
        The place where the ray ends.
    fields : list of strings, optional
        If you want the object to pre-retrieve a set of fields, supply them
        here.  This is not necessary.
    kwargs : dict of items
        Any additional values are passed as field parameters that can be
        accessed by generated fields.

    Examples
    --------

    >>> pf = load("RedshiftOutput0005")
    >>> ray = pf.h.ray((0.2, 0.74, 0.11), (0.4, 0.91, 0.31))
    >>> print ray["Density"], ray["t"], ray["dts"]
    """
    _type_name = "ray"
    _con_args = ('start_point', 'end_point')
    sort_by = 't'
    def __init__(self, start_point, end_point, fields=None, pf=None, **kwargs):
        AMR1DData.__init__(self, pf, fields, **kwargs)
        self.start_point = np.array(start_point, dtype='float64')
        self.end_point = np.array(end_point, dtype='float64')
        self.vec = self.end_point - self.start_point
        #self.vec /= np.sqrt(np.dot(self.vec, self.vec))
        self._set_center(self.start_point)
        self.set_field_parameter('center', self.start_point)
        self._dts, self._ts = {}, {}
        #self._refresh_data()

    def _get_list_of_grids(self):
        # Get the value of the line at each LeftEdge and RightEdge
        LE = self.pf.h.grid_left_edge
        RE = self.pf.h.grid_right_edge
        p = np.zeros(self.pf.h.num_grids, dtype='bool')
        # Check left faces first
        for i in range(3):
            i1 = (i+1) % 3
            i2 = (i+2) % 3
            vs = self._get_line_at_coord(LE[:,i], i)
            p = p | ( ( (LE[:,i1] <= vs[:,i1]) & (RE[:,i1] >= vs[:,i1]) ) \
                    & ( (LE[:,i2] <= vs[:,i2]) & (RE[:,i2] >= vs[:,i2]) ) )
            vs = self._get_line_at_coord(RE[:,i], i)
            p = p | ( ( (LE[:,i1] <= vs[:,i1]) & (RE[:,i1] >= vs[:,i1]) ) \
                    & ( (LE[:,i2] <= vs[:,i2]) & (RE[:,i2] >= vs[:,i2]) ) )
        p = p | ( np.all( LE <= self.start_point, axis=1 )
                & np.all( RE >= self.start_point, axis=1 ) )
        p = p | ( np.all( LE <= self.end_point,   axis=1 )
                & np.all( RE >= self.end_point,   axis=1 ) )
        self._grids = self.hierarchy.grids[p]

    def _get_line_at_coord(self, v, index):
        # t*self.vec + self.start_point = self.end_point
        t = (v - self.start_point[index])/self.vec[index]
        t = t.reshape((t.shape[0],1))
        return self.start_point + t*self.vec

    @restore_grid_state
    def _get_data_from_grid(self, grid, field):
        mask = np.logical_and(self._get_cut_mask(grid),
                              grid.child_mask)
        if field == 'dts': return self._dts[grid.id][mask]
        if field == 't': return self._ts[grid.id][mask]
        gf = grid[field]
        if not iterable(gf):
            gf = gf * np.ones(grid.child_mask.shape)
        return gf[mask]

    @cache_mask
    def _get_cut_mask(self, grid):
        mask = np.zeros(grid.ActiveDimensions, dtype='int')
        dts = np.zeros(grid.ActiveDimensions, dtype='float64')
        ts = np.zeros(grid.ActiveDimensions, dtype='float64')
        VoxelTraversal(mask, ts, dts, grid.LeftEdge, grid.RightEdge,
                       grid.dds, self.center, self.vec)
        self._dts[grid.id] = np.abs(dts)
        self._ts[grid.id] = np.abs(ts)
        return mask

class AMRStreamlineBase(AMR1DData):
    """
    This is a streamline, which is a set of points defined as
    being parallel to some vector field.

    This object is typically accessed through the Streamlines.path
    function.  The resulting arrays have their dimensionality
    reduced to one, and an ordered list of points at an (x,y)
    tuple along `axis` are available, as is the `t` field, which
    corresponds to a unitless measurement along the ray from start
    to end.

    Parameters
    ----------
    positions : array-like
        List of streamline positions
    length : float
        The magnitude of the distance; dts will be divided by this
    fields : list of strings, optional
        If you want the object to pre-retrieve a set of fields, supply them
        here.  This is not necessary.
    pf : Parameter file object
        Passed in to access the hierarchy
    kwargs : dict of items
        Any additional values are passed as field parameters that can be
        accessed by generated fields.

    Examples
    --------

    >>> from yt.visualization.api import Streamlines
    >>> streamlines = Streamlines(pf, [0.5]*3)
    >>> streamlines.integrate_through_volume()
    >>> stream = streamlines.path(0)
    >>> matplotlib.pylab.semilogy(stream['t'], stream['Density'], '-x')

    """
    _type_name = "streamline"
    _con_args = ('positions')
    sort_by = 't'
    def __init__(self, positions, length = 1.0, fields=None, pf=None, **kwargs):
        AMR1DData.__init__(self, pf, fields, **kwargs)
        self.positions = positions
        self.dts = np.empty_like(positions[:,0])
        self.dts[:-1] = np.sqrt(np.sum((self.positions[1:]-
                                        self.positions[:-1])**2,axis=1))
        self.dts[-1] = self.dts[-2]
        self.length = length
        self.dts /= length
        self.ts = np.add.accumulate(self.dts)
        self._set_center(self.positions[0])
        self.set_field_parameter('center', self.positions[0])
        self._dts, self._ts = {}, {}
        #self._refresh_data()

    def _get_list_of_grids(self):
        # Get the value of the line at each LeftEdge and RightEdge
        LE = self.pf.h.grid_left_edge
        RE = self.pf.h.grid_right_edge
        # Check left faces first
        min_streampoint = np.min(self.positions, axis=0)
        max_streampoint = np.max(self.positions, axis=0)
        p = np.all((min_streampoint <= RE) & (max_streampoint > LE), axis=1)
        self._grids = self.hierarchy.grids[p]

    @restore_grid_state
    def _get_data_from_grid(self, grid, field):
        # No child masking here; it happens inside the mask cut
        mask = self._get_cut_mask(grid)
        if field == 'dts': return self._dts[grid.id]
        if field == 't': return self._ts[grid.id]
        return grid[field].flat[mask]

    @cache_mask
    def _get_cut_mask(self, grid):
        #pdb.set_trace()
        points_in_grid = np.all(self.positions > grid.LeftEdge, axis=1) & \
                         np.all(self.positions <= grid.RightEdge, axis=1)
        pids = np.where(points_in_grid)[0]
        mask = np.zeros(points_in_grid.sum(), dtype='int')
        dts = np.zeros(points_in_grid.sum(), dtype='float64')
        ts = np.zeros(points_in_grid.sum(), dtype='float64')
        for mi, (i, pos) in enumerate(zip(pids, self.positions[points_in_grid])):
            if not points_in_grid[i]: continue
            ci = ((pos - grid.LeftEdge)/grid.dds).astype('int')
            if grid.child_mask[ci[0], ci[1], ci[2]] == 0: continue
            for j in range(3):
                ci[j] = min(ci[j], grid.ActiveDimensions[j]-1)
            mask[mi] = np.ravel_multi_index(ci, grid.ActiveDimensions)
            dts[mi] = self.dts[i]
            ts[mi] = self.ts[i]
        self._dts[grid.id] = dts
        self._ts[grid.id] = ts
        return mask

class AMR2DData(AMRData, GridPropertiesMixin, ParallelAnalysisInterface):
    _key_fields = ['px','py','pdx','pdy']
    """
    Class to represent a set of :class:`AMRData` that's 2-D in nature, and
    thus does not have as many actions as the 3-D data types.
    """
    _spatial = False
    def __init__(self, axis, fields, pf=None, **kwargs):
        """
        Prepares the AMR2DData, normal to *axis*.  If *axis* is 4, we are not
        aligned with any axis.
        """
        ParallelAnalysisInterface.__init__(self)
        self.axis = axis
        AMRData.__init__(self, pf, fields, **kwargs)
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
        for field in fields_to_get:
            if self.field_data.has_key(field): continue
            if field not in self.hierarchy.field_list:
                if self._generate_field(field):
                    continue # A "True" return means we did it
            # To ensure that we use data from this object as much as possible,
            # we're going to have to set the same thing several times
            data = [self._get_data_from_grid(grid, field)
                    for grid in self._get_grids()]
            if len(data) == 0:
                data = np.array([])
            else:
                data = np.concatenate(data)
            # Now the next field can use this field
            self[field] = self.comm.par_combine_object(data, op='cat',
                                                       datatype='array')

    def _get_pw(self, fields, center, width, origin, axes_unit, plot_type):
        axis = self.axis
        if fields == None:
            if self.fields == None:
                raise SyntaxError("The fields keyword argument must be set")
        else:
            self.fields = ensure_list(fields)
        from yt.visualization.plot_window import \
            GetWindowParameters, PWViewerMPL
        from yt.visualization.fixed_resolution import FixedResolutionBuffer
        (bounds, center, units) = GetWindowParameters(axis, center, width, self.pf)
        if axes_unit is None and units != ('1', '1'):
            axes_unit = units
        pw = PWViewerMPL(self, bounds, origin=origin, frb_generator=FixedResolutionBuffer,
                         plot_type=plot_type)
        pw.set_axes_unit(axes_unit)
        return pw

    def to_frb(self, width, resolution, center=None, height=None,
               periodic = False):
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
        height : height specifier
            This will be the physical height of the FRB, by default it is equal
            to width.  Note that this will not make any corrections to
            resolution for the aspect ratio.
        resolution : int or tuple of ints
            The number of pixels on a side of the final FRB.  If iterable, this
            will be the width then the height.
        center : array-like of floats, optional
            The center of the FRB.  If not specified, defaults to the center of
            the current object.
        periodic : bool
            Should the returned Fixed Resolution Buffer be periodic?  (default:
            False).

        Returns
        -------
        frb : :class:`~yt.visualization.fixed_resolution.FixedResolutionBuffer`
            A fixed resolution buffer, which can be queried for fields.

        Examples
        --------

        >>> proj = pf.h.proj(0, "Density")
        >>> frb = proj.to_frb( (100.0, 'kpc'), 1024)
        >>> write_image(np.log10(frb["Density"]), 'density_100kpc.png')
        """
        if center is None:
            center = self.get_field_parameter("center")
            if center is None:
                center = (self.pf.domain_right_edge
                        + self.pf.domain_left_edge)/2.0
        if iterable(width):
            w, u = width
            width = w/self.pf[u]
        if height is None:
            height = width
        elif iterable(height):
            h, u = height
            height = h/self.pf[u]
        if not iterable(resolution):
            resolution = (resolution, resolution)
        from yt.visualization.fixed_resolution import FixedResolutionBuffer
        xax = x_dict[self.axis]
        yax = y_dict[self.axis]
        bounds = (center[xax] - width*0.5, center[xax] + width*0.5,
                  center[yax] - height*0.5, center[yax] + height*0.5)
        frb = FixedResolutionBuffer(self, bounds, resolution,
                                    periodic = periodic)
        return frb

    def interpolate_discretize(self, LE, RE, field, side, log_spacing=True):
        """
        This returns a uniform grid of points between *LE* and *RE*,
        interpolated using the nearest neighbor method, with *side* points on a
        side.
        """
        import matplotlib.delaunay.triangulate as de
        if log_spacing:
            zz = np.log10(self[field])
        else:
            zz = self[field]
        xi, yi = np.array( \
                 np.mgrid[LE[0]:RE[0]:side*1j, \
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

class AMRSliceBase(AMR2DData):
    """
    This is a data object corresponding to a slice through the simulation
    domain.

    This object is typically accessed through the `slice` object that hangs
    off of hierarchy objects.  AMRSlice is an orthogonal slice through the
    data, taking all the points at the finest resolution available and then
    indexing them.  It is more appropriately thought of as a slice
    'operator' than an object, however, as its field and coordinate can
    both change.

    Parameters
    ----------
    axis : int
        The axis along which to slice.  Can be 0, 1, or 2 for x, y, z.
    coord : float
        The coordinate along the axis at which to slice.  This is in
        "domain" coordinates.
    fields : list of strings, optional
        If you want the object to pre-retrieve a set of fields, supply them
        here.  This is not necessary.
    center : array_like, optional
        The 'center' supplied to fields that use it.  Note that this does
        not have to have `coord` as one value.  Strictly optional.
    node_name: string, optional
        The node in the .yt file to find or store this slice at.  Should
        probably not be used.
    kwargs : dict of items
        Any additional values are passed as field parameters that can be
        accessed by generated fields.

    Examples
    --------

    >>> pf = load("RedshiftOutput0005")
    >>> slice = pf.h.slice(0, 0.25)
    >>> print slice["Density"]
    """
    _top_node = "/Slices"
    _type_name = "slice"
    _con_args = ('axis', 'coord')
    #@time_execution
    def __init__(self, axis, coord, fields = None, center=None, pf=None,
                 node_name = False, **kwargs):
        AMR2DData.__init__(self, axis, fields, pf, **kwargs)
        self._set_center(center)
        self.coord = coord
        if node_name is False:
            self._refresh_data()
        else:
            if node_name is True: self._deserialize()
            else: self._deserialize(node_name)

    def reslice(self, coord):
        """
        Change the entire dataset, clearing out the current data and slicing at
        a new location.  Not terribly useful except for in-place plot changes.
        """
        mylog.debug("Setting coordinate to %0.5e" % coord)
        self.coord = coord
        self._refresh_data()

    def shift(self, val):
        """
        Moves the slice coordinate up by either a floating point value, or an
        integer number of indices of the finest grid.
        """
        if isinstance(val, types.FloatType):
            # We add the dx
            self.coord += val
        elif isinstance(val, types.IntType):
            # Here we assume that the grid is the max level
            level = self.hierarchy.max_level
            self.coord
            dx = self.hierarchy.select_grids(level)[0].dds[self.axis]
            self.coord += dx * val
        else:
            raise ValueError(val)
        self._refresh_data()

    def _generate_coords(self):
        points = []
        for grid in self._get_grids():
            points.append(self._generate_grid_coords(grid))
        if len(points) == 0:
            points = None
            t = self.comm.par_combine_object(None, datatype="array", op="cat")
        else:
            points = np.concatenate(points)
            # We have to transpose here so that _par_combine_object works
            # properly, as it and the alltoall assume the long axis is the last
            # one.
            t = self.comm.par_combine_object(points.transpose(),
                        datatype="array", op="cat")
        self['px'] = t[0,:]
        self['py'] = t[1,:]
        self['pz'] = t[2,:]
        self['pdx'] = t[3,:]
        self['pdy'] = t[4,:]
        self['pdz'] = t[3,:] # Does not matter!

        # Now we set the *actual* coordinates
        self[axis_names[x_dict[self.axis]]] = t[0,:]
        self[axis_names[y_dict[self.axis]]] = t[1,:]
        self[axis_names[self.axis]] = t[2,:]

        self.ActiveDimensions = (t.shape[1], 1, 1)

    def _get_list_of_grids(self):
        goodI = ((self.pf.h.grid_right_edge[:,self.axis] > self.coord)
              &  (self.pf.h.grid_left_edge[:,self.axis] <= self.coord ))
        self._grids = self.pf.h.grids[goodI] # Using sources not hierarchy

    def __cut_mask_child_mask(self, grid):
        mask = grid.child_mask.copy()
        return mask

    def _generate_grid_coords(self, grid):
        xaxis = x_dict[self.axis]
        yaxis = y_dict[self.axis]
        ds, dx, dy = grid.dds[self.axis], grid.dds[xaxis], grid.dds[yaxis]
        sl_ind = int((self.coord-self.pf.domain_left_edge[self.axis])/ds) - \
                     grid.get_global_startindex()[self.axis]
        sl = [slice(None), slice(None), slice(None)]
        sl[self.axis] = slice(sl_ind, sl_ind + 1)
        #sl.reverse()
        sl = tuple(sl)
        nx = grid.child_mask.shape[xaxis]
        ny = grid.child_mask.shape[yaxis]
        mask = self.__cut_mask_child_mask(grid)[sl]
        cm = np.where(mask.ravel()== 1)
        cmI = np.indices((nx,ny))
        ind = cmI[0, :].ravel()   # xind
        npoints = cm[0].shape
        # create array of "npoints" ones that will be reused later
        points = np.ones(npoints, 'float64')
        # calculate xpoints array
        t = points * ind[cm] * dx + (grid.LeftEdge[xaxis] + 0.5 * dx)
        # calculate ypoints array
        ind = cmI[1, :].ravel()   # yind
        del cmI   # no longer needed
        t = np.vstack( (t, points * ind[cm] * dy + \
                (grid.LeftEdge[yaxis] + 0.5 * dy))
            )
        del ind, cm   # no longer needed
        # calculate zpoints array
        t = np.vstack((t, points * self.coord))
        # calculate dx array
        t = np.vstack((t, points * dx * 0.5))
        # calculate dy array
        t = np.vstack((t, points * dy * 0.5))
        # return [xpoints, ypoints, zpoints, dx, dy] as (5, npoints) array
        return t.swapaxes(0, 1)

    @restore_grid_state
    def _get_data_from_grid(self, grid, field):
        # So what's our index of slicing?  This is what we need to figure out
        # first, so we can deal with our data in the fastest way.
        dx = grid.dds[self.axis]
        sl_ind = int((self.coord-self.pf.domain_left_edge[self.axis])/dx) - \
                     grid.get_global_startindex()[self.axis]
        sl = [slice(None), slice(None), slice(None)]
        sl[self.axis] = slice(sl_ind, sl_ind + 1)
        sl = tuple(sl)
        if self.pf.field_info.has_key(field) and self.pf.field_info[field].particle_type:
            return grid[field]
        elif field in self.pf.field_info and self.pf.field_info[field].not_in_all:
            dv = grid[field][sl]
        elif not grid.has_key(field):
            conv_factor = 1.0
            if self.pf.field_info.has_key(field):
                conv_factor = self.pf.field_info[field]._convert_function(self)
            dv = self.hierarchy.io._read_data_slice(grid, field, self.axis, sl_ind) * conv_factor
        else:
            dv = grid[field]
            if dv.size == 1: dv = np.ones(grid.ActiveDimensions)*dv
            dv = dv[sl]
        mask = self.__cut_mask_child_mask(grid)[sl]
        dataVals = dv.ravel()[mask.ravel() == 1]
        return dataVals

    def _gen_node_name(self):
        return "%s/%s_%s" % \
            (self._top_node, self.axis, self.coord)

    def __get_quantities(self):
        if self.__quantities is None:
            self.__quantities = DerivedQuantityCollection(self)
        return self.__quantities
    __quantities = None
    quantities = property(__get_quantities)

    @property
    def _mrep(self):
        return MinimalSliceData(self)

    def hub_upload(self):
        self._mrep.upload()

    def to_pw(self, fields=None, center='c', width=None, axes_unit=None,
               origin='center-window'):
        r"""Create a :class:`~yt.visualization.plot_window.PWViewerMPL` from this
        object.

        This is a bare-bones mechanism of creating a plot window from this
        object, which can then be moved around, zoomed, and on and on.  All
        behavior of the plot window is relegated to that routine.
        """
        pw = self._get_pw(fields, center, width, origin, axes_unit, 'Slice')
        return pw

class AMRCuttingPlaneBase(AMR2DData):
    """
    This is a data object corresponding to an oblique slice through the
    simulation domain.

    This object is typically accessed through the `cutting` object
    that hangs off of hierarchy objects.  AMRCuttingPlane is an oblique
    plane through the data, defined by a normal vector and a coordinate.
    It attempts to guess an 'up' vector, which cannot be overridden, and
    then it pixelizes the appropriate data onto the plane without
    interpolation.

    Parameters
    ----------
    normal : array_like
        The vector that defines the desired plane.  For instance, the
        angular momentum of a sphere.
    center : array_like, optional
        The center of the cutting plane.
    fields : list of strings, optional
        If you want the object to pre-retrieve a set of fields, supply them
        here.  This is not necessary.
    node_name: string, optional
        The node in the .yt file to find or store this slice at.  Should
        probably not be used.
    kwargs : dict of items
        Any additional values are passed as field parameters that can be
        accessed by generated fields.

    Notes
    -----

    This data object in particular can be somewhat expensive to create.
    It's also important to note that unlike the other 2D data objects, this
    oject provides px, py, pz, as some cells may have a height from the
    plane.

    Examples
    --------

    >>> pf = load("RedshiftOutput0005")
    >>> cp = pf.h.cutting([0.1, 0.2, -0.9], [0.5, 0.42, 0.6])
    >>> print cp["Density"]
    """
    _plane = None
    _top_node = "/CuttingPlanes"
    _key_fields = AMR2DData._key_fields + ['pz','pdz']
    _type_name = "cutting"
    _con_args = ('normal', 'center')
    def __init__(self, normal, center, fields = None, node_name = None,
                 north_vector = None, **kwargs):
        AMR2DData.__init__(self, 4, fields, **kwargs)
        self._set_center(center)
        self.set_field_parameter('center',center)
        # Let's set up our plane equation
        # ax + by + cz + d = 0
        self.orienter = Orientation(normal, north_vector = north_vector)
        self._norm_vec = self.orienter.normal_vector
        self._d = -1.0 * np.dot(self._norm_vec, self.center)
        self._x_vec = self.orienter.unit_vectors[0]
        self._y_vec = self.orienter.unit_vectors[1]
        self._rot_mat = np.array([self._x_vec,self._y_vec,self._norm_vec])
        self._inv_mat = np.linalg.pinv(self._rot_mat)
        self.set_field_parameter('cp_x_vec',self._x_vec)
        self.set_field_parameter('cp_y_vec',self._y_vec)
        self.set_field_parameter('cp_z_vec',self._norm_vec)
        if node_name is False:
            self._refresh_data()
        else:
            if node_name is True: self._deserialize()
            else: self._deserialize(node_name)

    @property
    def normal(self):
        return self._norm_vec

    def _get_list_of_grids(self):
        # Recall that the projection of the distance vector from a point
        # onto the normal vector of a plane is:
        # D = (a x_0 + b y_0 + c z_0 + d)/sqrt(a^2+b^2+c^2)
        # @todo: Convert to using corners
        LE = self.pf.h.grid_left_edge
        RE = self.pf.h.grid_right_edge
        vertices = np.array([[LE[:,0],LE[:,1],LE[:,2]],
                             [RE[:,0],RE[:,1],RE[:,2]],
                             [LE[:,0],LE[:,1],RE[:,2]],
                             [RE[:,0],RE[:,1],LE[:,2]],
                             [LE[:,0],RE[:,1],RE[:,2]],
                             [RE[:,0],LE[:,1],LE[:,2]],
                             [LE[:,0],RE[:,1],LE[:,2]],
                             [RE[:,0],LE[:,1],RE[:,2]]])
        # This gives us shape: 8, 3, n_grid
        D = np.sum(self._norm_vec.reshape((1,3,1)) * vertices, axis=1) + self._d
        self.D = D
        self._grids = self.hierarchy.grids[
            np.where(np.logical_not(np.all(D<0,axis=0) | np.all(D>0,axis=0) )) ]

    @cache_mask
    def _get_cut_mask(self, grid):
        # This is slow.  Suggestions for improvement would be great...
        ss = grid.ActiveDimensions
        D = np.ones(ss) * self._d
        x = grid.LeftEdge[0] + grid.dds[0] * \
                (np.arange(grid.ActiveDimensions[0], dtype='float64')+0.5)
        y = grid.LeftEdge[1] + grid.dds[1] * \
                (np.arange(grid.ActiveDimensions[1], dtype='float64')+0.5)
        z = grid.LeftEdge[2] + grid.dds[2] * \
                (np.arange(grid.ActiveDimensions[2], dtype='float64')+0.5)
        D += (x * self._norm_vec[0]).reshape(ss[0],1,1)
        D += (y * self._norm_vec[1]).reshape(1,ss[1],1)
        D += (z * self._norm_vec[2]).reshape(1,1,ss[2])
        diag_dist = np.sqrt(np.sum(grid.dds**2.0))
        cm = (np.abs(D) <= 0.5*diag_dist) # Boolean
        return cm

    def _generate_coords(self):
        points = []
        for grid in self._get_grids():
            points.append(self._generate_grid_coords(grid))
        if len(points) == 0: points = None
        else: points = np.concatenate(points)
        t = self.comm.par_combine_object(points, datatype="array", op="cat")
        pos = (t[:,0:3] - self.center)
        self['px'] = np.dot(pos, self._x_vec)
        self['py'] = np.dot(pos, self._y_vec)
        self['pz'] = np.dot(pos, self._norm_vec)
        self['pdx'] = t[:,3] * 0.5
        self['pdy'] = t[:,3] * 0.5
        self['pdz'] = t[:,3] * 0.5

    def _generate_grid_coords(self, grid):
        pointI = self._get_point_indices(grid)
        coords = [grid[ax][pointI].ravel() for ax in 'xyz']
        coords.append(np.ones(coords[0].shape, 'float64') * just_one(grid['dx']))
        return np.array(coords).swapaxes(0,1)

    def _get_data_from_grid(self, grid, field):
        if not self.pf.field_info[field].particle_type:
            pointI = self._get_point_indices(grid)
            if grid[field].size == 1: # dx, dy, dz, cellvolume
                t = grid[field] * np.ones(grid.ActiveDimensions)
                return t[pointI].ravel()
            return grid[field][pointI].ravel()
        else:
            return grid[field]

    def interpolate_discretize(self, *args, **kwargs):
        pass

    @cache_point_indices
    def _get_point_indices(self, grid, use_child_mask=True):
        k = np.zeros(grid.ActiveDimensions, dtype='bool')
        k = (k | self._get_cut_mask(grid))
        if use_child_mask: k = (k & grid.child_mask)
        return np.where(k)

    def _gen_node_name(self):
        cen_name = ("%s" % (self.center,)).replace(" ","_")[1:-1]
        L_name = ("%s" % self._norm_vec).replace(" ","_")[1:-1]
        return "%s/c%s_L%s" % \
            (self._top_node, cen_name, L_name)

    def to_pw(self, fields=None, center='c', width=None, axes_unit=None):
        r"""Create a :class:`~yt.visualization.plot_window.PWViewerMPL` from this
        object.

        This is a bare-bones mechanism of creating a plot window from this
        object, which can then be moved around, zoomed, and on and on.  All
        behavior of the plot window is relegated to that routine.
        """
        normal = self.normal
        center = self.center
        if fields == None:
            if self.fields == None:
                raise SyntaxError("The fields keyword argument must be set")
        else:
            self.fields = ensure_list(fields)
        from yt.visualization.plot_window import \
            GetObliqueWindowParameters, PWViewerMPL
        from yt.visualization.fixed_resolution import ObliqueFixedResolutionBuffer
        (bounds, center_rot, units) = GetObliqueWindowParameters(normal, center, width, self.pf)
        if axes_unit is None and units != ('1', '1'):
            axes_units = units
        pw = PWViewerMPL(self, bounds, origin='center-window', periodic=False, oblique=True,
                         frb_generator=ObliqueFixedResolutionBuffer, plot_type='OffAxisSlice')
        pw.set_axes_unit(axes_unit)
        return pw

    def to_frb(self, width, resolution, height=None):
        r"""This function returns an ObliqueFixedResolutionBuffer generated
        from this object.

        An ObliqueFixedResolutionBuffer is an object that accepts a
        variable-resolution 2D object and transforms it into an NxM bitmap that
        can be plotted, examined or processed.  This is a convenience function
        to return an FRB directly from an existing 2D data object.  Unlike the
        corresponding to_frb function for other AMR2DData objects, this does
        not accept a 'center' parameter as it is assumed to be centered at the
        center of the cutting plane.

        Parameters
        ----------
        width : width specifier
            This can either be a floating point value, in the native domain
            units of the simulation, or a tuple of the (value, unit) style.
            This will be the width of the FRB.
        height : height specifier, optional
            This will be the height of the FRB, by default it is equal to width.
        resolution : int or tuple of ints
            The number of pixels on a side of the final FRB.

        Returns
        -------
        frb : :class:`~yt.visualization.fixed_resolution.ObliqueFixedResolutionBuffer`
            A fixed resolution buffer, which can be queried for fields.

        Examples
        --------

        >>> v, c = pf.h.find_max("Density")
        >>> sp = pf.h.sphere(c, (100.0, 'au'))
        >>> L = sp.quantities["AngularMomentumVector"]()
        >>> cutting = pf.h.cutting(L, c)
        >>> frb = cutting.to_frb( (1.0, 'pc'), 1024)
        >>> write_image(np.log10(frb["Density"]), 'density_1pc.png')
        """
        if iterable(width):
            w, u = width
            width = w/self.pf[u]
        if height is None:
            height = width
        elif iterable(height):
            h, u = height
            height = h/self.pf[u]
        if not iterable(resolution):
            resolution = (resolution, resolution)
        from yt.visualization.fixed_resolution import ObliqueFixedResolutionBuffer
        bounds = (-width/2.0, width/2.0, -height/2.0, height/2.0)
        frb = ObliqueFixedResolutionBuffer(self, bounds, resolution)
        return frb

class AMRFixedResCuttingPlaneBase(AMR2DData):
    """
    The fixed resolution Cutting Plane slices at an oblique angle,
    where we use the *normal* vector at the *center* to define the
    viewing plane.  The plane is *width* units wide.  The 'up'
    direction is guessed at automatically if not given.

    AMRFixedResCuttingPlaneBase is an oblique plane through the data,
    defined by a normal vector and a coordinate.  It trilinearly
    interpolates the data to a fixed resolution slice.  It differs from
    the other data objects as it doesn't save the grid data, only the
    interpolated data.
    """
    _top_node = "/FixedResCuttingPlanes"
    _type_name = "fixed_res_cutting"
    _con_args = ('normal', 'center', 'width', 'dims')
    def __init__(self, normal, center, width, dims, fields = None,
                 node_name = None, **kwargs):
        #
        # Taken from Cutting Plane
        #
        AMR2DData.__init__(self, 4, fields, **kwargs)
        self._set_center(center)
        self.width = width
        self.dims = dims
        self.dds = self.width / self.dims
        self.bounds = np.array([0.0,1.0,0.0,1.0])

        self.set_field_parameter('center', center)
        # Let's set up our plane equation
        # ax + by + cz + d = 0
        self._norm_vec = normal/np.sqrt(np.dot(normal,normal))
        self._d = -1.0 * np.dot(self._norm_vec, self.center)
        # First we try all three, see which has the best result:
        vecs = np.identity(3)
        _t = np.cross(self._norm_vec, vecs).sum(axis=1)
        ax = _t.argmax()
        self._x_vec = np.cross(vecs[ax,:], self._norm_vec).ravel()
        self._x_vec /= np.sqrt(np.dot(self._x_vec, self._x_vec))
        self._y_vec = np.cross(self._norm_vec, self._x_vec).ravel()
        self._y_vec /= np.sqrt(np.dot(self._y_vec, self._y_vec))
        self._rot_mat = np.array([self._x_vec,self._y_vec,self._norm_vec])
        self._inv_mat = np.linalg.pinv(self._rot_mat)
        self.set_field_parameter('cp_x_vec',self._x_vec)
        self.set_field_parameter('cp_y_vec',self._y_vec)
        self.set_field_parameter('cp_z_vec',self._norm_vec)

        # Calculate coordinates of each pixel
        _co = self.dds * \
              (np.mgrid[-self.dims/2 : self.dims/2,
                        -self.dims/2 : self.dims/2] + 0.5)
        self._coord = self.center + np.outer(_co[0,:,:], self._x_vec) + \
                      np.outer(_co[1,:,:], self._y_vec)
        self._pixelmask = np.ones(self.dims*self.dims, dtype='int8')

        if node_name is False:
            self._refresh_data()
        else:
            if node_name is True: self._deserialize()
            else: self._deserialize(node_name)

    @property
    def normal(self):
        return self._norm_vec

    def _get_list_of_grids(self):
        # Just like the Cutting Plane but restrict the grids to be
        # within width/2 of the center.
        vertices = self.hierarchy.gridCorners
        # Shape = (8,3,n_grid)
        D = np.sum(self._norm_vec.reshape((1,3,1)) * vertices, axis=1) + self._d
        valid_grids = np.where(np.logical_not(np.all(D<0,axis=0) |
                                              np.all(D>0,axis=0) ))[0]
        # Now restrict these grids to a rect. prism that bounds the slice
        sliceCorners = np.array([ \
            self.center + 0.5*self.width * (+self._x_vec + self._y_vec),
            self.center + 0.5*self.width * (+self._x_vec - self._y_vec),
            self.center + 0.5*self.width * (-self._x_vec - self._y_vec),
            self.center + 0.5*self.width * (-self._x_vec + self._y_vec) ])
        sliceLeftEdge = sliceCorners.min(axis=0)
        sliceRightEdge = sliceCorners.max(axis=0)
        # Check for bounding box and grid overlap
        leftOverlap = np.less(self.hierarchy.gridLeftEdge[valid_grids],
                              sliceRightEdge).all(axis=1)
        rightOverlap = np.greater(self.hierarchy.gridRightEdge[valid_grids],
                                  sliceLeftEdge).all(axis=1)
        self._grids = self.hierarchy.grids[valid_grids[
            np.where(leftOverlap & rightOverlap)]]
        self._grids = self._grids[::-1]

    def _generate_coords(self):
        self['px'] = self._coord[:,0].ravel()
        self['py'] = self._coord[:,1].ravel()
        self['pz'] = self._coord[:,2].ravel()
        self['pdx'] = self.dds * 0.5
        self['pdy'] = self.dds * 0.5
        #self['pdz'] = self.dds * 0.5

    def _get_data_from_grid(self, grid, field):
        if not self.pf.field_info[field].particle_type:
            pointI = self._get_point_indices(grid)
            if len(pointI) == 0: return
            vc = self._calc_vertex_centered_data(grid, field)
            bds = np.array(zip(grid.LeftEdge,
                               grid.RightEdge)).ravel()
            interp = TrilinearFieldInterpolator(vc, bds, ['x', 'y', 'z'])
            self[field][pointI] = interp( \
                dict(x=self._coord[pointI,0],
                     y=self._coord[pointI,1],
                     z=self._coord[pointI,2])).ravel()

            # Mark these pixels to speed things up
            self._pixelmask[pointI] = 0

            return
        else:
            raise SyntaxError("Making a fixed resolution slice with "
                              "particles isn't supported yet.")

    def reslice(self, normal, center, width):

        # Cleanup
        del self._coord
        del self._pixelmask

        self.center = center
        self.width = width
        self.dds = self.width / self.dims
        self.set_field_parameter('center', center)
        self._norm_vec = normal/np.sqrt(np.dot(normal,normal))
        self._d = -1.0 * np.dot(self._norm_vec, self.center)
        # First we try all three, see which has the best result:
        vecs = np.identity(3)
        _t = np.cross(self._norm_vec, vecs).sum(axis=1)
        ax = _t.argmax()
        self._x_vec = np.cross(vecs[ax,:], self._norm_vec).ravel()
        self._x_vec /= np.sqrt(np.dot(self._x_vec, self._x_vec))
        self._y_vec = np.cross(self._norm_vec, self._x_vec).ravel()
        self._y_vec /= np.sqrt(np.dot(self._y_vec, self._y_vec))
        self.set_field_parameter('cp_x_vec',self._x_vec)
        self.set_field_parameter('cp_y_vec',self._y_vec)
        self.set_field_parameter('cp_z_vec',self._norm_vec)
        # Calculate coordinates of each pixel
        _co = self.dds * \
              (np.mgrid[-self.dims/2 : self.dims/2,
                        -self.dims/2 : self.dims/2] + 0.5)

        self._coord = self.center + np.outer(_co[0,:,:], self._x_vec) + \
                      np.outer(_co[1,:,:], self._y_vec)
        self._pixelmask = np.ones(self.dims*self.dims, dtype='int8')

        self._refresh_data()
        return

    #@time_execution
    def get_data(self, fields = None):
        """
        Iterates over the list of fields and generates/reads them all.
        """
        self._get_list_of_grids()
        if not self.has_key('pdx'):
            self._generate_coords()
        if fields == None:
            fields_to_get = self.fields[:]
        else:
            fields_to_get = ensure_list(fields)
        temp_data = {}
        _size = self.dims * self.dims
        for field in fields_to_get:
            if self.field_data.has_key(field): continue
            if field not in self.hierarchy.field_list:
                if self._generate_field(field):
                    continue # A "True" return means we did it
            if not self._vc_data.has_key(field):
                self._vc_data[field] = {}
            self[field] = np.zeros(_size, dtype='float64')
            for grid in self._get_grids():
                self._get_data_from_grid(grid, field)
            self[field] = self.comm.mpi_allreduce(\
                self[field], op='sum').reshape([self.dims]*2).transpose()

    def interpolate_discretize(self, *args, **kwargs):
        pass

    @cache_vc_data
    def _calc_vertex_centered_data(self, grid, field):
        #return grid.retrieve_ghost_zones(1, field, smoothed=False)
        return grid.get_vertex_centered_data(field)

    def _get_point_indices(self, grid):
        if self._pixelmask.max() == 0: return []
        k = planar_points_in_volume(self._coord, self._pixelmask,
                                    grid.LeftEdge, grid.RightEdge,
                                    grid.child_mask, just_one(grid['dx']))
        return k

    def _gen_node_name(self):
        cen_name = ("%s" % (self.center,)).replace(" ","_")[1:-1]
        L_name = ("%s" % self._norm_vec).replace(" ","_")[1:-1]
        return "%s/c%s_L%s" % \
            (self._top_node, cen_name, L_name)

class AMRQuadTreeProjBase(AMR2DData):
    """
    This is a data object corresponding to a line integral through the
    simulation domain.

    This object is typically accessed through the `proj` object that
    hangs off of hierarchy objects.  AMRQuadProj is a projection of a
    `field` along an `axis`.  The field can have an associated
    `weight_field`, in which case the values are multiplied by a weight
    before being summed, and then divided by the sum of that weight; the
    two fundamental modes of operating are direct line integral (no
    weighting) and average along a line of sight (weighting.)  What makes
    `proj` different from the standard projection mechanism is that it
    utilizes a quadtree data structure, rather than the old mechanism for
    projections.  It will not run in parallel, but serial runs should be
    substantially faster.  Note also that lines of sight are integrated at
    every projected finest-level cell.

    Parameters
    ----------
    axis : int
        The axis along which to slice.  Can be 0, 1, or 2 for x, y, z.
    field : string
        This is the field which will be "projected" along the axis.  If
        multiple are specified (in a list) they will all be projected in
        the first pass.
    weight_field : string
        If supplied, the field being projected will be multiplied by this
        weight value before being integrated, and at the conclusion of the
        projection the resultant values will be divided by the projected
        `weight_field`.
    max_level : int
        If supplied, only cells at or below this level will be projected.
    center : array_like, optional
        The 'center' supplied to fields that use it.  Note that this does
        not have to have `coord` as one value.  Strictly optional.
    source : `yt.data_objects.api.AMRData`, optional
        If specified, this will be the data source used for selecting
        regions to project.
    node_name: string, optional
        The node in the .yt file to find or store this slice at.  Should
        probably not be used.
    field_cuts : list of strings, optional
        If supplied, each of these strings will be evaluated to cut a
        region of a grid out.  They can be of the form "grid['Temperature']
        > 100" for instance.
    preload_style : string
        Either 'level', 'all', or None (default).  Defines how grids are
        loaded -- either level by level, or all at once.  Only applicable
        during parallel runs.
    serialize : bool, optional
        Whether we should store this projection in the .yt file or not.
    kwargs : dict of items
        Any additional values are passed as field parameters that can be
        accessed by generated fields.

    Examples
    --------

    >>> pf = load("RedshiftOutput0005")
    >>> qproj = pf.h.quad_proj(0, "Density")
    >>> print qproj["Density"]
    """
    _top_node = "/Projections"
    _key_fields = AMR2DData._key_fields + ['weight_field']
    _type_name = "proj"
    _con_args = ('axis', 'field', 'weight_field')
    def __init__(self, axis, field, weight_field = None,
                 max_level = None, center = None, pf = None,
                 source=None, node_name = None, field_cuts = None,
                 preload_style=None, serialize=True,
                 style = "integrate", **kwargs):
        AMR2DData.__init__(self, axis, field, pf, node_name = None, **kwargs)
        self.proj_style = style
        if style == "mip":
            self.func = np.max
        elif style == "integrate":
            self.func = np.sum # for the future
        else:
            raise NotImplementedError(style)
        self.weight_field = weight_field
        self._field_cuts = field_cuts
        self.serialize = serialize
        self._set_center(center)
        if center is not None: self.set_field_parameter('center',center)
        self._node_name = node_name
        self._initialize_source(source)
        self._grids = self.source._grids
        if max_level == None:
            max_level = self.hierarchy.max_level
        if self.source is not None:
            max_level = min(max_level, self.source.grid_levels.max())
        self._max_level = max_level
        self._weight = weight_field
        self.preload_style = preload_style
        self._deserialize(node_name)
        self._refresh_data()
        if self._okay_to_serialize and self.serialize: self._serialize(node_name=self._node_name)

    @property
    def _mrep(self):
        return MinimalProjectionData(self)

    def hub_upload(self):
        self._mrep.upload()

    def _convert_field_name(self, field):
        if field == "weight_field": return "weight_field_%s" % self._weight
        if field in self._key_fields: return field
        return "%s_%s" % (field, self._weight)

    def _initialize_source(self, source = None):
        if source is None:
            source = self.pf.h.all_data()
            self._check_region = False
            #self._okay_to_serialize = (not check)
        else:
            self._distributed = False
            self._okay_to_serialize = False
            self._check_region = True
            for k, v in source.field_parameters.items():
                if k not in self.field_parameters:
                    self.set_field_parameter(k,v)
        self.source = source
        if self._field_cuts is not None:
            # Override if field cuts are around; we don't want to serialize!
            self._check_region = True
            self._okay_to_serialize = False
        if self._node_name is not None:
            self._node_name = "%s/%s" % (self._top_node,self._node_name)
            self._okay_to_serialize = True

    def _get_tree(self, nvals):
        xax = x_dict[self.axis]
        yax = y_dict[self.axis]
        xd = self.pf.domain_dimensions[xax]
        yd = self.pf.domain_dimensions[yax]
        bounds = (self.pf.domain_left_edge[xax],
                  self.pf.domain_right_edge[yax],
                  self.pf.domain_left_edge[xax],
                  self.pf.domain_right_edge[yax])
        return QuadTree(np.array([xd,yd], dtype='int64'), nvals,
                        bounds, style = self.proj_style)

    def _get_dls(self, grid, fields):
        # Place holder for a time when maybe we will not be doing just
        # a single dx for every field.
        dls = []
        convs = []
        for field in fields + [self._weight]:
            if field is None: continue
            dls.append(just_one(grid['d%s' % axis_names[self.axis]]))
            convs.append(self.pf.units[self.pf.field_info[field].projection_conversion])
        dls = np.array(dls)
        convs = np.array(convs)
        if self.proj_style == "mip":
            dls[:] = 1.0
            convs[:] = 1.0
        return dls, convs

    def to_pw(self, fields=None, center='c', width=None, axes_unit=None,
               origin='center-window'):
        r"""Create a :class:`~yt.visualization.plot_window.PWViewerMPL` from this
        object.

        This is a bare-bones mechanism of creating a plot window from this
        object, which can then be moved around, zoomed, and on and on.  All
        behavior of the plot window is relegated to that routine.
        """
        pw = self._get_pw(fields, center, width, origin, axes_unit, 'Projection')
        return pw

    def get_data(self, fields = None):
        if fields is None: fields = ensure_list(self.fields)[:]
        else: fields = ensure_list(fields)
        # We need a new tree for every single set of fields we add
        self._obtain_fields(fields, self._node_name)
        fields = [f for f in fields if f not in self.field_data]
        if len(fields) == 0: return
        tree = self._get_tree(len(fields))
        coord_data = []
        field_data = []
        dxs = []
        # We do this here, but I am not convinced it should be done here
        # It is probably faster, as it consolidates IO, but if we did it in
        # _project_level, then it would be more memory conservative
        if self.preload_style == 'all':
            dependencies = self.get_dependencies(fields)
            mylog.debug("Preloading %s grids and getting %s",
                            len([g for g in self.source._get_grid_objs()]),
                            dependencies)
            self.comm.preload([g for g in self._get_grid_objs()],
                          dependencies, self.hierarchy.io)
        # By changing the remove-from-tree method to accumulate, we can avoid
        # having to do this by level, and instead do it by CPU file
        for level in range(0, self._max_level+1):
            if self.preload_style == 'level':
                self.comm.preload([g for g in self._get_grid_objs()
                                 if g.Level == level],
                              self.get_dependencies(fields), self.hierarchy.io)
            self._add_level_to_tree(tree, level, fields)
            mylog.debug("End of projecting level level %s, memory usage %0.3e",
                        level, get_memory_usage()/1024.)
        # Note that this will briefly double RAM usage
        if self.proj_style == "mip":
            merge_style = -1
            op = "max"
        elif self.proj_style == "integrate":
            merge_style = 1
            op = "sum"
        else:
            raise NotImplementedError
        #tree = self.comm.merge_quadtree_buffers(tree, merge_style=merge_style)
        buf = list(tree.tobuffer())
        del tree
        new_buf = [buf.pop(0)]
        new_buf.append(self.comm.mpi_allreduce(buf.pop(0), op=op))
        new_buf.append(self.comm.mpi_allreduce(buf.pop(0), op=op))
        tree = self._get_tree(len(fields))
        tree.frombuffer(new_buf[0], new_buf[1], new_buf[2], merge_style)
        coord_data, field_data, weight_data, dxs, dys = [], [], [], [], []
        for level in range(0, self._max_level + 1):
            npos, nvals, nwvals = tree.get_all_from_level(level, False)
            coord_data.append(npos)
            if self._weight is not None: nvals /= nwvals[:,None]
            field_data.append(nvals)
            weight_data.append(nwvals)
            gs = self.source.select_grids(level)
            if len(gs) > 0:
                dx = gs[0].dds[x_dict[self.axis]]
                dy = gs[0].dds[y_dict[self.axis]]
            else:
                dx = dy = 0.0
            dxs.append(np.ones(nvals.shape[0], dtype='float64') * dx)
            dys.append(np.ones(nvals.shape[0], dtype='float64') * dy)
        coord_data = np.concatenate(coord_data, axis=0).transpose()
        field_data = np.concatenate(field_data, axis=0).transpose()
        if self._weight is None:
            dls, convs = self._get_dls(self._grids[0], fields)
            field_data *= convs[:,None]
        weight_data = np.concatenate(weight_data, axis=0).transpose()
        dxs = np.concatenate(dxs, axis=0).transpose()
        dys = np.concatenate(dys, axis=0).transpose()
        # We now convert to half-widths and center-points
        data = {}
        data['pdx'] = dxs
        data['pdy'] = dys
        ox = self.pf.domain_left_edge[x_dict[self.axis]]
        oy = self.pf.domain_left_edge[y_dict[self.axis]]
        data['px'] = (coord_data[0,:]+0.5) * data['pdx'] + ox
        data['py'] = (coord_data[1,:]+0.5) * data['pdy'] + oy
        data['weight_field'] = weight_data
        del coord_data
        data['pdx'] *= 0.5
        data['pdy'] *= 0.5
        data['fields'] = field_data
        # Now we run the finalizer, which is ignored if we don't need it
        field_data = np.vsplit(data.pop('fields'), len(fields))
        for fi, field in enumerate(fields):
            self[field] = field_data[fi].ravel()
            if self.serialize: self._store_fields(field, self._node_name)
        for i in data.keys(): self[i] = data.pop(i)
        mylog.info("Projection completed")

    def _add_grid_to_tree(self, tree, grid, fields, zero_out, dls):
        # We build up the fields to add
        if self._weight is None or fields is None:
            weight_data = np.ones(grid.ActiveDimensions, dtype='float64')
            if zero_out: weight_data[grid.child_indices] = 0
            masked_data = [fd.astype('float64') * weight_data
                           for fd in self._get_data_from_grid(grid, fields)]
            wdl = 1.0
        else:
            fields_to_get = list(set(fields + [self._weight]))
            field_data = dict(zip(
                fields_to_get, self._get_data_from_grid(grid, fields_to_get)))
            weight_data = field_data[self._weight].copy().astype('float64')
            if zero_out: weight_data[grid.child_indices] = 0
            masked_data  = [field_data[field].copy().astype('float64') * weight_data
                                for field in fields]
            del field_data
            wdl = dls[-1]
        full_proj = [self.func(field, axis=self.axis) * dl
                     for field, dl in zip(masked_data, dls)]
        weight_proj = self.func(weight_data, axis=self.axis) * wdl
        if (self._check_region and not self.source._is_fully_enclosed(grid)) or self._field_cuts is not None:
            used_data = self._get_points_in_region(grid).astype('bool')
            used_points = np.logical_or.reduce(used_data, self.axis)
        else:
            used_data = np.array([1.0], dtype='bool')
            used_points = slice(None)
        xind, yind = [arr[used_points].ravel()
                      for arr in np.indices(full_proj[0].shape)]
        start_index = grid.get_global_startindex()
        xpoints = (xind + (start_index[x_dict[self.axis]])).astype('int64')
        ypoints = (yind + (start_index[y_dict[self.axis]])).astype('int64')
        to_add = np.array([d[used_points].ravel() for d in full_proj], order='F')
        tree.add_array_to_tree(grid.Level, xpoints, ypoints,
                    to_add, weight_proj[used_points].ravel())

    def _add_level_to_tree(self, tree, level, fields):
        grids_to_project = [g for g in self._get_grid_objs()
                            if g.Level == level]
        grids_to_initialize = [g for g in self._grids if (g.Level == level)]
        zero_out = (level != self._max_level)
        if len(grids_to_initialize) == 0: return
        pbar = get_pbar('Initializing tree % 2i / % 2i ' \
                          % (level, self._max_level), len(grids_to_initialize))
        start_index = np.empty(2, dtype="int64")
        dims = np.empty(2, dtype="int64")
        xax = x_dict[self.axis]
        yax = y_dict[self.axis]
        for pi, grid in enumerate(grids_to_initialize):
            dims[0] = grid.ActiveDimensions[xax]
            dims[1] = grid.ActiveDimensions[yax]
            ind = grid.get_global_startindex()
            start_index[0] = ind[xax]
            start_index[1] = ind[yax]
            tree.initialize_grid(level, start_index, dims)
            pbar.update(pi)
        pbar.finish()
        if len(grids_to_project) > 0:
            dls, convs = self._get_dls(grids_to_project[0], fields)
            pbar = get_pbar('Projecting  level % 2i / % 2i ' \
                              % (level, self._max_level), len(grids_to_project))
            for pi, grid in enumerate(grids_to_project):
                self._add_grid_to_tree(tree, grid, fields, zero_out, dls)
                pbar.update(pi)
                grid.clear_data()
            pbar.finish()
        return

    def _get_points_in_region(self, grid):
        pointI = self.source._get_point_indices(grid, use_child_mask=False)
        point_mask = np.zeros(grid.ActiveDimensions)
        point_mask[pointI] = 1.0
        if self._field_cuts is not None:
            for cut in self._field_cuts:
                point_mask *= eval(cut)
        return point_mask

    @restore_grid_state
    def _get_data_from_grid(self, grid, fields):
        fields = ensure_list(fields)
        if self._check_region:
            bad_points = self._get_points_in_region(grid)
        else:
            bad_points = 1.0
        return [grid[field] * bad_points for field in fields]

    def _gen_node_name(self):
        return  "%s/%s" % \
            (self._top_node, self.axis)


class AMRProjBase(AMR2DData):
    """
    This is a data object corresponding to a line integral through the
    simulation domain.

    This object is typically accessed through the `proj` object that
    hangs off of hierarchy objects.  AMRProj is a projection of a `field`
    along an `axis`.  The field can have an associated `weight_field`, in
    which case the values are multiplied by a weight before being summed,
    and then divided by the sum of that weight; the two fundamental modes
    of operating are direct line integral (no weighting) and average along
    a line of sight (weighting.)  Note also that lines of sight are
    integrated at every projected finest-level cell

    Parameters
    ----------
    axis : int
        The axis along which to slice.  Can be 0, 1, or 2 for x, y, z.
    field : string
        This is the field which will be "projected" along the axis.  If
        multiple are specified (in a list) they will all be projected in
        the first pass.
    weight_field : string
        If supplied, the field being projected will be multiplied by this
        weight value before being integrated, and at the conclusion of the
        projection the resultant values will be divided by the projected
        `weight_field`.
    max_level : int
        If supplied, only cells at or below this level will be projected.
    center : array_like, optional
        The 'center' supplied to fields that use it.  Note that this does
        not have to have `coord` as one value.  Strictly optional.
    source : `yt.data_objects.api.AMRData`, optional
        If specified, this will be the data source used for selecting
        regions to project.
    node_name: string, optional
        The node in the .yt file to find or store this slice at.  Should
        probably not be used.
    field_cuts : list of strings, optional
        If supplied, each of these strings will be evaluated to cut a
        region of a grid out.  They can be of the form "grid['Temperature']
        > 100" for instance.
    preload_style : string
        Either 'level' (default) or 'all'.  Defines how grids are loaded --
        either level by level, or all at once.  Only applicable during
        parallel runs.
    serialize : bool, optional
        Whether we should store this projection in the .yt file or not.
    kwargs : dict of items
        Any additional values are passed as field parameters that can be
        accessed by generated fields.

    Examples
    --------

    >>> pf = load("RedshiftOutput0005")
    >>> proj = pf.h.proj(0, "Density")
    >>> print proj["Density"]
    """
    _top_node = "/Projections"
    _key_fields = AMR2DData._key_fields + ['weight_field']
    _type_name = "overlap_proj"
    _con_args = ('axis', 'field', 'weight_field')
    def __init__(self, axis, field, weight_field = None,
                 max_level = None, center = None, pf = None,
                 source=None, node_name = None, field_cuts = None,
                 preload_style='level', serialize=True,**kwargs):
        AMR2DData.__init__(self, axis, field, pf, node_name = None, **kwargs)
        self.proj_style = "integrate"
        self.weight_field = weight_field
        self._field_cuts = field_cuts
        self.serialize = serialize
        self._set_center(center)
        if center is not None: self.set_field_parameter('center',center)
        self._node_name = node_name
        self._initialize_source(source)
        self._grids = self.source._grids
        if max_level == None:
            max_level = self.hierarchy.max_level
        if self.source is not None:
            max_level = min(max_level, self.source.grid_levels.max())
        self._max_level = max_level
        self._weight = weight_field
        self.preload_style = preload_style
        self.func = np.sum # for the future
        self.__retval_coords = {}
        self.__retval_fields = {}
        self.__retval_coarse = {}
        self.__overlap_masks = {}
        self._deserialize(node_name)
        self._refresh_data()
        if self._okay_to_serialize and self.serialize: self._serialize(node_name=self._node_name)

    def _convert_field_name(self, field):
        if field == "weight_field": return "weight_field_%s" % self._weight
        if field in self._key_fields: return field
        return "%s_%s" % (field, self._weight)

    def _initialize_source(self, source = None):
        if source is None:
            check, source = self.partition_hierarchy_2d(self.axis)
            self._check_region = check
            #self._okay_to_serialize = (not check)
        else:
            self._distributed = False
            self._okay_to_serialize = False
            self._check_region = True
            for k, v in source.field_parameters.items():
                if k not in self.field_parameters:
                    self.set_field_parameter(k,v)
        self.source = source
        if self._field_cuts is not None:
            # Override if field cuts are around; we don't want to serialize!
            self._check_region = True
            self._okay_to_serialize = False
        if self._node_name is not None:
            self._node_name = "%s/%s" % (self._top_node,self._node_name)
            self._okay_to_serialize = True

    #@time_execution
    def __calculate_overlap(self, level):
        s = self.source
        mylog.info("Generating overlap masks for level %s", level)
        i = 0
        pbar = get_pbar("Reading and masking grids ", len(s._grids))
        mylog.debug("Examining level %s", level)
        grids = s.select_grid_indices(level)
        RE = s.grid_right_edge[grids]
        LE = s.grid_left_edge[grids]
        for grid in s._grids[grids]:
            pbar.update(i)
            self.__overlap_masks[grid.id] = \
                grid._generate_overlap_masks(self.axis, LE, RE)
            i += 1
        pbar.finish()
        mylog.info("Finished calculating overlap.")

    def _get_dls(self, grid, fields):
        # Place holder for a time when maybe we will not be doing just
        # a single dx for every field.
        dls = []
        convs = []
        for field in fields + [self._weight]:
            if field is None: continue
            dls.append(just_one(grid['d%s' % axis_names[self.axis]]))
            convs.append(self.pf.units[self.pf.field_info[field].projection_conversion])
        return np.array(dls), np.array(convs)

    def __project_level(self, level, fields):
        grids_to_project = self.source.select_grids(level)
        dls, convs = self._get_dls(grids_to_project[0], fields)
        zero_out = (level != self._max_level)
        pbar = get_pbar('Projecting  level % 2i / % 2i ' \
                          % (level, self._max_level), len(grids_to_project))
        for pi, grid in enumerate(grids_to_project):
            g_coords, g_fields = self._project_grid(grid, fields, zero_out)
            self.__retval_coords[grid.id] = g_coords
            self.__retval_fields[grid.id] = g_fields
            for fi in range(len(fields)): g_fields[fi] *= dls[fi]
            if self._weight is not None: g_coords[3] *= dls[-1]
            pbar.update(pi)
            grid.clear_data()
        pbar.finish()
        self.__combine_grids_on_level(level) # In-place
        if level > 0 and level <= self._max_level:
            self.__refine_to_level(level) # In-place
        coord_data = []
        field_data = []
        for grid in grids_to_project:
            coarse = self.__retval_coords[grid.id][2]==0 # Where childmask = 0
            fine = ~coarse
            coord_data.append([pi[fine] for pi in self.__retval_coords[grid.id]])
            field_data.append([pi[fine] for pi in self.__retval_fields[grid.id]])
            self.__retval_coords[grid.id] = [pi[coarse] for pi in self.__retval_coords[grid.id]]
            self.__retval_fields[grid.id] = [pi[coarse] for pi in self.__retval_fields[grid.id]]
        coord_data = np.concatenate(coord_data, axis=1)
        field_data = np.concatenate(field_data, axis=1)
        if self._weight is not None:
            field_data = field_data / coord_data[3,:].reshape((1,coord_data.shape[1]))
        else:
            field_data *= convs[...,np.newaxis]
        mylog.info("Level %s done: %s final", \
                   level, coord_data.shape[1])
        pdx = grids_to_project[0].dds[x_dict[self.axis]] # this is our dl
        pdy = grids_to_project[0].dds[y_dict[self.axis]] # this is our dl
        return coord_data, pdx, pdy, field_data

    def __combine_grids_on_level(self, level):
        grids = self.source.select_grids(level)
        grids_i = self.source.select_grid_indices(level)
        pbar = get_pbar('Combining   level % 2i / % 2i ' \
                          % (level, self._max_level), len(grids))
        # We have an N^2 check, so we try to be as quick as possible
        # and to skip as many as possible
        for pi, grid1 in enumerate(grids):
            pbar.update(pi)
            if self.__retval_coords[grid1.id][0].shape[0] == 0: continue
            for grid2 in self.source._grids[grids_i][self.__overlap_masks[grid1.id]]:
                if self.__retval_coords[grid2.id][0].shape[0] == 0 \
                  or grid1.id == grid2.id:
                    continue
                args = [] # First is source, then destination
                args += self.__retval_coords[grid2.id] + [self.__retval_fields[grid2.id]]
                args += self.__retval_coords[grid1.id] + [self.__retval_fields[grid1.id]]
                args.append(1) # Refinement factor
                args.append(np.ones(args[0].shape, dtype='int64'))
                kk = CombineGrids(*args)
                goodI = args[-1].astype('bool')
                self.__retval_coords[grid2.id] = \
                    [coords[goodI] for coords in self.__retval_coords[grid2.id]]
                self.__retval_fields[grid2.id] = \
                    [fields[goodI] for fields in self.__retval_fields[grid2.id]]
        pbar.finish()

    def __refine_to_level(self, level):
        grids = self.source.select_grids(level)
        grids_up = self.source.select_grid_indices(level - 1)
        pbar = get_pbar('Refining to level % 2i / % 2i ' \
                          % (level, self._max_level), len(grids))
        for pi, grid1 in enumerate(grids):
            pbar.update(pi)
            for parent in ensure_list(grid1.Parent):
                if parent.id not in self.__overlap_masks: continue
                for grid2 in self.source._grids[grids_up][self.__overlap_masks[parent.id]]:
                    if self.__retval_coords[grid2.id][0].shape[0] == 0: continue
                    args = []
                    args += self.__retval_coords[grid2.id] + [self.__retval_fields[grid2.id]]
                    args += self.__retval_coords[grid1.id] + [self.__retval_fields[grid1.id]]
                    # Refinement factor, which is same in all directions.  Note
                    # that this complicated rounding is because sometimes
                    # epsilon differences in dds between the grids causes this
                    # to round to up or down from the expected value.
                    args.append(int(np.rint(grid2.dds / grid1.dds)[0]))
                    args.append(np.ones(args[0].shape, dtype='int64'))
                    kk = CombineGrids(*args)
                    goodI = args[-1].astype('bool')
                    self.__retval_coords[grid2.id] = \
                        [coords[goodI] for coords in self.__retval_coords[grid2.id]]
                    self.__retval_fields[grid2.id] = \
                        [fields[goodI] for fields in self.__retval_fields[grid2.id]]
        for grid1 in self.source.select_grids(level-1):
            if not self._check_region and self.__retval_coords[grid1.id][0].size != 0:
                mylog.error("Something messed up, and %s still has %s points of data",
                            grid1, self.__retval_coords[grid1.id][0].size)
                mylog.error("Please contact the yt-users mailing list.")
                raise ValueError(grid1, self.__retval_coords[grid1.id])
        pbar.finish()

    #@time_execution
    def get_data(self, fields = None):
        if fields is None: fields = ensure_list(self.fields)[:]
        else: fields = ensure_list(fields)
        self._obtain_fields(fields, self._node_name)
        fields = [f for f in fields if f not in self.field_data]
        if len(fields) == 0: return
        coord_data = []
        field_data = []
        pdxs = []
        pdys = []
        # We do this here, but I am not convinced it should be done here
        # It is probably faster, as it consolidates IO, but if we did it in
        # _project_level, then it would be more memory conservative
        if self.preload_style == 'all':
            print "Preloading %s grids and getting %s" % (
                    len(self.source._grids), self.get_dependencies(fields))
            self.comm.preload(self.source._grids,
                          self.get_dependencies(fields), self.hierarchy.io)
        for level in range(0, self._max_level+1):
            if self.preload_style == 'level':
                self.comm.preload(self.source.select_grids(level),
                              self.get_dependencies(fields), self.hierarchy.io)
            self.__calculate_overlap(level)
            my_coords, my_pdx, my_pdy, my_fields = \
                self.__project_level(level, fields)
            coord_data.append(my_coords)
            field_data.append(my_fields)
            pdxs.append(my_pdx * np.ones(my_coords.shape[1], dtype='float64'))
            pdys.append(my_pdx * np.ones(my_coords.shape[1], dtype='float64'))
            if self._check_region and False:
                check=self.__cleanup_level(level - 1)
                if len(check) > 0: all_data.append(check)
            # Now, we should clean up after ourselves...
            for grid in self.source.select_grids(level - 1):
                del self.__retval_coords[grid.id]
                del self.__retval_fields[grid.id]
                del self.__overlap_masks[grid.id]
            mylog.debug("End of projecting level level %s, memory usage %0.3e",
                        level, get_memory_usage()/1024.)
        coord_data = np.concatenate(coord_data, axis=1)
        field_data = np.concatenate(field_data, axis=1)
        pdxs = np.concatenate(pdxs, axis=1)
        pdys = np.concatenate(pdys, axis=1)
        # We now convert to half-widths and center-points
        data = {}
        data['pdx'] = pdxs; del pdxs
        data['pdy'] = pdys; del pdys
        ox = self.pf.domain_left_edge[x_dict[self.axis]]
        oy = self.pf.domain_left_edge[y_dict[self.axis]]
        data['px'] = (coord_data[0,:]+0.5) * data['pdx'] + ox
        data['py'] = (coord_data[1,:]+0.5) * data['pdx'] + oy
        data['weight_field'] = coord_data[3,:].copy()
        del coord_data
        data['pdx'] *= 0.5
        data['pdy'] *= 0.5
        data['fields'] = field_data
        # Now we run the finalizer, which is ignored if we don't need it
        data = self.comm.par_combine_object(data, datatype='dict', op='cat')
        field_data = np.vsplit(data.pop('fields'), len(fields))
        for fi, field in enumerate(fields):
            self[field] = field_data[fi].ravel()
            if self.serialize: self._store_fields(field, self._node_name)
        for i in data.keys(): self[i] = data.pop(i)
        mylog.info("Projection completed")

    def add_fields(self, fields, weight = "CellMassMsun"):
        pass

    def to_pw(self, fields=None, center='c', width=None, axes_unit=None,
               origin='center-window'):
        r"""Create a :class:`~yt.visualization.plot_window.PWViewerMPL` from this
        object.

        This is a bare-bones mechanism of creating a plot window from this
        object, which can then be moved around, zoomed, and on and on.  All
        behavior of the plot window is relegated to that routine.
        """
        pw = self._get_pw(fields, center, width, origin, axes_unit, 'Projection')
        return pw

    def _project_grid(self, grid, fields, zero_out):
        # We split this next bit into two sections to try to limit the IO load
        # on the system.  This way, we perserve grid state (@restore_grid_state
        # in _get_data_from_grid *and* we attempt not to load weight data
        # independently of the standard field data.
        if self._weight is None:
            weight_data = np.ones(grid.ActiveDimensions, dtype='float64')
            if zero_out: weight_data[grid.child_indices] = 0
            masked_data = [fd.astype('float64') * weight_data
                           for fd in self._get_data_from_grid(grid, fields)]
        else:
            fields_to_get = list(set(fields + [self._weight]))
            field_data = dict(zip(
                fields_to_get, self._get_data_from_grid(grid, fields_to_get)))
            weight_data = field_data[self._weight].copy().astype('float64')
            if zero_out: weight_data[grid.child_indices] = 0
            masked_data  = [field_data[field].copy().astype('float64') * weight_data
                                for field in fields]
            del field_data
        # if we zero it out here, then we only have to zero out the weight!
        full_proj = [self.func(field, axis=self.axis) for field in masked_data]
        weight_proj = self.func(weight_data, axis=self.axis)
        if (self._check_region and not self.source._is_fully_enclosed(grid)) or self._field_cuts is not None:
            used_data = self._get_points_in_region(grid).astype('bool')
            used_points = np.where(np.logical_or.reduce(used_data, self.axis))
        else:
            used_data = np.array([1.0], dtype='bool')
            used_points = slice(None)
        if zero_out:
            subgrid_mask = np.logical_and.reduce(
                                np.logical_or(grid.child_mask,
                                             ~used_data),
                                self.axis).astype('int64')
        else:
            subgrid_mask = np.ones(full_proj[0].shape, dtype='int64')
        xind, yind = [arr[used_points].ravel() for arr in np.indices(full_proj[0].shape)]
        start_index = grid.get_global_startindex()
        xpoints = (xind + (start_index[x_dict[self.axis]])).astype('int64')
        ypoints = (yind + (start_index[y_dict[self.axis]])).astype('int64')
        return ([xpoints, ypoints,
                subgrid_mask[used_points].ravel(),
                weight_proj[used_points].ravel()],
                [data[used_points].ravel() for data in full_proj])

    def _get_points_in_region(self, grid):
        pointI = self.source._get_point_indices(grid, use_child_mask=False)
        point_mask = np.zeros(grid.ActiveDimensions)
        point_mask[pointI] = 1.0
        if self._field_cuts is not None:
            for cut in self._field_cuts:
                point_mask *= eval(cut)
        return point_mask

    @restore_grid_state
    def _get_data_from_grid(self, grid, fields):
        fields = ensure_list(fields)
        if self._check_region:
            bad_points = self._get_points_in_region(grid)
        else:
            bad_points = 1.0
        return [grid[field] * bad_points for field in fields]

    def _gen_node_name(self):
        return  "%s/%s" % \
            (self._top_node, self.axis)

class AMRFixedResProjectionBase(AMR2DData):
    """
    This is a data structure that projects grids, but only to fixed (rather
    than variable) resolution.

    This object is typically accessed through the `fixed_res_proj` object
    that hangs off of hierarchy objects.  This projection mechanism is much
    simpler than the standard, variable-resolution projection.  Rather than
    attempt to identify the highest-resolution element along every possible
    line of sight, this data structure simply deposits every cell into one
    of a fixed number of bins.  It is suitable for inline analysis, and it
    should scale nicely.

    Parameters
    ----------
    axis : int
        The axis along which to project.  Can be 0, 1, or 2 for x, y, z.
    level : int
        This is the level to which values will be projected.  Note that
        the pixel size in the projection will be identical to a cell at
        this level of refinement in the simulation.
    left_edge : array of ints
        The left edge, in level-local integer coordinates, of the
        projection
    dims : array of ints
        The dimensions of the projection (which, in concert with the
        left_edge, serves to define its right edge.)
    fields : list of strings, optional
        If you want the object to pre-retrieve a set of fields, supply them
        here.  This is not necessary.
    kwargs : dict of items
        Any additional values are passed as field parameters that can be
        accessed by generated fields.

    Examples
    --------

    >>> pf = load("RedshiftOutput0005")
    >>> fproj = pf.h.fixed_res_proj(1, [0, 0, 0], [64, 64, 64], ["Density"])
    >>> print fproj["Density"]
    """
    _top_node = "/Projections"
    _type_name = "fixed_res_proj"
    _con_args = ('axis', 'field', 'weight_field')
    def __init__(self, axis, level, left_edge, dims,
                 fields = None, pf=None, **kwargs):
        AMR2DData.__init__(self, axis, fields, pf, **kwargs)
        self.left_edge = np.array(left_edge)
        self.level = level
        self.dds = self.pf.h.select_grids(self.level)[0].dds.copy()
        self.dims = np.array([dims]*2)
        self.ActiveDimensions = np.array([dims]*3, dtype='int32')
        self.right_edge = self.left_edge + self.ActiveDimensions*self.dds
        self.global_startindex = np.rint((self.left_edge - self.pf.domain_left_edge)
                                         /self.dds).astype('int64')
        self._dls = {}
        self.domain_width = np.rint((self.pf.domain_right_edge -
                    self.pf.domain_left_edge)/self.dds).astype('int64')
        self._refresh_data()

    def _get_list_of_grids(self):
        if self._grids is not None: return
        if np.any(self.left_edge < self.pf.domain_left_edge) or \
           np.any(self.right_edge > self.pf.domain_right_edge):
            grids,ind = self.pf.hierarchy.get_periodic_box_grids(
                            self.left_edge, self.right_edge)
        else:
            grids,ind = self.pf.hierarchy.get_box_grids(
                            self.left_edge, self.right_edge)
        level_ind = (self.pf.hierarchy.grid_levels.ravel()[ind] <= self.level)
        sort_ind = np.argsort(self.pf.h.grid_levels.ravel()[ind][level_ind])
        self._grids = self.pf.hierarchy.grids[ind][level_ind][(sort_ind,)][::-1]

    def _generate_coords(self):
        xax = x_dict[self.axis]
        yax = y_dict[self.axis]
        ci = self.left_edge + self.dds*0.5
        cf = self.left_edge + self.dds*(self.ActiveDimensions-0.5)
        cx = np.mgrid[ci[xax]:cf[xax]:self.ActiveDimensions[xax]*1j]
        cy = np.mgrid[ci[yax]:cf[yax]:self.ActiveDimensions[yax]*1j]
        blank = np.ones( (self.ActiveDimensions[xax],
                          self.ActiveDimensions[yax]), dtype='float64')
        self['px'] = cx[None,:] * blank
        self['py'] = cx[:,None] * blank
        self['pdx'] = self.dds[xax]
        self['pdy'] = self.dds[yax]

    #@time_execution
    def get_data(self, fields = None):
        """
        Iterates over the list of fields and generates/reads them all.
        """
        self._get_list_of_grids()
        if not self.has_key('pdx'):
            self._generate_coords()
        if fields == None:
            fields_to_get = [f for f in self.fields if f not in self._key_fields]
        else:
            fields_to_get = ensure_list(fields)
        if len(fields_to_get) == 0: return
        temp_data = {}
        for field in fields_to_get:
            self[field] = np.zeros(self.dims, dtype='float64')
        dls = self.__setup_dls(fields_to_get)
        for i,grid in enumerate(self._get_grids()):
            mylog.debug("Getting fields from %s", i)
            self._get_data_from_grid(grid, fields_to_get, dls)
        mylog.info("IO completed; summing")
        for field in fields_to_get:
            self[field] = self.comm.mpi_allreduce(self[field], op='sum')
            conv = self.pf.units[self.pf.field_info[field].projection_conversion]
            self[field] *= conv

    def __setup_dls(self, fields):
        dls = {}
        for level in range(self.level+1):
            dls[level] = []
            grid = self.select_grids(level)[0]
            for field in fields:
                if field is None: continue
                dls[level].append(float(just_one(grid['d%s' % axis_names[self.axis]])))
        return dls

    @restore_grid_state
    def _get_data_from_grid(self, grid, fields, dls):
        g_fields = [grid[field].astype("float64") for field in fields]
        c_fields = [self[field] for field in fields]
        ref_ratio = self.pf.refine_by**(self.level - grid.Level)
        FillBuffer(ref_ratio,
            grid.get_global_startindex(), self.global_startindex,
            c_fields, g_fields,
            self.ActiveDimensions, grid.ActiveDimensions,
            grid.child_mask, self.domain_width, dls[grid.Level],
            self.axis)

class AMR3DData(AMRData, GridPropertiesMixin, ParallelAnalysisInterface):
    _key_fields = ['x','y','z','dx','dy','dz']
    """
    Class describing a cluster of data points, not necessarily sharing any
    particular attribute.
    """
    _spatial = False
    _num_ghost_zones = 0
    def __init__(self, center, fields, pf = None, **kwargs):
        """
        Returns an instance of AMR3DData, or prepares one.  Usually only
        used as a base class.  Note that *center* is supplied, but only used
        for fields and quantities that require it.
        """
        ParallelAnalysisInterface.__init__(self)
        AMRData.__init__(self, pf, fields, **kwargs)
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
            points.append((np.ones(
                grid.ActiveDimensions,dtype='float64')*grid['dx'])\
                    [self._get_point_indices(grid)])
            t = np.concatenate([t,points])
            del points
        self['dx'] = t
        #self['dy'] = t
        #self['dz'] = t
        mylog.info("Done with coordinates")

    @restore_grid_state
    def _generate_grid_coords(self, grid, field=None):
        pointI = self._get_point_indices(grid)
        dx = np.ones(pointI[0].shape[0], 'float64') * grid.dds[0]
        tr = np.array([grid['x'][pointI].ravel(), \
                grid['y'][pointI].ravel(), \
                grid['z'][pointI].ravel(), \
                grid["RadiusCode"][pointI].ravel(),
                dx, grid["GridIndices"][pointI].ravel()], 'float64').swapaxes(0,1)
        return tr

    def get_data(self, fields=None, in_grids=False, force_particle_read = False):
        if self._grids == None:
            self._get_list_of_grids()
        if len(self._grids) == 0:
            raise YTNoDataInObjectError(self)
        points = []
        if not fields:
            fields_to_get = self.fields[:]
        else:
            fields_to_get = ensure_list(fields)
        mylog.debug("Going to obtain %s", fields_to_get)
        for field in fields_to_get:
            if self.field_data.has_key(field):
                continue
            # There are a lot of 'ands' here, but I think they are all
            # necessary.
            if force_particle_read == False and \
               self.pf.field_info.has_key(field) and \
               self.pf.field_info[field].particle_type and \
               self.pf.h.io._particle_reader and \
               not isinstance(self, AMRBooleanRegionBase):
                try:
                    self.particles.get_data(field)
                    if field not in self.field_data:
                        self._generate_field(field)
                    continue
                except KeyError:
                    # This happens for fields like ParticleRadiuskpc
                    pass
            if field not in self.hierarchy.field_list and not in_grids:
                if self._generate_field(field):
                    continue # True means we already assigned it
            mylog.info("Getting field %s from %s", field, len(self._grids))
            self[field] = np.concatenate(
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
            if grid.NumberOfParticles == 0: return np.array([], dtype='int64')
            pointI = self._get_particle_indices(grid)
            if self.pf.field_info[field].vector_field:
                f = grid[field]
                return np.array([f[i,:][pointI] for i in range(3)])
            if self._is_fully_enclosed(grid): return grid[field].ravel()
            return grid[field][pointI].ravel()
        if field in self.pf.field_info and self.pf.field_info[field].vector_field:
            pointI = self._get_point_indices(grid)
            f = grid[field]
            return np.array([f[i,:][pointI] for i in range(3)])
        else:
            tr = grid[field]
            if tr.size == 1: # dx, dy, dz, cellvolume
                tr = tr * np.ones(grid.ActiveDimensions, dtype='float64')
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
                new_field = np.ones(grid.ActiveDimensions, dtype=dtype) * default_val
            new_field[pointI] = self[field][i:i+np]
            grid[field] = new_field
            i += np

    def _is_fully_enclosed(self, grid):
        return np.all(self._get_cut_mask)

    def _get_point_indices(self, grid, use_child_mask=True):
        k = np.zeros(grid.ActiveDimensions, dtype='bool')
        k = (k | self._get_cut_mask(grid))
        if use_child_mask: k = (k & grid.child_mask)
        return np.where(k)

    def _get_cut_particle_mask(self, grid):
        if self._is_fully_enclosed(grid):
            return True
        fake_grid = FakeGridForParticles(grid)
        return self._get_cut_mask(fake_grid)

    def _get_particle_indices(self, grid):
        k = np.zeros(grid.NumberOfParticles, dtype='bool')
        k = (k | self._get_cut_particle_mask(grid))
        return np.where(k)

    def cut_region(self, field_cuts):
        """
        Return an InLineExtractedRegion, where the grid cells are cut on the
        fly with a set of field_cuts.  It is very useful for applying
        conditions to the fields in your data object.

        Examples
        --------
        To find the total mass of gas above 10^6 K in your volume:

        >>> pf = load("RedshiftOutput0005")
        >>> ad = pf.h.all_data()
        >>> cr = ad.cut_region(["grid['Temperature'] > 1e6"])
        >>> print cr.quantities["TotalQuantity"]("CellMassMsun")

        """
        return InLineExtractedRegionBase(self, field_cuts)

    def extract_region(self, indices):
        """
        Return an ExtractedRegion where the points contained in it are defined
        as the points in `this` data object with the given *indices*.
        """
        fp = self.field_parameters.copy()
        return ExtractedRegionBase(self, indices, **fp)

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
        pb = get_pbar("Extracting ", len(list(self._get_grid_objs())))
        for i, g in enumerate(self._get_grid_objs()):
            pb.update(i)
            my_verts = self._extract_isocontours_from_grid(
                            g, field, value, sample_values)
            if sample_values is not None:
                my_verts, svals = my_verts
                samples.append(svals)
            verts.append(my_verts)
        pb.finish()
        verts = np.concatenate(verts).transpose()
        verts = self.comm.par_combine_object(verts, op='cat', datatype='array')
        verts = verts.transpose()
        if sample_values is not None:
            samples = np.concatenate(samples)
            samples = self.comm.par_combine_object(samples, op='cat',
                                datatype='array')
        if rescale:
            mi = np.min(verts, axis=0)
            ma = np.max(verts, axis=0)
            verts = (verts - mi) / (ma - mi).max()
        if filename is not None and self.comm.rank == 0:
            if hasattr(filename, "write"): f = filename
            for v1 in verts:
                f.write("v %0.16e %0.16e %0.16e\n" % (v1[0], v1[1], v1[2]))
            for i in range(len(verts)/3):
                f.write("f %s %s %s\n" % (i*3+1, i*3+2, i*3+3))
            if not hasattr(filename, "write"): f.close()
        if sample_values is not None:
            return verts, samples
        return verts


    @restore_grid_state
    def _extract_isocontours_from_grid(self, grid, field, value,
                                       sample_values = None):
        mask = self._get_cut_mask(grid) * grid.child_mask
        vals = grid.get_vertex_centered_data(field, no_ghost = False)
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
            ff = np.ones(vals.shape, dtype="float64")
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
            cons = np.logspace(np.log10(min_val),np.log10(max_val),
                               num_levels+1)
        else:
            cons = np.linspace(min_val, max_val, num_levels+1)
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
                grid[field] = np.ones(grid.ActiveDimensions)*default_value
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

class ExtractedRegionBase(AMR3DData):
    """
    An arbitrarily defined data container that allows for selection
    of all data meeting certain criteria.

    In order to create an arbitrarily selected set of data, the
    ExtractedRegion takes a `base_region` and a set of `indices`
    and creates a region within the `base_region` consisting of
    all data indexed by the `indices`. Note that `indices` must be
    precomputed. This does not work well for parallelized
    operations.

    Parameters
    ----------
    base_region : yt data source
        A previously selected data source.
    indices : array_like
        An array of indices

    Other Parameters
    ----------------
    force_refresh : bool
       Force a refresh of the data. Defaults to True.

    Examples
    --------
    """
    _type_name = "extracted_region"
    _con_args = ('_base_region', '_indices')
    def __init__(self, base_region, indices, force_refresh=True, **kwargs):
        cen = kwargs.pop("center", None)
        if cen is None: cen = base_region.get_field_parameter("center")
        AMR3DData.__init__(self, center=cen,
                            fields=None, pf=base_region.pf, **kwargs)
        self._base_region = base_region # We don't weakly reference because
                                        # It is not cyclic
        if isinstance(indices, types.DictType):
            self._indices = indices
            self._grids = self._base_region.pf.h.grids[self._indices.keys()]
        else:
            self._grids = None
            self._base_indices = indices
        if force_refresh: self._refresh_data()

    def _get_cut_particle_mask(self, grid):
        # Override to provide a warning
        mylog.warning("Returning all particles from an Extracted Region.  This could be incorrect!")
        return True

    def _get_list_of_grids(self):
        # Okay, so what we're going to want to do is get the pointI from
        # region._get_point_indices(grid) for grid in base_region._grids,
        # and then construct an array of those, which we will select along indices.
        if self._grids != None: return
        grid_vals, xi, yi, zi = [], [], [], []
        for grid in self._base_region._grids:
            xit,yit,zit = self._base_region._get_point_indices(grid)
            grid_vals.append(np.ones(xit.shape, dtype='int') * (grid.id-grid._id_offset))
            xi.append(xit)
            yi.append(yit)
            zi.append(zit)
        grid_vals = np.concatenate(grid_vals)[self._base_indices]
        grid_order = np.argsort(grid_vals)
        # Note: grid_vals is still unordered
        grid_ids = np.unique(grid_vals)
        xi = np.concatenate(xi)[self._base_indices][grid_order]
        yi = np.concatenate(yi)[self._base_indices][grid_order]
        zi = np.concatenate(zi)[self._base_indices][grid_order]
        bc = np.bincount(grid_vals)
        splits = []
        for i,v in enumerate(bc):
            if v > 0: splits.append(v)
        splits = np.add.accumulate(splits)
        xis, yis, zis = [np.array_split(aa, splits) for aa in [xi,yi,zi]]
        self._indices = {}
        h = self._base_region.pf.h
        for grid_id, x, y, z in itertools.izip(grid_ids, xis, yis, zis):
            # grid_id needs no offset
            ll = h.grids[grid_id].ActiveDimensions.prod() \
               - (np.logical_not(h.grids[grid_id].child_mask)).sum()
            # This means we're completely enclosed, except for child masks
            if x.size == ll:
                self._indices[grid_id] = None
            else:
                # This will slow things down a bit, but conserve memory
                self._indices[grid_id] = \
                    np.zeros(h.grids[grid_id].ActiveDimensions, dtype='bool')
                self._indices[grid_id][(x,y,z)] = True
        self._grids = h.grids[self._indices.keys()]

    def _is_fully_enclosed(self, grid):
        if self._indices[grid.id-grid._id_offset] is None or \
            (self._indices[grid.id-grid._id_offset][0].size ==
             grid.ActiveDimensions.prod()):
            return True
        return False

    def _get_cut_mask(self, grid):
        cm = np.zeros(grid.ActiveDimensions, dtype='bool')
        cm[self._get_point_indices(grid, False)] = True
        return cm

    __empty_array = np.array([], dtype='bool')
    def _get_point_indices(self, grid, use_child_mask=True):
        # Yeah, if it's not true, we don't care.
        tr = self._indices.get(grid.id-grid._id_offset, self.__empty_array)
        if tr is None: tr = np.where(grid.child_mask)
        else: tr = np.where(tr)
        return tr

    def __repr__(self):
        # We'll do this the slow way to be clear what's going on
        s = "%s (%s): " % (self.__class__.__name__, self.pf)
        s += ", ".join(["%s=%s" % (i, getattr(self,i))
                       for i in self._con_args if i != "_indices"])
        return s

    def join(self, other):
        ng = {}
        gs = set(self._indices.keys() + other._indices.keys())
        for g in gs:
            grid = self.pf.h.grids[g]
            if g in other._indices and g in self._indices:
                # We now join the indices
                ind = np.zeros(grid.ActiveDimensions, dtype='bool')
                ind[self._indices[g]] = True
                ind[other._indices[g]] = True
                if ind.prod() == grid.ActiveDimensions.prod(): ind = None
            elif g in self._indices:
                ind = self._indices[g]
            elif g in other._indices:
                ind = other._indices[g]
            # Okay we have indices
            if ind is not None: ind = ind.copy()
            ng[g] = ind
        gl = self.pf.h.grids[list(gs)]
        gc = self.pf.h.grid_collection(
            self._base_region.get_field_parameter("center"), gl)
        return self.pf.h.extracted_region(gc, ng)

class InLineExtractedRegionBase(AMR3DData):
    """
    In-line extracted regions accept a base region and a set of field_cuts to
    determine which points in a grid should be included.
    """
    _type_name = "cut_region"
    _con_args = ("_base_region", "_field_cuts")
    def __init__(self, base_region, field_cuts, **kwargs):
        cen = base_region.get_field_parameter("center")
        AMR3DData.__init__(self, center=cen,
                            fields=None, pf=base_region.pf, **kwargs)
        self._base_region = base_region # We don't weakly reference because
                                        # It is not cyclic
        self._field_cuts = ensure_list(field_cuts)[:]
        self._refresh_data()

    def _get_list_of_grids(self):
        self._grids = self._base_region._grids

    def _is_fully_enclosed(self, grid):
        return False

    @cache_mask
    def _get_cut_mask(self, grid):
        point_mask = np.ones(grid.ActiveDimensions, dtype='bool')
        point_mask *= self._base_region._get_cut_mask(grid)
        for cut in self._field_cuts:
            point_mask *= eval(cut)
        return point_mask

class AMRCylinderBase(AMR3DData):
    """
    By providing a *center*, a *normal*, a *radius* and a *height* we
    can define a cylinder of any proportion.  Only cells whose centers are
    within the cylinder will be selected.
    """
    _type_name = "disk"
    _con_args = ('center', '_norm_vec', '_radius', '_height')
    def __init__(self, center, normal, radius, height, fields=None,
                 pf=None, **kwargs):
        AMR3DData.__init__(self, center, fields, pf, **kwargs)
        self._norm_vec = np.array(normal)/np.sqrt(np.dot(normal,normal))
        self.set_field_parameter("normal", self._norm_vec)
        self._height = fix_length(height, self.pf)
        self._radius = fix_length(radius, self.pf)
        self._d = -1.0 * np.dot(self._norm_vec, self.center)
        self._refresh_data()

    def _get_list_of_grids(self):
        H = np.sum(self._norm_vec.reshape((1,3,1)) * self.pf.h.grid_corners,
                   axis=1) + self._d
        D = np.sqrt(np.sum((self.pf.h.grid_corners -
                           self.center.reshape((1,3,1)))**2.0,axis=1))
        R = np.sqrt(D**2.0-H**2.0)
        self._grids = self.hierarchy.grids[
            ( (np.any(np.abs(H)<self._height,axis=0))
            & (np.any(R<self._radius,axis=0)
            & (np.logical_not((np.all(H>0,axis=0) | (np.all(H<0, axis=0)))) )
            ) ) ]
        self._grids = self.hierarchy.grids

    def _is_fully_enclosed(self, grid):
        corners = grid._corners.reshape((8,3,1))
        H = np.sum(self._norm_vec.reshape((1,3,1)) * corners,
                   axis=1) + self._d
        D = np.sqrt(np.sum((corners -
                           self.center.reshape((1,3,1)))**2.0,axis=1))
        R = np.sqrt(D**2.0-H**2.0)
        return (np.all(np.abs(H) < self._height, axis=0) \
            and np.all(R < self._radius, axis=0))

    @cache_mask
    def _get_cut_mask(self, grid):
        if self._is_fully_enclosed(grid):
            return True
        else:
            h = grid['x'] * self._norm_vec[0] \
              + grid['y'] * self._norm_vec[1] \
              + grid['z'] * self._norm_vec[2] \
              + self._d
            d = np.sqrt(
                (grid['x'] - self.center[0])**2.0
              + (grid['y'] - self.center[1])**2.0
              + (grid['z'] - self.center[2])**2.0
                )
            r = np.sqrt(d**2.0-h**2.0)
            cm = ( (np.abs(h) <= self._height)
                 & (r <= self._radius))
        return cm

class AMRInclinedBox(AMR3DData):
    """
    A rectangular prism with arbitrary alignment to the computational
    domain.  *origin* is the origin of the box, while *box_vectors* is an
    array of ordering [ax, ijk] that describes the three vectors that
    describe the box.  No checks are done to ensure that the box satisfies
    a right-hand rule, but if it doesn't, behavior is undefined.
    """
    _type_name="inclined_box"
    _con_args = ('origin','box_vectors')

    def __init__(self, origin, box_vectors, fields=None,
                 pf=None, **kwargs):
        self.origin = np.array(origin)
        self.box_vectors = np.array(box_vectors, dtype='float64')
        self.box_lengths = (self.box_vectors**2.0).sum(axis=1)**0.5
        center = origin + 0.5*self.box_vectors.sum(axis=0)
        AMR3DData.__init__(self, center, fields, pf, **kwargs)
        self._setup_rotation_parameters()
        self._refresh_data()

    def _setup_rotation_parameters(self):
        xv = self.box_vectors[0,:]
        yv = self.box_vectors[1,:]
        zv = self.box_vectors[2,:]
        self._x_vec = xv / np.sqrt(np.dot(xv, xv))
        self._y_vec = yv / np.sqrt(np.dot(yv, yv))
        self._z_vec = zv / np.sqrt(np.dot(zv, zv))
        self._rot_mat = np.array([self._x_vec,self._y_vec,self._z_vec])
        self._inv_mat = np.linalg.pinv(self._rot_mat)

    def _get_list_of_grids(self):
        if self._grids is not None: return
        GLE = self.pf.h.grid_left_edge
        GRE = self.pf.h.grid_right_edge
        goodI = find_grids_in_inclined_box(self.box_vectors, self.center,
                                           GLE, GRE)
        cgrids = self.pf.h.grids[goodI.astype('bool')]
       # find_grids_in_inclined_box seems to be broken.
        cgrids = self.pf.h.grids[:]
        grids = []
        for i,grid in enumerate(cgrids):
            v = grid_points_in_volume(self.box_lengths, self.origin,
                                      self._rot_mat, grid.LeftEdge,
                                      grid.RightEdge, grid.dds,
                                      grid.child_mask, 1)
            if v: grids.append(grid)
        self._grids = np.empty(len(grids), dtype='object')
        for gi, g in enumerate(grids): self._grids[gi] = g


    def _is_fully_enclosed(self, grid):
        # This should be written at some point.
        # We'd rotate all eight corners into the space of the box, then check to
        # see if all are enclosed.
        return False

    def _get_cut_mask(self, grid):
        if self._is_fully_enclosed(grid):
            return True
        pm = np.zeros(grid.ActiveDimensions, dtype='int32')
        grid_points_in_volume(self.box_lengths, self.origin,
                              self._rot_mat, grid.LeftEdge,
                              grid.RightEdge, grid.dds, pm, 0)
        return pm


class AMRRegionBase(AMR3DData):
    """A 3D region of data with an arbitrary center.

    Takes an array of three *left_edge* coordinates, three
    *right_edge* coordinates, and a *center* that can be anywhere
    in the domain. If the selected region extends past the edges
    of the domain, no data will be found there, though the
    object's `left_edge` or `right_edge` are not modified.

    Parameters
    ----------
    center : array_like
        The center of the region
    left_edge : array_like
        The left edge of the region
    right_edge : array_like
        The right edge of the region
    """
    _type_name = "region"
    _con_args = ('center', 'left_edge', 'right_edge')
    _dx_pad = 0.5
    def __init__(self, center, left_edge, right_edge, fields = None,
                 pf = None, **kwargs):
        AMR3DData.__init__(self, center, fields, pf, **kwargs)
        self.left_edge = left_edge
        self.right_edge = right_edge
        self._refresh_data()

    def _get_list_of_grids(self):
        self._grids, ind = self.pf.hierarchy.get_box_grids(self.left_edge,
                                                           self.right_edge)

    def _is_fully_enclosed(self, grid):
        return np.all( (grid._corners <= self.right_edge)
                     & (grid._corners >= self.left_edge))

    @cache_mask
    def _get_cut_mask(self, grid):
        if self._is_fully_enclosed(grid):
            return True
        else:
            dxp, dyp, dzp = self._dx_pad * grid.dds
            cm = ( (grid['x'] - dxp < self.right_edge[0])
                 & (grid['x'] + dxp > self.left_edge[0])
                 & (grid['y'] - dyp < self.right_edge[1])
                 & (grid['y'] + dyp > self.left_edge[1])
                 & (grid['z'] - dzp < self.right_edge[2])
                 & (grid['z'] + dzp > self.left_edge[2]) )
        return cm

class AMRRegionStrictBase(AMRRegionBase):
    """
    AMRRegion without any dx padding for cell selection
    """
    _type_name = "region_strict"
    _dx_pad = 0.0

class AMRPeriodicRegionBase(AMR3DData):
    """
    AMRRegions are rectangular prisms of data.
    """
    _type_name = "periodic_region"
    _con_args = ('center', 'left_edge', 'right_edge')
    _dx_pad = 0.5
    def __init__(self, center, left_edge, right_edge, fields = None,
                 pf = None, **kwargs):
        """A 3D region of data that with periodic boundary
        conditions if the selected region extends beyond the
        simulation domain.

        Takes an array of three *left_edge* coordinates, three
        *right_edge* coordinates, and a *center* that can be anywhere
        in the domain. The selected region can extend past the edges
        of the domain, in which case periodic boundary conditions will
        be applied to fill the region.

        Parameters
        ----------
        center : array_like
            The center of the region
        left_edge : array_like
            The left edge of the region
        right_edge : array_like
            The right edge of the region

        """
        AMR3DData.__init__(self, center, fields, pf, **kwargs)
        self.left_edge = np.array(left_edge)
        self.right_edge = np.array(right_edge)
        self._refresh_data()
        self.offsets = (np.mgrid[-1:1:3j,-1:1:3j,-1:1:3j] * \
                        (self.pf.domain_right_edge -
                         self.pf.domain_left_edge)[:,None,None,None])\
                       .transpose().reshape(27,3) # cached and in order

    def _get_list_of_grids(self):
        self._grids, ind = self.pf.hierarchy.get_periodic_box_grids(self.left_edge,
                                                                    self.right_edge)

    def _is_fully_enclosed(self, grid):
        for off_x, off_y, off_z in self.offsets:
            region_left = [self.left_edge[0]+off_x,
                           self.left_edge[1]+off_y,self.left_edge[2]+off_z]
            region_right = [self.right_edge[0]+off_x,
                            self.right_edge[1]+off_y,self.right_edge[2]+off_z]
            if (np.all((grid._corners <= region_right) &
                       (grid._corners >= region_left))):
                return True
        return False

    @cache_mask
    def _get_cut_mask(self, grid):
        if self._is_fully_enclosed(grid):
            return True
        else:
            cm = np.zeros(grid.ActiveDimensions,dtype='bool')
            dxp, dyp, dzp = self._dx_pad * grid.dds
            for off_x, off_y, off_z in self.offsets:
                cm = cm | ( (grid['x'] - dxp + off_x < self.right_edge[0])
                          & (grid['x'] + dxp + off_x > self.left_edge[0])
                          & (grid['y'] - dyp + off_y < self.right_edge[1])
                          & (grid['y'] + dyp + off_y > self.left_edge[1])
                          & (grid['z'] - dzp + off_z < self.right_edge[2])
                          & (grid['z'] + dzp + off_z > self.left_edge[2]) )
            return cm

class AMRPeriodicRegionStrictBase(AMRPeriodicRegionBase):
    """
    AMRPeriodicRegion without any dx padding for cell selection
    """
    _type_name = "periodic_region_strict"
    _dx_pad = 0.0
    def __init__(self, center, left_edge, right_edge, fields = None,
                 pf = None, **kwargs):
        AMRPeriodicRegionBase.__init__(self, center, left_edge, right_edge,
                                       fields = None, pf = None, **kwargs)


class AMRGridCollectionBase(AMR3DData):
    """
    By selecting an arbitrary *grid_list*, we can act on those grids.
    Child cells are not returned.
    """
    _type_name = "grid_collection"
    _con_args = ("center", "grid_list")
    def __init__(self, center, grid_list, fields = None,
                 pf = None, **kwargs):
        AMR3DData.__init__(self, center, fields, pf, **kwargs)
        self._grids = np.array(grid_list)
        self.grid_list = self._grids

    def _get_list_of_grids(self):
        pass

    def _is_fully_enclosed(self, grid):
        return True

    @cache_mask
    def _get_cut_mask(self, grid):
        return np.ones(grid.ActiveDimensions, dtype='bool')

    def _get_point_indices(self, grid, use_child_mask=True):
        k = np.ones(grid.ActiveDimensions, dtype='bool')
        if use_child_mask:
            k[grid.child_indices] = False
        pointI = np.where(k == True)
        return pointI

class AMRMaxLevelCollection(AMR3DData):
    """
    By selecting an arbitrary *max_level*, we can act on those grids.
    Child cells are masked when the level of the grid is below the max
    level.
    """
    _type_name = "grid_collection_max_level"
    _con_args = ("center", "max_level")
    def __init__(self, center, max_level, fields = None,
                 pf = None, **kwargs):
        AMR3DData.__init__(self, center, fields, pf, **kwargs)
        self.max_level = max_level
        self._refresh_data()

    def _get_list_of_grids(self):
        if self._grids is not None: return
        gi = (self.pf.h.grid_levels <= self.max_level)[:,0]
        self._grids = self.pf.h.grids[gi]

    def _is_fully_enclosed(self, grid):
        return True

    @cache_mask
    def _get_cut_mask(self, grid):
        return np.ones(grid.ActiveDimensions, dtype='bool')

    def _get_point_indices(self, grid, use_child_mask=True):
        k = np.ones(grid.ActiveDimensions, dtype='bool')
        if use_child_mask and grid.Level < self.max_level:
            k[grid.child_indices] = False
        pointI = np.where(k == True)
        return pointI


class AMRSphereBase(AMR3DData):
    """
    A sphere f points defined by a *center* and a *radius*.

    Parameters
    ----------
    center : array_like
        The center of the sphere.
    radius : float
        The radius of the sphere.

    Examples
    --------
    >>> pf = load("DD0010/moving7_0010")
    >>> c = [0.5,0.5,0.5]
    >>> sphere = pf.h.sphere(c,1.*pf['kpc'])
    """
    _type_name = "sphere"
    _con_args = ('center', 'radius')
    def __init__(self, center, radius, fields = None, pf = None, **kwargs):
        AMR3DData.__init__(self, center, fields, pf, **kwargs)
        # Unpack the radius, if necessary
        radius = fix_length(radius, self.pf)
        if radius < self.hierarchy.get_smallest_dx():
            raise YTSphereTooSmall(pf, radius, self.hierarchy.get_smallest_dx())
        self.set_field_parameter('radius',radius)
        self.radius = radius
        self.DW = self.pf.domain_right_edge - self.pf.domain_left_edge
        self._refresh_data()

    def _get_list_of_grids(self, field = None):
        grids,ind = self.hierarchy.find_sphere_grids(self.center, self.radius)
        # Now we sort by level
        grids = grids.tolist()
        grids.sort(key=lambda x: (x.Level, x.LeftEdge[0], x.LeftEdge[1], x.LeftEdge[2]))
        self._grids = np.empty(len(grids), dtype='object')
        for gi, g in enumerate(grids): self._grids[gi] = g

    def _is_fully_enclosed(self, grid):
        return False

    @restore_grid_state # Pains me not to decorate with cache_mask here
    def _get_cut_mask(self, grid, field=None):
        # We have the *property* center, which is not necessarily
        # the same as the field_parameter
        if self._is_fully_enclosed(grid):
            return True # We do not want child masking here
        if not isinstance(grid, (FakeGridForParticles, GridChildMaskWrapper)) \
           and grid.id in self._cut_masks:
            return self._cut_masks[grid.id]
        cm = ( (grid["RadiusCode"]<=self.radius) & grid.child_mask )
        if not isinstance(grid, (FakeGridForParticles, GridChildMaskWrapper)):
            self._cut_masks[grid.id] = cm
        return cm

class AMREllipsoidBase(AMR3DData):
    """
    By providing a *center*,*A*,*B*,*C*,*e0*,*tilt* we
    can define a ellipsoid of any proportion.  Only cells whose
    centers are within the ellipsoid will be selected.

    Parameters
    ----------
    center : array_like
        The center of the ellipsoid.
    A : float
        The magnitude of the largest semi-major axis of the ellipsoid.
    B : float
        The magnitude of the medium semi-major axis of the ellipsoid.
    C : float
        The magnitude of the smallest semi-major axis of the ellipsoid.
    e0 : array_like (automatically normalized)
        the direction of the largest semi-major axis of the ellipsoid
    tilt : float
        After the rotation about the z-axis to allign e0 to x in the x-y
        plane, and then rotating about the y-axis to align e0 completely
        to the x-axis, tilt is the angle in radians remaining to
        rotate about the x-axis to align both e1 to the y-axis and e2 to
        the z-axis.
    Examples
    --------
    >>> pf = load("DD####/DD####")
    >>> c = [0.5,0.5,0.5]
    >>> ell = pf.h.ellipsoid(c, 0.1, 0.1, 0.1, np.array([0.1, 0.1, 0.1]), 0.2)
    """
    _type_name = "ellipsoid"
    _con_args = ('center', '_A', '_B', '_C', '_e0', '_tilt')
    def __init__(self, center, A, B, C, e0, tilt, fields=None,
                 pf=None, **kwargs):
        AMR3DData.__init__(self, np.array(center), fields, pf, **kwargs)
        # make sure the magnitudes of semi-major axes are in order
        if A<B or B<C:
            raise YTEllipsoidOrdering(pf, A, B, C)
        # make sure the smallest side is not smaller than dx
        if C < self.hierarchy.get_smallest_dx():
            raise YTSphereTooSmall(pf, C, self.hierarchy.get_smallest_dx())
        self._A = A
        self._B = B
        self._C = C
        self._e0 = e0 = e0 / (e0**2.0).sum()**0.5
        self._tilt = tilt

        # find the t1 angle needed to rotate about z axis to align e0 to x
        t1 = np.arctan(e0[1] / e0[0])
        # rotate e0 by -t1
        RZ = get_rotation_matrix(t1, (0,0,1)).transpose()
        r1 = (e0 * RZ).sum(axis = 1)
        # find the t2 angle needed to rotate about y axis to align e0 to x
        t2 = np.arctan(-r1[2] / r1[0])
        """
        calculate the original e1
        given the tilt about the x axis when e0 was aligned
        to x after t1, t2 rotations about z, y
        """
        RX = get_rotation_matrix(-tilt, (1, 0, 0)).transpose()
        RY = get_rotation_matrix(-t2,   (0, 1, 0)).transpose()
        RZ = get_rotation_matrix(-t1,   (0, 0, 1)).transpose()
        e1 = ((0, 1, 0) * RX).sum(axis=1)
        e1 = (e1 * RY).sum(axis=1)
        e1 = (e1 * RZ).sum(axis=1)
        e2 = np.cross(e0, e1)

        self._e1 = e1
        self._e2 = e2

        self.set_field_parameter('A', A)
        self.set_field_parameter('B', B)
        self.set_field_parameter('C', C)
        self.set_field_parameter('e0', e0)
        self.set_field_parameter('e1', e1)
        self.set_field_parameter('e2', e2)
        self.DW = self.pf.domain_right_edge - self.pf.domain_left_edge
        self._refresh_data()

        """
        Having another function find_ellipsoid_grids is too much work,
        can just use the sphere one and forget about checking orientation
        but feed in the A parameter for radius
        """
    def _get_list_of_grids(self, field=None):
        """
        This returns the grids that are possibly within the ellipse
        """
        grids, ind = self.hierarchy.find_sphere_grids(self.center, self._A)
        # Now we sort by level
        grids = grids.tolist()
        grids.sort(key=lambda x: (x.Level,
                                  x.LeftEdge[0],
                                  x.LeftEdge[1],
                                  x.LeftEdge[2]))
        self._grids = np.array(grids, dtype='object')

    def _is_fully_enclosed(self, grid):
        """
        check if all grid corners are inside the ellipsoid
        """
        return False

    @restore_grid_state  # Pains me not to decorate with cache_mask here
    def _get_cut_mask(self, grid, field=None):
        """
        This checks if each cell is inside the ellipsoid
        """
        # We have the *property* center, which is not necessarily
        # the same as the field_parameter
        if self._is_fully_enclosed(grid):
            return True  # We do not want child masking here
        if not isinstance(grid, (FakeGridForParticles, GridChildMaskWrapper)) \
           and grid.id in self._cut_masks:
            return self._cut_masks[grid.id]

        dot_evecx = np.zeros(grid.ActiveDimensions)
        dot_evecy = np.zeros(grid.ActiveDimensions)
        dot_evecz = np.zeros(grid.ActiveDimensions)

        for i, ax in enumerate('xyz'):
            # distance to center
            ar = grid[ax]-self.center[i]
            # correct for periodicity
            vec = np.array([ar, ar + self.DW[i], ar - self.DW[i]])
            ind = np.argmin(np.abs(vec), axis=0)
            vec = np.choose(ind, vec)
            # sum up to get the dot product with e_vectors
            dot_evecx += vec * self._e0[i] / self._A
            dot_evecy += vec * self._e1[i] / self._B
            dot_evecz += vec * self._e2[i] / self._C

        # Calculate the eqn of ellipsoid, if it is inside
        # then result should be <= 1.0
        cm = ((dot_evecx**2 +
               dot_evecy**2 +
               dot_evecz**2 <= 1.0) & grid.child_mask)
        if not isinstance(grid, (FakeGridForParticles, GridChildMaskWrapper)):
            self._cut_masks[grid.id] = cm
        return cm


class AMRCoveringGridBase(AMR3DData):
    """A 3D region with all data extracted to a single, specified resolution.

    Parameters
    ----------
    level : int
        The resolution level data is uniformly gridded at
    left_edge : array_like
        The left edge of the region to be extracted
    dims : array_like
        Number of cells along each axis of resulting covering_grid
    fields : array_like, optional
        A list of fields that you'd like pre-generated for your object

    Examples
    --------
    cube = pf.h.covering_grid(2, left_edge=[0.0, 0.0, 0.0], \
                              dims=[128, 128, 128])
    """
    _spatial = True
    _type_name = "covering_grid"
    _con_args = ('level', 'left_edge', 'ActiveDimensions')
    def __init__(self, level, left_edge, dims, fields = None,
                 pf = None, num_ghost_zones = 0, use_pbar = True, **kwargs):
        AMR3DData.__init__(self, center=kwargs.pop("center", None),
                           fields=fields, pf=pf, **kwargs)
        self.left_edge = np.array(left_edge)
        self.level = level
        dims = np.array(dims)
        rdx = self.pf.domain_dimensions*self.pf.refine_by**level
        rdx[np.where(dims - 2 * num_ghost_zones <= 1)] = 1   # issue 602
        self.dds = self.pf.domain_width / rdx.astype("float64")
        self.ActiveDimensions = np.array(dims, dtype='int32')
        self.right_edge = self.left_edge + self.ActiveDimensions*self.dds
        self._num_ghost_zones = num_ghost_zones
        self._use_pbar = use_pbar
        self.global_startindex = np.rint((self.left_edge-self.pf.domain_left_edge)/self.dds).astype('int64')
        self.domain_width = np.rint((self.pf.domain_right_edge -
                    self.pf.domain_left_edge)/self.dds).astype('int64')
        self._refresh_data()

    def _get_list_of_grids(self, buffer = 0.0):
        if self._grids is not None: return
        if np.any(self.left_edge - buffer < self.pf.domain_left_edge) or \
           np.any(self.right_edge + buffer > self.pf.domain_right_edge):
            grids,ind = self.pf.hierarchy.get_periodic_box_grids_below_level(
                            self.left_edge - buffer,
                            self.right_edge + buffer, self.level)
        else:
            grids,ind = self.pf.hierarchy.get_box_grids_below_level(
                self.left_edge - buffer,
                self.right_edge + buffer, self.level)
        sort_ind = np.argsort(self.pf.h.grid_levels.ravel()[ind])
        self._grids = self.pf.hierarchy.grids[ind][(sort_ind,)][::-1]

    def _refresh_data(self):
        AMR3DData._refresh_data(self)
        self['dx'] = self.dds[0] * np.ones(self.ActiveDimensions, dtype='float64')
        self['dy'] = self.dds[1] * np.ones(self.ActiveDimensions, dtype='float64')
        self['dz'] = self.dds[2] * np.ones(self.ActiveDimensions, dtype='float64')

    def get_data(self, fields=None):
        if self._grids is None:
            self._get_list_of_grids()
        if fields is None:
            fields = self.fields[:]
        else:
            fields = ensure_list(fields)
        obtain_fields = []
        for field in fields:
            if self.field_data.has_key(field): continue
            if field not in self.hierarchy.field_list:
                try:
                    #print "Generating", field
                    self._generate_field(field)
                    continue
                except NeedsOriginalGrid, ngt_exception:
                    pass
            elif self.pf.field_info[field].particle_type:
                region = self.pf.h.region(self.center,
                            self.left_edge, self.right_edge)
                self.field_data[field] = region[field]
                continue
            obtain_fields.append(field)
            self[field] = np.zeros(self.ActiveDimensions, dtype='float64') -999
        if len(obtain_fields) == 0: return
        mylog.debug("Getting fields %s from %s possible grids",
                   obtain_fields, len(self._grids))
        if self._use_pbar: pbar = \
                get_pbar('Searching grids for values ', len(self._grids))
        count = self.ActiveDimensions.prod()
        for i, grid in enumerate(self._grids):
            if self._use_pbar: pbar.update(i)
            count -= self._get_data_from_grid(grid, obtain_fields)
            if count <= 0: break
        if self._use_pbar: pbar.finish()
        if count > 0 or np.any(self[obtain_fields[0]] == -999):
            # and self.dx < self.hierarchy.grids[0].dx:
            n_bad = np.where(self[obtain_fields[0]]==-999)[0].size
            mylog.error("Covering problem: %s cells are uncovered", n_bad)
            raise KeyError(n_bad)

    def _generate_field(self, field):
        if self.pf.field_info.has_key(field):
            # First we check the validator; this might even raise!
            self.pf.field_info[field].check_available(self)
            self[field] = self.pf.field_info[field](self)
        else: # Can't find the field, try as it might
            raise KeyError(field)

    def flush_data(self, field=None):
        """
        Any modifications made to the data in this object are pushed back
        to the originating grids, except the cells where those grids are both
        below the current level `and` have child cells.
        """
        self._get_list_of_grids()
        # We don't generate coordinates here.
        if field == None:
            fields_to_get = self.fields[:]
        else:
            fields_to_get = ensure_list(field)
        for grid in self._grids:
            self._flush_data_to_grid(grid, fields_to_get)

    @restore_grid_state
    def _get_data_from_grid(self, grid, fields):
        ll = int(grid.Level == self.level)
        ref_ratio = self.pf.refine_by**(self.level - grid.Level)
        g_fields = [gf.astype("float64")
                    if gf.dtype != "float64"
                    else gf for gf in (grid[field] for field in fields)]
        c_fields = [self[field] for field in fields]
        count = FillRegion(ref_ratio,
            grid.get_global_startindex(), self.global_startindex,
            c_fields, g_fields,
            self.ActiveDimensions, grid.ActiveDimensions,
            grid.child_mask, self.domain_width, ll, 0)
        return count

    def _flush_data_to_grid(self, grid, fields):
        ll = int(grid.Level == self.level)
        ref_ratio = self.pf.refine_by**(self.level - grid.Level)
        g_fields = []
        for field in fields:
            if not grid.has_key(field): grid[field] = \
               np.zeros(grid.ActiveDimensions, dtype=self[field].dtype)
            g_fields.append(grid[field])
        c_fields = [self[field] for field in fields]
        FillRegion(ref_ratio,
            grid.get_global_startindex(), self.global_startindex,
            c_fields, g_fields,
            self.ActiveDimensions, grid.ActiveDimensions,
            grid.child_mask, self.domain_width, ll, 1)

    @property
    def LeftEdge(self):
        return self.left_edge

    @property
    def RightEdge(self):
        return self.right_edge

class AMRSmoothedCoveringGridBase(AMRCoveringGridBase):
    """A 3D region with all data extracted and interpolated to a
    single, specified resolution. (Identical to covering_grid,
    except that it interpolates.)

    Smoothed covering grids start at level 0, interpolating to
    fill the region to level 1, replacing any cells actually
    covered by level 1 data, and then recursively repeating this
    process until it reaches the specified `level`.

    Parameters
    ----------
    level : int
        The resolution level data is uniformly gridded at
    left_edge : array_like
        The left edge of the region to be extracted
    dims : array_like
        Number of cells along each axis of resulting covering_grid.
    fields : array_like, optional
        A list of fields that you'd like pre-generated for your object

    Examples
    --------

    >>> cube = pf.h.smoothed_covering_grid(2, left_edge=[0.0, 0.0, 0.0], \
    ...                          dims=[128, 128, 128])
    """
    _type_name = "smoothed_covering_grid"
    def __init__(self, *args, **kwargs):
        self._base_dx = (
              (self.pf.domain_right_edge - self.pf.domain_left_edge) /
               self.pf.domain_dimensions.astype("float64"))
        AMRCoveringGridBase.__init__(self, *args, **kwargs)
        self._final_start_index = self.global_startindex

    def _get_list_of_grids(self):
        if self._grids is not None: return
        buffer = ((self.pf.domain_right_edge - self.pf.domain_left_edge)
                 / self.pf.domain_dimensions).max()
        AMRCoveringGridBase._get_list_of_grids(self, buffer)
        # We reverse the order to ensure that coarse grids are first
        self._grids = self._grids[::-1]

    def get_data(self, field=None):
        self._get_list_of_grids()
        # We don't generate coordinates here.
        if field == None:
            fields = self.fields[:]
        else:
            fields = ensure_list(field)
        fields_to_get = []
        for field in fields:
            if self.field_data.has_key(field): continue
            if field not in self.hierarchy.field_list:
                try:
                    #print "Generating", field
                    self._generate_field(field)
                    continue
                except NeedsOriginalGrid, ngt_exception:
                    pass
            elif self.pf.field_info[field].particle_type:
                region = self.pf.h.region(self.center,
                            self.left_edge, self.right_edge)
                self.field_data[field] = region[field]
                continue
            fields_to_get.append(field)
        if len(fields_to_get) == 0: return
        # Note that, thanks to some trickery, we have different dimensions
        # on the field than one might think from looking at the dx and the
        # L/R edges.
        # We jump-start our task here
        mylog.debug("Getting fields %s from %s possible grids",
                   fields_to_get, len(self._grids))
        self._update_level_state(0, fields_to_get)
        if self._use_pbar: pbar = \
                get_pbar('Searching grids for values ', len(self._grids))
        # The grids are assumed to be pre-sorted
        last_level = 0
        for gi, grid in enumerate(self._grids):
            if self._use_pbar: pbar.update(gi)
            if grid.Level > last_level and grid.Level <= self.level:
                mylog.debug("Updating level state to %s", last_level + 1)
                self._update_level_state(last_level + 1)
                self._refine(1, fields_to_get)
                last_level = grid.Level
            self._get_data_from_grid(grid, fields_to_get)
        while last_level < self.level:
            mylog.debug("Grid-free refinement %s to %s", last_level, last_level + 1)
            self._update_level_state(last_level + 1)
            self._refine(1, fields_to_get)
            last_level += 1
        if self.level > 0:
            for field in fields_to_get:
                self[field] = self[field][1:-1,1:-1,1:-1]
                if np.any(self[field] == -999):
                    # and self.dx < self.hierarchy.grids[0].dx:
                    n_bad = (self[field]==-999).sum()
                    mylog.error("Covering problem: %s cells are uncovered", n_bad)
                    raise KeyError(n_bad)
        if self._use_pbar: pbar.finish()

    def _update_level_state(self, level, fields = None):
        dx = self._base_dx / self.pf.refine_by**level
        self.field_data['cdx'] = dx[0]
        self.field_data['cdy'] = dx[1]
        self.field_data['cdz'] = dx[2]
        LL = self.left_edge - self.pf.domain_left_edge
        self._old_global_startindex = self.global_startindex
        self.global_startindex = np.rint(LL / dx).astype('int64') - 1
        self.domain_width = np.rint((self.pf.domain_right_edge -
                    self.pf.domain_left_edge)/dx).astype('int64')
        if level == 0 and self.level > 0:
            # We use one grid cell at LEAST, plus one buffer on all sides
            idims = np.rint((self.ActiveDimensions*self.dds)/dx).astype('int64') + 2
            fields = ensure_list(fields)
            for field in fields:
                self.field_data[field] = np.zeros(idims,dtype='float64')-999
            self._cur_dims = idims.astype("int32")
        elif level == 0 and self.level == 0:
            DLE = self.pf.domain_left_edge
            self.global_startindex = np.array(np.floor(LL/ dx), dtype='int64')
            idims = np.rint((self.ActiveDimensions*self.dds)/dx).astype('int64')
            fields = ensure_list(fields)
            for field in fields:
                self.field_data[field] = np.zeros(idims,dtype='float64')-999
            self._cur_dims = idims.astype("int32")

    def _refine(self, dlevel, fields):
        rf = float(self.pf.refine_by**dlevel)

        input_left = (self._old_global_startindex + 0.5) * rf
        dx = np.fromiter((self['cd%s' % ax] for ax in 'xyz'), count=3, dtype='float64')
        output_dims = np.rint((self.ActiveDimensions*self.dds)/dx+0.5).astype('int32') + 2
        self._cur_dims = output_dims

        for field in fields:
            output_field = np.zeros(output_dims, dtype="float64")
            output_left = self.global_startindex + 0.5
            ghost_zone_interpolate(rf, self[field], input_left,
                                   output_field, output_left)
            self.field_data[field] = output_field

    @restore_field_information_state
    def _get_data_from_grid(self, grid, fields):
        g_fields = [gf.astype("float64")
                    if gf.dtype != "float64"
                    else gf for gf in (grid[field] for field in fields)]
        c_fields = [self.field_data[field] for field in fields]
        count = FillRegion(1,
            grid.get_global_startindex(), self.global_startindex,
            c_fields, g_fields,
            self._cur_dims, grid.ActiveDimensions,
            grid.child_mask, self.domain_width, 1, 0)
        return count

    def flush_data(self, *args, **kwargs):
        raise KeyError("Can't do this")

class AMRBooleanRegionBase(AMR3DData):
    """
    This will build a hybrid region based on the boolean logic
    of the regions.

    Parameters
    ----------
    regions : list
        A list of region objects and strings describing the boolean logic
        to use when building the hybrid region. The boolean logic can be
        nested using parentheses.

    Examples
    --------
    >>> re1 = pf.h.region([0.5, 0.5, 0.5], [0.4, 0.4, 0.4],
        [0.6, 0.6, 0.6])
    >>> re2 = pf.h.region([0.5, 0.5, 0.5], [0.45, 0.45, 0.45],
        [0.55, 0.55, 0.55])
    >>> sp1 = pf.h.sphere([0.575, 0.575, 0.575], .03)
    >>> toroid_shape = pf.h.boolean([re1, "NOT", re2])
    >>> toroid_shape_with_hole = pf.h.boolean([re1, "NOT", "(", re2, "OR",
        sp1, ")"])
    """
    _type_name = "boolean"
    _con_args = ("regions",)
    def __init__(self, regions, fields = None, pf = None, **kwargs):
        # Center is meaningless, but we'll define it all the same.
        AMR3DData.__init__(self, [0.5]*3, fields, pf, **kwargs)
        self.regions = regions
        self._all_regions = []
        self._some_overlap = []
        self._all_overlap = []
        self._cut_masks = {}
        self._get_all_regions()
        self._make_overlaps()
        self._get_list_of_grids()

    def _get_all_regions(self):
        # Before anything, we simply find out which regions are involved in all
        # of this process, uniquely.
        for item in self.regions:
            if isinstance(item, types.StringType): continue
            self._all_regions.append(item)
            # So cut_masks don't get messed up.
            item._boolean_touched = True
        self._all_regions = np.unique(self._all_regions)

    def _make_overlaps(self):
        # Using the processed cut_masks, we'll figure out what grids
        # are left in the hybrid region.
        pbar = get_pbar("Building boolean", len(self._all_regions))
        for i, region in enumerate(self._all_regions):
            try:
                region._get_list_of_grids()
                alias = region
            except AttributeError:
                alias = region.data
            for grid in alias._grids:
                if grid in self._some_overlap or grid in self._all_overlap:
                    continue
                # Get the cut_mask for this grid in this region, and see
                # if there's any overlap with the overall cut_mask.
                overall = self._get_cut_mask(grid)
                local = force_array(alias._get_cut_mask(grid),
                    grid.ActiveDimensions)
                # Below we don't want to match empty masks.
                if overall.sum() == 0 and local.sum() == 0: continue
                # The whole grid is in the hybrid region if a) its cut_mask
                # in the original region is identical to the new one and b)
                # the original region cut_mask is all ones.
                if (local == np.bitwise_and(overall, local)).all() and \
                        (local == True).all():
                    self._all_overlap.append(grid)
                    continue
                if (overall == local).any():
                    # Some of local is in overall
                    self._some_overlap.append(grid)
                    continue
            pbar.update(i)
        pbar.finish()

    def __repr__(self):
        # We'll do this the slow way to be clear what's going on
        s = "%s (%s): " % (self.__class__.__name__, self.pf)
        s += "["
        for i, region in enumerate(self.regions):
            if region in ["OR", "AND", "NOT", "(", ")"]:
                s += region
            else:
                s += region.__repr__()
            if i < (len(self.regions) - 1): s += ", "
        s += "]"
        return s

    def _is_fully_enclosed(self, grid):
        return (grid in self._all_overlap)

    def _get_list_of_grids(self):
        self._grids = np.array(self._some_overlap + self._all_overlap,
            dtype='object')

    def _get_cut_mask(self, grid, field=None):
        if self._is_fully_enclosed(grid):
            return True # We do not want child masking here
        if not isinstance(grid, (FakeGridForParticles, GridChildMaskWrapper)) \
                and grid.id in self._cut_masks:
            return self._cut_masks[grid.id]
        # If we get this far, we have to generate the cut_mask.
        return self._get_level_mask(self.regions, grid)

    def _get_level_mask(self, ops, grid):
        level_masks = []
        end = 0
        for i, item in enumerate(ops):
            if end > 0 and i < end:
                # We skip over things inside parentheses on this level.
                continue
            if isinstance(item, AMRData):
                # Add this regions cut_mask to level_masks
                level_masks.append(force_array(item._get_cut_mask(grid),
                    grid.ActiveDimensions))
            elif item == "AND" or item == "NOT" or item == "OR":
                level_masks.append(item)
            elif item == "(":
                # recurse down, and we'll append the results, which
                # should be a single cut_mask
                open_count = 0
                for ii, item in enumerate(ops[i + 1:]):
                    # We look for the matching closing parentheses to find
                    # where we slice ops.
                    if item == "(":
                        open_count += 1
                    if item == ")" and open_count > 0:
                        open_count -= 1
                    elif item == ")" and open_count == 0:
                        end = i + ii + 1
                        break
                level_masks.append(force_array(self._get_level_mask(ops[i + 1:end],
                    grid), grid.ActiveDimensions))
                end += 1
            elif isinstance(item.data, AMRData):
                level_masks.append(force_array(item.data._get_cut_mask(grid),
                    grid.ActiveDimensions))
            else:
                mylog.error("Item in the boolean construction unidentified.")
        # Now we do the logic on our level_mask.
        # There should be no nested logic anymore.
        # The first item should be a cut_mask,
        # so that will be our starting point.
        this_cut_mask = level_masks[0]
        for i, item in enumerate(level_masks):
            # I could use a slice above, but I'll keep i consistent instead.
            if i == 0: continue
            if item == "AND":
                # So, the next item in level_masks we want to AND.
                np.bitwise_and(this_cut_mask, level_masks[i+1], this_cut_mask)
            if item == "NOT":
                # It's convenient to remember that NOT == AND NOT
                np.bitwise_and(this_cut_mask, np.invert(level_masks[i+1]),
                    this_cut_mask)
            if item == "OR":
                np.bitwise_or(this_cut_mask, level_masks[i+1], this_cut_mask)
        if not isinstance(grid, FakeGridForParticles):
            self._cut_masks[grid.id] = this_cut_mask
        return this_cut_mask

class AMRSurfaceBase(AMRData, ParallelAnalysisInterface):
    r"""This surface object identifies isocontours on a cell-by-cell basis,
    with no consideration of global connectedness, and returns the vertices
    of the Triangles in that isocontour.

    This object simply returns the vertices of all the triangles
    calculated by the marching cubes algorithm; for more complex
    operations, such as identifying connected sets of cells above a given
    threshold, see the extract_connected_sets function.  This is more
    useful for calculating, for instance, total isocontour area, or
    visualizing in an external program (such as `MeshLab
    <http://meshlab.sf.net>`_.)  The object has the properties .vertices
    and will sample values if a field is requested.  The values are
    interpolated to the center of a given face.

    Parameters
    ----------
    data_source : AMR3DDataObject
        This is the object which will used as a source
    surface_field : string
        Any field that can be obtained in a data object.  This is the field
        which will be isocontoured.
    field_value : float
        The value at which the isocontour should be calculated.

    References
    ----------

    .. [1] Marching Cubes: http://en.wikipedia.org/wiki/Marching_cubes

    Examples
    --------
    This will create a data object, find a nice value in the center, and
    output the vertices to "triangles.obj" after rescaling them.

    >>> sp = pf.h.sphere("max", (10, "kpc")
    >>> surf = pf.h.surface(sp, "Density", 5e-27)
    >>> print surf["Temperature"]
    >>> print surf.vertices
    >>> bounds = [(sp.center[i] - 5.0/pf['kpc'],
    ...            sp.center[i] + 5.0/pf['kpc']) for i in range(3)]
    >>> surf.export_ply("my_galaxy.ply", bounds = bounds)
    """
    _type_name = "surface"
    _con_args = ("data_source", "surface_field", "field_value")
    vertices = None
    def __init__(self, data_source, surface_field, field_value):
        ParallelAnalysisInterface.__init__(self)
        self.data_source = data_source
        self.surface_field = surface_field
        self.field_value = field_value
        self.vertex_samples = YTFieldData()
        center = data_source.get_field_parameter("center")
        AMRData.__init__(self, center = center, fields = None, pf =
                         data_source.pf)
        self._grids = self.data_source._grids.copy()

    def get_data(self, fields = None, sample_type = "face"):
        if isinstance(fields, list) and len(fields) > 1:
            for field in fields: self.get_data(field)
            return
        elif isinstance(fields, list):
            fields = fields[0]
        # Now we have a "fields" value that is either a string or None
        pb = get_pbar("Extracting (sampling: %s)" % fields,
                      len(list(self._get_grid_objs())))
        verts = []
        samples = []
        for i,g in enumerate(self._get_grid_objs()):
            pb.update(i)
            my_verts = self._extract_isocontours_from_grid(
                            g, self.surface_field, self.field_value,
                            fields, sample_type)
            if fields is not None:
                my_verts, svals = my_verts
                samples.append(svals)
            verts.append(my_verts)
        pb.finish()
        verts = np.concatenate(verts).transpose()
        verts = self.comm.par_combine_object(verts, op='cat', datatype='array')
        self.vertices = verts
        if fields is not None:
            samples = np.concatenate(samples)
            samples = self.comm.par_combine_object(samples, op='cat',
                                datatype='array')
            if sample_type == "face":
                self[fields] = samples
            elif sample_type == "vertex":
                self.vertex_samples[fields] = samples


    @restore_grid_state
    def _extract_isocontours_from_grid(self, grid, field, value,
                                       sample_values = None,
                                       sample_type = "face"):
        mask = self.data_source._get_cut_mask(grid) * grid.child_mask
        vals = grid.get_vertex_centered_data(field, no_ghost = False)
        if sample_values is not None:
            svals = grid.get_vertex_centered_data(sample_values)
        else:
            svals = None
        sample_type = {"face":1, "vertex":2}[sample_type]
        my_verts = march_cubes_grid(value, vals, mask, grid.LeftEdge,
                                    grid.dds, svals, sample_type)
        return my_verts

    def calculate_flux(self, field_x, field_y, field_z, fluxing_field = None):
        r"""This calculates the flux over the surface.

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

        >>> sp = pf.h.sphere("max", (10, "kpc")
        >>> surf = pf.h.surface(sp, "Density", 5e-27)
        >>> flux = surf.calculate_flux(
        ...     "x-velocity", "y-velocity", "z-velocity", "Metal_Density")
        """
        flux = 0.0
        pb = get_pbar("Fluxing %s" % fluxing_field,
                len(list(self._get_grid_objs())))
        for i, g in enumerate(self._get_grid_objs()):
            pb.update(i)
            flux += self._calculate_flux_in_grid(g,
                    field_x, field_y, field_z, fluxing_field)
        pb.finish()
        flux = self.comm.mpi_allreduce(flux, op="sum")
        return flux

    @restore_grid_state
    def _calculate_flux_in_grid(self, grid,
                    field_x, field_y, field_z, fluxing_field = None):
        mask = self.data_source._get_cut_mask(grid) * grid.child_mask
        vals = grid.get_vertex_centered_data(self.surface_field)
        if fluxing_field is None:
            ff = np.ones(vals.shape, dtype="float64")
        else:
            ff = grid.get_vertex_centered_data(fluxing_field)
        xv, yv, zv = [grid.get_vertex_centered_data(f) for f in
                     [field_x, field_y, field_z]]
        return march_cubes_grid_flux(self.field_value, vals, xv, yv, zv,
                    ff, mask, grid.LeftEdge, grid.dds)

    @property
    def triangles(self):
        if self.vertices is None:
            self.get_data()
        vv = np.empty((self.vertices.shape[1]/3, 3, 3), dtype="float64")
        for i in range(3):
            for j in range(3):
                vv[:,i,j] = self.vertices[j,i::3]
        return vv

    def export_obj(self, filename, transparency = 1.0, dist_fac = None,
                   color_field = None, emit_field = None, color_map = "algae", 
                   color_log = True, emit_log = True, plot_index = None, 
                   color_field_max = None, color_field_min = None, 
                   emit_field_max = None, emit_field_min = None):
        r"""This exports the surface to the OBJ format, suitable for visualization
        in many different programs (e.g., Blender).  NOTE: this exports an .obj file 
        and an .mtl file, both with the general 'filename' as a prefix.  
        The .obj file points to the .mtl file in its header, so if you move the 2 
        files, make sure you change the .obj header to account for this. ALSO NOTE: 
        the emit_field needs to be a combination of the other 2 fields used to 
        have the emissivity track with the color.

        Parameters
        ----------
        filename : string
            The file this will be exported to.  This cannot be a file-like object.
            Note - there are no file extentions included - both obj & mtl files 
            are created.
        transparency : float
            This gives the transparency of the output surface plot.  Values
            from 0.0 (invisible) to 1.0 (opaque).
        dist_fac : float
            Divide the axes distances by this amount.
        color_field : string
            Should a field be sample and colormapped?
        emit_field : string
            Should we track the emissivity of a field?
              NOTE: this should be a combination of the other 2 fields being used.
        color_map : string
            Which color map should be applied?
        color_log : bool
            Should the color field be logged before being mapped?
        emit_log : bool
            Should the emitting field be logged before being mapped?
        plot_index : integer
            Index of plot for multiple plots.  If none, then only 1 plot.
        color_field_max : float
            Maximum value of the color field across all surfaces.
        color_field_min : float
            Minimum value of the color field across all surfaces.
        emit_field_max : float
            Maximum value of the emitting field across all surfaces.
        emit_field_min : float
            Minimum value of the emitting field across all surfaces.

        Examples
        --------

        >>> sp = pf.h.sphere("max", (10, "kpc"))
        >>> trans = 1.0
        >>> distf = 3.1e18*1e3 # distances into kpc
        >>> surf = pf.h.surface(sp, "Density", 5e-27)
        >>> surf.export_obj("my_galaxy", transparency=trans, dist_fac = distf)

        >>> sp = pf.h.sphere("max", (10, "kpc"))
        >>> mi, ma = sp.quantities['Extrema']('Temperature')[0]
        >>> rhos = [1e-24, 1e-25]
        >>> trans = [0.5, 1.0]
        >>> distf = 3.1e18*1e3 # distances into kpc
        >>> for i, r in enumerate(rhos):
        ...     surf = pf.h.surface(sp,'Density',r)
        ...     surf.export_obj("my_galaxy", transparency=trans[i], 
        ...                      color_field='Temperature', dist_fac = distf, 
        ...                      plot_index = i, color_field_max = ma, 
        ...                      color_field_min = mi)

        >>> sp = pf.h.sphere("max", (10, "kpc"))
        >>> rhos = [1e-24, 1e-25]
        >>> trans = [0.5, 1.0]
        >>> distf = 3.1e18*1e3 # distances into kpc
        >>> def _Emissivity(field, data):
        ...     return (data['Density']*data['Density']*np.sqrt(data['Temperature']))
        >>> add_field("Emissivity", function=_Emissivity, units=r"\rm{g K}/\rm{cm}^{6}")
        >>> for i, r in enumerate(rhos):
        ...     surf = pf.h.surface(sp,'Density',r)
        ...     surf.export_obj("my_galaxy", transparency=trans[i], 
        ...                      color_field='Temperature', emit_field = 'Emissivity', 
        ...                      dist_fac = distf, plot_index = i)

        """
        if self.vertices is None:
            self.get_data(color_field,"face")
        elif color_field is not None:
            if color_field not in self.field_data:
                self[color_field]
        if emit_field is not None:
            if color_field not in self.field_data:
                self[emit_field]
        only_on_root(self._export_obj, filename, transparency, dist_fac, color_field, emit_field, 
                             color_map, color_log, emit_log, plot_index, color_field_max, 
                             color_field_min, emit_field_max, emit_field_min)

    def _color_samples_obj(self, cs, em, color_log, emit_log, color_map, arr, 
                           color_field_max, color_field_min, 
                           emit_field_max, emit_field_min): # this now holds for obj files
        if color_log: cs = np.log10(cs)
        if emit_log: em = np.log10(em)
        if color_field_min is None:
            mi = cs.min()
        else:
            mi = color_field_min
            if color_log: mi = np.log10(mi)
        if color_field_max is None:
            ma = cs.max()
        else:
            ma = color_field_max
            if color_log: ma = np.log10(ma)
        cs = (cs - mi) / (ma - mi)
        # to get color indicies for OBJ formatting
        from yt.visualization._colormap_data import color_map_luts
        lut = color_map_luts[color_map]
        x = np.mgrid[0.0:1.0:lut[0].shape[0]*1j]
        arr["cind"][:] = (np.interp(cs,x,x)*(lut[0].shape[0]-1)).astype("uint8")
        # now, get emission
        if emit_field_min is None:
            emi = em.min()
        else:
            emi = emit_field_min
            if emit_log: emi = np.log10(emi)
        if emit_field_max is None:
            ema = em.max()
        else:
            ema = emit_field_max
            if emit_log: ema = np.log10(ema)
        em = (em - emi)/(ema - emi)
        x = np.mgrid[0.0:255.0:2j] # assume 1 emissivity per color
        arr["emit"][:] = (np.interp(em,x,x))*2.0 # for some reason, max emiss = 2

    @parallel_root_only
    def _export_obj(self, filename, transparency, dist_fac = None, 
                    color_field = None, emit_field = None, color_map = "algae", 
                    color_log = True, emit_log = True, plot_index = None, 
                    color_field_max = None, color_field_min = None, 
                    emit_field_max = None, emit_field_min = None):
        if plot_index is None:
            plot_index = 0
        if isinstance(filename, file):
            fobj = filename + '.obj'
            fmtl = filename + '.mtl'
        else:
            if plot_index == 0:
                fobj = open(filename + '.obj', "w")
                fmtl = open(filename + '.mtl', 'w')
                cc = 1
            else:
                # read in last vertex
                linesave = ''
                for line in fileinput.input(filename + '.obj'):
                    if line[0] == 'f':
                        linesave = line
                p = [m.start() for m in finditer(' ', linesave)]
                cc = int(linesave[p[len(p)-1]:])+1
                fobj = open(filename + '.obj', "a")
                fmtl = open(filename + '.mtl', 'a')
        ftype = [("cind", "uint8"), ("emit", "float")]
        vtype = [("x","float"),("y","float"), ("z","float")]
        if plot_index == 0:
            fobj.write("# yt OBJ file\n")
            fobj.write("# www.yt-project.com\n")
            fobj.write("mtllib " + filename + '.mtl\n\n')  # use this material file for the faces
            fmtl.write("# yt MLT file\n")
            fmtl.write("# www.yt-project.com\n\n")
        #(0) formulate vertices
        nv = self.vertices.shape[1] # number of groups of vertices
        f = np.empty(nv/self.vertices.shape[0], dtype=ftype) # store sets of face colors
        v = np.empty(nv, dtype=vtype) # stores vertices
        if color_field is not None:
            cs = self[color_field]
        else:
            cs = np.empty(self.vertices.shape[1]/self.vertices.shape[0])
        if emit_field is not None:
            em = self[emit_field]
        else:
            em = np.empty(self.vertices.shape[1]/self.vertices.shape[0])            
        self._color_samples_obj(cs, em, color_log, emit_log, color_map, f, 
                                color_field_max, color_field_min, 
                                emit_field_max, emit_field_min) # map color values to color scheme
        from yt.visualization._colormap_data import color_map_luts # import colors for mtl file
        lut = color_map_luts[color_map] # enumerate colors
        # interpolate emissivity to enumerated colors
        emiss = np.interp(np.mgrid[0:lut[0].shape[0]],np.mgrid[0:len(cs)],f["emit"][:])
        if dist_fac is None: # then normalize by bounds
            DLE = self.pf.domain_left_edge
            DRE = self.pf.domain_right_edge
            bounds = [(DLE[i], DRE[i]) for i in range(3)]
            for i, ax in enumerate("xyz"):
                # Do the bounds first since we cast to f32
                tmp = self.vertices[i,:]
                np.subtract(tmp, bounds[i][0], tmp)
                w = bounds[i][1] - bounds[i][0]
                np.divide(tmp, w, tmp)
                np.subtract(tmp, 0.5, tmp) # Center at origin.
                v[ax][:] = tmp   
        else:
            for i, ax in enumerate("xyz"):
                tmp = self.vertices[i,:]
                np.divide(tmp, dist_fac, tmp)
                v[ax][:] = tmp
        #(1) write all colors per surface to mtl file
        for i in range(0,lut[0].shape[0]): 
            omname = "material_" + str(i) + '_' + str(plot_index)  # name of the material
            fmtl.write("newmtl " + omname +'\n') # the specific material (color) for this face
            fmtl.write("Ka %.6f %.6f %.6f\n" %(0.0, 0.0, 0.0)) # ambient color, keep off
            fmtl.write("Kd %.6f %.6f %.6f\n" %(lut[0][i], lut[1][i], lut[2][i])) # color of face
            fmtl.write("Ks %.6f %.6f %.6f\n" %(0.0, 0.0, 0.0)) # specular color, keep off
            fmtl.write("d %.6f\n" %(transparency))  # transparency
            fmtl.write("em %.6f\n" %(emiss[i])) # emissivity per color
            fmtl.write("illum 2\n") # not relevant, 2 means highlights on?
            fmtl.write("Ns %.6f\n\n" %(0.0)) #keep off, some other specular thing
        #(2) write vertices
        for i in range(0,self.vertices.shape[1]):
            fobj.write("v %.6f %.6f %.6f\n" %(v["x"][i], v["y"][i], v["z"][i]))    
        fobj.write("#done defining vertices\n\n")
        #(3) define faces and materials for each face
        for i in range(0,self.triangles.shape[0]):
            omname = 'material_' + str(f["cind"][i]) + '_' + str(plot_index) # which color to use
            fobj.write("usemtl " + omname + '\n') # which material to use for this face (color)
            fobj.write("f " + str(cc) + ' ' + str(cc+1) + ' ' + str(cc+2) + '\n\n') # vertices to color
            cc = cc+3
        fmtl.close()
        fobj.close()


    def export_ply(self, filename, bounds = None, color_field = None,
                   color_map = "algae", color_log = True, sample_type = "face"):
        r"""This exports the surface to the PLY format, suitable for visualization
        in many different programs (e.g., MeshLab).

        Parameters
        ----------
        filename : string
            The file this will be exported to.  This cannot be a file-like object.
        bounds : list of tuples
            The bounds the vertices will be normalized to.  This is of the format:
            [(xmin, xmax), (ymin, ymax), (zmin, zmax)].  Defaults to the full
            domain.
        color_field : string
            Should a field be sample and colormapped?
        color_map : string
            Which color map should be applied?
        color_log : bool
            Should the color field be logged before being mapped?

        Examples
        --------

        >>> sp = pf.h.sphere("max", (10, "kpc")
        >>> surf = pf.h.surface(sp, "Density", 5e-27)
        >>> print surf["Temperature"]
        >>> print surf.vertices
        >>> bounds = [(sp.center[i] - 5.0/pf['kpc'],
        ...            sp.center[i] + 5.0/pf['kpc']) for i in range(3)]
        >>> surf.export_ply("my_galaxy.ply", bounds = bounds)
        """
        if self.vertices is None:
            self.get_data(color_field, sample_type)
        elif color_field is not None:
            if sample_type == "face" and \
                color_field not in self.field_data:
                self[color_field]
            elif sample_type == "vertex" and \
                color_field not in self.vertex_data:
                self.get_data(color_field, sample_type)
        self._export_ply(filename, bounds, color_field, color_map, color_log,
                         sample_type)

    def _color_samples(self, cs, color_log, color_map, arr):
            if color_log: cs = np.log10(cs)
            mi, ma = cs.min(), cs.max()
            cs = (cs - mi) / (ma - mi)
            from yt.visualization.image_writer import map_to_colors
            cs = map_to_colors(cs, color_map)
            arr["red"][:] = cs[0,:,0]
            arr["green"][:] = cs[0,:,1]
            arr["blue"][:] = cs[0,:,2]

    @parallel_root_only
    def _export_ply(self, filename, bounds = None, color_field = None,
                   color_map = "algae", color_log = True, sample_type = "face"):
        if isinstance(filename, file):
            f = filename
        else:
            f = open(filename, "wb")
        if bounds is None:
            DLE = self.pf.domain_left_edge
            DRE = self.pf.domain_right_edge
            bounds = [(DLE[i], DRE[i]) for i in range(3)]
        nv = self.vertices.shape[1]
        vs = [("x", "<f"), ("y", "<f"), ("z", "<f"),
              ("red", "uint8"), ("green", "uint8"), ("blue", "uint8") ]
        fs = [("ni", "uint8"), ("v1", "<i4"), ("v2", "<i4"), ("v3", "<i4"),
              ("red", "uint8"), ("green", "uint8"), ("blue", "uint8") ]
        f.write("ply\n")
        f.write("format binary_little_endian 1.0\n")
        f.write("element vertex %s\n" % (nv))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if color_field is not None and sample_type == "vertex":
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            v = np.empty(self.vertices.shape[1], dtype=vs)
            cs = self.vertex_samples[color_field]
            self._color_samples(cs, color_log, color_map, v)
        else:
            v = np.empty(self.vertices.shape[1], dtype=vs[:3])
        f.write("element face %s\n" % (nv/3))
        f.write("property list uchar int vertex_indices\n")
        if color_field is not None and sample_type == "face":
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            # Now we get our samples
            cs = self[color_field]
            arr = np.empty(cs.shape[0], dtype=np.dtype(fs))
            self._color_samples(cs, color_log, color_map, arr)
        else:
            arr = np.empty(nv/3, np.dtype(fs[:-3]))
        for i, ax in enumerate("xyz"):
            # Do the bounds first since we cast to f32
            tmp = self.vertices[i,:]
            np.subtract(tmp, bounds[i][0], tmp)
            w = bounds[i][1] - bounds[i][0]
            np.divide(tmp, w, tmp)
            np.subtract(tmp, 0.5, tmp) # Center at origin.
            v[ax][:] = tmp
        f.write("end_header\n")
        v.tofile(f)
        arr["ni"][:] = 3
        vi = np.arange(nv, dtype="<i")
        vi.shape = (nv/3, 3)
        arr["v1"][:] = vi[:,0]
        arr["v2"][:] = vi[:,1]
        arr["v3"][:] = vi[:,2]
        arr.tofile(f)
        if filename is not f:
            f.close()

    def export_sketchfab(self, title, description, api_key = None,
                            color_field = None, color_map = "algae",
                            color_log = True, bounds = None):
        r"""This exports Surfaces to SketchFab.com, where they can be viewed
        interactively in a web browser.

        SketchFab.com is a proprietary web service that provides WebGL
        rendering of models.  This routine will use temporary files to
        construct a compressed binary representation (in .PLY format) of the
        Surface and any optional fields you specify and upload it to
        SketchFab.com.  It requires an API key, which can be found on your
        SketchFab.com dashboard.  You can either supply the API key to this
        routine directly or you can place it in the variable
        "sketchfab_api_key" in your ~/.yt/config file.  This function is
        parallel-safe.

        Parameters
        ----------
        title : string
            The title for the model on the website
        description : string
            How you want the model to be described on the website
        api_key : string
            Optional; defaults to using the one in the config file
        color_field : string
            If specified, the field by which the surface will be colored
        color_map : string
            The name of the color map to use to map the color field
        color_log : bool
            Should the field be logged before being mapped to RGB?
        bounds : list of tuples
            [ (xmin, xmax), (ymin, ymax), (zmin, zmax) ] within which the model
            will be scaled and centered.  Defaults to the full domain.

        Returns
        -------
        URL : string
            The URL at which your model can be viewed.

        Examples
        --------

        >>> from yt.mods import *
        >>> pf = load("redshift0058")
        >>> dd = pf.h.sphere("max", (200, "kpc"))
        >>> rho = 5e-27
        >>> bounds = [(dd.center[i] - 100.0/pf['kpc'],
        ...            dd.center[i] + 100.0/pf['kpc']) for i in range(3)]
        ...
        >>> surf = pf.h.surface(dd, "Density", rho)
        >>> rv = surf.export_sketchfab(
        ...     title = "Testing Upload",
        ...     description = "A simple test of the uploader",
        ...     color_field = "Temperature",
        ...     color_map = "hot",
        ...     color_log = True,
        ...     bounds = bounds)
        ...
        """
        api_key = api_key or ytcfg.get("yt","sketchfab_api_key")
        if api_key in (None, "None"):
            raise YTNoAPIKey("SketchFab.com", "sketchfab_api_key")
        import zipfile, json
        from tempfile import TemporaryFile

        ply_file = TemporaryFile()
        self.export_ply(ply_file, bounds, color_field, color_map, color_log,
                        sample_type = "vertex")
        ply_file.seek(0)
        # Greater than ten million vertices and we throw an error but dump
        # to a file.
        if self.vertices.shape[1] > 1e7:
            tfi = 0
            fn = "temp_model_%03i.ply" % tfi
            while os.path.exists(fn):
                fn = "temp_model_%03i.ply" % tfi
                tfi += 1
            open(fn, "wb").write(ply_file.read())
            raise YTTooManyVertices(self.vertices.shape[1], fn)

        zfs = TemporaryFile()
        with zipfile.ZipFile(zfs, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("yt_export.ply", ply_file.read())
        zfs.seek(0)

        zfs.seek(0)
        data = {
            'title': title,
            'token': api_key,
            'description': description,
            'fileModel': zfs,
            'filenameModel': "yt_export.zip",
        }
        upload_id = self._upload_to_sketchfab(data)
        upload_id = self.comm.mpi_bcast(upload_id, root = 0)
        return upload_id

    @parallel_root_only
    def _upload_to_sketchfab(self, data):
        import urllib2, json
        from yt.utilities.poster.encode import multipart_encode
        from yt.utilities.poster.streaminghttp import register_openers
        register_openers()
        datamulti, headers = multipart_encode(data)
        request = urllib2.Request("https://api.sketchfab.com/v1/models",
                        datamulti, headers)
        rv = urllib2.urlopen(request).read()
        rv = json.loads(rv)
        upload_id = rv.get("result", {}).get("id", None)
        if upload_id:
            mylog.info("Model uploaded to: https://sketchfab.com/show/%s",
                       upload_id)
        else:
            mylog.error("Problem uploading.")
        return upload_id

# Many of these items are set up specifically to ensure that
# we are not breaking old pickle files.  This means we must only call the
# _reconstruct_object and that we cannot mandate any additional arguments to
# the reconstruction function.
#
# In the future, this would be better off being set up to more directly
# reference objects or retain state, perhaps with a context manager.
#
# One final detail: time series or multiple parameter files in a single pickle
# seems problematic.

class ReconstructedObject(tuple):
    pass

def _check_nested_args(arg, ref_pf):
    if not isinstance(arg, (tuple, list, ReconstructedObject)):
        return arg
    elif isinstance(arg, ReconstructedObject) and ref_pf == arg[0]:
        return arg[1]
    narg = [_check_nested_args(a, ref_pf) for a in arg]
    return narg

def _get_pf_by_hash(hash):
    from yt.data_objects.static_output import _cached_pfs
    for pf in _cached_pfs.values():
        if pf._hash() == hash: return pf
    return None

def _reconstruct_object(*args, **kwargs):
    pfid = args[0]
    dtype = args[1]
    pf = _get_pf_by_hash(pfid)
    if not pf:
        pfs = ParameterFileStore()
        pf = pfs.get_pf_hash(pfid)
    field_parameters = args[-1]
    # will be much nicer when we can do pfid, *a, fp = args
    args = args[2:-1]
    new_args = [_check_nested_args(a, pf) for a in args]
    cls = getattr(pf.h, dtype)
    obj = cls(*new_args)
    obj.field_parameters.update(field_parameters)
    return ReconstructedObject((pf, obj))
