"""
Various non-grid data containers.

@author: U{Matthew Turk<http://www.stanford.edu/~mturk/>}
@organization: U{KIPAC<http://www-group.slac.stanford.edu/KIPAC/>}
@contact: U{mturk@slac.stanford.edu<mailto:mturk@slac.stanford.edu>}
@license:
  Copyright (C) 2007 Matthew Turk.  All Rights Reserved.

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

from yt.lagos import *

def restore_grid_state(func):
    def save_state(self, grid, field=None):
        old_params = grid.field_parameters
        grid.field_parameters = self.field_parameters
        tr = func(self, grid, field)
        grid.field_parameters = old_params
        return tr
    return save_state


class EnzoData:
    """
    Generic EnzoData container.  By itself, will attempt to
    generate field, read fields (method defined by derived classes)
    and deal with passing back and forth field parameters.
    """
    _grids = None
    _num_ghost_zones = 0

    def __init__(self, pf, fields):
        """
        @param pf: The parameterfile associated with this container
        @type hierarchy: L{EnzoOutput<EnzoOutput>}
        @param fields: Fields represented in the data
        @type fields: list of strings
        """
        if pf != None:
            self.pf = pf
            self.hierarchy = pf.hierarchy
        if fields == None: fields = []
        self.fields = ensure_list(fields)
        self.data = {}
        self.field_parameters = {}
        self.__set_default_field_parameters()

    def __set_default_field_parameters(self):
        self.set_field_parameter("center",na.zeros(3,dtype='float64'))
        self.set_field_parameter("bulk_velocity",na.zeros(3,dtype='float64'))

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
        return self.field_parameters.has_key(name)

    def convert(self, datatype):
        return self.hierarchy[datatype]

    def clear_data(self):
        """
        Clears out all data from the EnzoData instance, freeing memory.
        """
        del self.data
        self.data = {}

    def has_key(self, key):
        return self.data.has_key(key)

    def _refresh_data(self):
        """
        Wipes data and rereads/regenerates it from the self.fields.
        """
        self.clear_data()
        self.get_data()

    def __getitem__(self, key):
        """
        Returns a single field.  Will add if necessary.
        """
        if not self.data.has_key(key):
            if key not in self.fields:
                self.fields.append(key)
            self.get_data(key)
        return self.data[key]

    def __setitem__(self, key, val):
        """
        Sets a field to be some other value.
        """
        if key not in self.fields: self.fields.append(key)
        self.data[key] = val

    def __delitem__(self, key):
        """
        Sets a field to be some other value.
        """
        try:
            del self.fields[self.fields.index(key)]
        except ValueError:
            pass
        del self.data[key]

    def _generate_field_in_grids(self, fieldName):
        pass

class Enzo1DData(EnzoData):
    _spatial = False
    def __init__(self, pf, fields):
        EnzoData.__init__(self, pf, fields)
        self._grids = None

    def _generate_field_in_grids(self, field, num_ghost_zones=0):
        for grid in self._grids:
            temp = grid[field]

    def _generate_field(self, field):
        if fieldInfo.has_key(field):
            # First we check the validator
            try:
                fieldInfo[field].check_available(self)
            except NeedsGridType, ngt_exception:
                # We leave this to be implementation-specific
                self._generate_field_in_grids(field, ngt_exception.ghost_zones)
                return False
            else:
                self[field] = fieldInfo[field](self)
                return True
        else: # Can't find the field, try as it might
            raise exceptions.KeyError(field)


class EnzoOrthoRayBase(Enzo1DData):
    def __init__(self, axis, coords, fields=None, pf=None):
        Enzo1DData.__init__(self, pf, fields)
        self.axis = axis
        self.px_ax = x_dict[self.axis]
        self.py_ax = y_dict[self.axis]
        self.px_dx = 'd%s'%(axis_names[self.px_ax])
        self.py_dx = 'd%s'%(axis_names[self.py_ax])
        self.px, self.py = coords
        self._refresh_data()

    def _get_list_of_grids(self):
        y = na.where( (self.px > self.pf.hierarchy.gridLeftEdge[:,self.px_ax])
                    & (self.px < self.pf.hierarchy.gridRightEdge[:,self.px_ax])
                    & (self.py > self.pf.hierarchy.gridLeftEdge[:,self.py_ax])
                    & (self.py < self.pf.hierarchy.gridRightEdge[:,self.py_ax]))
        self._grids = self.hierarchy.grids[y]

    def get_data(self, fields=None, in_grids=False):
        if self._grids == None:
            self._get_list_of_grids()
        points = []
        #if not self.has_key('dx'):
            #self._generate_coords()
        if not fields:
            fields_to_get = self.fields
        else:
            fields_to_get = ensure_list(fields)
        mylog.debug("Going to obtain %s (%s)", fields_to_get, self.fields)
        for field in fields_to_get:
            if self.data.has_key(field):
                continue
            mylog.info("Getting field %s from %s", field, len(self._grids))
            if field not in self.hierarchy.field_list and not in_grids:
                if self._generate_field(field):
                    continue # True means we already assigned it
            self[field] = na.concatenate(
                [self._get_data_from_grid(grid, field)
                 for grid in self._grids])

    def _get_data_from_grid(self, grid, field):
        # We are orthogonal, so we can feel free to make assumptions
        # for the sake of speed.
        gdx = just_one(grid[self.px_dx])
        gdy = just_one(grid[self.py_dx])
        x_coord = int((self.px - grid.LeftEdge[self.px_ax])/gdx)
        y_coord = int((self.py - grid.LeftEdge[self.py_ax])/gdy)
        sl = [None,None,None]
        sl[self.px_ax] = slice(x_coord,x_coord+1,None)
        sl[self.py_ax] = slice(y_coord,y_coord+1,None)
        sl[self.axis] = slice(None)
        if not iterable(grid[field]):
            gf = grid[field] * na.ones(grid.child_mask[sl].shape)
        else:
            gf = grid[field][sl]
        return gf[na.where(grid.child_mask[sl])]

class Enzo2DData(EnzoData):
    """
    Class to represent a set of EnzoData that's 2-D in nature, and thus
    does not have as many actions as the 3-D data types.
    """
    _spatial = False
    def __init__(self, axis, fields, pf=None):
        """
        Prepares the Enzo2DData.

        @param axis: Axis to slice along
        @type axis: integer (0,1,2, 4)
        @param fields: fields to be processed or generated
        @type fields: list of strings
        """
        self.axis = axis
        EnzoData.__init__(self, pf, fields)

    @time_execution
    def get_data(self, fields = None):
        """
        Iterates over the list of fields and generates/reads them all.

        @keyword field: field to add (list or single)
        @type field: string or list of strings
        """
        # We get it for the values in fields and coords
        # We take a 3-tuple of the coordinate we want to slice through, as well
        # as the axis we're slicing along
        self._get_list_of_grids()
        if not self.has_key('dx'):
            self._generate_coords()
        if fields == None:
            fields_to_get = self.fields
        else:
            fields_to_get = ensure_list(fields)
        for field in fields_to_get:
            if self.data.has_key(field):
                continue
            rvs=[]
            if field not in self.hierarchy.field_list:
                if self._generate_field(field):
                    continue # A "True" return means we did it
            self[field] = na.concatenate(
                [self._get_data_from_grid(grid, field)
                 for grid in self._grids])


    def _generate_field(self, field):
        if fieldInfo.has_key(field):
            # First we check the validator
            try:
                fieldInfo[field].check_available(self)
            except NeedsGridType, ngt_exception:
                # We leave this to be implementation-specific
                self._generate_field_in_grids(field, ngt_exception.ghost_zones)
                return False
            else:
                self[field] = fieldInfo[field](self)
                return True
        else: # Can't find the field, try as it might
            raise exceptions.KeyError(field)

    def _generate_field_in_grids(self, field, num_ghost_zones=0):
        for grid in self._grids:
            temp = grid[field]

    def interpolate_discretize(self, LE, RE, field, side, logSpacing=True):
        """
        This returns a uniform grid of points, interpolated using the nearest
        neighbor method.

        @note: Requires Delaunay triangulation, which is not included in
        most/all scipy binaries.
        @param LE: Left Edge of interpolation region
        @type LE: array of Floats
        @param RE: Right Edge of interpolation region
        @type RE: array of Floats
        @param field: The field to discretize
        @type field: string
        @param side: The number of points on a side
        @type side: int
        """
        import scipy.sandbox.delaunay as de
        if logSpacing:
            zz = na.log10(self[field])
        else:
            zz = self[field]
        xi, yi = na.array( \
                 na.mgrid[LE[0]:RE[0]:side*1j, \
                          LE[1]:RE[1]:side*1j], 'float64')
        zi = de.Triangulation(self['px'],self['py']).nn_interpolator(zz)\
                 [LE[0]:RE[0]:side*1j, \
                  LE[1]:RE[1]:side*1j]
        if logSpacing:
            zi = 10**(zi)
        return [xi,yi,zi]
        #return [xx,yy,zz]

class EnzoSliceBase(Enzo2DData):
    """
    EnzoSlice is an orthogonal slice through the data, taking all the points
    at the finest resolution available and then indexing them.  It is more
    appropriately thought of as a slice 'operator' than an object,
    however, as its field and coordinate can both change.
    """

    @time_execution
    def __init__(self, axis, coord, fields = None, center=None, pf=None):
        """
        @param axis: axis to which this data is parallel
        @type axis: integer (0,1,2)
        @param coord: three points defining the center
        @type coord: na.array
        @keyword fields: fields to be processed or generated
        @type fields: list of strings
        """
        Enzo2DData.__init__(self, axis, fields, pf)
        self.center = center
        self.coord = coord
        self._refresh_data()

    def reslice(self, coord):
        """
        Change the entire dataset, clearing out the current data and slicing at
        a new location.  Not terribly useful except for in-place plot changes.

        @param coord: New coordinate for slice
        @type coord: float
        """
        mylog.debug("Setting coordinate to %0.5e" % coord)
        self.coord = coord
        self._refresh_data()

    def shift(self, val):
        """
        Moves the slice coordinate up by either a floating point value, or an
        integer number of indices of the finest grid.

        @param val: shift amount
        @type val: integer (number of cells) or float (distance)
        """
        if isinstance(val, types.FloatType):
            # We add the dx
            self.coord += val
        elif isinstance(val, types.IntType):
            # Here we assume that the grid is the max level
            level = self.hierarchy.max_level
            self.coord
            dx = self.hierarchy.gridDxs[self.hierarchy.levelIndices[level][0]]
            self.coord += dx * val
        else:
            raise ValueError(val)
        self.refreshData()

    def _generate_coords(self):
        points = []
        for grid in self._grids:
            points.append(self._generate_grid_coords(grid))
        t = na.concatenate(points)
        self['px'] = t[:,0]
        self['py'] = t[:,1]
        self['pz'] = t[:,2]
        self['pdx'] = t[:,3]
        self['pdy'] = t[:,3]
        self['pdz'] = t[:,3]

        # Now we set the *actual* coordinates
        self[axis_names[x_dict[self.axis]]] = t[:,0]
        self[axis_names[y_dict[self.axis]]] = t[:,1]
        self[axis_names[self.axis]] = t[:,2]
        self['dx'] = t[:,3]
        self['dy'] = t[:,3]
        self['dz'] = t[:,3]

        self.ActiveDimensions = (t.shape[0], 1, 1)

    def _get_list_of_grids(self):
        self._grids, ind = self.hierarchy.find_slice_grids(self.coord, self.axis)

    def _generate_grid_coords(self, grid):
        xaxis = x_dict[self.axis]
        yaxis = y_dict[self.axis]
        wantedIndex = int(((self.coord-grid.LeftEdge[self.axis])/grid.dx))
        sl = [slice(None), slice(None), slice(None)]
        sl[self.axis] = slice(wantedIndex, wantedIndex + 1)
        #sl.reverse()
        sl = tuple(sl)
        nx = grid.child_mask.shape[xaxis]
        ny = grid.child_mask.shape[yaxis]
        cm = na.where(grid.child_mask[sl].ravel() == 1)
        cmI = na.indices((nx,ny))
        xind = cmI[0,:].ravel()
        xpoints = na.ones(cm[0].shape, 'float64')
        xpoints *= xind[cm]*grid.dx+(grid.LeftEdge[xaxis] + 0.5*grid.dx)
        yind = cmI[1,:].ravel()
        ypoints = na.ones(cm[0].shape, 'float64')
        ypoints *= yind[cm]*grid.dx+(grid.LeftEdge[yaxis] + 0.5*grid.dx)
        zpoints = na.ones(xpoints.shape, 'float64') * self.coord
        dx = na.ones(xpoints.shape, 'float64') * grid.dx/2.0
        t = na.array([xpoints, ypoints, zpoints, dx]).swapaxes(0,1)
        return t

    @restore_grid_state
    def _get_data_from_grid(self, grid, field):
        # So what's our index of slicing?  This is what we need to figure out
        # first, so we can deal with our data in the fastest way.
        wantedIndex = int(((self.coord-grid.LeftEdge[self.axis])/grid.dx))
        sl = [slice(None), slice(None), slice(None)]
        sl[self.axis] = slice(wantedIndex, wantedIndex + 1)
        slHERE = tuple(sl)
        sl.reverse()
        slHDF = tuple(sl)
        if fieldInfo.has_key(field) and fieldInfo[field].variable_length:
            return grid[field]
        if not grid.has_key(field):
            conv_factor = 1.0
            if fieldInfo.has_key(field):
                conv_factor = fieldInfo[field]._convert_function(self)
            dv = self._read_data_slice(grid, field, slHDF) * conv_factor
        else:
            dv = grid[field]
            if dv.size == 1: dv = na.ones(grid.ActiveDimensions)*dv
            dv = dv[slHERE]
        cm = na.where(grid.child_mask[slHERE].ravel() == 1)
        dataVals = dv.ravel()[cm]
        return dataVals

    def _generate_field_in_grids(self, field, num_ghost_zones=0):
        for grid in self._grids:
            self.__touch_grid_field(grid, field)

    @restore_grid_state
    def __touch_grid_field(self, grid, field):
        grid[field]

class EnzoCuttingPlaneBase(Enzo2DData):
    """
    EnzoCuttingPlane is an oblique plane through the data,
    defined by a normal vector and a coordinate.  It attempts to guess
    an 'up' vector, which cannot be overridden, and then it pixelizes
    the appropriate data onto the plane without interpolation.
    """
    _plane = None
    def __init__(self, normal, center, fields = None):
        """
        @param normal: Vector normal to which the plane will be defined
        @type normal: List or array of floats
        @param center: The center point of the plane
        @type center: List or array of floats
        """
        Enzo2DData.__init__(self, 4, fields)
        self.center = center
        self.set_field_parameter('center',center)
        self._cut_masks = {}
        # Let's set up our plane equation
        # ax + by + cz + d = 0
        self._norm_vec = normal/na.sqrt(na.dot(normal,normal))
        self._d = -1.0 * na.dot(self._norm_vec, self.center)
        # First we try all three, see which has the best result:
        vecs = na.identity(3)
        _t = na.cross(self._norm_vec, vecs).sum(axis=1)
        ax = nd.maximum_position(_t)
        self._x_vec = na.cross(vecs[ax,:], self._norm_vec).ravel()
        self._x_vec /= na.sqrt(na.dot(self._x_vec, self._x_vec))
        self._y_vec = na.cross(self._norm_vec, self._x_vec).ravel()
        self._y_vec /= na.sqrt(na.dot(self._y_vec, self._y_vec))
        self._rot_mat = na.array([self._x_vec,self._y_vec,self._norm_vec])
        self._inv_mat = na.linalg.pinv(self._rot_mat)
        self._refresh_data()

    def _get_list_of_grids(self):
        # Recall that the projection of the distance vector from a point
        # onto the normal vector of a plane is:
        # D = (a x_0 + b y_0 + c z_0 + d)/sqrt(a^2+b^2+c^2)
        LE = self.pf.h.gridLeftEdge
        RE = self.pf.h.gridRightEdge
        vertices = na.array([[LE[:,0],LE[:,1],LE[:,2]],
                             [RE[:,0],RE[:,1],RE[:,2]],
                             [LE[:,0],LE[:,1],RE[:,2]],
                             [RE[:,0],RE[:,1],LE[:,2]],
                             [LE[:,0],RE[:,1],RE[:,2]],
                             [RE[:,0],LE[:,1],LE[:,2]],
                             [LE[:,0],RE[:,1],LE[:,2]],
                             [RE[:,0],LE[:,1],RE[:,2]]])
        # This gives us shape: 8, 3, n_grid
        D = na.sum(self._norm_vec.reshape((1,3,1)) * vertices, axis=1) + self._d
        self.D = D
        self._grids = self.hierarchy.grids[
            na.where(na.logical_not(na.all(D<0,axis=0) | na.all(D>0,axis=0) )) ]

    def _get_cut_mask(self, grid):
        if self._cut_masks.has_key(grid.id):
            return self._cut_masks[grid.id]
        # This is slow.  Suggestions for improvement would be great...
        ss = grid.ActiveDimensions
        D = na.ones(ss) * self._d
        D += (grid['x'][:,0,0] * self._norm_vec[0]).reshape(ss[0],1,1)
        D += (grid['y'][0,:,0] * self._norm_vec[1]).reshape(1,ss[1],1)
        D += (grid['z'][0,0,:] * self._norm_vec[2]).reshape(1,1,ss[2])
        diag_dist = na.sqrt(grid.dx**2.0
                          + grid.dy**2.0
                          + grid.dz**2.0)
        cm = na.where(na.abs(D) <= 0.5*diag_dist)
        self._cut_masks[grid.id] = cm
        return cm

    def _generate_coords(self):
        points = []
        for grid in self._grids:
            points.append(self._generate_grid_coords(grid))
        t = na.concatenate(points)
        pos = (t[:,0:3] - self.center)
        self['px'] = na.dot(pos, self._x_vec)
        self['py'] = na.dot(pos, self._y_vec)
        self['pz'] = na.dot(pos, self._norm_vec)
        self['pdx'] = t[:,3] * 0.5
        self['pdy'] = t[:,3] * 0.5
        self['pdz'] = t[:,3] * 0.5

    def _generate_grid_coords(self, grid):
        pointI = self._get_point_indices(grid)
        coords = [grid[ax][pointI].ravel() for ax in 'xyz']
        coords.append(na.ones(coords[0].shape, 'float64') * grid['dx'])
        return na.array(coords).swapaxes(0,1)

    def _get_data_from_grid(self, grid, field):
        if not fieldInfo[field].variable_length:
            pointI = self._get_point_indices(grid)
            if grid[field].size == 1: # dx, dy, dz, cellvolume
                t = grid[field] * na.ones(grid.ActiveDimensions)
                return t[pointI].ravel()
            return grid[field][pointI].ravel()
        else:
            return grid[field]

    def interpolate_discretize(self, *args, **kwargs):
        pass

    def _get_point_indices(self, grid, use_child_mask=True):
        k = na.zeros(grid.ActiveDimensions, dtype='bool')
        k[self._get_cut_mask(grid)] = True
        if use_child_mask:
            pointI = na.where(k & grid.child_mask)
        else:
            pointI = na.where(k)
        return pointI

class EnzoProjBase(Enzo2DData):
    def __init__(self, axis, field, weight_field = None,
                 max_level = None, center = None, pf = None,
                 source=None, type=0):
        """
        EnzoProj is a line integral of a field along an axis.  The field
        can be weighted, in which case some degree of averaging takes place.

        @param axis: axis to project along
        @type axis: integer
        @param field: the field to project (NOT multiple)
        @type field: string
        @keyword weight_field: the field to weight by
        @type weight_field: string
        @keyword max_level: the maximum level to project through
        @keyword type: The type of projection: 0 for sum, 1 for MIP
        @keyword source: The data source, particularly for parallel projections.
        @type source: L{EnzoData<EnzoData>}
        """
        Enzo2DData.__init__(self, axis, field, pf)
        if not source:
            source = EnzoGridCollection(center, self.hierarchy.grids)
        self.source = source
        if max_level == None:
            max_level = self.hierarchy.maxLevel
        self._max_level = max_level
        self._weight = weight_field
        self.center = center
        if type == 1:
            self.type="MIP"
            self.func = na.max
        else:
            self.type="SUM"
            self.func = na.sum
        if not self._deserialize():
            self.__calculate_memory()
            self._refresh_data()

    @time_execution
    def __calculate_memory(self):
        level_mem = {}
        h = self.hierarchy
        s = self.source
        i = 0
        mylog.info("Calculating memory usage")
        for level in range(self._max_level+1):
            level_mem[level] = 0
            mylog.debug("Examining level %s", level)
            grids = s.levelIndices[level]
            num_grids = len(grids)
            RE = s.gridRightEdge[grids]
            LE = s.gridLeftEdge[grids]
            for grid in s._grids[grids]:
                if (i%1e3) == 0:
                    mylog.debug("Reading and masking %s / %s", i, len(s._grids))
                grid._generate_overlap_masks(self.axis, LE, RE)
                grid._overlap_grids[self.axis] = \
                  s._grids[grids][na.where(grid.overlap_masks[self.axis])]
                level_mem[level] += \
                          grid.ActiveDimensions.prod() / \
                          grid.ActiveDimensions[self.axis]
                i += 1
        for level in range(self._max_level+1):
            gI = s.levelIndices[level]
            mylog.debug("%s cells and %s grids for level %s", \
                level_mem[level], len(gI), level)
        mylog.debug("We need %s cells total",
                    na.add.reduce(level_mem.values()))
        self.__memory_per_level = level_mem

    def _serialize(self):
        mylog.info("Serializing data...")
        node_name = "%s_%s_%s" % (self.fields[0], self._weight, self.axis)
        mylog.info("nodeName: %s", node_name)
        projArray = na.array([self['px'], self['py'],
                              self['pdx'], self['pdy'], self[self.fields[0]]])
        self.hierarchy.save_data(projArray, "/Projections", node_name)
        mylog.info("Done serializing...")

    def _deserialize(self):
        node_name = "%s_%s_%s" % (self.fields[0], self._weight, self.axis)
        mylog.debug("Trying to get node %s", node_name)
        array=self.hierarchy.get_data("/Projections", node_name)
        if array == None:
            mylog.debug("Didn't find it!")
            return
        self['px'] = array[0,:]
        self['py'] = array[1,:]
        self['pdx'] = array[2,:]
        self['pdy']= array[3,:]
        self[self.fields[0]] = array[4,:]
        return True

    def __project_level(self, level, field):
        grids_to_project = self.source.select_grids(level)
        zero_out = (level != self._max_level)
        pbar = get_pbar('Projecting  level % 2i / % 2i ' \
                          % (level, self._max_level), len(grids_to_project))
        for pi, grid in enumerate(grids_to_project):
            grid.retVal = self._project_grid(grid, field, zero_out)
            pbar.update(pi)
        pbar.finish()
        self.__combine_grids_on_level(level) # In-place
        if level > 0 and level <= self._max_level:
            self.__refine_to_level(level) # In-place
        all_data = [ [grid.retVal[j] for grid in grids_to_project] for j in range(5)]
        for grid in grids_to_project:
            cI = na.where(grid.retVal[3]==0) # Where childmask = 0
            grid.coarseData = [grid.retVal[j][cI] for j in range(5)]
        levelData = [na.concatenate(all_data[i]) for i in range(5)]
        dblI = na.where((levelData[0]>-1) & (levelData[3]==1))
        if self._weight != None:
            weightedData = levelData[2][dblI] / levelData[4][dblI]
        else:
            weightedData = levelData[2][dblI]
        mylog.debug("Level %s done: %s final of %s", \
                   level, len(dblI[0]), \
                   levelData[0].shape[0])
        dx = grids_to_project[0].dx * na.ones(len(dblI[0]), dtype='float64')
        return [levelData[0][dblI], levelData[1][dblI], weightedData, dx]

    def __setup_weave_dict_combine(self, grid1, grid2):
        # Recall: x, y, val, mask, weight
        my_dict = {}
        my_dict['g1data_x'] = grid1.retVal[0]
        my_dict['g1data_y'] = grid1.retVal[1]
        my_dict['g1data_vals'] = grid1.retVal[2]
        my_dict['g1data_mask'] = grid1.retVal[3]
        my_dict['g1data_wgt'] = grid1.retVal[4]
        my_dict['g1points'] = grid1.retVal[0].size

        my_dict['g2data_x'] = grid2.retVal[0]
        my_dict['g2data_y'] = grid2.retVal[1]
        my_dict['g2data_vals'] = grid2.retVal[2]
        my_dict['g2data_mask'] = grid2.retVal[3]
        my_dict['g2data_wgt'] = grid2.retVal[4]
        my_dict['g2points'] = grid2.retVal[0].size
        return my_dict

    def __combine_grids_on_level(self, level):
        grids = self.source.select_grids(level)
        pbar = get_pbar('Combining   level % 2i / % 2i ' \
                          % (level, self._max_level), len(grids))
        # We have an N^2 check, so we try to be as quick as possible
        # and to skip as many as possible
        for pi, grid1 in enumerate(grids):
            pbar.update(pi)
            if grid1.retVal[0].shape[0] == 0: continue
            for grid2 in grid1._overlap_grids[self.axis].tolist():
                if grid2.retVal[0].shape[0] == 0 \
                  or grid1.id == grid2.id:
                    continue
                my_dict = self.__setup_weave_dict_combine(grid1, grid2)
                weave.inline(ProjectionCombineSameLevel,
                             my_dict.keys(), local_dict=my_dict,
                             compiler='gcc',
                             type_converters=converters.blitz,
                             auto_downcast=0, verbose=2)
                goodI = na.where(grid2.retVal[0] > -1)
                grid2.retVal = [grid2.retVal[i][goodI] for i in range(5)]
        pbar.finish()

    def __setup_weave_dict_refine(self, grid1, grid2):
        # Recall: x, y, val, mask, weight
        my_dict = {}
        my_dict['finedata_x'] = grid1.retVal[0]
        my_dict['finedata_y'] = grid1.retVal[1]
        my_dict['finedata_vals'] = grid1.retVal[2]
        my_dict['finedata_wgt'] = grid1.retVal[4]
        my_dict['fpoints'] = grid1.retVal[0].size
        my_dict['coarsedata_x'] = grid2.coarseData[0]
        my_dict['coarsedata_y'] = grid2.coarseData[1]
        my_dict['coarsedata_vals'] = grid2.coarseData[2]
        my_dict['coarsedata_wgt'] = grid2.coarseData[4]
        my_dict['cpoints'] = grid2.coarseData[0].size
        my_dict['flagged'] = na.zeros(grid2.coarseData[0].size)
        my_dict['rf'] = int(grid2.dx/grid1.dx)
        return my_dict

    def __refine_to_level(self, level):
        grids = self.source.select_grids(level)
        pbar = get_pbar('Refining to level % 2i / % 2i ' \
                          % (level, self._max_level), len(grids))
        for pi, grid1 in enumerate(grids):
            pbar.update(pi)
            o_grids = grid1.Parent._overlap_grids[self.axis].tolist()
            for grid2 in o_grids:
                if grid2.coarseData[0].shape[0] == 0: continue # Already refined
                my_dict = self.__setup_weave_dict_refine(grid1, grid2)
                weave.inline(ProjectionRefineCoarseData,
                             my_dict.keys(), local_dict=my_dict,
                             compiler='gcc',
                             type_converters=converters.blitz,
                             auto_downcast=0, verbose=2)
                goodI = na.where(my_dict['flagged'] == 0)
                grid2.coarseData = [grid2.coarseData[i][goodI] for i in range(5)]
        for grid1 in self.source.select_grids(level-1):
            if grid1.coarseData[0].shape[0] != 0:
                mylog.error("Something messed up, and %s still has %s points of data",
                            grid1, grid1.coarseData[0].size)
                print grid1.coarseData[0]
                raise ValueError(grid1)
        pbar.finish()

    @time_execution
    def get_data(self, field = None):
        if not field: field = self.fields[0]
        all_data = []
        s = self.source
        for level in range(0, self._max_level+1):
            all_data.append(self.__project_level(level, field))
        all_data = na.concatenate(all_data, axis=1)
        # We now convert to half-widths and center-points
        self['pdx'] = all_data[3,:]
        self['px'] = (all_data[0,:]+0.5) * self['pdx']
        self['py'] = (all_data[1,:]+0.5) * self['pdx']
        self['pdx'] *= 0.5
        self['pdy'] = self['pdx'].copy()
        self[field] = all_data[2,:]
        # Now, we should clean up after ourselves...
        [grid.clear_data() for grid in s._grids]

    def _project_grid(self, grid, field, zero_out):
        if self._weight == None:
            masked_data = self._get_data_from_grid(grid, field)
            weight_data = na.ones(masked_data.shape)
        else:
            weight_data = self._get_data_from_grid(grid, self._weight)
            masked_data = self._get_data_from_grid(grid, field) * weight_data
        if zero_out:   # Not sure how to do this here
            masked_data[grid.child_indices] = 0
            weight_data[grid.child_indices] = 0
        #@todo: Fix the func to set up a path length too
        dl = just_one(grid['d%s' % axis_names[self.axis]])
        dx = just_one(grid['d%s' % axis_names[x_dict[self.axis]]])
        dy = just_one(grid['d%s' % axis_names[y_dict[self.axis]]])
        full_proj = self.func(masked_data,axis=self.axis)*dl
        weight_proj = self.func(weight_data,axis=self.axis)*dl
        used_data = self._get_points_in_region(grid)
        used_points = na.where(na.logical_or.reduce(used_data, self.axis))
        if zero_out:
            subgrid_mask = na.logical_and.reduce(grid.child_mask, self.axis).astype('int64')
        else:
            subgrid_mask = na.ones(full_proj.shape, dtype='int64')
        xind, yind = [arr[used_points].ravel() for arr in na.indices(full_proj.shape)]
        start_index = grid.get_global_startindex()
        xpoints = xind + (start_index[x_dict[self.axis]])
        ypoints = yind + (start_index[y_dict[self.axis]])
        return [xpoints, ypoints,
                full_proj[used_points].ravel(),
                subgrid_mask[used_points].ravel(),
                weight_proj[used_points].ravel()]

    def _get_points_in_region(self, grid):
        pointI = self.source._get_point_indices(grid, use_child_mask=False)
        point_mask = na.zeros(grid.ActiveDimensions)
        point_mask[pointI] = 1.0
        return point_mask

    @restore_grid_state
    def _get_data_from_grid(self, grid, field):
        bad_points = self._get_points_in_region(grid)
        d = grid[field] * bad_points
        return d

class Enzo3DData(EnzoData):
    """
    Class describing a cluster of data points, not necessarily sharing any
    particular attribute.
    """
    _spatial = False
    _num_ghost_zones = 0
    def __init__(self, center, fields, pf = None):
        """
        Returns an instance of Enzo3DData, or prepares one.  Usually only
        used as a base class.

        @param hierarchy: hierarchy we are associated with
        @type hierarchy: L{EnzoHierarchy<EnzoHierarchy>}
        @param center: center of the region
        @type center: array of floats
        @param fields: fields to read/generate
        @type fields: list of strings
        """
        EnzoData.__init__(self, pf, fields)
        self.center = center
        self.set_field_parameter("center",center)
        self.coords = None
        self.dx = None
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
        dx = na.ones(pointI[0].shape[0], 'float64') * grid.dx
        tr = na.array([grid['x'][pointI].ravel(), \
                grid['y'][pointI].ravel(), \
                grid['z'][pointI].ravel(), \
                grid["RadiusCode"][pointI].ravel(),
                dx, grid["GridIndices"][pointI].ravel()], 'float64').swapaxes(0,1)
        return tr

    def get_data(self, fields=None, in_grids=False):
        if self._grids == None:
            self._get_list_of_grids()
        points = []
        #if not self.has_key('dx'):
            #self._generate_coords()
        if not fields:
            fields_to_get = self.fields
        else:
            fields_to_get = ensure_list(fields)
        mylog.debug("Going to obtain %s (%s)", fields_to_get, self.fields)
        for field in fields_to_get:
            if self.data.has_key(field):
                continue
            mylog.info("Getting field %s from %s", field, len(self._grids))
            if field not in self.hierarchy.field_list and not in_grids:
                if self._generate_field(field):
                    continue # True means we already assigned it
            self[field] = na.concatenate(
                [self._get_data_from_grid(grid, field)
                 for grid in self._grids])

    @restore_grid_state
    def _get_data_from_grid(self, grid, field):
        if fieldInfo.has_key(field) and fieldInfo[field].variable_length:
            tr = grid[field]
            return tr
        else:
            pointI = self._get_point_indices(grid)
            if grid[field].size == 1: # dx, dy, dz, cellvolume
                t = grid[field] * na.ones(grid.ActiveDimensions)
                return t[pointI].ravel()
            return grid[field][pointI].ravel()

    def _flush_data_to_grids(self, field, default_val, dtype='float32'):
        """
        A dangerous, thusly underscored, thing to do to a data object,
        we can flush back any changes in a given field that have been made
        with a default value for the rest of the grid.
        """
        i = 0
        for grid in self._grids:
            pointI = self._get_point_indices(grid)
            new_field = na.ones(grid.ActiveDimensions, dtype=dtype) * default_val
            np = pointI[0].ravel().size
            new_field[pointI] = self[field][i:i+np]
            if grid.data.has_key(field): del grid.data[field]
            grid[field] = new_field
            i += np

    def _generate_field(self, field):
        if fieldInfo.has_key(field):
            # First we check the validator
            try:
                fieldInfo[field].check_available(self)
            except NeedsGridType, ngt_exception:
                # We leave this to be implementation-specific
                self._generate_field_in_grids(field, ngt_exception.ghost_zones)
                return False
            else:
                self[field] = fieldInfo[field](self)
                return True
        else: # Can't find the field, try as it might
            raise exceptions.KeyError(field)

    def _generate_field_in_grids(self, field, num_ghost_zones=0):
        for grid in self._grids:
            self.__touch_grid_field(grid, field)

    @restore_grid_state
    def __touch_grid_field(self, grid, field):
        grid[field]

    @restore_grid_state
    def _get_point_indices(self, grid, field=None):
        k = na.zeros(grid.ActiveDimensions, dtype='bool')
        k[self._get_cut_mask(grid)] = True
        pointI = na.where(k & grid.child_mask)
        return pointI

    def extract_region(self, indices):
        """
        Return an ExtractedRegion where the points contained in it are defined
        as the points in *this* data object with the given indices

        @param indices: The indices of the points to keep
        @type indices: The return value of a numpy.where call
        """
        return ExtractedRegionBase(self, indices)

    def select_grids(self, level):
        """
        Select all grids on a given level.
        """
        grids = [g for g in self._grids if g.Level == level]
        return grids

    def __get_levelIndices(self):
        if self.__levelIndices: return self.__levelIndices
        # Otherwise, generate
        # We only have to do this once, so it's not terribly expensive:
        ll = {}
        for level in range(MAXLEVEL):
            t = [i for i in range(len(self._grids)) if self._grids[i].Level == level]
            ll[level] = na.array(t)
        self.__levelIndices = ll
        return self.__levelIndices

    def __set_levelIndices(self, val):
        self.__levelIndices = val

    def __del_levelIndices(self):
        del self.__levelIndices
        self.__levelIndices = None

    __levelIndices = None
    levelIndices = property(__get_levelIndices, __set_levelIndices,
                            __del_levelIndices)

    def __get_gridLeftEdge(self):
        if self.__gridLeftEdge == None:
            self.__gridLeftEdge = na.array([g.LeftEdge for g in self._grids])
        return self.__gridLeftEdge

    def __del_gridLeftEdge(self):
        del self.__gridLeftEdge
        self.__gridLeftEdge = None

    def __set_gridLeftEdge(self, val):
        self.__gridLeftEdge = val

    __gridLeftEdge = None
    gridLeftEdge = property(__get_gridLeftEdge, __set_gridLeftEdge,
                              __del_gridLeftEdge)

    def __get_gridRightEdge(self):
        if self.__gridRightEdge == None:
            self.__gridRightEdge = na.array([g.RightEdge for g in self._grids])
        return self.__gridRightEdge

    def __del_gridRightEdge(self):
        del self.__gridRightEdge
        self.__gridRightEdge = None

    def __set_gridRightEdge(self, val):
        self.__gridRightEdge = val

    __gridRightEdge = None
    gridRightEdge = property(__get_gridRightEdge, __set_gridRightEdge,
                             __del_gridRightEdge)

    def __get_gridLevels(self):
        if self.__gridLevels == None:
            self.__gridLevels = na.array([g.Level for g in self._grids])
        return self.__gridLevels

    def __del_gridLevels(self):
        del self.__gridLevels
        self.__gridLevels = None

    def __set_gridLevels(self, val):
        self.__gridLevels = val

    __gridLevels = None
    gridLevels = property(__get_gridLevels, __set_gridLevels,
                             __del_gridLevels)

    def extract_connected_sets(self, field, num_levels, min_val, max_val,
                                log_space=True, cumulative=True):
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
        for level in range(num_levels):
            contours[level] = {}
            if cumulative:
                mv = max_val
            else:
                mv = cons[level+1]
            cids = identify_contours(self, field, cons[level], mv)
            for cid, cid_ind in cids.items():
                contours[level][cid] = self.extract_region(cid_ind)
        return cons, contours

    def paint_grids(self, field, value, default_value=None):
        """
        This function paints every cell in our dataset with a given value.
        If default_value is given, the other values for the given in every grid
        are discarded and replaced with default_value.  Otherwise, the field is
        mandated to 'know how to exist' in the grid.

        @note: This only paints the cells *in the dataset*, so cells in grids
        with children are left untouched.
        @param field: The field to paint
        @param value: The value to paint
        @keyword default_value: The value to use for all other cells.  If 'None'
        then no value is used, and the grid is left untouched except in our cells.
        """
        for grid in self._grids:
            if default_value != None:
                grid[field] = na.ones(grid.ActiveDimensions)*value
            grid[field][self._get_point_indices()] = value


class ExtractedRegionBase(Enzo3DData):
    """
    ExtractedRegions are arbitrarily defined containers of data, useful
    for things like selection along a baryon field.
    """
    def __init__(self, base_region, indices):
        """
        @param base_region: The Enzo3DData we select points from
        @type base_region: L{Enzo3DData<Enzo3DData>}
        @param indices: The indices in the base container to take
        @type indices: The result of a numpy.where call
        """
        cen = base_region.get_field_parameter("center")
        Enzo3DData.__init__(self, center=cen,
                            fields=None, pf=base_region.pf)
        self._base_region = base_region
        self._base_indices = indices
        self._grids = None
        self._refresh_data()

    def _get_list_of_grids(self):
        # Okay, so what we're going to want to do is get the pointI from
        # region._get_point_indices(grid) for grid in base_region._grids,
        # and then construct an array of those, which we will select along indices.
        if self._grids != None: return
        grid_vals, xi, yi, zi = [], [], [], []
        for grid in self._base_region._grids:
            xit,yit,zit = self._base_region._get_point_indices(grid)
            grid_vals.append(na.ones(xit.shape) * grid.id-1)
            xi.append(xit)
            yi.append(yit)
            zi.append(zit)
        grid_vals = na.concatenate(grid_vals)
        xi = na.concatenate(xi)
        yi = na.concatenate(yi)
        zi = na.concatenate(zi)
        # We now have an identical set of indices that the base_region would
        # use to cut out the grids.  So what we want to do is take only
        # the points we want from these.
        self._indices = {}
        for grid in self._base_region._grids:
            ind_ind = na.where(grid_vals[self._base_indices] == grid.id-1)
            self._indices[grid.id-1] = ([xi[self._base_indices][ind_ind],
                                         yi[self._base_indices][ind_ind],
                                         zi[self._base_indices][ind_ind]])
        self._grids = self._base_region.pf.h.grids[self._indices.keys()]

    def _get_point_indices(self, grid):
        return self._indices[grid.id-1]

class EnzoRegionBase(Enzo3DData):
    """
    EnzoRegions are rectangular prisms of data.
    """
    def __init__(self, center, left_edge, right_edge, fields = None, pf = None):
        """
        @note: Center does not have to be (rightEdge - leftEdge) / 2.0
        @param center: The center for calculations that require it
        @type center: List or array of floats
        @param left_edge: The left boundary
        @type left_edge: list or array of floats
        @param right_edge: The right boundary
        @type right_edge: list or array of floats
        """
        Enzo3DData.__init__(self, center, fields, pf)
        self.left_edge = left_edge
        self.right_edge = right_edge
        self._refresh_data()

    def _get_list_of_grids(self):
        self._grids, ind = self.pf.hierarchy.get_box_grids(self.left_edge,
                                                           self.right_edge)

    def _get_cut_mask(self, grid):
        if na.all( (grid._corners < self.right_edge)
                 & (grid._corners >= self.left_edge)):
            return na.ones(grid.ActiveDimensions, dtype='bool')
        pointI = na.where(\
               ( (grid['x'] < self.right_edge[0])
               & (grid['x'] >= self.left_edge[0])
               & (grid['y'] < self.right_edge[1])
               & (grid['y'] >= self.left_edge[1])
               & (grid['z'] < self.right_edge[2])
               & (grid['z'] >= self.left_edge[2]) ))
        return pointI

class EnzoGridCollection(Enzo3DData):
    """
    An arbitrary selection of grids, within which we accept all points.
    """
    def __init__(self, center, grid_list, fields = None, connection_pool = True,
                 pf = None):
        """
        @param center: The center of the region, for derived fields
        @type center: List or array of floats
        @param grid_list: The grids we are composed of
        @type grid_list: List or array of Grid objects
        """
        Enzo3DData.__init__(self, center, fields, pf)
        self._grids = na.array(grid_list)
        self.fields = fields
        self.connection_pool = True

    def _get_list_of_grids(self):
        pass

    def _get_cut_mask(self, grid):
        return na.indices(grid.ActiveDimensions)

    def _get_point_indices(self, grid, use_child_mask=True):
        k = na.ones(grid.ActiveDimensions, dtype='bool')
        if use_child_mask:
            k[grid.child_indices] = False
        pointI = na.where(k == True)
        return pointI

class EnzoSphereBase(Enzo3DData):
    """
    A sphere of points
    """
    def __init__(self, center, radius, fields = None, pf = None):
        """
        @param center: center of the region
        @type center: array of floats
        @param radius: radius of the sphere in code units
        @type radius: float
        @keyword fields: fields to read/generate
        @type fields: list of strings
        """
        Enzo3DData.__init__(self, center, fields, pf)
        self._cut_masks = {}
        self.set_field_parameter('radius',radius)
        self.radius = radius
        self._refresh_data()

    def _get_list_of_grids(self, field = None):
        grids,ind = self.hierarchy.find_sphere_grids(self.center, self.radius)
        # Now we sort by level
        grids = grids.tolist()
        grids.sort(key=lambda x: (x.Level, x.LeftEdge[0], x.LeftEdge[1], x.LeftEdge[2]))
        self._grids = na.array(grids)

    @restore_grid_state
    def _get_cut_mask(self, grid, field=None):
        # We have the *property* center, which is not necessarily
        # the same as the field_parameter
        corner_radius = na.sqrt(((grid._corners - self.center)**2.0).sum(axis=1))
        if na.all(corner_radius <= self.radius):
            return na.ones(grid.ActiveDimensions, dtype='bool')
        if na.all(corner_radius > self.radius):
            return na.zeros(grid.ActiveDimensions, dtype='bool')
        if self._cut_masks.has_key(grid.id):
            return self._cut_masks[grid.id]
        pointI = na.where(
                  (grid["RadiusCode"]<=self.radius) &
                  (grid.child_mask==1) )
        self._cut_masks[grid.id] = pointI
        return pointI

class EnzoCoveringGrid(Enzo3DData):
    """
    Covering grids represent fixed-resolution data over a given region.
    In order to achieve this goal -- for instance in order to obtain ghost
    zones -- grids up to and including the indicated level are included.
    No interpolation is done (as that would affect the 'power' on small
    scales) on the input data.
    """
    _spatial = True
    def __init__(self, level, left_edge, right_edge, dims, fields = None,
                 pf = None, num_ghost_zones = 0):
        """
        @param level: The maximum level to consider when creating the grid
        @note: Level does not have to be related to the dx of the object.
        @param left_edge: The left edge of the covered region
        @type left_edge: List or array of floats
        @param right_edge: The right edge of the covered region
        @type right_edge: List or array of floats
        @param dims: The dimensions of the returned grid
        @type dims: List or array of integers
        @note: It is faster to feed all the fields in at the initialization
        """
        Enzo3DData.__init__(self, center=None, fields=fields, pf=pf)
        self.left_edge = na.array(left_edge)
        self.right_edge = na.array(right_edge)
        self.level = level
        self.ActiveDimensions = na.array(dims)
        self.dx, self.dy, self.dz = (self.right_edge-self.left_edge) \
                                  / self.ActiveDimensions
        self.data["dx"] = self.dx
        self.data["dy"] = self.dy
        self.data["dz"] = self.dz
        self._num_ghost_zones = num_ghost_zones
        self._refresh_data()

    def _get_list_of_grids(self):
        grids, ind = self.pf.hierarchy.get_box_grids(self.left_edge, self.right_edge)
        level_ind = na.where(self.pf.hierarchy.gridLevels.ravel()[ind] <= self.level)
        sort_ind = na.argsort(self.pf.h.gridLevels.ravel()[ind][level_ind])
        self._grids = self.pf.hierarchy.grids[ind][level_ind][(sort_ind,)]

    def extract_region(self, indices):
        mylog.error("Sorry, dude, do it yourself, it's already in 3-D.")

    def __setup_weave_dict(self, grid):
        return {
            'nx_g': int(grid.ActiveDimensions[0]),
            'ny_g': int(grid.ActiveDimensions[1]),
            'nz_g': int(grid.ActiveDimensions[2]),
            'leftEdgeGrid': na.array(grid.LeftEdge),
            'rf': int(grid.dx / self.dx),
            'dx_g': float(grid.dx),
            'dy_g': float(grid.dy),
            'dz_g': float(grid.dz),
            'dx_m': float(self.dx),
            'dy_m': float(self.dy),
            'dz_m': float(self.dz),
            'leftEdgeCube': na.array(self.left_edge),
            'cubeRightEdge': na.array(self.right_edge),
            'nx_m': int(self.ActiveDimensions[0]),
            'ny_m': int(self.ActiveDimensions[1]),
            'nz_m': int(self.ActiveDimensions[2]),
            'childMask' : grid.child_mask
        }

    def _refresh_data(self):
        Enzo3DData._refresh_data(self)
        self['dx'] = self.dx * na.ones(self.ActiveDimensions, dtype='float64')
        self['dy'] = self.dy * na.ones(self.ActiveDimensions, dtype='float64')
        self['dz'] = self.dz * na.ones(self.ActiveDimensions, dtype='float64')

    def get_data(self, field=None):
        self._get_list_of_grids()
        # We don't generate coordinates here.
        if field == None:
            fields_to_get = self.fields
        else:
            fields_to_get = ensure_list(field)
        for field in fields_to_get:
            if self.data.has_key(field):
                continue
            mylog.debug("Getting field %s from %s possible grids",
                       field, len(self._grids))
            self[field] = na.zeros(self.ActiveDimensions, dtype='float64') - 999
            for grid in self._grids:
                self._get_data_from_grid(grid, field)
                if not na.any(self[field] == -999): break
            if na.any(self[field] == -999) and self.dx < self.hierarchy.grids[0].dx:
                print "COVERING PROBLEM", na.where(self[field]==-999)[0].size
                raise KeyError

    def flush_data(self, field=None):
        """
        Any modifications made to the data in this object are pushed back
        to the originating grids, except the cells where those grids are both
        below the current level *and* have child cells.
        """
        self._get_list_of_grids()
        # We don't generate coordinates here.
        if field == None:
            fields_to_get = self.fields
        else:
            fields_to_get = ensure_list(field)
        for field in fields_to_get:
            mylog.debug("Flushing field %s to %s possible grids",
                       field, len(self._grids))
            grid_list = self._grids.tolist()
            for grid in grid_list:
                self._flush_data_to_grid(grid, field)

    @restore_grid_state
    def _get_data_from_grid(self, grid, fields):
        for field in ensure_list(fields):
            locals_dict = self.__setup_weave_dict(grid)
            locals_dict['fieldData'] = grid[field]
            locals_dict['cubeData'] = self[field]
            locals_dict['lastLevel'] = int(grid.Level == self.level)
            weave.inline(DataCubeRefineCoarseData,
                         locals_dict.keys(), local_dict=locals_dict,
                         compiler='gcc',
                         type_converters=converters.blitz,
                         auto_downcast=0, verbose=2)

    def _flush_data_to_grid(self, grid, fields):
        for field in ensure_list(fields):
            locals_dict = self.__setup_weave_dict(grid)
            locals_dict['fieldData'] = grid[field].copy()
            locals_dict['cubeData'] = self[field]
            locals_dict['lastLevel'] = int(grid.Level == self.level)
            weave.inline(DataCubeReplaceData,
                         locals_dict.keys(), local_dict=locals_dict,
                         compiler='gcc',
                         type_converters=converters.blitz,
                         auto_downcast=0, verbose=2)
            grid[field] = locals_dict['fieldData'].copy()
