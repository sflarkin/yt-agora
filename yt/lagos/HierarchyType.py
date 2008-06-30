"""
AMR hierarchy container class

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
from yt.funcs import *
import string, re, gc, time
import cPickle
#import yt.enki

_data_style_funcs = \
   { 4: (readDataHDF4, readAllDataHDF4, getFieldsHDF4, readDataSliceHDF4, getExceptionHDF4), \
     5: (readDataHDF5, readAllDataHDF5, getFieldsHDF5, readDataSliceHDF5, getExceptionHDF5), \
     6: (readDataPacked, readAllDataPacked, getFieldsPacked, readDataSlicePacked, getExceptionHDF5), \
     7: (readDataNative, readAllDataNative, None, readDataSliceNative, getExceptionHDF5) \
   }

class AMRHierarchy:
    def __init__(self, pf):
        self.parameter_file = weakref.proxy(pf)
        self._data_file = None
        self._setup_classes()
        self._initialize_grids()

        # For use with derived quantities depending on centers
        # Although really, I think perhaps we should take a closer look
        # at how "center" is used.
        self.center = None
        self.bulkVelocity = None

        self._initialize_level_stats()

        mylog.debug("Initializing data file")
        self._initialize_data_file()
        mylog.debug("Populating hierarchy")
        self._populate_hierarchy()
        mylog.debug("Done populating hierarchy")

    def _initialize_grids(self):
        mylog.debug("Allocating memory for %s grids", self.num_grids)
        self.gridDimensions = na.zeros((self.num_grids,3), 'int32')
        self.gridStartIndices = na.zeros((self.num_grids,3), 'int32')
        self.gridEndIndices = na.zeros((self.num_grids,3), 'int32')
        self.gridLeftEdge = na.zeros((self.num_grids,3), self.float_type)
        self.gridRightEdge = na.zeros((self.num_grids,3), self.float_type)
        self.gridLevels = na.zeros((self.num_grids,1), 'int32')
        self.gridDxs = na.zeros((self.num_grids,1), self.float_type)
        self.gridDys = na.zeros((self.num_grids,1), self.float_type)
        self.gridDzs = na.zeros((self.num_grids,1), self.float_type)
        self.gridTimes = na.zeros((self.num_grids,1), 'float64')
        self.gridNumberOfParticles = na.zeros((self.num_grids,1))
        mylog.debug("Done allocating")
        mylog.debug("Creating grid objects")
        self.grids = na.array([self.grid(i+1) for i in xrange(self.num_grids)])
        self.gridReverseTree = [-1] * self.num_grids
        self.gridTree = [ [] for i in range(self.num_grids)]
        mylog.debug("Done creating grid objects")

    def _initialize_level_stats(self):
        # Now some statistics:
        #   0 = number of grids
        #   1 = number of cells
        #   2 = blank
        desc = {'names': ['numgrids','numcells','level'],
                'formats':['Int32']*3}
        self.level_stats = blankRecordArray(desc, MAXLEVEL)
        self.level_stats['level'] = [i for i in range(MAXLEVEL)]
        self.level_stats['numgrids'] = [0 for i in range(MAXLEVEL)]
        self.level_stats['numcells'] = [0 for i in range(MAXLEVEL)]


    def _initialize_data_file(self):
        if not ytcfg.getboolean('lagos','serialize'): return
        fn = os.path.join(self.directory,"%s.yt" % self["CurrentTimeIdentifier"])
        if ytcfg.getboolean('lagos','onlydeserialize'):
            mode = 'r'
        else:
            mode = 'a'
        try:
            self._data_file = tables.openFile(fn, mode)
            my_name = self.get_data("/","MyName")
            if my_name is None:
                self.save_data(str(self.parameter_file), "/", "MyName")
            else:
                if str(my_name.read())!=str(self.parameter_file):
                    self._data_file.close()
                    self._data_file = None
        except:
            self._data_file = None
            pass

    def _setup_grid_corners(self):
        self.gridCorners = na.array([ # Unroll!
            [self.gridLeftEdge[:,0], self.gridLeftEdge[:,1], self.gridLeftEdge[:,2]],
            [self.gridRightEdge[:,0], self.gridLeftEdge[:,1], self.gridLeftEdge[:,2]],
            [self.gridRightEdge[:,0], self.gridRightEdge[:,1], self.gridLeftEdge[:,2]],
            [self.gridRightEdge[:,0], self.gridRightEdge[:,1], self.gridRightEdge[:,2]],
            [self.gridLeftEdge[:,0], self.gridRightEdge[:,1], self.gridRightEdge[:,2]],
            [self.gridLeftEdge[:,0], self.gridLeftEdge[:,1], self.gridRightEdge[:,2]],
            [self.gridRightEdge[:,0], self.gridLeftEdge[:,1], self.gridRightEdge[:,2]],
            [self.gridLeftEdge[:,0], self.gridRightEdge[:,1], self.gridLeftEdge[:,2]],
            ], dtype='float64')

    def save_data(self, array, node, name, set_attr=None, force=False):
        """
        Arbitrary numpy data will be saved to the region in the datafile
        described by *node* and *name*.  If data file does not exist, it throws
        no error and simply does not save.
        """
        if self._data_file is None: return
        if force:
            try:
                node_loc = self._data_file.getNode(node)
                if name in node_loc:
                    self._data_file.removeNode(node, name, recursive=True)
            except tables.exceptions.NoSuchNodeError:
                pass
        arr = self._data_file.createArray(node, name, array, createparents=True)
        if set_attr is not None:
            for i, j in set_attr.items(): arr.setAttr(i,j)
        self._data_file.flush()

    def get_data(self, node, name):
        """
        Return the dataset with a given *name* located at *node* in the
        datafile.
        """
        if self._data_file == None:
            return None
        try:
            return self._data_file.getNode(node, name)
        except tables.exceptions.NoSuchNodeError:
            return None

    def _close_data_file(self):
        if self._data_file:
            self._data_file.close()
            del self._data_file
            self._data_file = None

    def _setup_classes(self, dd):
        self.proj = classobj("AMRProj",(AMRProjBase,), dd)
        self.slice = classobj("AMRSlice",(AMRSliceBase,), dd)
        self.region = classobj("AMRRegion",(AMRRegionBase,), dd)
        self.covering_grid = classobj("AMRCoveringGrid",(AMRCoveringGrid,), dd)
        self.smoothed_covering_grid = classobj("AMRSmoothedCoveringGrid",(AMRSmoothedCoveringGrid,), dd)
        self.sphere = classobj("AMRSphere",(AMRSphereBase,), dd)
        self.cutting = classobj("AMRCuttingPlane",(AMRCuttingPlaneBase,), dd)
        self.ray = classobj("AMROrthoRay",(AMROrthoRayBase,), dd)
        self.disk = classobj("AMRCylinder",(AMRCylinderBase,), dd)

    def __deserialize_hierarchy(self, harray):
        mylog.debug("Cached entry found.")
        self.gridDimensions[:] = harray[:,0:3]
        self.gridStartIndices[:] = harray[:,3:6]
        self.gridEndIndices[:] = harray[:,6:9]
        self.gridLeftEdge[:] = harray[:,9:12]
        self.gridRightEdge[:] = harray[:,12:15]
        self.gridLevels[:] = harray[:,15:16]
        self.gridTimes[:] = harray[:,16:17]
        self.gridNumberOfParticles[:] = harray[:,17:18]

    def _get_data_reader_dict(self):
        dd = { 'readDataFast' : _data_style_funcs[self.data_style][0],
               'readAllData' : _data_style_funcs[self.data_style][1],
               'getFields' : _data_style_funcs[self.data_style][2],
               'readDataSlice' : _data_style_funcs[self.data_style][3],
               '_read_data' : _data_style_funcs[self.data_style][0],
               '_read_all_data' : _data_style_funcs[self.data_style][1],
               '_read_field_names' : _data_style_funcs[self.data_style][2],
               '_read_data_slice' : _data_style_funcs[self.data_style][3],
               '_read_exception' : _data_style_funcs[self.data_style][4](),
               'pf' : self.parameter_file, # Already weak
               'hierarchy': weakref.proxy(self) }
        return dd

    def select_grids(self, level):
        """
        Returns an array of grids at *level*.
        """
        return self.grids[self._select_level(level)]

    def _select_level(self, level):
        # We return a numarray of the indices of all the grids on a given level
        indices = na.where(self.gridLevels[:,0] == level)[0]
        return indices

    def get_smallest_dx(self):
        """
        Returns (in code units) the smallest cell size in the simulation.
        """
        return self.gridDxs.min()

    def print_stats(self):
        """
        Prints out (stdout) relevant information about the simulation
        """
        for i in xrange(MAXLEVEL):
            if (self.level_stats['numgrids'][i]) == 0:
                break
            print "% 3i\t% 6i\t% 11i" % \
                  (i, self.level_stats['numgrids'][i], self.level_stats['numcells'][i])
            dx = self.gridDxs[self.levelIndices[i][0]]
        print "-" * 28
        print "   \t% 6i\t% 11i" % (self.level_stats['numgrids'].sum(), self.level_stats['numcells'].sum())
        print "\n"
        try:
            print "z = %0.8f" % (self["CosmologyCurrentRedshift"])
        except:
            pass
        t_s = self["InitialTime"] * self["Time"]
        print "t = %0.8e = %0.8e s = %0.8e years" % \
            (self["InitialTime"], \
             t_s, t_s / (365*24*3600.0) )
        print "\nSmallest Cell:"
        u=[]
        for item in self.parameter_file.units.items():
            u.append((item[1],item[0]))
        u.sort()
        for unit in u:
            print "\tWidth: %0.3e %s" % (dx*unit[0], unit[1])

    def find_point(self, coord):
        """
        Returns the (objects, indices) of grids containing an (x,y,z) point
        """
        mask=na.ones(self.num_grids)
        for i in xrange(len(coord)):
            na.choose(na.greater(self.gridLeftEdge[:,i],coord[i]), (mask,0), mask)
            na.choose(na.greater(self.gridRightEdge[:,i],coord[i]), (0,mask), mask)
        ind = na.where(mask == 1)
        return self.grids[ind], ind

    def find_ray_grids(self, coord, axis):
        """
        Returns the (objects, indices) of grids that an (x,y) ray intersects
        along *axis*
        """
        # Let's figure out which grids are on the slice
        mask=na.ones(self.num_grids)
        # So if gRE > coord, we get a mask, if not, we get a zero
        #    if gLE > coord, we get a zero, if not, mask
        # Thus, if the coordinate is between the two edges, we win!
        na.choose(na.greater(self.gridRightEdge[:,x_dict[axis]],coord[0]),(0,mask),mask)
        na.choose(na.greater(self.gridLeftEdge[:,x_dict[axis]],coord[0]),(mask,0),mask)
        na.choose(na.greater(self.gridRightEdge[:,y_dict[axis]],coord[1]),(0,mask),mask)
        na.choose(na.greater(self.gridLeftEdge[:,y_dict[axis]],coord[1]),(mask,0),mask)
        ind = na.where(mask == 1)
        return self.grids[ind], ind

    def find_slice_grids(self, coord, axis):
        """
        Returns the (objects, indices) of grids that a slice intersects along
        *axis*
        """
        # Let's figure out which grids are on the slice
        mask=na.ones(self.num_grids)
        # So if gRE > coord, we get a mask, if not, we get a zero
        #    if gLE > coord, we get a zero, if not, mask
        # Thus, if the coordinate is between the edges, we win!
        #ind = na.where( na.logical_and(self.gridRightEdge[:,axis] > coord, \
                                       #self.gridLeftEdge[:,axis] < coord))
        na.choose(na.greater(self.gridRightEdge[:,axis],coord),(0,mask),mask)
        na.choose(na.greater(self.gridLeftEdge[:,axis],coord),(mask,0),mask)
        ind = na.where(mask == 1)
        return self.grids[ind], ind

    def find_sphere_grids(self, center, radius):
        """
        Returns objects, indices of grids within a sphere
        """
        centers = (self.gridRightEdge + self.gridLeftEdge)/2.0
        long_axis = na.maximum.reduce(self.gridRightEdge - self.gridLeftEdge, 1)
        t = centers - center
        dist = na.sqrt(t[:,0]**2+t[:,1]**2+t[:,2]**2)
        gridI = na.where(na.logical_and((self.gridDxs<=radius)[:,0],(dist < (radius + long_axis))) == 1)
        return self.grids[gridI], gridI

    def get_box_grids(self, leftEdge, rightEdge):
        """
        Gets back all the grids between a left edge and right edge
        """
        gridI = na.where((na.all(self.gridRightEdge > leftEdge, axis=1)
                        & na.all(self.gridLeftEdge < rightEdge, axis=1)) == True)
        return self.grids[gridI], gridI

    @time_execution
    def find_max(self, field, finestLevels = True):
        """
        Returns (value, center) of location of maximum for a given field.
        """
        if finestLevels:
            gI = na.where(self.gridLevels >= self.maxLevel - NUMTOCHECK)
        else:
            gI = na.where(self.gridLevels >= 0) # Slow but pedantic
        maxVal = -1e100
        for grid in self.grids[gI[0]]:
            mylog.debug("Checking %s (level %s)", grid.id, grid.Level)
            val, coord = grid.find_max(field)
            if val > maxVal:
                maxCoord = coord
                maxVal = val
                maxGrid = grid
        mc = na.array(maxCoord)
        pos=maxGrid.get_position(mc)
        pos[0] += 0.5*maxGrid.dx
        pos[1] += 0.5*maxGrid.dx
        pos[2] += 0.5*maxGrid.dx
        mylog.info("Max Value is %0.5e at %0.16f %0.16f %0.16f in grid %s at level %s %s", \
              maxVal, pos[0], pos[1], pos[2], maxGrid, maxGrid.Level, mc)
        self.center = pos
        # This probably won't work for anyone else
        self.bulkVelocity = (maxGrid["x-velocity"][maxCoord], \
                             maxGrid["y-velocity"][maxCoord], \
                             maxGrid["z-velocity"][maxCoord])
        self.parameters["Max%sValue" % (field)] = maxVal
        self.parameters["Max%sPos" % (field)] = "%s" % (pos)
        return maxVal, pos

    findMax = find_max

    @time_execution
    def find_min(self, field):
        """
        Returns (value, center) of location of minimum for a given field
        """
        gI = na.where(self.gridLevels >= 0) # Slow but pedantic
        minVal = 1e100
        for grid in self.grids[gI[0]]:
            mylog.debug("Checking %s (level %s)", grid.id, grid.Level)
            val, coord = grid.find_min(field)
            if val < minVal:
                minCoord = coord
                minVal = val
                minGrid = grid
        mc = na.array(minCoord)
        pos=minGrid.get_position(mc)
        pos[0] += 0.5*minGrid.dx
        pos[1] += 0.5*minGrid.dx
        pos[2] += 0.5*minGrid.dx
        mylog.info("Min Value is %0.5e at %0.16f %0.16f %0.16f in grid %s at level %s", \
              minVal, pos[0], pos[1], pos[2], minGrid, minGrid.Level)
        self.center = pos
        # This probably won't work for anyone else
        self.binkVelocity = (minGrid["x-velocity"][minCoord], \
                             minGrid["y-velocity"][minCoord], \
                             minGrid["z-velocity"][minCoord])
        self.parameters["Min%sValue" % (field)] = minVal
        self.parameters["Min%sPos" % (field)] = "%s" % (pos)
        return minVal, pos

    def _get_parameters(self):
        return self.parameter_file.parameters
    parameters=property(_get_parameters)

    def __getitem__(self, item):
        return self.parameter_file[item]


class EnzoHierarchy(AMRHierarchy):
    eiTopGrid = None
    _strip_path = False
    @time_execution
    def __init__(self, pf, data_style=None):
        """
        This is the grid structure as Enzo sees it, with some added bonuses.
        It's primarily used as a class factory, to generate data objects and
        access grids.

        It should never be created directly -- you should always access it via
        calls to an affiliated :class:`~yt.lagos.EnzoStaticOutput`.

        On instantiation, it processes the hierarchy and generates the grids.
        """
        # Expect filename to be the name of the parameter file, not the
        # hierarchy
        self.data_style = data_style
        self.hierarchy_filename = os.path.abspath(pf.parameter_filename) \
                               + ".hierarchy"
        self.__hierarchy_lines = open(self.hierarchy_filename).readlines()
        if len(self.__hierarchy_lines) == 0:
            raise IOError(-1,"File empty", self.hierarchy_filename)
        self.boundary_filename = os.path.abspath(pf.parameter_filename) \
                               + ".boundary"
        self.directory = os.path.dirname(self.hierarchy_filename)
        # Now we search backwards from the end of the file to find out how many
        # grids we have, which allows us to preallocate memory
        self.__hierarchy_string = open(self.hierarchy_filename).read()
        for line in reversed(self.__hierarchy_lines):
            if line.startswith("Grid ="):
                self.num_grids = int(line.split("=")[-1])
                break
        self.__guess_data_style()
        # For some reason, r8 seems to want Float64
        if pf.has_key("CompilerPrecision") \
            and pf["CompilerPrecision"] == "r4":
            self.float_type = 'float32'
        else:
            self.float_type = 'float64'

        AMRHierarchy.__init__(self, pf)

        del self.__hierarchy_string, self.__hierarchy_lines

    def _setup_classes(self):
        dd = self._get_data_reader_dict()
        self.grid = classobj("EnzoGrid",(EnzoGridBase,), dd)
        AMRHierarchy._setup_classes(self, dd)

    def __guess_data_style(self):
        if self.data_style: return
        for i in xrange(len(self.__hierarchy_lines)-1,0,-1):
            line = self.__hierarchy_lines[i]
            if line.startswith("BaryonFileName") or \
               line.startswith("FileName "):
                testGrid = line.split("=")[-1].strip().rstrip()
                break
        if testGrid[0] != os.path.sep:
            testGrid = os.path.join(self.directory, testGrid)
        if not os.path.exists(testGrid):
            testGrid = os.path.join(self.directory,
                                    os.path.basename(testGrid))
            mylog.debug("Your data uses the annoying hardcoded path.")
            self._strip_path = True
        try:
            a = SD.SD(testGrid)
            self.data_style = 4
            mylog.debug("Detected HDF4")
        except:
            a = tables.openFile(testGrid, 'r')
            for b in a.iterNodes("/"):
                c = "%s" % (b)
                break
            if c.startswith("/Grid"):
                mylog.debug("Detected packed HDF5")
                self.data_style = 6
            else:
                mylog.debug("Detected unpacked HDF5")
                self.data_style = 5
            a.close()

    def __setup_filemap(self, grid):
        if not self.data_style == 6:
            return
        self.cpu_map[grid.filename].append(grid)

    def __del__(self):
        self._close_data_file()
        try:
            del self.eiTopGrid
        except:
            pass
        for gridI in xrange(self.num_grids):
            for g in self.gridTree[gridI]:
                del g
        del self.gridReverseTree
        del self.gridLeftEdge, self.gridRightEdge
        del self.gridLevels, self.gridStartIndices, self.gridEndIndices
        del self.gridTimes
        del self.gridTree

    def __set_all_filenames(self, fns):
        if self._strip_path:
            for fnI, fn in enumerate(fns):
                self.grids[fnI].filename = os.path.sep.join([self.directory,
                                                     os.path.basename(fn)])
        elif fns[0][0] == os.path.sep:
            for fnI, fn in enumerate(fns):
                self.grids[fnI].filename = fn
        else:
            for fnI, fn in enumerate(fns):
                self.grids[fnI].filename = os.path.sep.join([self.directory,
                                                             fn])
        mylog.debug("Done with baryon filenames")
        for g in self.grids:
            self.__setup_filemap(g)
        mylog.debug("Done with filemap")

    def __parse_hierarchy_file(self):
        def __split_convert(vals, func, toAdd, curGrid):
            """
            Quick function to split up a parameter and convert it and toss onto a grid
            """
            j = 0
            for v in vals.split():
                toAdd[curGrid-1,j] = func(v)
                j+=1
        for line_index, line in enumerate(self.__hierarchy_lines):
            # We can do this the slow, 'reliable' way by stripping
            # or we can manually pad all our strings, which speeds it up by a
            # factor of about ten
            #param, vals = map(strip,line.split("="))
            if (line_index % 1e5) == 0:
                mylog.debug("Parsing line % 9i / % 9i",
                            line_index, len(self.__hierarchy_lines))
            if len(line) < 2:
                continue
            param, vals = line.split("=")
            param = param.rstrip() # This slows things down considerably...
                                   # or so I used to think...
            if param == "Grid":
                curGrid = int(vals)
                self.grids[curGrid-1] = self.grid(curGrid)
            elif param == "GridDimension":
                __split_convert(vals, float, self.gridDimensions, curGrid)
            elif param == "GridStartIndex":
                __split_convert(vals, int, self.gridStartIndices, curGrid)
            elif param == "GridEndIndex":
                __split_convert(vals, int, self.gridEndIndices, curGrid)
            elif param == "GridLeftEdge":
                __split_convert(vals, float, self.gridLeftEdge, curGrid)
            elif param == "GridRightEdge":
                __split_convert(vals, float, self.gridRightEdge, curGrid)
            elif param == "Level":
                __split_convert(vals, int, self.gridLevels, curGrid)
            elif param == "Time":
                __split_convert(vals, float, self.gridTimes, curGrid)
            elif param == "NumberOfParticles":
                __split_convert(vals, int, self.gridNumberOfParticles, curGrid)
            elif param == "FileName":
                self.grids[curGrid-1].set_filename(vals[1:-1])
            elif param == "BaryonFileName":
                self.grids[curGrid-1].set_filename(vals[1:-1])
        mylog.info("Caching hierarchy information")
        allArrays = na.zeros((self.num_grids,18),'float64')
        allArrays[:,0:3] = self.gridDimensions[:]
        allArrays[:,3:6] = self.gridStartIndices[:]
        allArrays[:,6:9] = self.gridEndIndices[:]
        allArrays[:,9:12] = self.gridLeftEdge[:]
        allArrays[:,12:15] = self.gridRightEdge[:]
        allArrays[:,15:16] = self.gridLevels[:]
        allArrays[:,16:17] = self.gridTimes[:]
        allArrays[:,17:18] = self.gridNumberOfParticles[:]
        if self.num_grids > 1000:
            self.save_data(allArrays, "/","Hierarchy")
        del allArrays

    def __obtain_filenames(self):
        mylog.debug("Copied to local array.")
        # This needs to go elsewhere:
        # Now get the baryon filenames
        mylog.debug("Getting baryon filenames")
        re_BaryonFileName = constructRegularExpressions("BaryonFileName",('s'))
        fn_results = re.findall(re_BaryonFileName, self.__hierarchy_string)
        if len(fn_results):
            self.__set_all_filenames(fn_results)
            return
        re_BaryonFileName = constructRegularExpressions("FileName",('s'))
        fn_results = re.findall(re_BaryonFileName, self.__hierarchy_string)
        self.__set_all_filenames(fn_results)
        # This is pretty bad, but we do it to save a significant amount of time
        # in larger runs.

    def __setup_grid_tree(self):
        mylog.debug("No cached tree found, creating")
        self.grids[0].Level = 0  # Bootstrap
        self.gridLevels[0] = 0   # Bootstrap
        p = re.compile(r"Pointer: Grid\[(\d*)\]->NextGrid(Next|This)Level = (\d*)$", re.M)
        # Now we assemble the grid tree
        # This is where all the time is spent.
        for m in p.finditer(self.__hierarchy_string):
            secondGrid = int(m.group(3))-1 # zero-index versus one-index
            if secondGrid == -1:
                continue
            firstGrid = int(m.group(1))-1
            if m.group(2) == "Next":
                self.gridTree[firstGrid].append(weakref.proxy(self.grids[secondGrid]))
                self.gridReverseTree[secondGrid] = firstGrid + 1
                self.grids[secondGrid].Level = self.grids[firstGrid].Level + 1
                self.gridLevels[secondGrid] = self.gridLevels[firstGrid] + 1
            elif m.group(2) == "This":
                parent = self.gridReverseTree[firstGrid]
                if parent and parent > -1:
                    self.gridTree[parent-1].append(weakref.proxy(self.grids[secondGrid]))
                    self.gridReverseTree[secondGrid] = parent
                self.grids[secondGrid].Level = self.grids[firstGrid].Level
                self.gridLevels[secondGrid] = self.gridLevels[firstGrid]
        pTree = [ [ grid.id - 1 for grid in self.gridTree[i] ] for i in range(self.num_grids) ]
        self.gridReverseTree[0] = -1
        self.save_data(cPickle.dumps(pTree), "/", "Tree")
        self.save_data(na.array(self.gridReverseTree), "/", "ReverseTree")
        self.save_data(self.gridLevels, "/", "Levels")

    @time_execution
    def _populate_hierarchy(self):
        """
        Instantiates all of the grid objects, with their appropriate
        parameters.  This is the work-horse.
        """
        if self.data_style == 6:
            self.cpu_map = defaultdict(lambda: [][:])
            self.file_access = {}
        harray = self.get_data("/", "Hierarchy")
        if self.num_grids <= 1000:
            mylog.info("Skipping serialization!")
        if harray and self.num_grids > 1000:
            self.__deserialize_hierarchy(harray)
        else:
            self.__parse_hierarchy_file()
        self.__obtain_filenames()
        treeArray = self.get_data("/", "Tree")
        if treeArray == None:
            self.__setup_grid_tree()
        else:
            mylog.debug("Grabbing serialized tree data")
            pTree = cPickle.loads(treeArray.read())
            self.gridReverseTree = list(self.get_data("/","ReverseTree"))
            self.gridTree = [ [ weakref.proxy(self.grids[i]) for i in pTree[j] ]
                for j in range(self.num_grids) ]
            self.gridLevels = self.get_data("/","Levels")[:]
            mylog.debug("Grabbed")
        for i,v in enumerate(self.gridReverseTree):
            # For multiple grids on the root level
            if v == -1: self.gridReverseTree[i] = None
        mylog.debug("Tree created")
        self.maxLevel = self.gridLevels.max()
        self.max_level = self.maxLevel
        # Now we do things that we need all the grids to do
        #self.fieldList = self.grids[0].getFields()
        # The rest of this can probably be done with list comprehensions, but
        # I think this way is clearer.
        mylog.debug("Preparing grids")
        for i, grid in enumerate(self.grids):
            if (i%1e4) == 0: mylog.debug("Prepared % 7i / % 7i grids", i, self.num_grids)
            grid._prepare_grid()
        self.__setup_grid_dxs()
        mylog.debug("Prepared")
        self.__setup_field_lists()
        self.levelIndices = {}
        self.levelNum = {}
        ad = self.gridEndIndices - self.gridStartIndices + 1
        for level in xrange(self.maxLevel+1):
            self.level_stats[level]['numgrids'] = na.where(self.gridLevels==level)[0].size
            li = na.where(self.gridLevels[:,0] == level)
            self.level_stats[level]['numcells'] = ad[li].prod(axis=1).sum()
            self.levelIndices[level] = self._select_level(level)
            self.levelNum[level] = len(self.levelIndices[level])
        mylog.debug("Hierarchy fully populated.")

    def __setup_grid_dxs(self):
        mylog.debug("Setting up corners and dxs")
        self._setup_grid_corners()
        dx = (self.gridRightEdge[:,0] - self.gridLeftEdge[:,0]) / \
             (self.gridEndIndices[:,0]-self.gridStartIndices[:,0]+1)
        dy = (self.gridRightEdge[:,1] - self.gridLeftEdge[:,1]) / \
             (self.gridEndIndices[:,1]-self.gridStartIndices[:,1]+1)
        dz = (self.gridRightEdge[:,2] - self.gridLeftEdge[:,2]) / \
             (self.gridEndIndices[:,2]-self.gridStartIndices[:,2]+1)
        self.gridDxs[:,0] = dx[:]
        self.gridDys[:,0] = dy[:]
        self.gridDzs[:,0] = dz[:]
        mylog.debug("Flushing to grids")
        for grid in self.grids:
            grid._setup_dx()
        mylog.debug("Done flushing to grids")
        if ytcfg.getboolean("lagos","ReconstructHierarchy") == True:
            mylog.debug("Reconstructing hierarchy")
            for level in range(self.maxLevel+1):
                grids_to_recon = self.select_grids(level)
                pbar = None
                if len(self.grids) > 3e5:
                    pbar = get_pbar('Reconsructing  level % 2i / % 2i ' \
                                      % (level, self.maxLevel),
                                      len(grids_to_recon))
                for i,grid in enumerate(grids_to_recon):
                    if pbar: pbar.update(i)
                    if grid.Parent: grid._guess_properties_from_parent()
                if pbar: pbar.finish()

    def __setup_field_lists(self):
        field_list = self.get_data("/", "DataFields")
        if field_list == None:
            mylog.info("Gathering a field list (this may take a moment.)")
            field_list = sets.Set()
            if self.num_grids > 40:
                starter = na.random.randint(0, 20)
                random_sample = na.mgrid[starter:len(self.grids)-1:20j].astype("int32")
                mylog.debug("Checking grids: %s", random_sample.tolist())
            else:
                random_sample = na.mgrid[0:max(len(self.grids)-1,1)].astype("int32")
            for grid in self.grids[(random_sample,)]:
                gf = grid.getFields()
                mylog.debug("Grid %s has: %s", grid.id, gf)
                field_list = field_list.union(sets.Set(gf))
        self.field_list = list(field_list)
        for field in self.field_list:
            if field in fieldInfo: continue
            mylog.info("Adding %s to list of fields", field)
            add_field(field, lambda a, b: None)
        self.derived_field_list = []
        for field in fieldInfo:
            try:
                fd = fieldInfo[field].get_dependencies(pf = self.parameter_file)
            except:
                continue
            available = na.all([f in self.field_list for f in fd.requested])
            if available: self.derived_field_list.append(field)
        for field in self.field_list:
            if field not in self.derived_field_list:
                self.derived_field_list.append(field)


    @time_execution
    def export_particles_pb(self, filename, filter = 1, indexboundary = 0, fields = None, scale=1.0):
        """
        Exports all the star particles, or a subset, to pb-format *filename*
        for viewing in partiview.  Filters based on particle_type=*filter*,
        particle_index>=*indexboundary*, and exports *fields*, if supplied.
        Otherwise, index, position(x,y,z).  Optionally *scale* by a given
        factor before outputting.
        """
        import struct
        pbf_magic = 0xffffff98
        header_fmt = 'Iii'
        fmt = 'ifff'
        f = open(filename,"w")
        if fields:
            fmt += len(fields)*'f'
            padded_fields = string.join(fields,"\0") + "\0"
            header_fmt += "%ss" % len(padded_fields)
            args = [pbf_magic, struct.calcsize(header_fmt), len(fields), padded_fields]
            fields = ["particle_index","particle_position_x","particle_position_y","particle_position_z"] \
                   + fields
            format = 'Int32,Float32,Float32,Float32' + ',Float32'*(len(fields)-4)
        else:
            args = [pbf_magic, struct.calcsize(header_fmt), 0]
            fields = ["particle_index","particle_position_x","particle_position_y","particle_position_z"]
            format = 'Int32,Float32,Float32,Float32'
        f.write(struct.pack(header_fmt, *args))
        tot = 0
        sc = na.array([1.0] + [scale] * 3 + [1.0]*(len(fields)-4))
        gI = na.where(self.gridNumberOfParticles.ravel() > 0)
        for g in self.grids[gI]:
            pI = na.where(na.logical_and((g["particle_type"] == filter),(g["particle_index"] >= indexboundary)) == 1)
            tot += pI[0].shape[0]
            toRec = []
            for field, scale in zip(fields, sc):
                toRec.append(scale*g[field][pI])
            particle_info = rec.array(toRec,formats=format)
            particle_info.tofile(f)
        f.close()
        mylog.info("Wrote %s particles to %s", tot, filename)

    @time_execution
    def export_boxes_pv(self, filename):
        """
        Exports the grid structure in partiview text format.
        """
        f=open(filename,"w")
        for l in xrange(self.maxLevel):
            f.write("add object g%s = l%s\n" % (l,l))
            ind = self._select_level(l)
            for i in ind:
                f.write("add box -n %s -l %s %s,%s %s,%s %s,%s\n" % \
                    (i+1, self.gridLevels.ravel()[i],
                     self.gridLeftEdge[i,0], self.gridRightEdge[i,0],
                     self.gridLeftEdge[i,1], self.gridRightEdge[i,1],
                     self.gridLeftEdge[i,2], self.gridRightEdge[i,2]))


scanf_regex = {}
scanf_regex['e'] = r"[-+]?\d+\.?\d*?|\.\d+[eE][-+]?\d+?"
scanf_regex['g'] = scanf_regex['e']
scanf_regex['f'] = scanf_regex['e']
scanf_regex['F'] = scanf_regex['e']
#scanf_regex['g'] = r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?"
#scanf_regex['f'] = r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?"
#scanf_regex['F'] = r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?"
scanf_regex['i'] = r"[-+]?(0[xX][\dA-Fa-f]+|0[0-7]*|\d+)"
scanf_regex['d'] = r"[-+]?\d+"
scanf_regex['s'] = r"\S+"

def constructRegularExpressions(param, toReadTypes):
    re_e=r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?"
    re_i=r"[-+]?(0[xX][\dA-Fa-f]+|0[0-7]*|\d+)"
    rs = "^%s\s*=\s*" % (param)
    for t in toReadTypes:
        rs += "(%s)\s*" % (scanf_regex[t])
    rs +="$"
    return re.compile(rs,re.M)

class OrionHierarchy(AMRHierarchy):
    def __init__(self,pf,data_style=7):
        self.field_indexes = {}
        self.parameter_file = weakref.proxy(pf)
        header_filename = os.path.join(pf.fullplotdir,'Header')
        self.directory = pf.fullpath
        self.data_style = data_style
        self._setup_classes()
        self.readGlobalHeader(header_filename)
        AMRHierarchy.__init__(self,pf)
        self._setup_field_list()

    def readGlobalHeader(self,filename):
        """
        read the global header file for an Orion plotfile output.
        
        
        """
        counter = 0
        header_file = open(filename,'r')
        self.__global_header_lines = header_file.readlines()

        # parse the file
        self.orion_version = self.__global_header_lines[0].rstrip()
        self.n_fields      = int(self.__global_header_lines[1])

        counter = self.n_fields+2
        for i,line in enumerate(self.__global_header_lines[2:counter]):
            self.field_indexes[line.rstrip()] =i
        self.field_list = []
        for f in self.field_indexes:
            self.field_list.append(orion2ytFieldsDict.get(f,f))

        self.dimension = int(self.__global_header_lines[counter])
        if self.dimension != 3:
            raise RunTimeError("Orion must be in 3D to use yt.")
        counter += 1
        self.Time = float(self.__global_header_lines[counter])
        counter += 1
        self.finest_grid_level = int(self.__global_header_lines[counter])
        self.n_levels = self.finest_grid_level + 1
        counter += 1
        self.domainLeftEdge_unnecessary = na.array(map(float,self.__global_header_lines[counter].split()))
        counter += 1
        self.domainRightEdge_unnecessary = na.array(map(float,self.__global_header_lines[counter].split()))
        counter += 1
        self.refinementFactor_unnecessary = na.array(map(int,self.__global_header_lines[counter].split()))
        counter += 1
        self.globalIndexSpace_unnecessary = self.__global_header_lines[counter]
        #domain_re.search(self.__global_header_lines[counter]).groups()
        counter += 1
        self.timestepsPerLevel_unnecessary = self.__global_header_lines[counter]
        counter += 1
        self.dx = na.zeros((self.n_levels,3))
        for i,line in enumerate(self.__global_header_lines[counter:counter+self.n_levels]):
            self.dx[i] = na.array(map(int,line.split()))
        counter += self.n_levels
        self.geometry = int(self.__global_header_lines[counter])
        if self.geometry != 0:
            raise RunTimeError("yt only supports cartesian coordinates.")
        counter += 1

        # this is just to debug. eventually it should go away.
        linebreak = int(self.__global_header_lines[counter])
        if linebreak != 0:
            raise RunTimeError("INTERNAL ERROR! This should be a zero.")
        counter += 1

        # each level is one group with ngrids on it. each grid has 3 lines of 2 reals
        self.levels = []
        grid_counter = 0
        file_finder_pattern = r"FabOnDisk: (Cell_D_[0-9]{4}) (\d+)\n"
        re_file_finder = re.compile(file_finder_pattern)
        dim_finder_pattern = r"\(\((\d+,\d+,\d+)\) \((\d+,\d+,\d+)\) \(\d+,\d+,\d+\)\)\n"
        re_dim_finder = re.compile(dim_finder_pattern)
        
        for level in range(0,self.n_levels):
            tmp = self.__global_header_lines[counter].split()
            # should this be grid_time or level_time??
            lev,ngrids,grid_time = int(tmp[0]),int(tmp[1]),float(tmp[2])
            counter += 1
            nsteps = int(self.__global_header_lines[counter])
            counter += 1
            self.levels.append(OrionLevel(lev,ngrids))
            # open level header, extract file names and offsets for
            # each grid
            fn = os.path.join(self.parameter_file.fullplotdir,'Level_%i'%level)
            level_header_file = open(os.path.join(fn,'Cell_H'),'r').read()
            grid_file_offset = re_file_finder.findall(level_header_file)
            start_stop_index = re_dim_finder.findall(level_header_file)
            for grid in range(0,ngrids):
                gfn = os.path.join(fn,grid_file_offset[grid][0])
                gfo = int(grid_file_offset[grid][1])
                xlo,xhi = map(float,self.__global_header_lines[counter].split())
                counter+=1
                ylo,yhi = map(float,self.__global_header_lines[counter].split())
                counter+=1
                zlo,zhi = map(float,self.__global_header_lines[counter].split())
                counter+=1
                lo = na.array([xlo,ylo,zlo])
                hi = na.array([xhi,yhi,zhi])
                dims,start,stop = self.__calculate_grid_dimensions(start_stop_index[grid])
                self.levels[-1].grids.append(self.grid(lo,hi,grid_counter,level,gfn, gfo, dims,start,stop))
                grid_counter += 1 # this is global, and shouldn't be reset
                                  # for each level
            self.levels[-1]._fileprefix = self.__global_header_lines[counter]
            counter+=1
            self.num_grids = grid_counter
            self.float_type = 'float64'

        self.maxLevel = self.n_levels - 1 
        self.max_level = self.n_levels - 1
        header_file.close()

    def __calculate_grid_dimensions(self,start_stop):
        start = na.array(map(int,start_stop[0].split(',')))
        stop = na.array(map(int,start_stop[1].split(',')))
        dimension = stop - start + 1
        return dimension,start,stop
        

    def _initialize_grids(self):
        mylog.debug("Allocating memory for %s grids", self.num_grids)
        self.gridDimensions = na.zeros((self.num_grids,3), 'int32')
        self.gridStartIndices = na.zeros((self.num_grids,3), 'int32')
        self.gridEndIndices = na.zeros((self.num_grids,3), 'int32')
        self.gridTimes = na.zeros((self.num_grids,1), 'float64')
        self.gridNumberOfParticles = na.zeros((self.num_grids,1))
        mylog.debug("Done allocating")
        mylog.debug("Creating grid objects")
        self.grids = na.concatenate([level.grids for level in self.levels])
        self.gridLevels = na.concatenate([level.ngrids*[level.level] for level in self.levels])
        self.gridLevels = self.gridLevels.reshape((self.num_grids,1))
        gridDcs = na.concatenate([level.ngrids*[self.dx[level.level]] for level in self.levels],axis=0)
        self.gridDxs = gridDcs[:,0].reshape((self.num_grids,1))
        self.gridDys = gridDcs[:,1].reshape((self.num_grids,1))
        self.gridDzs = gridDcs[:,2].reshape((self.num_grids,1))
        left_edges = []
        right_edges = []
        for level in self.levels:
            left_edges += [g.LeftEdge for g in level.grids]
            right_edges += [g.RightEdge for g in level.grids]
        self.gridLeftEdge = na.array(left_edges)
        self.gridRightEdge = na.array(right_edges)
        self.gridReverseTree = [] * self.num_grids
        self.gridReverseTree = [ [] for i in range(self.num_grids)]
        self.gridTree = [ [] for i in range(self.num_grids)]
        mylog.debug("Done creating grid objects")

    def _populate_hierarchy(self):
        self.__setup_grid_tree()
        self._setup_grid_corners()
        for i, grid in enumerate(self.grids):
            if (i%1e4) == 0: mylog.debug("Prepared % 7i / % 7i grids", i, self.num_grids)
            grid._prepare_grid()
            grid._setup_dx()

    def __setup_grid_tree(self):
        for i, grid in enumerate(self.grids):
            children = self._get_grid_children(grid)
            for child in children:
                self.gridReverseTree[child.id].append(i)
                self.gridTree[i].append(weakref.proxy(child))

    def _setup_classes(self):
        dd = self._get_data_reader_dict()
        dd["field_indexes"] = self.field_indexes
        self.grid = classobj("OrionGrid",(OrionGridBase,), dd)
        AMRHierarchy._setup_classes(self, dd)

    def _get_grid_children(self, grid):
        mask = na.zeros(self.num_grids, dtype='bool')
        grids, grid_ind = self.get_box_grids(grid.LeftEdge, grid.RightEdge)
        mask[grid_ind] = True
        mask = na.logical_and(mask, (self.gridLevels == (grid.Level+1)).flat)
        return self.grids[mask]

    def _setup_field_list(self):
        self.derived_field_list = []
        for field in fieldInfo:
            try:
                fd = fieldInfo[field].get_dependencies(pf = self.parameter_file)
            except:
                continue
            available = na.all([f in self.field_list for f in fd.requested])
            if available: self.derived_field_list.append(field)
        for field in self.field_list:
            if field not in self.derived_field_list:
                self.derived_field_list.append(field)

class OrionLevel:
    def __init__(self,level,ngrids):
        self.level = level
        self.ngrids = ngrids
        self.grids = []
    

