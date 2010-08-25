"""
AMR hierarchy container class

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2007-2009 Matthew Turk.  All Rights Reserved.

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
import string, re, gc, time, cPickle, pdb
from itertools import chain, izip
try:
    import yt.ramses_reader as ramses_reader
except ImportError:
    mylog.warning("Ramses Reader not imported")

def num_deep_inc(f):
    def wrap(self, *args, **kwargs):
        self.num_deep += 1
        rv = f(self, *args, **kwargs)
        self.num_deep -= 1
        return rv
    return wrap

class AMRHierarchy(ObjectFindingMixin, ParallelAnalysisInterface):
    float_type = 'float64'

    def __init__(self, pf, data_style):
        self.parameter_file = weakref.proxy(pf)
        self.pf = self.parameter_file

        self._initialize_state_variables()

        mylog.debug("Initializing data storage.")
        self._initialize_data_storage()

        mylog.debug("Counting grids.")
        self._count_grids()

        # Must be defined in subclass
        mylog.debug("Setting up classes.")
        self._setup_classes()

        mylog.debug("Counting grids.")
        self._initialize_grid_arrays()

        mylog.debug("Parsing hierarchy.")
        self._parse_hierarchy()

        mylog.debug("Constructing grid objects.")
        self._populate_grid_objects()

        mylog.debug("Initializing data grid data IO")
        self._setup_data_io()

        mylog.debug("Detecting fields.")
        self._detect_fields()

        mylog.debug("Adding unknown detected fields")
        self._setup_unknown_fields()

        mylog.debug("Setting up derived fields")
        self._setup_derived_fields()

        mylog.debug("Re-examining hierarchy")
        self._initialize_level_stats()

    def _get_parameters(self):
        return self.parameter_file.parameters
    parameters=property(_get_parameters)

    def select_grids(self, level):
        """
        Returns an array of grids at *level*.
        """
        return self.grids[self.grid_levels.flat == level]

    def get_levels(self):
        for level in range(self.max_level+1):
            yield self.select_grids(level)

    def _initialize_state_variables(self):
        self._parallel_locking = False
        self._data_file = None
        self._data_mode = None
        self._max_locations = {}
        self.num_grids = None

    def _initialize_grid_arrays(self):
        mylog.debug("Allocating arrays for %s grids", self.num_grids)
        self.grid_dimensions = na.ones((self.num_grids,3), 'int32')
        self.grid_left_edge = na.zeros((self.num_grids,3), self.float_type)
        self.grid_right_edge = na.ones((self.num_grids,3), self.float_type)
        self.grid_levels = na.zeros((self.num_grids,1), 'int32')
        self.grid_particle_count = na.zeros((self.num_grids,1), 'int32')

    def _setup_classes(self, dd):
        # Called by subclass
        self.object_types = []
        self.objects = []
        for name, cls in sorted(data_object_registry.items()):
            cname = cls.__name__
            if cname.endswith("Base"): cname = cname[:-4]
            self._add_object_class(name, cname, cls, dd)
        self.object_types.sort()

    # Now all the object related stuff

    def all_data(self, find_max=False):
        pf = self.parameter_file
        if find_max: c = self.find_max("Density")[1]
        else: c = (pf["DomainRightEdge"] + pf["DomainLeftEdge"])/2.0
        return self.region(c, 
            pf["DomainLeftEdge"], pf["DomainRightEdge"])

    def clear_all_data(self):
        """
        This routine clears all the data currently being held onto by the grids
        and the data io handler.
        """
        for g in self.grids: g.clear_data()
        self.io.queue.clear()

    def _get_data_reader_dict(self):
        dd = { 'pf' : self.parameter_file, # Already weak
               'hierarchy': weakref.proxy(self) }
        return dd

    def _initialize_data_storage(self):
        if not ytcfg.getboolean('lagos','serialize'): return
        fn = self.pf.storage_filename
        if fn is None:
            if os.path.isfile(os.path.join(self.directory,
                                "%s.yt" % self.pf["CurrentTimeIdentifier"])):
                fn = os.path.join(self.directory,"%s.yt" % self.pf["CurrentTimeIdentifier"])
            else:
                fn = os.path.join(self.directory,
                        "%s.yt" % self.parameter_file.basename)
        dir_to_check = os.path.dirname(fn)
        # We have four options:
        #    Writeable, does not exist      : create, open as append
        #    Writeable, does exist          : open as append
        #    Not writeable, does not exist  : do not attempt to open
        #    Not writeable, does exist      : open as read-only
        exists = os.path.isfile(fn)
        if not exists:
            writeable = os.access(dir_to_check, os.W_OK)
        else:
            writeable = os.access(fn, os.W_OK)
        writeable = writeable and not ytcfg.getboolean('lagos','onlydeserialize')
        # We now have our conditional stuff
        self._barrier()
        if not writeable and not exists: return
        if writeable:
            self._data_mode = 'a'
            if not exists: self.__create_data_file(fn)
        else:
            self._data_mode = 'r'

        self.__data_filename = fn
        self._data_file = h5py.File(fn, self._data_mode)

    def __create_data_file(self, fn):
        # Note that this used to be parallel_root_only; it no longer is,
        # because we have better logic to decide who owns the file.
        f = h5py.File(fn, 'a')
        f.close()

    def _setup_data_io(self):
        self.io = io_registry[self.data_style]()

    def _save_data(self, array, node, name, set_attr=None, force=False, passthrough = False):
        """
        Arbitrary numpy data will be saved to the region in the datafile
        described by *node* and *name*.  If data file does not exist, it throws
        no error and simply does not save.
        """

        if self._data_mode != 'a': return
        if "ArgsError" in dir(h5py.h5):
            exception = h5py.h5.ArgsError
        else:
            exception = h5py.h5.H5Error
        try:
            node_loc = self._data_file[node]
            if name in node_loc.listnames() and force:
                mylog.info("Overwriting node %s/%s", node, name)
                del self._data_file[node][name]
            elif name in node_loc.listnames() and passthrough:
                return
        except exception:
            pass
        myGroup = self._data_file['/']
        for q in node.split('/'):
            if q: myGroup = myGroup.require_group(q)
        arr = myGroup.create_dataset(name,data=array)
        if set_attr is not None:
            for i, j in set_attr.items(): arr.attrs[i] = j
        self._data_file.flush()

    def _reload_data_file(self, *args, **kwargs):
        if self._data_file is None: return
        self._data_file.close()
        del self._data_file
        self._data_file = h5py.File(self.__data_filename, self._data_mode)

    def _reset_save_data(self,round_robin=False):
        if round_robin:
            self.save_data = self._save_data
        else:
            self.save_data = parallel_splitter(self._save_data, self._reload_data_file)
    
    save_data = parallel_splitter(_save_data, _reload_data_file)

    def save_object(self, obj, name):
        """
        Save an object (*obj*) to the data_file using the Pickle protocol,
        under the name *name* on the node /Objects.
        """
        s = cPickle.dumps(obj, protocol=-1)
        self.save_data(s, "/Objects", name, force = True)

    def load_object(self, name):
        """
        Load and return and object from the data_file using the Pickle protocol,
        under the name *name* on the node /Objects.
        """
        obj = self.get_data("/Objects", name)
        if obj is None:
            return
        obj = cPickle.loads(obj.value)
        if iterable(obj) and len(obj) == 2:
            obj = obj[1] # Just the object, not the pf
        if hasattr(obj, '_fix_pickle'): obj._fix_pickle()
        return obj

    def get_data(self, node, name):
        """
        Return the dataset with a given *name* located at *node* in the
        datafile.
        """
        if self._data_file == None:
            return None
        if node[0] != "/": node = "/%s" % node

        myGroup = self._data_file['/']
        for group in node.split('/'):
            if group:
                if group not in myGroup.listnames():
                    return None
                myGroup = myGroup[group]
        if name not in myGroup.listnames():
            return None

        full_name = "%s/%s" % (node, name)
        try:
            return self._data_file[full_name][:]
        except TypeError:
            return self._data_file[full_name]

    def _close_data_file(self):
        if self._data_file:
            self._data_file.close()
            del self._data_file
            self._data_file = None

    def _deserialize_hierarchy(self, harray):
        # THIS IS BROKEN AND NEEDS TO BE FIXED
        mylog.debug("Cached entry found.")
        self.gridDimensions[:] = harray[:,0:3]
        self.gridStartIndices[:] = harray[:,3:6]
        self.gridEndIndices[:] = harray[:,6:9]
        self.gridLeftEdge[:] = harray[:,9:12]
        self.gridRightEdge[:] = harray[:,12:15]
        self.gridLevels[:] = harray[:,15:16]
        self.gridTimes[:] = harray[:,16:17]
        self.gridNumberOfParticles[:] = harray[:,17:18]

    def get_smallest_dx(self):
        """
        Returns (in code units) the smallest cell size in the simulation.
        """
        return self.select_grids(self.grid_levels.max())[0].dds[0]

    def _add_object_class(self, name, class_name, base, dd):
        self.object_types.append(name)
        obj = classobj(class_name, (base,), dd)
        setattr(self, name, obj)

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
        for level in xrange(self.max_level+1):
            self.level_stats[level]['numgrids'] = na.sum(self.grid_levels == level)
            li = (self.grid_levels[:,0] == level)
            self.level_stats[level]['numcells'] = self.grid_dimensions[li,:].prod(axis=1).sum()

    @property
    def grid_corners(self):
        return na.array([
          [self.grid_left_edge[:,0], self.grid_left_edge[:,1], self.grid_left_edge[:,2]],
          [self.grid_right_edge[:,0], self.grid_left_edge[:,1], self.grid_left_edge[:,2]],
          [self.grid_right_edge[:,0], self.grid_right_edge[:,1], self.grid_left_edge[:,2]],
          [self.grid_right_edge[:,0], self.grid_right_edge[:,1], self.grid_right_edge[:,2]],
          [self.grid_left_edge[:,0], self.grid_right_edge[:,1], self.grid_right_edge[:,2]],
          [self.grid_left_edge[:,0], self.grid_left_edge[:,1], self.grid_right_edge[:,2]],
          [self.grid_right_edge[:,0], self.grid_left_edge[:,1], self.grid_right_edge[:,2]],
          [self.grid_left_edge[:,0], self.grid_right_edge[:,1], self.grid_left_edge[:,2]],
        ], dtype='float64')

    def print_stats(self):
        """
        Prints out (stdout) relevant information about the simulation
        """
        for level in xrange(MAXLEVEL):
            if (self.level_stats['numgrids'][level]) == 0:
                break
            print "% 3i\t% 6i\t% 11i" % \
                  (level, self.level_stats['numgrids'][level],
                   self.level_stats['numcells'][level])
            dx = self.select_grids(level)[0].dds[0]
        print "-" * 28
        print "   \t% 6i\t% 11i" % (self.level_stats['numgrids'].sum(), self.level_stats['numcells'].sum())
        print "\n"
        try:
            print "z = %0.8f" % (self["CosmologyCurrentRedshift"])
        except:
            pass
        t_s = self.pf["InitialTime"] * self.pf["Time"]
        print "t = %0.8e = %0.8e s = %0.8e years" % \
            (self.pf["InitialTime"], \
             t_s, t_s / (365*24*3600.0) )
        print "\nSmallest Cell:"
        u=[]
        for item in self.parameter_file.units.items():
            u.append((item[1],item[0]))
        u.sort()
        for unit in u:
            print "\tWidth: %0.3e %s" % (dx*unit[0], unit[1])


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

# These next two functions are taken from
# http://www.reddit.com/r/Python/comments/6hj75/reverse_file_iterator/c03vms4
# Credit goes to "Brian" on Reddit

def rblocks(f, blocksize=4096):
    """Read file as series of blocks from end of file to start.

    The data itself is in normal order, only the order of the blocks is reversed.
    ie. "hello world" -> ["ld","wor", "lo ", "hel"]
    Note that the file must be opened in binary mode.
    """
    if 'b' not in f.mode.lower():
        raise Exception("File must be opened using binary mode.")
    size = os.stat(f.name).st_size
    fullblocks, lastblock = divmod(size, blocksize)

    # The first(end of file) block will be short, since this leaves 
    # the rest aligned on a blocksize boundary.  This may be more 
    # efficient than having the last (first in file) block be short
    f.seek(-lastblock,2)
    yield f.read(lastblock)

    for i in range(fullblocks-1,-1, -1):
        f.seek(i * blocksize)
        yield f.read(blocksize)

def rlines(f, keepends=False):
    """Iterate through the lines of a file in reverse order.

    If keepends is true, line endings are kept as part of the line.
    """
    buf = ''
    for block in rblocks(f):
        buf = block + buf
        lines = buf.splitlines(keepends)
        # Return all lines except the first (since may be partial)
        if lines:
            lines.reverse()
            buf = lines.pop() # Last line becomes end of new first line.
            for line in lines:
                yield line
    yield buf  # First line.

class GadgetHierarchy(AMRHierarchy):
    grid = GadgetGrid

    def __init__(self, pf, data_style='gadget_hdf5'):
        self.field_info = GadgetFieldContainer()
        self.directory = os.path.dirname(pf.parameter_filename)
        self.data_style = data_style
        self._handle = h5py.File(pf.parameter_filename)
        AMRHierarchy.__init__(self, pf, data_style)
        self._handle.close()

    def _initialize_data_storage(self):
        pass

    def _detect_fields(self):
        #example string:
        #"(S'VEL'\np1\nS'ID'\np2\nS'MASS'\np3\ntp4\n."
        #fields are surrounded with '
        fields_string=self._handle['root'].attrs['fieldnames']
        #splits=fields_string.split("'")
        #pick out the odd fields
        #fields= [splits[j] for j in range(1,len(splits),2)]
        self.field_list = cPickle.loads(fields_string)
    
    def _setup_classes(self):
        dd = self._get_data_reader_dict()
        AMRHierarchy._setup_classes(self, dd)
        self.object_types.sort()

    def _count_grids(self):
        fh = self._handle #shortcut
        #nodes in the hdf5 file are the same as grids
        #in yt
        #the num of levels and total nodes is already saved
        self._levels   = self.pf._get_param('maxlevel')
        self.num_grids = self.pf._get_param('numnodes')
        
    def _parse_hierarchy(self):
        #for every box, define a self.grid(level,edge1,edge2) 
        #with particle counts, dimensions
        f = self._handle #shortcut
        
        root = f['root']
        grids,numnodes = self._walk_nodes(None,root,[])
        dims = [self.pf.max_grid_size for grid in grids]
        LE = [grid.LeftEdge for grid in grids]
        RE = [grid.RightEdge for grid in grids]
        levels = [grid.Level for grid in grids]
        counts = [(grid.N if grid.IsLeaf else 0) for grid in grids]
        self.grids = na.array(grids,dtype='object')
        self.grid_dimensions[:] = na.array(dims, dtype='int64')
        self.grid_left_edge[:] = na.array(LE, dtype='float64')
        self.grid_right_edge[:] = na.array(RE, dtype='float64')
        self.grid_levels.flat[:] = na.array(levels, dtype='int32')
        self.grid_particle_count.flat[:] = na.array(counts, dtype='int32')
            
    def _walk_nodes(self,parent,node,grids,idx=0):
        pi = cPickle.loads
        loc = node.attrs['h5address']
        
        kwargs = {}
        kwargs['Address'] = loc
        kwargs['Parent'] = parent
        kwargs['Axis']  = self.pf._get_param('divideaxis',location=loc)
        kwargs['Level']  = self.pf._get_param('level',location=loc)
        kwargs['LeftEdge'] = self.pf._get_param('leftedge',location=loc) 
        kwargs['RightEdge'] = self.pf._get_param('rightedge',location=loc)
        kwargs['IsLeaf'] = self.pf._get_param('isleaf',location=loc)
        kwargs['N'] = self.pf._get_param('n',location=loc)
        kwargs['NumberOfParticles'] = self.pf._get_param('n',location=loc)
        dx = self.pf._get_param('dx',location=loc)
        dy = self.pf._get_param('dy',location=loc)
        dz = self.pf._get_param('dz',location=loc)
        divdims = na.array([1,1,1])
        if not kwargs['IsLeaf']: 
            divdims[kwargs['Axis']] = 2
        kwargs['ActiveDimensions'] = divdims
        #Active dimensions:
        #This is the number of childnodes, along with dimensiolaity
        #ie, binary tree can be (2,1,1) but octree is (2,2,2)
        
        idx+=1
        #pdb.set_trace()
        children = []
        if not kwargs['IsLeaf']:
            for child in node.values():
                children,idx=self._walk_nodes(node,child,children,idx=idx)
        
        kwargs['Children'] = children
        grid = self.grid(idx,self.pf.parameter_filename,self,**kwargs)
        grids += children
        grids += [grid,]
        return grids,idx

    def _populate_grid_objects(self):
        for g in self.grids:
            g._prepare_grid()
        self.max_level = cPickle.loads(self._handle['root'].attrs['maxlevel'])
    
    def _setup_unknown_fields(self):
        pass

    def _setup_derived_fields(self):
        self.derived_field_list = []

    def _get_grid_children(self, grid):
        #given a grid, use it's address to find subchildren
        pass

class GadgetHierarchyOld(AMRHierarchy):
    #Kept here to compare for the time being
    grid = GadgetGrid

    def __init__(self, pf, data_style):
        self.directory = pf.fullpath
        self.data_style = data_style
        AMRHierarchy.__init__(self, pf, data_style)

    def _count_grids(self):
        # We actually construct our octree here!
        # ...but we do read in our particles, it seems.
        LE = na.zeros(3, dtype='float64')
        RE = na.ones(3, dtype='float64')
        base_grid = ProtoGadgetGrid(0, LE, RE, self.pf.particles)
        self.proto_grids = base_grid.refine(8)
        self.num_grids = len(self.proto_grids)
        self.max_level = max( (g.level for g in self.proto_grids) )

    def _setup_classes(self):
        dd = self._get_data_reader_dict()
        AMRHierarchy._setup_classes(self, dd)
        self.object_types.sort()

    def _parse_hierarchy(self):
        grids = []
        # We need to fill in dims, LE, RE, level, count
        dims, LE, RE, levels, counts = [], [], [], [], []
        self.proto_grids.sort(key = lambda a: a.level)
        for i, pg in enumerate(self.proto_grids):
            g = self.grid(i, self, pg)
            pg.real_grid = g
            grids.append(g)
            dims.append(g.ActiveDimensions)
            LE.append(g.LeftEdge)
            RE.append(g.RightEdge)
            levels.append(g.Level)
            counts.append(g.NumberOfParticles)
        del self.proto_grids
        self.grids = na.array(grids, dtype='object')
        self.grid_dimensions[:] = na.array(dims, dtype='int64')
        self.grid_left_edge[:] = na.array(LE, dtype='float64')
        self.grid_right_edge[:] = na.array(RE, dtype='float64')
        self.grid_levels.flat[:] = na.array(levels, dtype='int32')
        self.grid_particle_count.flat[:] = na.array(counts, dtype='int32')

    def _populate_grid_objects(self):
        # We don't need to do anything here
        for g in self.grids: g._setup_dx()

    def _detect_fields(self):
        self.field_list = ['particle_position_%s' % ax for ax in 'xyz']

    def _setup_unknown_fields(self):
        pass

    def _setup_derived_fields(self):
        self.derived_field_list = []

class ChomboHierarchy(AMRHierarchy):

    grid = ChomboGrid
    
    def __init__(self,pf,data_style='chombo_hdf5'):
        self.data_style = data_style
        self.field_info = ChomboFieldContainer()
        self.field_indexes = {}
        self.parameter_file = weakref.proxy(pf)
        # for now, the hierarchy file is the parameter file!
        self.hierarchy_filename = self.parameter_file.parameter_filename
        self.directory = os.path.dirname(self.hierarchy_filename)
        self._fhandle = h5py.File(self.hierarchy_filename)

        self.float_type = self._fhandle['/level_0']['data:datatype=0'].dtype.name
        self._levels = self._fhandle.listnames()[1:]
        AMRHierarchy.__init__(self,pf,data_style)

        self._fhandle.close()

    def _initialize_data_storage(self):
        pass

    def _detect_fields(self):
        ncomp = int(self._fhandle['/'].attrs['num_components'])
        self.field_list = [c[1] for c in self._fhandle['/'].attrs.listitems()[-ncomp:]]
    
    def _setup_classes(self):
        dd = self._get_data_reader_dict()
        AMRHierarchy._setup_classes(self, dd)
        self.object_types.sort()

    def _count_grids(self):
        self.num_grids = 0
        for lev in self._levels:
            self.num_grids += self._fhandle[lev]['Processors'].len()
        
    def _parse_hierarchy(self):
        f = self._fhandle # shortcut
        
        # this relies on the first Group in the H5 file being
        # 'Chombo_global'
        levels = f.listnames()[1:]
        self.grids = []
        i = 0
        for lev in levels:
            level_number = int(re.match('level_(\d+)',lev).groups()[0])
            boxes = f[lev]['boxes'].value
            dx = f[lev].attrs['dx']
            for level_id, box in enumerate(boxes):
                si = na.array([box['lo_%s' % ax] for ax in 'ijk'])
                ei = na.array([box['hi_%s' % ax] for ax in 'ijk'])
                pg = self.grid(len(self.grids),self,level=level_number,
                               start = si, stop = ei)
                self.grids.append(pg)
                self.grids[-1]._level_id = level_id
                self.grid_left_edge[i] = dx*si.astype(self.float_type)
                self.grid_right_edge[i] = dx*(ei.astype(self.float_type) + 1)
                self.grid_particle_count[i] = 0
                self.grid_dimensions[i] = ei - si + 1
                i += 1
        self.grids = na.array(self.grids, dtype='object')

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

class TigerHierarchy(AMRHierarchy):

    grid = TigerGrid

    def __init__(self, pf, data_style):
        self.directory = pf.fullpath
        self.data_style = data_style
        AMRHierarchy.__init__(self, pf, data_style)

    def _count_grids(self):
        # Tiger is unigrid
        self.ngdims = [i/j for i,j in
                izip(self.pf.root_size, self.pf.max_grid_size)]
        self.num_grids = na.prod(self.ngdims)
        self.max_level = 0

    def _setup_classes(self):
        dd = self._get_data_reader_dict()
        AMRHierarchy._setup_classes(self, dd)
        self.object_types.sort()

    def _parse_hierarchy(self):
        grids = []
        # We need to fill in dims, LE, RE, level, count
        dims, LE, RE, levels, counts = [], [], [], [], []
        DLE = self.pf["DomainLeftEdge"]
        DRE = self.pf["DomainRightEdge"] 
        DW = DRE - DLE
        gds = DW / self.ngdims
        rd = [self.pf.root_size[i]-self.pf.max_grid_size[i] for i in range(3)]
        glx, gly, glz = na.mgrid[DLE[0]:DRE[0]-gds[0]:self.ngdims[0]*1j,
                                 DLE[1]:DRE[1]-gds[1]:self.ngdims[1]*1j,
                                 DLE[2]:DRE[2]-gds[2]:self.ngdims[2]*1j]
        gdx, gdy, gdz = na.mgrid[0:rd[0]:self.ngdims[0]*1j,
                                 0:rd[1]:self.ngdims[1]*1j,
                                 0:rd[2]:self.ngdims[2]*1j]
        LE, RE, levels, counts = [], [], [], []
        i = 0
        for glei, gldi in izip(izip(glx.flat, gly.flat, glz.flat),
                               izip(gdx.flat, gdy.flat, gdz.flat)):
            gld = na.array(gldi)
            gle = na.array(glei)
            gre = gle + gds
            g = self.grid(i, self, gle, gre, gld, gld+self.pf.max_grid_size)
            grids.append(g)
            dims.append(self.pf.max_grid_size)
            LE.append(g.LeftEdge)
            RE.append(g.RightEdge)
            levels.append(g.Level)
            counts.append(g.NumberOfParticles)
            i += 1
        self.grids = na.array(grids, dtype='object')
        self.grid_dimensions[:] = na.array(dims, dtype='int64')
        self.grid_left_edge[:] = na.array(LE, dtype='float64')
        self.grid_right_edge[:] = na.array(RE, dtype='float64')
        self.grid_levels.flat[:] = na.array(levels, dtype='int32')
        self.grid_particle_count.flat[:] = na.array(counts, dtype='int32')

    def _populate_grid_objects(self):
        # We don't need to do anything here
        for g in self.grids: g._setup_dx()

    def _detect_fields(self):
        self.file_mapping = {"Density" : "rhob",
                             "Temperature" : "temp"}

    @property
    def field_list(self):
        return self.file_mapping.keys()

    def _setup_unknown_fields(self):
        for field in self.field_list:
            add_tiger_field(field, lambda a, b: None)

    def _setup_derived_fields(self):
        self.derived_field_list = []

