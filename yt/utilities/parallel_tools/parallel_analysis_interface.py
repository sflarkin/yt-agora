"""
Parallel data mapping techniques for yt

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
Homepage: http://yt-project.org/
License:
  Copyright (C) 2008-2011 Matthew Turk.  All Rights Reserved.

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

import cPickle
import cStringIO
import itertools
import logging
import numpy as na
import sys

from yt.funcs import *

from yt.config import ytcfg
from yt.utilities.definitions import \
    x_dict, y_dict
import yt.utilities.logger
from yt.utilities.amr_utils import \
    QuadTree, merge_quadtrees

exe_name = os.path.basename(sys.executable)
# At import time, we determined whether or not we're being run in parallel.
if exe_name in \
        ["mpi4py", "embed_enzo",
         "python"+sys.version[:3]+"-mpi"] \
    or "--parallel" in sys.argv or '_parallel' in dir(sys) \
    or any(["ipengine" in arg for arg in sys.argv]):
    from mpi4py import MPI
    parallel_capable = (MPI.COMM_WORLD.size > 1)
    if parallel_capable:
        mylog.info("Parallel computation enabled: %s / %s",
                   MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size)
        ytcfg["yt","__parallel_rank"] = str(MPI.COMM_WORLD.rank)
        ytcfg["yt","__parallel_size"] = str(MPI.COMM_WORLD.size)
        ytcfg["yt","__parallel"] = "True"
        if exe_name == "embed_enzo" or \
            ("_parallel" in dir(sys) and sys._parallel == True):
            ytcfg["yt","inline"] = "True"
        # I believe we do not need to turn this off manually
        #ytcfg["yt","StoreParameterFiles"] = "False"
        # Now let's make sure we have the right options set.
        if MPI.COMM_WORLD.rank > 0:
            if ytcfg.getboolean("yt","serialize"):
                ytcfg["yt","onlydeserialize"] = "True"
            if ytcfg.getboolean("yt","LogFile"):
                ytcfg["yt","LogFile"] = "False"
                yt.utilities.logger.disable_file_logging()
        yt.utilities.logger.uncolorize_logging()
        # Even though the uncolorize function already resets the format string,
        # we reset it again so that it includes the processor.
        f = logging.Formatter("P%03i %s" % (MPI.COMM_WORLD.rank,
                                            yt.utilities.logger.ufstring))
        if len(yt.utilities.logger.rootLogger.handlers) > 0:
            yt.utilities.logger.rootLogger.handlers[0].setFormatter(f)
        if ytcfg.getboolean("yt", "parallel_traceback"):
            sys.excepthook = traceback_writer_hook("_%03i" % MPI.COMM_WORLD.rank)
    if ytcfg.getint("yt","LogLevel") < 20:
        yt.utilities.logger.ytLogger.warning(
          "Log Level is set low -- this could affect parallel performance!")

else:
    parallel_capable = False

# Set up translation table
if parallel_capable:
    dtype_names = dict(
            float32 = MPI.FLOAT,
            float64 = MPI.DOUBLE,
            int32   = MPI.INT,
            int64   = MPI.LONG
    )
else:
    dtype_names = dict(
            float32 = "MPI.FLOAT",
            float64 = "MPI.DOUBLE",
            int32   = "MPI.INT",
            int64   = "MPI.LONG"
    )

# Because the dtypes will == correctly but do not hash the same, we need this
# function for dictionary access.
def get_mpi_type(dtype):
    for dt, val in dtype_names.items():
        if dt == dtype: return val

class ObjectIterator(object):
    """
    This is a generalized class that accepts a list of objects and then
    attempts to intelligently iterate over them.
    """
    def __init__(self, pobj, just_list = False, attr='_grids'):
        self.pobj = pobj
        if hasattr(pobj, attr) and getattr(pobj, attr) is not None:
            gs = getattr(pobj, attr)
        else:
            gs = getattr(pobj._data_source, attr)
        if hasattr(gs[0], 'proc_num'):
            # This one sort of knows about MPI, but not quite
            self._objs = [g for g in gs if g.proc_num ==
                          ytcfg.getint('yt','__parallel_rank')]
            self._use_all = True
        else:
            self._objs = gs
            if hasattr(self._objs[0], 'filename'):
                self._objs = sorted(self._objs, key = lambda g: g.filename)
            self._use_all = False
        self.ng = len(self._objs)
        self.just_list = just_list

    def __iter__(self):
        for obj in self._objs: yield obj
        
class ParallelObjectIterator(ObjectIterator):
    """
    This takes an object, *pobj*, that implements ParallelAnalysisInterface,
    and then does its thing, calling initliaze and finalize on the object.
    """
    def __init__(self, pobj, just_list = False, attr='_grids',
                 round_robin=False):
        ObjectIterator.__init__(self, pobj, just_list, attr=attr)
        self._offset = MPI.COMM_WORLD.rank
        self._skip = MPI.COMM_WORLD.size
        # Note that we're doing this in advance, and with a simple means
        # of choosing them; more advanced methods will be explored later.
        if self._use_all:
            self.my_obj_ids = na.arange(len(self._objs))
        else:
            if not round_robin:
                self.my_obj_ids = na.array_split(
                                na.arange(len(self._objs)), self._skip)[self._offset]
            else:
                self.my_obj_ids = na.arange(len(self._objs))[self._offset::self._skip]
        
    def __iter__(self):
        for gid in self.my_obj_ids:
            yield self._objs[gid]
        if not self.just_list: self.pobj._finalize_parallel()

def parallel_simple_proxy(func):
    """
    This is a decorator that broadcasts the result of computation on a single
    processor to all other processors.  To do so, it uses the _processing and
    _distributed flags in the object to check for blocks.  Meant only to be
    used on objects that subclass
    :class:`~yt.utilities.parallel_tools.parallel_analysis_interface.ParallelAnalysisInterface`.
    """
    if not parallel_capable: return func
    @wraps(func)
    def single_proc_results(self, *args, **kwargs):
        retval = None
        if self._processing or not self._distributed:
            return func(self, *args, **kwargs)
        if self._owner == MPI.COMM_WORLD.rank:
            self._processing = True
            retval = func(self, *args, **kwargs)
            self._processing = False
        retval = MPI.COMM_WORLD.bcast(retval, root=self._owner)
        #MPI.COMM_WORLD.Barrier()
        return retval
    return single_proc_results

class ParallelDummy(type):
    """
    This is a base class that, on instantiation, replaces all attributes that
    don't start with ``_`` with
    :func:`~yt.utilities.parallel_tools.parallel_analysis_interface.parallel_simple_proxy`-wrapped
    attributes.  Used as a metaclass.
    """
    def __init__(cls, name, bases, d):
        super(ParallelDummy, cls).__init__(name, bases, d)
        skip = d.pop("dont_wrap", [])
        extra = d.pop("extra_wrap", [])
        for attrname in d:
            if attrname.startswith("_") or attrname in skip:
                if attrname not in extra: continue
            attr = getattr(cls, attrname)
            if type(attr) == types.MethodType:
                setattr(cls, attrname, parallel_simple_proxy(attr))

def parallel_passthrough(func):
    """
    If we are not run in parallel, this function passes the input back as
    output; otherwise, the function gets called.  Used as a decorator.
    """
    @wraps(func)
    def passage(self, data):
        if not self._distributed: return data
        return func(self, data)
    return passage

def parallel_blocking_call(func):
    """
    This decorator blocks on entry and exit of a function.
    """
    @wraps(func)
    def barrierize(*args, **kwargs):
        mylog.debug("Entering barrier before %s", func.func_name)
        MPI.COMM_WORLD.Barrier()
        retval = func(*args, **kwargs)
        mylog.debug("Entering barrier after %s", func.func_name)
        MPI.COMM_WORLD.Barrier()
        return retval
    if parallel_capable:
        return barrierize
    else:
        return func

def parallel_splitter(f1, f2):
    """
    This function returns either the function *f1* or *f2* depending on whether
    or not we're the root processor.  Mainly used in class definitions.
    """
    @wraps(f1)
    def in_order(*args, **kwargs):
        if MPI.COMM_WORLD.rank == 0:
            f1(*args, **kwargs)
        MPI.COMM_WORLD.Barrier()
        if MPI.COMM_WORLD.rank != 0:
            f2(*args, **kwargs)
    if not parallel_capable: return f1
    return in_order

def parallel_root_only(func):
    """
    This decorator blocks and calls the function on the root processor,
    but does not broadcast results to the other processors.
    """
    @wraps(func)
    def root_only(*args, **kwargs):
        if MPI.COMM_WORLD.rank == 0:
            try:
                func(*args, **kwargs)
                all_clear = 1
            except:
                traceback.print_last()
                all_clear = 0
        else:
            all_clear = None
        #MPI.COMM_WORLD.Barrier()
        all_clear = MPI.COMM_WORLD.bcast(all_clear, root=0)
        if not all_clear: raise RuntimeError
    if parallel_capable: return root_only
    return func

class ParallelAnalysisInterface(object):
    """
    This is an interface specification providing several useful utility
    functions for analyzing something in parallel.
    """
    _grids = None
    _distributed = parallel_capable

    def _get_objs(self, attr, *args, **kwargs):
        if self._distributed:
            rr = kwargs.pop("round_robin", False)
            self._initialize_parallel(*args, **kwargs)
            return ParallelObjectIterator(self, attr=attr,
                    round_robin=rr)
        return ObjectIterator(self, attr=attr)

    def _get_grids(self, *args, **kwargs):
        if self._distributed:
            self._initialize_parallel(*args, **kwargs)
            return ParallelObjectIterator(self, attr='_grids')
        return ObjectIterator(self, attr='_grids')

    def _get_grid_objs(self):
        if self._distributed:
            return ParallelObjectIterator(self, True, attr='_grids')
        return ObjectIterator(self, True, attr='_grids')

    def _initialize_parallel(self):
        pass

    def _finalize_parallel(self):
        pass

    def _partition_hierarchy_2d(self, axis):
        if not self._distributed:
           return False, self.hierarchy.grid_collection(self.center, self.hierarchy.grids)

        xax, yax = x_dict[axis], y_dict[axis]
        cc = MPI.Compute_dims(MPI.COMM_WORLD.size, 2)
        mi = MPI.COMM_WORLD.rank
        cx, cy = na.unravel_index(mi, cc)
        x = na.mgrid[0:1:(cc[0]+1)*1j][cx:cx+2]
        y = na.mgrid[0:1:(cc[1]+1)*1j][cy:cy+2]

        DLE, DRE = self.pf.domain_left_edge.copy(), self.pf.domain_right_edge.copy()
        LE = na.ones(3, dtype='float64') * DLE
        RE = na.ones(3, dtype='float64') * DRE
        LE[xax] = x[0] * (DRE[xax]-DLE[xax]) + DLE[xax]
        RE[xax] = x[1] * (DRE[xax]-DLE[xax]) + DLE[xax]
        LE[yax] = y[0] * (DRE[yax]-DLE[yax]) + DLE[yax]
        RE[yax] = y[1] * (DRE[yax]-DLE[yax]) + DLE[yax]
        mylog.debug("Dimensions: %s %s", LE, RE)

        reg = self.hierarchy.region_strict(self.center, LE, RE)
        return True, reg

    def _partition_hierarchy_3d(self, ds, padding=0.0, rank_ratio = 1):
        LE, RE = na.array(ds.left_edge), na.array(ds.right_edge)
        # We need to establish if we're looking at a subvolume, in which case
        # we *do* want to pad things.
        if (LE == self.pf.domain_left_edge).all() and \
                (RE == self.pf.domain_right_edge).all():
            subvol = False
        else:
            subvol = True
        if not self._distributed and not subvol:
            return False, LE, RE, ds
        if not self._distributed and subvol:
            return True, LE, RE, \
            self.hierarchy.periodic_region_strict(self.center,
                LE-padding, RE+padding)
        elif ytcfg.getboolean("yt", "inline"):
            # At this point, we want to identify the root grid tile to which
            # this processor is assigned.
            # The only way I really know how to do this is to get the level-0
            # grid that belongs to this processor.
            grids = self.pf.h.select_grids(0)
            root_grids = [g for g in grids
                          if g.proc_num == MPI.COMM_WORLD.rank]
            if len(root_grids) != 1: raise RuntimeError
            #raise KeyError
            LE = root_grids[0].LeftEdge
            RE = root_grids[0].RightEdge
            return True, LE, RE, self.hierarchy.region(self.center, LE, RE)

        cc = MPI.Compute_dims(MPI.COMM_WORLD.size / rank_ratio, 3)
        mi = MPI.COMM_WORLD.rank % (MPI.COMM_WORLD.size / rank_ratio)
        cx, cy, cz = na.unravel_index(mi, cc)
        x = na.mgrid[LE[0]:RE[0]:(cc[0]+1)*1j][cx:cx+2]
        y = na.mgrid[LE[1]:RE[1]:(cc[1]+1)*1j][cy:cy+2]
        z = na.mgrid[LE[2]:RE[2]:(cc[2]+1)*1j][cz:cz+2]

        LE = na.array([x[0], y[0], z[0]], dtype='float64')
        RE = na.array([x[1], y[1], z[1]], dtype='float64')

        if padding > 0:
            return True, \
                LE, RE, self.hierarchy.periodic_region_strict(self.center,
                LE-padding, RE+padding)

        return False, LE, RE, self.hierarchy.region_strict(self.center, LE, RE)

    def _partition_region_3d(self, left_edge, right_edge, padding=0.0,
            rank_ratio = 1):
        """
        Given a region, it subdivides it into smaller regions for parallel
        analysis.
        """
        LE, RE = left_edge[:], right_edge[:]
        if not self._distributed:
            return LE, RE, re
        
        cc = MPI.Compute_dims(MPI.COMM_WORLD.size / rank_ratio, 3)
        mi = MPI.COMM_WORLD.rank % (MPI.COMM_WORLD.size / rank_ratio)
        cx, cy, cz = na.unravel_index(mi, cc)
        x = na.mgrid[LE[0]:RE[0]:(cc[0]+1)*1j][cx:cx+2]
        y = na.mgrid[LE[1]:RE[1]:(cc[1]+1)*1j][cy:cy+2]
        z = na.mgrid[LE[2]:RE[2]:(cc[2]+1)*1j][cz:cz+2]

        LE = na.array([x[0], y[0], z[0]], dtype='float64')
        RE = na.array([x[1], y[1], z[1]], dtype='float64')

        if padding > 0:
            return True, \
                LE, RE, self.hierarchy.periodic_region(self.center, LE-padding,
                    RE+padding)

        return False, LE, RE, self.hierarchy.region(self.center, LE, RE)

    def _partition_hierarchy_3d_bisection_list(self):
        """
        Returns an array that is used to drive _partition_hierarchy_3d_bisection,
        below.
        """

        def factor(n):
            if n == 1: return [1]
            i = 2
            limit = n**0.5
            while i <= limit:
                if n % i == 0:
                    ret = factor(n/i)
                    ret.append(i)
                    return ret
                i += 1
            return [n]

        cc = MPI.Compute_dims(MPI.COMM_WORLD.size, 3)
        si = MPI.COMM_WORLD.size
        
        factors = factor(si)
        xyzfactors = [factor(cc[0]), factor(cc[1]), factor(cc[2])]
        
        # Each entry of cuts is a two element list, that is:
        # [cut dim, number of cuts]
        cuts = []
        # The higher cuts are in the beginning.
        # We're going to do our best to make the cuts cyclic, i.e. x, then y,
        # then z, etc...
        lastdim = 0
        for f in factors:
            nextdim = (lastdim + 1) % 3
            while True:
                if f in xyzfactors[nextdim]:
                    cuts.append([nextdim, f])
                    topop = xyzfactors[nextdim].index(f)
                    temp = xyzfactors[nextdim].pop(topop)
                    lastdim = nextdim
                    break
                nextdim = (nextdim + 1) % 3
        return cuts

    def _barrier(self):
        if not self._distributed: return
        mylog.debug("Opening MPI Barrier on %s", MPI.COMM_WORLD.rank)
        MPI.COMM_WORLD.Barrier()

    def _mpi_exit_test(self, data=False):
        # data==True -> exit. data==False -> no exit
        mine, statuses = self._mpi_info_dict(data)
        if True in statuses.values():
            raise RuntimeError("Fatal error. Exiting.")
        return None

    @parallel_passthrough
    def _mpi_minimum_array_long(self, data):
        """
        Specifically for parallelHOP. For the identical array on each task,
        it merges the arrays together, taking the lower value at each index.
        """
        self._barrier()
        size = data.size # They're all the same size, of course
        if MPI.COMM_WORLD.rank == 0:
            new_data = na.empty(size, dtype='int64')
            for i in range(1, MPI.COMM_WORLD.size):
                MPI.COMM_WORLD.Recv([new_data, MPI.LONG], i, 0)
                data = na.minimum(data, new_data)
            del new_data
        else:
            MPI.COMM_WORLD.Send([data, MPI.LONG], 0, 0)
        # Redistribute from root
        MPI.COMM_WORLD.Bcast([data, MPI.LONG], root=0)
        return data

    @parallel_passthrough
    def _mpi_catdict(self, data):
        field_keys = data.keys()
        field_keys.sort()
        size = data[field_keys[0]].shape[-1]
        sizes = na.zeros(MPI.COMM_WORLD.size, dtype='int64')
        outsize = na.array(size, dtype='int64')
        MPI.COMM_WORLD.Allgather([outsize, 1, MPI.LONG],
                                 [sizes, 1, MPI.LONG] )
        # This nested concatenate is to get the shapes to work out correctly;
        # if we just add [0] to sizes, it will broadcast a summation, not a
        # concatenation.
        offsets = na.add.accumulate(na.concatenate([[0], sizes]))[:-1]
        arr_size = MPI.COMM_WORLD.allreduce(size, op=MPI.SUM)
        for key in field_keys:
            dd = data[key]
            rv = _alltoallv_array(dd, arr_size, offsets, sizes)
            data[key] = rv
        return data

    @parallel_passthrough
    def _mpi_joindict(self, data):
        #self._barrier()
        if MPI.COMM_WORLD.rank == 0:
            for i in range(1,MPI.COMM_WORLD.size):
                data.update(MPI.COMM_WORLD.recv(source=i, tag=0))
        else:
            MPI.COMM_WORLD.send(data, dest=0, tag=0)
        data = MPI.COMM_WORLD.bcast(data, root=0)
        #self._barrier()
        return data

    @parallel_passthrough
    def _mpi_maxdict(self, data):
        """
        For each key in data, find the maximum value across all tasks, and
        then broadcast it back.
        """
        self._barrier()
        if MPI.COMM_WORLD.rank == 0:
            for i in range(1,MPI.COMM_WORLD.size):
                temp_data = MPI.COMM_WORLD.recv(source=i, tag=0)
                for key in temp_data:
                    try:
                        old_value = data[key]
                    except KeyError:
                        # This guarantees the new value gets added.
                        old_value = None
                    if old_value < temp_data[key]:
                        data[key] = temp_data[key]
        else:
            MPI.COMM_WORLD.send(data, dest=0, tag=0)
        data = MPI.COMM_WORLD.bcast(data, root=0)
        self._barrier()
        return data

    def _mpi_maxdict_dict(self, data):
        """
        Similar to above, but finds maximums for dicts of dicts. This is
        specificaly for a part of chainHOP.
        """
        if not self._distributed:
            top_keys = []
            bot_keys = []
            vals = []
            for top_key in data:
                for bot_key in data[top_key]:
                    top_keys.append(top_key)
                    bot_keys.append(bot_key)
                    vals.append(data[top_key][bot_key])
            top_keys = na.array(top_keys, dtype='int64')
            bot_keys = na.array(bot_keys, dtype='int64')
            vals = na.array(vals, dtype='float64')
            return (top_keys, bot_keys, vals)
        self._barrier()
        size = 0
        top_keys = []
        bot_keys = []
        vals = []
        for top_key in data:
            for bot_key in data[top_key]:
                top_keys.append(top_key)
                bot_keys.append(bot_key)
                vals.append(data[top_key][bot_key])
        top_keys = na.array(top_keys, dtype='int64')
        bot_keys = na.array(bot_keys, dtype='int64')
        vals = na.array(vals, dtype='float64')
        del data
        if MPI.COMM_WORLD.rank == 0:
            for i in range(1,MPI.COMM_WORLD.size):
                size = MPI.COMM_WORLD.recv(source=i, tag=0)
                mylog.info('Global Hash Table Merge %d of %d size %d' % \
                    (i,MPI.COMM_WORLD.size, size))
                recv_top_keys = na.empty(size, dtype='int64')
                recv_bot_keys = na.empty(size, dtype='int64')
                recv_vals = na.empty(size, dtype='float64')
                MPI.COMM_WORLD.Recv([recv_top_keys, MPI.LONG], source=i, tag=0)
                MPI.COMM_WORLD.Recv([recv_bot_keys, MPI.LONG], source=i, tag=0)
                MPI.COMM_WORLD.Recv([recv_vals, MPI.DOUBLE], source=i, tag=0)
                top_keys = na.concatenate([top_keys, recv_top_keys])
                bot_keys = na.concatenate([bot_keys, recv_bot_keys])
                vals = na.concatenate([vals, recv_vals])
        else:
            size = top_keys.size
            MPI.COMM_WORLD.send(size, dest=0, tag=0)
            MPI.COMM_WORLD.Send([top_keys, MPI.LONG], dest=0, tag=0)
            MPI.COMM_WORLD.Send([bot_keys, MPI.LONG], dest=0, tag=0)
            MPI.COMM_WORLD.Send([vals, MPI.DOUBLE], dest=0, tag=0)
        # We're going to decompose the dict into arrays, send that, and then
        # reconstruct it. When data is too big the pickling of the dict fails.
        if MPI.COMM_WORLD.rank == 0:
            size = top_keys.size
        # Broadcast them using array methods
        size = MPI.COMM_WORLD.bcast(size, root=0)
        if MPI.COMM_WORLD.rank != 0:
            top_keys = na.empty(size, dtype='int64')
            bot_keys = na.empty(size, dtype='int64')
            vals = na.empty(size, dtype='float64')
        MPI.COMM_WORLD.Bcast([top_keys,MPI.LONG], root=0)
        MPI.COMM_WORLD.Bcast([bot_keys,MPI.LONG], root=0)
        MPI.COMM_WORLD.Bcast([vals, MPI.DOUBLE], root=0)
        return (top_keys, bot_keys, vals)

    @parallel_passthrough
    def _mpi_catlist(self, data):
        self._barrier()
        if MPI.COMM_WORLD.rank == 0:
            data = self.__mpi_recvlist(data)
        else:
            MPI.COMM_WORLD.send(data, dest=0, tag=0)
        mylog.debug("Opening MPI Broadcast on %s", MPI.COMM_WORLD.rank)
        data = MPI.COMM_WORLD.bcast(data, root=0)
        self._barrier()
        return data

    @parallel_passthrough
    def _mpi_catarray(self, data):
        if data is None:
            ncols = -1
            size = 0
        else:
            if len(data) == 0:
                ncols = -1
                size = 0
            elif len(data.shape) == 1:
                ncols = 1
                size = data.shape[0]
            else:
                ncols, size = data.shape
        ncols = MPI.COMM_WORLD.allreduce(ncols, op=MPI.MAX)
        if size == 0:
            data = na.zeros((ncols,0), dtype='float64') # This only works for
        size = data.shape[-1]
        sizes = na.zeros(MPI.COMM_WORLD.size, dtype='int64')
        outsize = na.array(size, dtype='int64')
        MPI.COMM_WORLD.Allgather([outsize, 1, MPI.LONG],
                                 [sizes, 1, MPI.LONG] )
        # This nested concatenate is to get the shapes to work out correctly;
        # if we just add [0] to sizes, it will broadcast a summation, not a
        # concatenation.
        offsets = na.add.accumulate(na.concatenate([[0], sizes]))[:-1]
        arr_size = MPI.COMM_WORLD.allreduce(size, op=MPI.SUM)
        data = _alltoallv_array(data, arr_size, offsets, sizes)
        return data

    @parallel_passthrough
    def _mpi_bcast_pickled(self, data):
        #self._barrier()
        data = MPI.COMM_WORLD.bcast(data, root=0)
        return data

    def _should_i_write(self):
        if not self._distributed: return True
        return (MPI.COMM_WORLD == 0)

    def _preload(self, grids, fields, io_handler):
        # This will preload if it detects we are parallel capable and
        # if so, we load *everything* that we need.  Use with some care.
        mylog.debug("Preloading %s from %s grids", fields, len(grids))
        if not self._distributed: return
        io_handler.preload(grids, fields)

    @parallel_passthrough
    def _mpi_double_array_max(self,data):
        """
        Finds the na.maximum of a distributed array and returns the result
        back to all. The array should be the same length on all tasks!
        """
        self._barrier()
        if MPI.COMM_WORLD.rank == 0:
            recv_data = na.empty(data.size, dtype='float64')
            for i in xrange(1, MPI.COMM_WORLD.size):
                MPI.COMM_WORLD.Recv([recv_data, MPI.DOUBLE], source=i, tag=0)
                data = na.maximum(data, recv_data)
        else:
            MPI.COMM_WORLD.Send([data, MPI.DOUBLE], dest=0, tag=0)
        MPI.COMM_WORLD.Bcast([data, MPI.DOUBLE], root=0)
        return data

    @parallel_passthrough
    def _mpi_allsum(self, data, dtype=None):
        if isinstance(data, na.ndarray) and data.dtype != na.bool:
            if dtype is None:
                dtype = data.dtype
            if dtype != data.dtype:
                data = data.astype(dtype)
            temp = data.copy()
            MPI.COMM_WORLD.Allreduce([temp,get_mpi_type(dtype)], 
                                     [data,get_mpi_type(dtype)], op=MPI.SUM)
            return data
        else:
            # We use old-school pickling here on the assumption the arrays are
            # relatively small ( < 1e7 elements )
            return MPI.COMM_WORLD.allreduce(data, op=MPI.SUM)

    @parallel_passthrough
    def _mpi_allmax(self, data):
        self._barrier()
        return MPI.COMM_WORLD.allreduce(data, op=MPI.MAX)

    @parallel_passthrough
    def _mpi_allmin(self, data):
        self._barrier()
        return MPI.COMM_WORLD.allreduce(data, op=MPI.MIN)

    ###
    # Non-blocking stuff.
    ###

    def _mpi_nonblocking_recv(self, data, source, tag=0, dtype=None):
        if not self._distributed: return -1
        if dtype is None: dtype = data.dtype
        mpi_type = get_mpi_type(dtype)
        return MPI.COMM_WORLD.Irecv([data, mpi_type], source, tag)

    def _mpi_nonblocking_send(self, data, dest, tag=0, dtype=None):
        if not self._distributed: return -1
        if dtype is None: dtype = data.dtype
        mpi_type = get_mpi_type(dtype)
        return MPI.COMM_WORLD.Isend([data, mpi_type], dest, tag)

    def _mpi_Request_Waitall(self, hooks):
        if not self._distributed: return
        MPI.Request.Waitall(hooks)

    def _mpi_Request_Waititer(self, hooks):
        for i in xrange(len(hooks)):
            req = MPI.Request.Waitany(hooks)
            yield req

    def _mpi_Request_Testall(self, hooks):
        """
        This returns False if any of the request hooks are un-finished,
        and True if they are all finished.
        """
        if not self._distributed: return True
        return MPI.Request.Testall(hooks)

    ###
    # End non-blocking stuff.
    ###

    ###
    # Parallel rank and size properties.
    ###

    @property
    def _par_size(self):
        if not self._distributed: return 1
        return MPI.COMM_WORLD.size

    @property
    def _par_rank(self):
        if not self._distributed: return 0
        return MPI.COMM_WORLD.rank

    def _mpi_info_dict(self, info):
        if not self._distributed: return 0, {0:info}
        self._barrier()
        data = None
        if MPI.COMM_WORLD.rank == 0:
            data = {0:info}
            for i in range(1, MPI.COMM_WORLD.size):
                data[i] = MPI.COMM_WORLD.recv(source=i, tag=0)
        else:
            MPI.COMM_WORLD.send(info, dest=0, tag=0)
        mylog.debug("Opening MPI Broadcast on %s", MPI.COMM_WORLD.rank)
        data = MPI.COMM_WORLD.bcast(data, root=0)
        self._barrier()
        return MPI.COMM_WORLD.rank, data

    def _get_dependencies(self, fields):
        deps = []
        fi = self.pf.field_info
        for field in fields:
            deps += ensure_list(fi[field].get_dependencies(pf=self.pf).requested)
        return list(set(deps))

    def _claim_object(self, obj):
        if not self._distributed: return
        obj._owner = MPI.COMM_WORLD.rank
        obj._distributed = True

    def _do_not_claim_object(self, obj):
        if not self._distributed: return
        obj._owner = -1
        obj._distributed = True

    def _write_on_root(self, fn):
        if not self._distributed: return open(fn, "w")
        if MPI.COMM_WORLD.rank == 0:
            return open(fn, "w")
        else:
            return cStringIO.StringIO()

    def _get_filename(self, prefix, rank=None):
        if not self._distributed: return prefix
        if rank == None:
            return "%s_%04i" % (prefix, MPI.COMM_WORLD.rank)
        else:
            return "%s_%04i" % (prefix, rank)

    def _is_mine(self, obj):
        if not obj._distributed: return True
        return (obj._owner == MPI.COMM_WORLD.rank)

    def _send_quadtree(self, target, buf, tgd, args):
        sizebuf = na.zeros(1, 'int64')
        sizebuf[0] = buf[0].size
        MPI.COMM_WORLD.Send([sizebuf, MPI.LONG], dest=target)
        MPI.COMM_WORLD.Send([buf[0], MPI.INT], dest=target)
        MPI.COMM_WORLD.Send([buf[1], MPI.DOUBLE], dest=target)
        MPI.COMM_WORLD.Send([buf[2], MPI.DOUBLE], dest=target)
        
    def _recv_quadtree(self, target, tgd, args):
        sizebuf = na.zeros(1, 'int64')
        MPI.COMM_WORLD.Recv(sizebuf, source=target)
        buf = [na.empty((sizebuf[0],), 'int32'),
               na.empty((sizebuf[0], args[2]),'float64'),
               na.empty((sizebuf[0],),'float64')]
        MPI.COMM_WORLD.Recv([buf[0], MPI.INT], source=target)
        MPI.COMM_WORLD.Recv([buf[1], MPI.DOUBLE], source=target)
        MPI.COMM_WORLD.Recv([buf[2], MPI.DOUBLE], source=target)
        return buf

    @parallel_passthrough
    def merge_quadtree_buffers(self, qt):
        # This is a modified version of pairwise reduction from Lisandro Dalcin,
        # in the reductions demo of mpi4py
        size = MPI.COMM_WORLD.size
        rank = MPI.COMM_WORLD.rank

        mask = 1

        args = qt.get_args() # Will always be the same
        tgd = na.array([args[0], args[1]], dtype='int64')
        sizebuf = na.zeros(1, 'int64')

        while mask < size:
            if (mask & rank) != 0:
                target = (rank & ~mask) % size
                #print "SENDING FROM %02i to %02i" % (rank, target)
                buf = qt.tobuffer()
                self._send_quadtree(target, buf, tgd, args)
                #qt = self._recv_quadtree(target, tgd, args)
            else:
                target = (rank | mask)
                if target < size:
                    #print "RECEIVING FROM %02i on %02i" % (target, rank)
                    buf = self._recv_quadtree(target, tgd, args)
                    qto = QuadTree(tgd, args[2])
                    qto.frombuffer(*buf)
                    merge_quadtrees(qt, qto)
                    del qto
                    #self._send_quadtree(target, qt, tgd, args)
            mask <<= 1

        if rank == 0:
            buf = qt.tobuffer()
            sizebuf[0] = buf[0].size
        MPI.COMM_WORLD.Bcast([sizebuf, MPI.LONG], root=0)
        if rank != 0:
            buf = [na.empty((sizebuf[0],), 'int32'),
                   na.empty((sizebuf[0], args[2]),'float64'),
                   na.empty((sizebuf[0],),'float64')]
        MPI.COMM_WORLD.Bcast([buf[0], MPI.INT], root=0)
        MPI.COMM_WORLD.Bcast([buf[1], MPI.DOUBLE], root=0)
        MPI.COMM_WORLD.Bcast([buf[2], MPI.DOUBLE], root=0)
        self.refined = buf[0]
        if rank != 0:
            qt = QuadTree(tgd, args[2])
            qt.frombuffer(*buf)
        return qt

__tocast = 'c'

def _send_array(arr, dest, tag = 0):
    if not isinstance(arr, na.ndarray):
        MPI.COMM_WORLD.send((None,None), dest=dest, tag=tag)
        MPI.COMM_WORLD.send(arr, dest=dest, tag=tag)
        return
    tmp = arr.view(__tocast) # Cast to CHAR
    # communicate type and shape
    MPI.COMM_WORLD.send((arr.dtype.str, arr.shape), dest=dest, tag=tag)
    MPI.COMM_WORLD.Send([arr, MPI.CHAR], dest=dest, tag=tag)
    del tmp

def _recv_array(source, tag = 0):
    dt, ne = MPI.COMM_WORLD.recv(source=source, tag=tag)
    if dt is None and ne is None:
        return MPI.COMM_WORLD.recv(source=source, tag=tag)
    arr = na.empty(ne, dtype=dt)
    tmp = arr.view(__tocast)
    MPI.COMM_WORLD.Recv([tmp, MPI.CHAR], source=source, tag=tag)
    return arr

def _bcast_array(arr, root = 0):
    if MPI.COMM_WORLD.rank == root:
        tmp = arr.view(__tocast) # Cast to CHAR
        MPI.COMM_WORLD.bcast((arr.dtype.str, arr.shape), root=root)
    else:
        dt, ne = MPI.COMM_WORLD.bcast(None, root=root)
        arr = na.empty(ne, dtype=dt)
        tmp = arr.view(__tocast)
    MPI.COMM_WORLD.Bcast([tmp, MPI.CHAR], root=root)
    return arr

def _alltoallv_array(send, total_size, offsets, sizes):
    if len(send.shape) > 1:
        recv = []
        for i in range(send.shape[0]):
            recv.append(_alltoallv_array(send[i,:].copy(), total_size, offsets, sizes))
        recv = na.array(recv)
        return recv
    offset = offsets[MPI.COMM_WORLD.rank]
    tmp_send = send.view(__tocast)
    recv = na.empty(total_size, dtype=send.dtype)
    recv[offset:offset+send.size] = send[:]
    dtr = send.dtype.itemsize / tmp_send.dtype.itemsize # > 1
    roff = [off * dtr for off in offsets]
    rsize = [siz * dtr for siz in sizes]
    tmp_recv = recv.view(__tocast)
    MPI.COMM_WORLD.Allgatherv((tmp_send, tmp_send.size, MPI.CHAR),
                              (tmp_recv, (rsize, roff), MPI.CHAR))
    return recv
    
