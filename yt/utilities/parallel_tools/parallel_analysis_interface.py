"""
Parallel data mapping techniques for yt

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2008-2009 Matthew Turk.  All Rights Reserved.

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
        yt.utilities.logger.rootLogger.handlers[0].setFormatter(f)
        if ytcfg.getboolean("yt", "parallel_traceback"):
            sys.excepthook = traceback_writer_hook("_%03i" % MPI.COMM_WORLD.rank)
    if ytcfg.getint("yt","LogLevel") < 20:
        yt.utilities.logger.ytLogger.warning(
          "Log Level is set low -- this could affect parallel performance!")

else:
    parallel_capable = False

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
        MPI.COMM_WORLD.Barrier()
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
        MPI.COMM_WORLD.Barrier()
        if MPI.COMM_WORLD.rank == 0:
            f1(*args, **kwargs)
        MPI.COMM_WORLD.Barrier()
        if MPI.COMM_WORLD.rank != 0:
            f2(*args, **kwargs)
        MPI.COMM_WORLD.Barrier()
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
        MPI.COMM_WORLD.Barrier()
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

    def _partition_hierarchy_2d_inclined(self, unit_vectors, origin, widths,
                                         box_vectors, resolution = (1.0, 1.0)):
        if not self._distributed:
            ib = self.hierarchy.inclined_box(origin, box_vectors)
            return False, ib, resolution
        # We presuppose that unit_vectors is already unitary.  If it's not,
        # caveat emptor.
        uv = na.array(unit_vectors)
        inv_mat = na.linalg.pinv(uv)
        cc = MPI.Compute_dims(MPI.COMM_WORLD.size, 2)
        mi = MPI.COMM_WORLD.rank
        cx, cy = na.unravel_index(mi, cc)
        resolution = (1.0/cc[0], 1.0/cc[1])
        # We are rotating with respect to the *origin*, not the back center,
        # so we go from 0 .. width.
        px = na.mgrid[0.0:1.0:(cc[0]+1)*1j][cx] * widths[0]
        py = na.mgrid[0.0:1.0:(cc[1]+1)*1j][cy] * widths[1]
        nxo = inv_mat[0,0]*px + inv_mat[0,1]*py + origin[0]
        nyo = inv_mat[1,0]*px + inv_mat[1,1]*py + origin[1]
        nzo = inv_mat[2,0]*px + inv_mat[2,1]*py + origin[2]
        nbox_vectors = na.array(
                       [unit_vectors[0] * widths[0]/cc[0],
                        unit_vectors[1] * widths[1]/cc[1],
                        unit_vectors[2] * widths[2]],
                        dtype='float64')
        norigin = na.array([nxo, nyo, nzo])
        box = self.hierarchy.inclined_box(norigin, nbox_vectors)
        return True, box, resolution
        
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
        

    def _partition_hierarchy_3d_bisection(self, axis, bins, counts, top_bounds = None,\
        old_group = None, old_comm = None, cut=None, old_cc=None):
        """
        Partition the volume into evenly weighted subvolumes using the distribution
        in counts. The bisection happens in the MPI communicator group old_group.
        You may need to set "MPI_COMM_MAX" and "MPI_GROUP_MAX" environment 
        variables.
        """
        counts = counts.astype('int64')
        if not self._distributed:
            LE, RE = self.pf.domain_left_edge.copy(), self.pf.domain_right_edge.copy()
            return False, LE, RE, self.hierarchy.grid_collection(self.center, self.hierarchy.grids)
        
        # First time through the world is the current group.
        if old_group == None or old_comm == None:
            old_group = MPI.COMM_WORLD.Get_group()
            old_comm = MPI.COMM_WORLD
        
        # Figure out the gridding based on the deepness of cuts.
        if old_cc is None:
            cc = MPI.Compute_dims(MPI.COMM_WORLD.size, 3)
        else:
            cc = old_cc
        cc[cut[0]] /= cut[1]
        # Set the boundaries of the full bounding box for this group.
        if top_bounds == None:
            LE, RE = self.pf.domain_left_edge.copy(), self.pf.domain_right_edge.copy()
        else:
            LE, RE = top_bounds

        ra = old_group.Get_rank() # In this group, not WORLD, unless it's the first time.
        
        # First find the total number of particles in my group.
        parts = old_comm.allreduce(int(counts.sum()), op=MPI.SUM)
        # Now the full sum in the bins along this axis in this group.
        full_counts = na.empty(counts.size, dtype='int64')
        old_comm.Allreduce([counts, MPI.LONG], [full_counts, MPI.LONG], op=MPI.SUM)
        # Find the bin that passes the cut points.
        midpoints = [LE[axis]]
        sum = 0
        bin = 0
        for step in xrange(1,cut[1]):
            while sum < ((parts*step)/cut[1]):
                lastsum = sum
                sum += full_counts[bin]
                bin += 1
            # Bin edges
            left_edge = bins[bin-1]
            right_edge = bins[bin]
            # Find a better approx of the midpoint cut line using a linear approx.
            a = float(sum - lastsum) / (right_edge - left_edge)
            midpoints.append(left_edge + (0.5 - (float(lastsum) / parts / 2)) / a)
            #midpoint = (left_edge + right_edge) / 2.
        midpoints.append(RE[axis])
        # Now we need to split the members of this group into chunks. 
        # The values that go into the _ranks are the ranks of the tasks
        # in *this* communicator group, which go zero to size - 1. They are not
        # the same as the global ranks!
        groups = {}
        ranks = {}
        old_group_size = old_group.Get_size()
        for step in xrange(cut[1]):
            groups[step] = na.arange(step*old_group_size/cut[1], (step+1)*old_group_size/cut[1])
            # [ (start, stop, step), ]
            ranks[step] = [ (groups[step][0], groups[step][-1], 1), ] 
        
        # Based on where we are, adjust our LE or RE, depending on axis. At the
        # same time assign the new MPI group membership.
        for step in xrange(cut[1]):
            if ra in groups[step]:
                LE[axis] = midpoints[step]
                RE[axis] = midpoints[step+1]
                new_group = old_group.Range_incl(ranks[step])
                new_comm = old_comm.Create(new_group)
        
        if old_cc is not None:
            old_group.Free()
            old_comm.Free()
        
        new_top_bounds = (LE,RE)
        
        # Using the new boundaries, regrid.
        mi = new_comm.rank
        cx, cy, cz = na.unravel_index(mi, cc)
        x = na.mgrid[LE[0]:RE[0]:(cc[0]+1)*1j][cx:cx+2]
        y = na.mgrid[LE[1]:RE[1]:(cc[1]+1)*1j][cy:cy+2]
        z = na.mgrid[LE[2]:RE[2]:(cc[2]+1)*1j][cz:cz+2]

        my_LE = na.array([x[0], y[0], z[0]], dtype='float64')
        my_RE = na.array([x[1], y[1], z[1]], dtype='float64')
        
        # Return a new subvolume and associated stuff.
        return new_group, new_comm, my_LE, my_RE, new_top_bounds, cc,\
            self.hierarchy.region_strict(self.center, my_LE, my_RE)

    def _mpi_find_neighbor_3d(self, shift):
        """ Given a shift array, 1x3 long, find the task ID
        of that neighbor. For example, shift=[1,0,0] finds the neighbor
        immediately to the right in the positive x direction. Each task
        has 26 neighbors, of which some may be itself depending on the number
        and arrangement of tasks.
        """
        if not self._distributed: return 0
        shift = na.array(shift)
        cc = na.array(MPI.Compute_dims(MPI.COMM_WORLD.size, 3))
        mi = MPI.COMM_WORLD.rank
        si = MPI.COMM_WORLD.size
        # store some facts about myself
        mi_cx,mi_cy,mi_cz = na.unravel_index(mi,cc)
        mi_ar = na.array([mi_cx,mi_cy,mi_cz])
        # these are identical on all tasks
        # should these be calculated once and stored?
        #dLE = na.empty((si,3), dtype='float64') # positions not needed yet...
        #dRE = na.empty((si,3), dtype='float64')
        tasks = na.empty((cc[0],cc[1],cc[2]), dtype='int64')
        
        for i in range(si):
            cx,cy,cz = na.unravel_index(i,cc)
            tasks[cx,cy,cz] = i
            #x = na.mgrid[LE[0]:RE[0]:(cc[0]+1)*1j][cx:cx+2]
            #y = na.mgrid[LE[1]:RE[1]:(cc[1]+1)*1j][cy:cy+2]
            #z = na.mgrid[LE[2]:RE[2]:(cc[2]+1)*1j][cz:cz+2]
            #dLE[i, :] = na.array([x[0], y[0], z[0]], dtype='float64')
            #dRE[i, :] = na.array([x[1], y[1], z[1]], dtype='float64')
        
        # find the neighbor
        ne = (mi_ar + shift) % cc
        ne = tasks[ne[0],ne[1],ne[2]]
        return ne
        
        
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
    def _mpi_catrgb(self, data):
        self._barrier()
        data, final = data
        if MPI.COMM_WORLD.rank == 0:
            cc = MPI.Compute_dims(MPI.COMM_WORLD.size, 2)
            nsize = final[0]/cc[0], final[1]/cc[1]
            new_image = na.zeros((final[0], final[1], 6), dtype='float64')
            new_image[0:nsize[0],0:nsize[1],:] = data[:]
            for i in range(1,MPI.COMM_WORLD.size):
                cy, cx = na.unravel_index(i, cc)
                mylog.debug("Receiving image from % into bits %s:%s, %s:%s",
                    i, nsize[0]*cx,nsize[0]*(cx+1),
                       nsize[1]*cy,nsize[1]*(cy+1))
                buf = _recv_array(source=i, tag=0).reshape(
                    (nsize[0],nsize[1],6))
                new_image[nsize[0]*cy:nsize[0]*(cy+1),
                          nsize[1]*cx:nsize[1]*(cx+1),:] = buf[:]
            data = new_image
        else:
            _send_array(data.ravel(), dest=0, tag=0)
        data = MPI.COMM_WORLD.bcast(data)
        return (data, final)

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
        self._barrier()
        if MPI.COMM_WORLD.rank == 0:
            for i in range(1,MPI.COMM_WORLD.size):
                data.update(MPI.COMM_WORLD.recv(source=i, tag=0))
        else:
            MPI.COMM_WORLD.send(data, dest=0, tag=0)
        data = MPI.COMM_WORLD.bcast(data, root=0)
        self._barrier()
        return data

    @parallel_passthrough
    def _mpi_joindict_unpickled_double(self, data):
        self._barrier()
        size = 0
        if MPI.COMM_WORLD.rank == 0:
            for i in range(1,MPI.COMM_WORLD.size):
                size = MPI.COMM_WORLD.recv(source=i, tag=0)
                keys = na.empty(size, dtype='int64')
                values = na.empty(size, dtype='float64')
                MPI.COMM_WORLD.Recv([keys, MPI.LONG], i, 0)
                MPI.COMM_WORLD.Recv([values, MPI.DOUBLE], i, 0)
                for i,key in enumerate(keys):
                    data[key] = values[i]
            # Now convert root's data to arrays.
            size = len(data)
            root_keys = na.empty(size, dtype='int64')
            root_values = na.empty(size, dtype='float64')
            count = 0
            for key in data:
                root_keys[count] = key
                root_values[count] = data[key]
                count += 1
        else:
            MPI.COMM_WORLD.send(len(data), 0, 0)
            keys = na.empty(len(data), dtype='int64')
            values = na.empty(len(data), dtype='float64')
            count = 0
            for key in data:
                keys[count] = key
                values[count] = data[key]
                count += 1
            MPI.COMM_WORLD.Send([keys, MPI.LONG], 0, 0)
            MPI.COMM_WORLD.Send([values, MPI.DOUBLE], 0, 0)
        # Now send it back as arrays.
        size = MPI.COMM_WORLD.bcast(size, root=0)
        if MPI.COMM_WORLD.rank != 0:
            del keys, values
            root_keys = na.empty(size, dtype='int64')
            root_values = na.empty(size, dtype='float64')
        MPI.COMM_WORLD.Bcast([root_keys, MPI.LONG], root=0)
        MPI.COMM_WORLD.Bcast([root_values, MPI.DOUBLE], root=0)
        # Convert back to a dict.
        del data
        data = dict(itertools.izip(root_keys, root_values))
        return data

    @parallel_passthrough
    def _mpi_joindict_unpickled_long(self, data):
        self._barrier()
        size = 0
        if MPI.COMM_WORLD.rank == 0:
            for i in range(1,MPI.COMM_WORLD.size):
                size = MPI.COMM_WORLD.recv(source=i, tag=0)
                keys = na.empty(size, dtype='int64')
                values = na.empty(size, dtype='int64')
                MPI.COMM_WORLD.Recv([keys, MPI.LONG], i, 0)
                MPI.COMM_WORLD.Recv([values, MPI.LONG], i, 0)
                for i,key in enumerate(keys):
                    data[key] = values[i]
            # Now convert root's data to arrays.
            size = len(data)
            root_keys = na.empty(size, dtype='int64')
            root_values = na.empty(size, dtype='int64')
            count = 0
            for key in data:
                root_keys[count] = key
                root_values[count] = data[key]
                count += 1
        else:
            MPI.COMM_WORLD.send(len(data), 0, 0)
            keys = na.empty(len(data), dtype='int64')
            values = na.empty(len(data), dtype='int64')
            count = 0
            for key in data:
                keys[count] = key
                values[count] = data[key]
                count += 1
            MPI.COMM_WORLD.Send([keys, MPI.LONG], 0, 0)
            MPI.COMM_WORLD.Send([values, MPI.LONG], 0, 0)
        # Now send it back as arrays.
        size = MPI.COMM_WORLD.bcast(size, root=0)
        if MPI.COMM_WORLD.rank != 0:
            del keys, values
            root_keys = na.empty(size, dtype='int64')
            root_values = na.empty(size, dtype='int64')
        MPI.COMM_WORLD.Bcast([root_keys, MPI.LONG], root=0)
        MPI.COMM_WORLD.Bcast([root_values, MPI.LONG], root=0)
        # Convert back to a dict.
        del data
        data = dict(itertools.izip(root_keys,root_values))
        return data

    @parallel_passthrough
    def _mpi_concatenate_array_long(self, data):
        self._barrier()
        size = 0
        if MPI.COMM_WORLD.rank == 0:
            for i in range(1, MPI.COMM_WORLD.size):
                size = MPI.COMM_WORLD.recv(source=i, tag=0)
                new_data = na.empty(size, dtype='int64')
                MPI.COMM_WORLD.Recv([new_data, MPI.LONG], i, 0)
                data = na.concatenate((data, new_data))
            size = data.size
            del new_data
        else:
            MPI.COMM_WORLD.send(data.size, 0, 0)
            MPI.COMM_WORLD.Send([data, MPI.LONG], 0, 0)
        # Now we distribute the full array.
        size = MPI.COMM_WORLD.bcast(size, root=0)
        if MPI.COMM_WORLD.rank != 0:
            del data
            data = na.empty(size, dtype='int64')
        MPI.COMM_WORLD.Bcast([data, MPI.LONG], root=0)
        return data

    @parallel_passthrough
    def _mpi_concatenate_array_double(self, data):
        self._barrier()
        size = 0
        if MPI.COMM_WORLD.rank == 0:
            for i in range(1, MPI.COMM_WORLD.size):
                size = MPI.COMM_WORLD.recv(source=i, tag=0)
                new_data = na.empty(size, dtype='float64')
                MPI.COMM_WORLD.Recv([new_data, MPI.DOUBLE], i, 0)
                data = na.concatenate((data, new_data))
            size = data.size
            del new_data
        else:
            MPI.COMM_WORLD.send(data.size, 0, 0)
            MPI.COMM_WORLD.Send([data, MPI.DOUBLE], 0, 0)
        # Now we distribute the full array.
        size = MPI.COMM_WORLD.bcast(size, root=0)
        if MPI.COMM_WORLD.rank != 0:
            del data
            data = na.empty(size, dtype='float64')
        MPI.COMM_WORLD.Bcast([data, MPI.DOUBLE], root=0)
        return data

    @parallel_passthrough
    def _mpi_concatenate_array_on_root_double(self, data):
        self._barrier()
        size = 0
        if MPI.COMM_WORLD.rank == 0:
            for i in range(1, MPI.COMM_WORLD.size):
                size = MPI.COMM_WORLD.recv(source=i, tag=0)
                new_data = na.empty(size, dtype='float64')
                MPI.COMM_WORLD.Recv([new_data, MPI.DOUBLE], i, 0)
                data = na.concatenate((data, new_data))
        else:
            MPI.COMM_WORLD.send(data.size, 0, 0)
            MPI.COMM_WORLD.Send([data, MPI.DOUBLE], 0, 0)
        return data

    @parallel_passthrough
    def _mpi_concatenate_array_on_root_int(self, data):
        self._barrier()
        size = 0
        if MPI.COMM_WORLD.rank == 0:
            for i in range(1, MPI.COMM_WORLD.size):
                size = MPI.COMM_WORLD.recv(source=i, tag=0)
                new_data = na.empty(size, dtype='int32')
                MPI.COMM_WORLD.Recv([new_data, MPI.INT], i, 0)
                data = na.concatenate((data, new_data))
        else:
            MPI.COMM_WORLD.send(data.size, 0, 0)
            MPI.COMM_WORLD.Send([data, MPI.INT], 0, 0)
        return data

    @parallel_passthrough
    def _mpi_concatenate_array_on_root_long(self, data):
        self._barrier()
        size = 0
        if MPI.COMM_WORLD.rank == 0:
            for i in range(1, MPI.COMM_WORLD.size):
                size = MPI.COMM_WORLD.recv(source=i, tag=0)
                new_data = na.empty(size, dtype='int64')
                MPI.COMM_WORLD.Recv([new_data, MPI.LONG], i, 0)
                data = na.concatenate((data, new_data))
        else:
            MPI.COMM_WORLD.send(data.size, 0, 0)
            MPI.COMM_WORLD.Send([data, MPI.LONG], 0, 0)
        return data

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
    def _mpi_bcast_long_dict_unpickled(self, data):
        self._barrier()
        size = 0
        if MPI.COMM_WORLD.rank == 0:
            size = len(data)
        size = MPI.COMM_WORLD.bcast(size, root=0)
        root_keys = na.empty(size, dtype='int64')
        root_values = na.empty(size, dtype='int64')
        if MPI.COMM_WORLD.rank == 0:
            count = 0
            for key in data:
                root_keys[count] = key
                root_values[count] = data[key]
                count += 1
        MPI.COMM_WORLD.Bcast([root_keys, MPI.LONG], root=0)
        MPI.COMM_WORLD.Bcast([root_values, MPI.LONG], root=0)
        if MPI.COMM_WORLD.rank != 0:
            data = {}
            for i,key in enumerate(root_keys):
                data[key] = root_values[i]
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
#                 for j, top_key in enumerate(top_keys):
#                     if j%1000 == 0: mylog.info(j)
#                     # Make sure there's an entry for top_key in data
#                     try:
#                         test = data[top_key]
#                     except KeyError:
#                         data[top_key] = {}
#                     try:
#                         old_value = data[top_key][bot_keys[j]]
#                     except KeyError:
#                         # This guarantees the new value gets added.
#                         old_value = None
#                     if old_value < vals[j]:
#                         data[top_key][bot_keys[j]] = vals[j]
        else:
#             top_keys = []
#             bot_keys = []
#             vals = []
#             for top_key in data:
#                 for bot_key in data[top_key]:
#                     top_keys.append(top_key)
#                     bot_keys.append(bot_key)
#                     vals.append(data[top_key][bot_key])
#             top_keys = na.array(top_keys, dtype='int64')
#             bot_keys = na.array(bot_keys, dtype='int64')
#             vals = na.array(vals, dtype='float64')
            size = top_keys.size
            MPI.COMM_WORLD.send(size, dest=0, tag=0)
            MPI.COMM_WORLD.Send([top_keys, MPI.LONG], dest=0, tag=0)
            MPI.COMM_WORLD.Send([bot_keys, MPI.LONG], dest=0, tag=0)
            MPI.COMM_WORLD.Send([vals, MPI.DOUBLE], dest=0, tag=0)
        # Getting ghetto here, we're going to decompose the dict into arrays,
        # send that, and then reconstruct it. When data is too big the pickling
        # of the dict fails.
        if MPI.COMM_WORLD.rank == 0:
#             data = defaultdict(dict)
#             for i,top_key in enumerate(top_keys):
#                 try:
#                     old = data[top_key][bot_keys[i]]
#                 except KeyError:
#                     old = None
#                 if old < vals[i]:
#                     data[top_key][bot_keys[i]] = vals[i]
#             top_keys = []
#             bot_keys = []
#             vals = []
#             for top_key in data:
#                 for bot_key in data[top_key]:
#                     top_keys.append(top_key)
#                     bot_keys.append(bot_key)
#                     vals.append(data[top_key][bot_key])
#             del data
#             top_keys = na.array(top_keys, dtype='int64')
#             bot_keys = na.array(bot_keys, dtype='int64')
#             vals = na.array(vals, dtype='float64')
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
    def __mpi_recvlist(self, data):
        # First we receive, then we make a new list.
        data = ensure_list(data)
        for i in range(1,MPI.COMM_WORLD.size):
            buf = ensure_list(MPI.COMM_WORLD.recv(source=i, tag=0))
            data += buf
        return data

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
    def __mpi_recvarrays(self, data):
        # First we receive, then we make a new list.
        for i in range(1,MPI.COMM_WORLD.size):
            buf = _recv_array(source=i, tag=0)
            if buf is not None:
                if data is None: data = buf
                else: data = na.concatenate([data, buf])
        return data

    @parallel_passthrough
    def _mpi_cat_na_array(self,data):
        self._barrier()
        comm = MPI.COMM_WORLD
        if comm.rank == 0:
            for i in range(1,comm.size):
                buf = comm.recv(source=i, tag=0)
                data = na.concatenate([data,buf])
        else:
            comm.send(data, 0, tag = 0)
        data = comm.bcast(data, root=0)
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
            data = na.empty((ncols,0), dtype='float64') # This only works for
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
        self._barrier()
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
    def _mpi_allsum(self, data):
        self._barrier()
        # We use old-school pickling here on the assumption the arrays are
        # relatively small ( < 1e7 elements )
        if isinstance(data, na.ndarray):
            tr = na.zeros_like(data)
            if not data.flags.c_contiguous: data = data.copy()
            MPI.COMM_WORLD.Allreduce(data, tr, op=MPI.SUM)
            return tr
        else:
            return MPI.COMM_WORLD.allreduce(data, op=MPI.SUM)

    @parallel_passthrough
    def _mpi_Allsum_double(self, data):
        self._barrier()
        # Non-pickling float allsum of a float array, data.
        temp = data.copy()
        MPI.COMM_WORLD.Allreduce([temp, MPI.DOUBLE], [data, MPI.DOUBLE], op=MPI.SUM)
        del temp
        return data

    @parallel_passthrough
    def _mpi_Allsum_long(self, data):
        self._barrier()
        # Non-pickling float allsum of an int array, data.
        temp = data.copy()
        MPI.COMM_WORLD.Allreduce([temp, MPI.LONG], [data, MPI.LONG], op=MPI.SUM)
        del temp
        return data

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

    def _mpi_Irecv_long(self, data, source, tag=0):
        if not self._distributed: return -1
        return MPI.COMM_WORLD.Irecv([data, MPI.LONG], source, tag)

    def _mpi_Irecv_double(self, data, source, tag=0):
        if not self._distributed: return -1
        return MPI.COMM_WORLD.Irecv([data, MPI.DOUBLE], source, tag)

    def _mpi_Isend_long(self, data, dest, tag=0):
        if not self._distributed: return -1
        return MPI.COMM_WORLD.Isend([data, MPI.LONG], dest, tag)

    def _mpi_Isend_double(self, data, dest, tag=0):
        if not self._distributed: return -1
        return MPI.COMM_WORLD.Isend([data, MPI.DOUBLE], dest, tag)

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

    def _mpi_get_size(self):
        if not self._distributed: return 1
        return MPI.COMM_WORLD.size

    def _mpi_get_rank(self):
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
    
