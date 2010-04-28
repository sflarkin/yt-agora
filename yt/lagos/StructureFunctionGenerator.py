"""
Structure Function Generator

Author: Stephen Skory <stephenskory@yahoo.com>
Affiliation: UCSD Physics/CASS
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2010 Stephen Skory.  All Rights Reserved.

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
from yt.math_utils import *
from yt.performance_counters import yt_counters, time_function
try:
    from yt.extensions.kdtree import *
except ImportError:
    mylog.info("The Fortran kD-Tree did not import correctly. The structure function generator will not work correctly.")

import math, sys, itertools, inspect, types, time
from collections import defaultdict

class StructFcnGen(ParallelAnalysisInterface):
    def __init__(self, pf, fields, left_edge=None, right_edge=None,
            total_values=1000000, comm_size=10000, length_type="lin",
            length_number=10, length_range=None, vol_ratio = 1,
            salt=0):
        """
        *total_values* (int) How many total (global) pair calculations to run
        for each of the structure functions specified.
        A single integer. Default: 1,000,000.
        *comm_size* (int) How entries are sent during communication.
        Default: 10,000.
        Set the parameters used to search the simulational volume with randomly
        placed 'rulers'.
        *length_type* (str) controls the even spacing of the rulers lengths in
        logarithmic or linear space, set by "log" or "lin", respectively.
        A single string. Default: "lin"
        *length_number* (int) sets how many lengths to create, evenly spaced by the above
        parameter.
        A single integer. Default: "10"
        *length_range* (float) a min/max pair for the range of values to search the over
        the simulational volume.
        A single pair (a list or array). Default: [sqrt(3)dx, 1/2*shortest box edge],
        where dx is the smallest grid cell size.
        *salt* (int) a number that will be added to the random number generator
        seed. Use this if a different random series of numbers is desired when
        keeping everything else constant from this set: (MPI task count, 
        number of ruler lengths, ruler min/max, number of structure functions,
        number of point pairs per ruler length). Default: 0.
        """
        self._fsets = []
        self.fields = fields
        # MPI stuff.
        self.size = self._mpi_get_size()
        self.mine = self._mpi_get_rank()
        self.vol_ratio = vol_ratio
        self.total_values = int(total_values / self.size)
        # For communication.
        self.recv_hooks = []
        self.send_hooks = []
        self.done_hooks = []
        self.comm_size = min(int(comm_size), self.total_values)
        self.pf = pf
        self.nlevels = pf.h.max_level
        self.period = self.pf['DomainRightEdge'] - self.pf['DomainLeftEdge']
        self.min_edge = min(self.period)
        self.hierarchy = pf.h
        self.center = (pf["DomainRightEdge"] + pf["DomainLeftEdge"])/2.0
        # Figure out the range of ruler lengths.
        if length_range == None:
            length_range = [math.sqrt(3) * self.pf.h.get_smallest_dx(),
                self.min_edge/2.]
        else:
            if len(length_range) != 2:
                raise ValueError("length_range must have two values.")
            if length_range[1] <= length_range[0]:
                raise ValueError("length_range[1] must be larger than length_range[0]")
            if length_range[1] > self.min_edge/2.:
                length_range[1] = self.min_edge/2.
                mylog.info("Automatically adjusting length_range[1] to half the shortest box edge.")
        if length_range[0] == -1 or length_range[0] == -1.:
            mylog.info("Automatically adjusting length_range[0] to %1.5e." % \
                (math.sqrt(3) * self.pf.h.get_smallest_dx()))
            length_range[0] = math.sqrt(3) * self.pf.h.get_smallest_dx()
        # Make the list of ruler lengths.
        if length_type == "lin":
            self.lengths = na.linspace(length_range[0], length_range[1],
                length_number)
        elif length_type == "log":
            self.lengths = na.logspace(math.log10(length_range[0]),
                math.log10(length_range[1]), length_number)
        else:
            # Something went wrong.
            raise SyntaxError("length_type is either \"lin\" or \"log\".")
        # Subdivide the volume.
        if not left_edge or not right_edge:
            self.left_edge = self.pf['DomainLeftEdge']
            self.right_edge = self.pf['DomainRightEdge']
            padded, self.LE, self.RE, self.ds = self._partition_hierarchy_3d(padding=0.,
                rank_ratio = self.vol_ratio)
        else:
            self.left_edge = left_edge
            self.right_edge = right_edge
            # We do this twice, first with no 'buffer' to get the unbuffered
            # self.LE/RE, and then second to get a buffered self.ds.
            padded, self.LE, self.RE, temp = \
                self._partition_region_3d(left_edge, right_edge,
                    rank_ratio=self.vol_ratio)
            padded, temp, temp, self.ds = \
                self._partition_region_3d(left_edge - self.lengths[-1], \
                right_edge + self.lengths[-1], rank_ratio=self.vol_ratio)
        mylog.info("LE %s RE %s %s" % (str(self.LE), str(self.RE), str(self.ds)))
        self.width = self.ds.right_edge - self.ds.left_edge
        self.mt = na.random.mtrand.RandomState(seed = 1234 * self.mine + salt)
    
    def add_function(self, function, out_labels, sqrt):
        """
        Add a structure function to the list that will be evaluated at the
        generated pairs of points.
        """
        fargs = inspect.getargspec(function)
        if len(fargs.args) != 5:
            raise SyntaxError("The structure function %s needs five arguments." %\
                function.__name__)
        out_labels = list(out_labels)
        if len(out_labels) < 1:
            raise SyntaxError("Please specify at least one out_labels for function %s." %\
                function.__name__)
        sqrt = list(sqrt)
        if len(sqrt) != len(out_labels):
            raise SyntaxError("Please have the same number of elements in out_labels as in sqrt for function %s." %\
                function.__name__)
        self._fsets.append(StructSet(self, function, self.min_edge,
            out_labels, sqrt))
        return self._fsets[-1]

    def __getitem__(self, key):
        return self._fsets[key]
    
    def run_generator(self):
        """
        After all the structure functions have been added, run the generator.
        """
        yt_counters("run_generator")
        # We need a function!
        if len(self._fsets) == 0:
            mylog.error("You need to add at least one structure function!")
            return None
        # Do all the startup tasks to get the grid points.
        if self.nlevels == 1:
            yt_counters("build_sort")
            self._build_sort_array()
            self.sort_done = False
            yt_counters("build_sort")
        else:
            yt_counters("init_kd_tree")
            self._init_kd_tree()
            self.sort_done = True
            yt_counters("init_kd_tree")
        # Store the fields.
        self.stored_fields = {}
        yt_counters("getting data")
        for field in self.fields:
            self.stored_fields[field] = self.ds[field].copy()
        self.ds.clear_data()
        # If the arrays haven't been sorted yet and need to be, do that.
        if not self.sort_done:
            for field in self.fields:
                self.stored_fields[field] = self.stored_fields[field][self.sort]
            del self.sort
            self.sort_done = True
        yt_counters("getting data")
        self._build_fields_vals()
        yt_counters("big loop over lengths")
        t_waiting = 0.
        for bigloop, length in enumerate(self.lengths):
            self._build_points_array()
            if self.mine == 0:
                mylog.info("Doing length %1.5e" % length)
            # Things stop when this value below equals total_values.
            self.generated_points = 0
            self.gen_array = na.zeros(self.size, dtype='int64')
            self.comm_cycle_count = 0
            self.final_comm_cycle_count = 0
            self.sent_done = False
            self._setup_done_hooks_on_root()
            # While everyone else isn't done or I'm not done, we loop.
            while self._should_cycle():
                self._setup_recv_arrays()
                self._send_arrays()
                t0 = time.time()
                self._mpi_Request_Waitall(self.send_hooks)
                self._mpi_Request_Waitall(self.recv_hooks)
                t1 = time.time()
                t_waiting += (t1-t0)
                if (self.recv_points < -1.).any() or (self.recv_points > 1.).any(): # or \
                        #(na.abs(na.log10(na.abs(self.recv_points))) > 20).any():
                    raise ValueError("self.recv_points is no good!")
                self.points = self.recv_points.copy()
                self.fields_vals = self.recv_fields_vals.copy()
                self.gen_array = self.recv_gen_array.copy()
                self._eval_points(length)
                self.gen_array[self.mine] = self.generated_points
                self.comm_cycle_count += 1
                if self.generated_points == self.total_values:
                    self._send_done_to_root()
            if self.mine == 0:
                mylog.info("Length (%d of %d) %1.5e took %d communication cycles to complete." % \
                (bigloop+1, len(self.lengths), length, self.comm_cycle_count))
        yt_counters("big loop over lengths")
        if self.nlevels > 1:
            del fKD.pos, fKD.qv_many, fKD.nn_tags
            free_tree() # Frees the kdtree object.
        yt_counters("allsum")
        self._allsum_bin_hits()
        mylog.info("Spent %f seconds waiting for communication." % t_waiting)
        yt_counters("allsum")
        yt_counters("run_generator")
    
    def _init_kd_tree(self):
        """
        Builds the kd tree of grid center points.
        """
        # Grid cell centers.
        mylog.info("Multigrid: Building kD-Tree.")
        xp = self.ds["x"]
        yp = self.ds["y"]
        zp = self.ds["z"]
        fKD.pos = na.asfortranarray(na.empty((3,xp.size), dtype='float64'))
        fKD.pos[0, :] = xp[:]
        fKD.pos[1, :] = yp[:]
        fKD.pos[2, :] = zp[:]
        fKD.nn = 1
        fKD.sort = False
        fKD.rearrange = True
        create_tree()

    def _build_sort_array(self):
        """
        When running on a unigrid simulation, the kD tree isn't necessary.
        But we need to ensure that the points are sorted in the usual manner
        allowing values to be found via array indices.
        """
        mylog.info("Unigrid: finding cell centers.")
        xp = self.ds["x"]
        yp = self.ds["y"]
        zp = self.ds["z"]
        self.sizes = [na.unique(xp).size, na.unique(yp).size, na.unique(zp).size]        
        self.sort = na.lexsort([zp, yp, xp])
        del xp, yp, zp
        self.ds.clear_data()
    
    def _build_fields_vals(self):
        """
        Builds an array to store the field values array.
        """
        self.fields_vals = na.empty((self.comm_size, len(self.fields)*2), \
            dtype='float64')
        # At the same time build a dict to label the columns.
        self.fields_columns = {}
        for i,field in enumerate(self.fields):
            self.fields_columns[field] = i

    def _build_points_array(self):
        """
        Initializes the array that contains the random points as all negatives
        to start with.
        """
        self.points = na.ones((self.comm_size, 6), dtype='float64') * -1.0
    
    def _setup_done_hooks_on_root(self):
        """
        Opens non-blocking receives on root pointing to all the other tasks
        """
        if self.mine != 0:
            return
        self.recv_done = {}
        for task in xrange(self.size):
            if task == self.mine: continue
            self.recv_done[task] = na.zeros(1, dtype='int64')
            self.done_hooks.append(self._mpi_Irecv_long(self.recv_done[task], \
                task, tag=15))
    
    def _send_done_to_root(self):
        """
        Tell the root process that I'm done.
        """
        # If I've already done this, don't do it again.
        if self.sent_done: return
        if self.mine !=0:
            # I send when I *think* things should finish.
            self.send_done = na.ones(1, dtype='int64') * \
                (self.size / self.vol_ratio -1) + self.comm_cycle_count
            self.done_hooks.append(self._mpi_Isend_long(self.send_done, \
                    0, tag=15))
        else:
            # As root, I need to mark myself!
            self.recv_done[0] = na.ones(1, dtype='int64') * \
                (self.size / self.vol_ratio -1) + self.comm_cycle_count
        self.sent_done = True
    
    def _should_cycle(self):
        """
        Determine if I should continue cycling the communication.
        """
        if self.mine == 0:
            # If other tasks aren't finished, this will return False.
            status = self._mpi_Request_Testall(self.done_hooks)
            # Convolve this with with root's status.
            status = status * (self.generated_points == self.total_values)
            if status == 1:
                # If they are all finished, meaning Testall returns True,
                # and root has made its points, we find
                # the biggest value in self.recv_done and stop there.
                status = max(self.recv_done.values())
        else:
            status = 0
        # Broadcast the status from root - we stop only if root thinks we should
        # stop.
        status = self._mpi_bcast_pickled(status)
        if status == 0: return True
        if self.comm_cycle_count < status:
            return True
        # If we've come this far, we're done.
        return False

    def _setup_recv_arrays(self):
        """
        Creates the recv buffers and calls a non-blocking MPI receive pointing
        to the left-hand neighbor.
        """
        self.recv_points = na.ones((self.comm_size, 6), dtype='float64') * -1.
        self.recv_fields_vals = na.zeros((self.comm_size, len(self.fields)*2), \
            dtype='float64')
        self.recv_gen_array = na.zeros(self.size, dtype='int64')
        self.recv_hooks.append(self._mpi_Irecv_double(self.recv_points, \
            (self.mine-1)%self.size, tag=10))
        self.recv_hooks.append(self._mpi_Irecv_double(self.recv_fields_vals, \
            (self.mine-1)%self.size, tag=20))
        self.recv_hooks.append(self._mpi_Irecv_long(self.recv_gen_array, \
            (self.mine-1)%self.size, tag=40))

    def _send_arrays(self):
        """
        Send the data arrays to the right-hand neighbor.
        """
        self.send_hooks.append(self._mpi_Isend_double(self.points,\
            (self.mine+1)%self.size, tag=10))
        self.send_hooks.append(self._mpi_Isend_double(self.fields_vals,\
            (self.mine+1)%self.size, tag=20))
        self.send_hooks.append(self._mpi_Isend_long(self.gen_array, \
            (self.mine+1)%self.size, tag=40))

    def _allsum_bin_hits(self):
        """
        Add up the hits to all the bins globally for all functions.
        """
        for fset in self._fsets:
            fset.too_low = self._mpi_allsum(fset.too_low)
            fset.too_high = self._mpi_allsum(fset.too_high)
            fset.binned = {}
            if self.mine == 0:
                mylog.info("Function %s had values out of range for these fields:" % \
                    fset.function.__name__)
                for i,field in enumerate(fset.out_labels):
                    mylog.info("Field %s had %d values too high and %d too low that were not binned." % \
                    (field, fset.too_high[i], fset.too_low[i]))
            for length in self.lengths:
                fset.length_bin_hits[length] = \
                    self._mpi_Allsum_long(fset.length_bin_hits[length])
                # Find out how many were successfully binned.
                fset.binned[length] = fset.length_bin_hits[length].sum()
                # Normalize the counts.
                fset.length_bin_hits[length] = \
                    fset.length_bin_hits[length].astype('float64') / \
                    fset.binned[length]
                # Return it to its original shape.
                fset.length_bin_hits[length] = \
                    fset.length_bin_hits[length].reshape(fset.bin_number)
    
    def _pick_random_points(self, length, size):
        """
        Picks out size random pairs separated by length *length*.
        """
        # First make random points inside this subvolume.
        r1 = na.empty((size,3), dtype='float64')
        for dim in range(3):
            r1[:,dim] = self.mt.uniform(low=self.ds.left_edge[dim],
                high=self.ds.right_edge[dim], size=size)
        # Next we find the second point, determined by a random
        # theta, phi angle.
        theta = self.mt.uniform(low=0, high=2.*math.pi, size=size)
        phi = self.mt.uniform(low=-math.pi/2., high=math.pi/2., size=size)
        r2 = na.empty((size,3), dtype='float64')
        r2[:,0] = r1[:,0] + length * na.cos(theta) * na.cos(phi)
        r2[:,1] = r1[:,1] + length * na.sin(theta) * na.cos(phi)
        r2[:,2] = r1[:,2] + length * na.sin(phi)
        # Reflect so it's inside the (full) volume.
        r2 %= self.period
        return (r1, r2)

    def _find_nearest_cell(self, points):
        """
        Finds the closest grid cell for each point in a vectorized manner.
        """
        if self.nlevels == 1:
            pos = (points - self.ds.left_edge) / self.width
            n = (self.sizes[2] * pos[:,2]).astype('int32')
            n += self.sizes[2] * (self.sizes[1] * pos[:,1]).astype('int32')
            n += self.sizes[2] * self.sizes[1] * (self.sizes[0] * pos[:,0]).astype('int32')
        else:
            fKD.qv_many = points.T
            fKD.nn_tags = na.asfortranarray(na.empty((1, points.shape[0]), dtype='int64'))
            find_many_nn_nearest_neighbors()
            # The -1 is for fortran counting.
            n = fKD.nn_tags[0,:] - 1
        return n

    def _get_fields_vals(self, points):
        """
        Given points, return the values for the fields we need for those
        points.
        """
        # First find the grid data index field.
        indices = self._find_nearest_cell(points)
        results = na.empty((len(indices), len(self.fields)), dtype='float64')
        # Put the field values into the columns of results.
        for field in self.fields:
            col = self.fields_columns[field]
            results[:, col] = self.stored_fields[field][indices]
        return results


    def _eval_points(self, length):
        # We need to loop over the points array at least once. Further
        # iterations only happen if we have added new points to the array,
        # but not as many as we want to, so we need to check again to see if
        # we can put more points into the buffer.
        added_points = True
        while added_points:
            # If we need to, add more points to the points array.
            if self.generated_points < self.total_values:
                # Look for 'empty' slots to put in new pairs.
                select = (self.points[:,0] < 0)
                ssum = select.sum()
                # We'll generate only as many points as we need to/can.
                size = min(ssum, self.total_values - self.generated_points)
                (new_r1,new_r2) = self._pick_random_points(length, size)
                self.generated_points += size
                # If size != select.sum(), we need to pad the end of new_r1/r2
                # which is what is effectively happening below.
                newpoints = na.ones((ssum, 6), dtype='float64') * -1.
                newpoints[:size,:3] = new_r1
                newpoints[:size,3:] = new_r2
                # Now we insert them into self.points.
                self.points[select] = newpoints
            else:
                added_points = False
            
            # If we have an empty buffer here, we can skip everything below.
            if (self.points < 0).all():
                added_points = False # Not strictly required, but clearer.
                break
            
            # Now we have a points array that is either full of unevaluated points,
            # or I don't need to make any new points and I'm just processing the
            # array. Start by finding the indices of the points I own.
            self.points.shape = (self.comm_size*2, 3) # Doesn't make a copy - fast!
            select = na.bitwise_or((self.points < self.ds.left_edge).any(axis=1),
                (self.points >= self.ds.right_edge).any(axis=1))
            select = na.invert(select)
            mypoints = self.points[select]
            if mypoints.size > 0:
                # Get the fields values.
                results = self._get_fields_vals(mypoints)
                # Put this into self.fields_vals.
                self.fields_vals.shape = (self.comm_size*2, len(self.fields))
                self.fields_vals[select] = results
            
            # Put our arrays back into their original shapes cheaply!
            if mypoints.size > 0:
                self.fields_vals.shape = (self.comm_size, len(self.fields)*2)
            self.points.shape = (self.comm_size, 6)
            
            # To run the structure functions, what is key is that the
            # second point in the pair is ours.
            second_points = self.points[:,3:]
            select = na.bitwise_or((second_points < self.ds.left_edge).any(axis=1),
                (second_points >= self.ds.right_edge).any(axis=1))
            select = na.invert(select)
            if select.any():
                points_to_eval = self.points[select]
                fields_to_eval = self.fields_vals[select]
                
                # Find the normal vector between our points.
                vec = na.abs(points_to_eval[:,:3] - points_to_eval[:,3:])
                norm = na.sqrt(na.sum(na.multiply(vec,vec), axis=1))
                # I wish there was a better way to do this, but I can't find it.
                for i, n in enumerate(norm):
                    vec[i] = na.divide(vec[i], n)
                
                # Now evaluate the functions.
                for fcn_set in self._fsets:
                    fcn_results = fcn_set._eval_st_fcn(fields_to_eval,points_to_eval,
                        vec)
                    fcn_set._bin_results(length, fcn_results)
                
                # Now clear the buffers at the processed points.
                self.points[select] = na.array([-1.]*6, dtype='float64')
                
            else:
                # We didn't clear any points, so we should move on with our
                # lives and pass this buffer along.
                added_points = False

    @parallel_blocking_call
    def write_out_means(self):
        """
        Writes out the weighted-average value for each structure function for
        each dimension for each ruler length to a text file. The data is written
        to files of the name 'function_name.txt' in the current working
        directory.
        """
        sep = 12
        for fset in self._fsets:
            fp = self._write_on_root("%s.txt" % fset.function.__name__)
            fset._avg_bin_hits()
            line = "# length".ljust(sep)
            line += "count".ljust(sep)
            for dim in fset.dims:
                line += ("%s" % fset.out_labels[dim]).ljust(sep)
            fp.write(line + "\n")
            for length in self.lengths:
                line = ("%1.5e" % length).ljust(sep)
                line += ("%d" % fset.binned[length]).ljust(sep)
                for dim in fset.dims:
                    if fset.sqrt[dim]:
                        line += ("%1.5e" % \
                            math.sqrt(fset.length_avgs[length][dim])).ljust(sep)
                    else:
                        line += ("%1.5e" % \
                            fset.length_avgs[length][dim]).ljust(sep)
                line += "\n"
                fp.write(line)
            fp.close()
    
    @parallel_root_only
    def write_out_arrays(self):
        """
        Writes out the raw probability bins and the bin edges to an HDF5 file
        for each of the structure functions. The files are named 
        'function_name.txt' and saved in the current working directory.
        """
        if self.mine == 0:
            for fset in self._fsets:
                f = h5py.File("%s.h5" % fset.function.__name__, "w")
                bin_names = []
                prob_names = []
                bin_counts = []
                for dim in fset.dims:
                    f.create_dataset("/bin_edges_%02d_%s" % \
                        (dim, fset.out_labels[dim]), \
                        data=fset.bin_edges[dim])
                    bin_names.append("/bin_edges_%02d_%s" % \
                    (dim, fset.out_labels[dim]))
                for i,length in enumerate(self.lengths):
                    f.create_dataset("/prob_bins_%05d" % i, \
                        data=fset.length_bin_hits[length])
                    prob_names.append("/prob_bins_%05d" % i)
                    bin_counts.append([fset.too_low.sum(), fset.binned[length],
                        fset.too_high.sum()])
                f.create_dataset("/bin_edges_names", data=bin_names)
                #f.create_dataset("/prob_names", data=prob_names)
                f.create_dataset("/lengths", data=self.lengths)
                f.create_dataset("/counts", data=bin_counts)
                f.close()

class StructSet(StructFcnGen):
    def __init__(self,sfg, function, min_edge, out_labels, sqrt):
        self.sfg = sfg # The overarching SFG class
        self.function = function # Function to eval between the two points.
        self.min_edge = min_edge # The length of the minimum edge of the box.
        self.out_labels = out_labels # For output.
        self.sqrt = sqrt # which columns to sqrt on output.
        # These below are used to track how many times the function returns
        # unbinned results.
        self.too_low = na.zeros(len(self.out_labels), dtype='int32')
        self.too_high = na.zeros(len(self.out_labels), dtype='int32')
        
    def set_pdf_params(self, bin_type="lin", bin_number=1000, bin_range=None):
        """
        Set the parameters used to build the Probability Distribution Function
        for each ruler length for this function. The values output by the
        function are slotted into the bins described here.
        *bin_type* (str) controls the edges of the bins spaced evenly in
        logarithmic or linear space, set by "log" or "lin", respectively.
        A single string, or list of strings for N-dim binning. Default: "lin".
        *bin_number* (int) sets how many bins to create, evenly spaced by the above
        parameter.
        A single integer, or a list of integers for N-dim binning. Default: 1000
        *bin_range* (float) A pair of values giving the range for the bins.
        A pair of floats (a list), or a list of pairs for N-dim binning.
        Default: None.
        """
        # This should be called after setSearchParams.
        if not hasattr(self.sfg, "lengths"):
            mylog.error("Please call setSearchParams() before calling setPDFParams().")
            return None
        # Make sure they're either all lists or only one is.
        input = [bin_type, bin_number, bin_range]
        lists = 0
        for thing in input:
            if type(thing) == types.ListType:
                lists += 1
        if lists > 1 and lists < 3:
            mylog.error("Either all the inputs need to be lists, or only one.")
            return None
        # Make sure they're all the same length if they're lists.
        if lists == 3:
            first_len = 0
            for thing in input:
                if first_len == 0:
                    first_len = len(thing)
                    if first_len == 0:
                        mylog.error("Input cannot be an empty list.")
                        return None
                    continue
                if first_len != len(thing):
                    mylog.error("All the inputs need to have the same length.")
                    return None
        # If they are not all lists, put the input into lists for convenience.
        if lists == 1:
            bin_type, bin_number = [bin_type], [bin_number]
            bin_range = [bin_range]
        self.bin_type = bin_type
        self.bin_number = na.array(bin_number) - 1
        self.dims = range(len(bin_type))
        # Create the dict that stores the arrays to store the bin hits, and
        # the arrays themselves.
        self.length_bin_hits = {}
        for length in self.sfg.lengths:
            # It's easier to index flattened, but will be unflattened later.
            self.length_bin_hits[length] = na.zeros(self.bin_number,
                dtype='int64').flatten()
        # Create the bin edges for each dimension.
        # self.bins is indexed by dimension
        self.bin_edges = {}
        for dim in self.dims:
            # Error check.
            if len(bin_range[dim]) != 2:
                raise ValueError("bin_range must have two values.")
            if bin_range[dim][1] <= bin_range[dim][0]:
                raise ValueError("bin_range[1] must be larger than bin_range[0]")
            # Make the edges for this dimension.
            if bin_type[dim] == "lin":
                self.bin_edges[dim] = na.linspace(bin_range[dim][0], bin_range[dim][1],
                    bin_number[dim])
            elif bin_type[dim] == "log":
                self.bin_edges[dim] = na.logspace(math.log10(bin_range[dim][0]),
                    math.log10(bin_range[dim][1]), bin_number[dim])
            else:
                raise SyntaxError("bin_edges is either \"lin\" or \"log\".")

    def _eval_st_fcn(self, results, points, vec):
        """
        Return the value of the structure function using the provided results.
        """
        return self.function(results[:,:len(self.sfg.fields)],
            results[:,len(self.sfg.fields):], points[:,:3], points[:,3:], vec)
        """
        NOTE - A function looks like:
        def stuff(a,b,r1,r2, vec):
            return [(a[0] - b[0])/(a[1] + b[1])]
        where a and b refer to different points in space and the indices
        are for the different fields, which are given when the function is
        added. The results need to be a list or array even if it's only one
        item.
        """

    def _bin_results(self, length, results):
        """
        Add hits to the bins corresponding to these results. length_hit_bins
        is flattened, so we need to figure out the offset for this hit by
        factoring the sizes of the other dimensions.
        """
        hit_bin = na.zeros(results.shape[0], dtype='int64')
        multi = 1
        good = na.ones(results.shape[0], dtype='bool')
        for dim in range(len(self.out_labels)):
            for d1 in range(dim):
                multi *= self.bin_edges[d1].size
            if dim == 0 and len(self.out_labels)==1:
                digi = na.digitize(results, self.bin_edges[dim])
            else:
                digi = na.digitize(results[:,dim], self.bin_edges[dim])
            too_low = (digi == 0)
            too_high = (digi == self.bin_edges[dim].size)
            self.too_low[dim] += (too_low).sum()
            self.too_high[dim] += (too_high).sum()
            newgood = na.bitwise_and(na.invert(too_low), na.invert(too_high))
            good = na.bitwise_and(good, newgood)
            hit_bin += na.multiply((digi - 1), multi)
        digi_bins = na.arange(self.length_bin_hits[length].size+1)
        hist, digi_bins = na.histogram(hit_bin[good], digi_bins)
        self.length_bin_hits[length] += hist

    def _dim_sum(self, a, dim):
        """
        Given a multidimensional array a, this finds the sum over all the
        elements leaving the dimension dim untouched.
        """
        dims = na.arange(len(a.shape))
        dims = na.flipud(dims)
        gt_dims = dims[dims > dim]
        lt_dims = dims[dims < dim]
        iter_dims = na.concatenate((gt_dims, lt_dims))
        for this_dim in iter_dims:
            a = a.sum(axis=this_dim)
        return a

    def _avg_bin_hits(self):
        """
        For each dimension and length of bin_hits return the weighted average.
        """
        self.length_avgs = defaultdict(dict)
        for length in self.sfg.lengths:
            for dim in self.dims:
                self.length_avgs[length][dim] = \
                    (self._dim_sum(self.length_bin_hits[length], dim) * \
                    ((self.bin_edges[dim][:-1] + self.bin_edges[dim][1:]) / 2.)).sum()
