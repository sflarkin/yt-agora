"""
Profile classes, to deal with generating and obtaining profiles



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import h5py
import numpy as np

from yt.funcs import *

from yt.data_objects.data_containers import YTFieldData
from yt.utilities.lib import bin_profile1d, bin_profile2d, bin_profile3d
from yt.utilities.lib import new_bin_profile1d, new_bin_profile2d, \
                             new_bin_profile3d
from yt.utilities.parallel_tools.parallel_analysis_interface import \
    ParallelAnalysisInterface, parallel_objects

def preserve_source_parameters(func):
    def save_state(*args, **kwargs):
        # Temporarily replace the 'field_parameters' for a
        # grid with the 'field_parameters' for the data source
        prof = args[0]
        source = args[1]
        if hasattr(source, 'field_parameters'):
            old_params = source.field_parameters
            source.field_parameters = prof._data_source.field_parameters
            tr = func(*args, **kwargs)
            source.field_parameters = old_params
        else:
            tr = func(*args, **kwargs)
        return tr
    return save_state

# Note we do not inherit from EnzoData.
# We could, but I think we instead want to deal with the root datasource.
class BinnedProfile(ParallelAnalysisInterface):
    def __init__(self, data_source, lazy_reader):
        ParallelAnalysisInterface.__init__(self)
        self._data_source = data_source
        self.pf = data_source.pf
        self.field_data = YTFieldData()
        self._lazy_reader = lazy_reader

    @property
    def hierarchy(self):
        return self.pf.hierarchy

    def _get_dependencies(self, fields):
        return ParallelAnalysisInterface._get_dependencies(
                    self, fields + self._get_bin_fields())

    def _initialize_parallel(self, fields):
        g_objs = [g for g in self._get_grid_objs()]
        self.comm.preload(g_objs, self.get_dependencies(fields),
                      self._data_source.hierarchy.io)

    def _lazy_add_fields(self, fields, weight, accumulation):
        self._ngrids = 0
        self.__data = {}         # final results will go here
        self.__weight_data = {}  # we need to track the weights as we go
        self.__std_data = {}
        for field in fields:
            self.__data[field] = self._get_empty_field()
            self.__weight_data[field] = self._get_empty_field()
            self.__std_data[field] = self._get_empty_field()
        self.__used = self._get_empty_field().astype('bool')
        #pbar = get_pbar('Binning grids', len(self._data_source._grids))
        for gi,grid in enumerate(self._get_grids(fields)):
            self._ngrids += 1
            #pbar.update(gi)
            try:
                args = self._get_bins(grid, check_cut=True)
            except YTEmptyProfileData:
                # No bins returned for this grid, so forget it!
                continue
            for field in fields:
                # We get back field values, weight values, used bins
                f, w, q, u = self._bin_field(grid, field, weight, accumulation,
                                          args=args, check_cut=True)
                self.__data[field] += f        # running total
                self.__weight_data[field] += w # running total
                self.__std_data[field][u] += w[u] * (q[u]/w[u] + \
                    (f[u]/w[u] -
                     self.__data[field][u]/self.__weight_data[field][u])**2) # running total
                self.__used = (self.__used | u)       # running 'or'
            grid.clear_data()
        # When the loop completes the parallel finalizer gets called
        #pbar.finish()
        ub = np.where(self.__used)
        for field in fields:
            if weight: # Now, at the end, we divide out.
                self.__data[field][ub] /= self.__weight_data[field][ub]
                self.__std_data[field][ub] /= self.__weight_data[field][ub]
            self[field] = self.__data[field]
            self["%s_std" % field] = np.sqrt(self.__std_data[field])
        self["UsedBins"] = self.__used
        del self.__data, self.__std_data, self.__weight_data, self.__used

    def _finalize_parallel(self):
        my_mean = {}
        my_weight = {}
        for key in self.__data:
            my_mean[key] = self._get_empty_field()
            my_weight[key] = self._get_empty_field()
        ub = np.where(self.__used)
        for key in self.__data:
            my_mean[key][ub] = self.__data[key][ub] / self.__weight_data[key][ub]
            my_weight[key][ub] = self.__weight_data[key][ub]
        for key in self.__data:
            self.__data[key] = self.comm.mpi_allreduce(self.__data[key], op='sum')
        for key in self.__weight_data:
            self.__weight_data[key] = self.comm.mpi_allreduce(self.__weight_data[key], op='sum')
        for key in self.__std_data:
            self.__std_data[key][ub] = my_weight[key][ub] * (self.__std_data[key][ub] / my_weight[key][ub] + \
                (my_mean[key][ub] - self.__data[key][ub]/self.__weight_data[key][ub])**2)
            self.__std_data[key] = self.comm.mpi_allreduce(self.__std_data[key], op='sum')
        self.__used = self.comm.mpi_allreduce(self.__used, op='sum')

    def _unlazy_add_fields(self, fields, weight, accumulation):
        for field in fields:
            f, w, q, u = self._bin_field(self._data_source, field, weight,
                                         accumulation, self._args, check_cut = False)
            if weight:
                f[u] /= w[u]
                q[u] = np.sqrt(q[u] / w[u])
            self[field] = f
            self["%s_std" % field] = q
        self["UsedBins"] = u

    def add_fields(self, fields, weight = "CellMassMsun", accumulation = False, fractional=False):
        """
        We accept a list of *fields* which will be binned if *weight* is not
        None and otherwise summed.  *accumulation* determines whether or not
        they will be accumulated from low to high along the appropriate axes.
        """
        # Note that the specification has to be the same for all of these
        fields = ensure_list(fields)
        if self._lazy_reader:
            self._lazy_add_fields(fields, weight, accumulation)
        else:
            self._unlazy_add_fields(fields, weight, accumulation)
        if fractional:
            for field in fields:
                self.field_data[field] /= self.field_data[field].sum()

    def keys(self):
        return self.field_data.keys()

    def __getitem__(self, key):
        # This raises a KeyError if it doesn't exist
        # This is because we explicitly want to add all fields
        return self.field_data[key]

    def __setitem__(self, key, value):
        self.field_data[key] = value

    def _get_field(self, source, field, check_cut):
        # This is where we will iterate to get all contributions to a field
        # which is how we will implement hybrid particle/cell fields
        # but...  we default to just the field.
        data = []
        pointI = None
        if check_cut:
            # This conditional is so that we can have variable-length
            # particle fields.  Note that we can't apply the
            # is_fully_enclosed to baryon fields, because child cells get
            # in the way.
            if field in self.pf.field_info \
                and self.pf.field_info[field].particle_type:
                if not self._data_source._is_fully_enclosed(source):
                    pointI = self._data_source._get_particle_indices(source)
            else:
                pointI = self._data_source._get_point_indices(source)
        data.append(source[field][pointI].ravel().astype('float64'))
        return np.concatenate(data, axis=0)

    def _fix_pickle(self):
        if isinstance(self._data_source, tuple):
            self._data_source = self._data_source[1]

# @todo: Fix accumulation with overriding
class BinnedProfile1D(BinnedProfile):
    """
    A 'Profile' produces either a weighted (or unweighted) average or a
    straight sum of a field in a bin defined by another field.  In the case
    of a weighted average, we have: p_i = sum( w_i * v_i ) / sum(w_i)

    We accept a *data_source*, which will be binned into *n_bins*
    by the field *bin_field* between the *lower_bound* and the
    *upper_bound*.  These bins may or may not be equally divided
    in *log_space*, and the *lazy_reader* flag controls whether we
    use a memory conservative approach. If *end_collect* is True,
    take all values outside the given bounds and store them in the
    0 and *n_bins*-1 values.
    """
    def __init__(self, data_source, n_bins, bin_field,
                 lower_bound, upper_bound,
                 log_space = True, lazy_reader=False,
                 end_collect=False):
        BinnedProfile.__init__(self, data_source, lazy_reader)
        self.bin_field = bin_field
        self._x_log = log_space
        self.end_collect = end_collect
        self.n_bins = n_bins

        # Get our bins
        if log_space:
            func = np.logspace
            lower_bound, upper_bound = np.log10(lower_bound), np.log10(upper_bound)
        else:
            func = np.linspace

        # These are the bin *edges*
        self._bins = func(lower_bound, upper_bound, n_bins + 1)

        # These are the bin *left edges*.  These are the x-axis values
        # we plot in the PlotCollection
        self[bin_field] = self._bins

        # If we are not being memory-conservative, grab all the bins
        # and the inverse indices right now.
        if not lazy_reader:
            self._args = self._get_bins(data_source)

    def _get_empty_field(self):
        return np.zeros(self[self.bin_field].size, dtype='float64')

    @preserve_source_parameters
    def _bin_field(self, source, field, weight, accumulation,
                   args, check_cut=False):
        mi, inv_bin_indices = args # Args has the indices to use as input
        # check_cut is set if source != self._data_source
        # (i.e., lazy_reader)
        source_data = self._get_field(source, field, check_cut)
        if weight: weight_data = self._get_field(source, weight, check_cut)
        else: weight_data = np.ones(source_data.shape, dtype='float64')
        self.total_stuff = source_data.sum()
        binned_field = self._get_empty_field()
        weight_field = self._get_empty_field()
        m_field = self._get_empty_field()
        q_field = self._get_empty_field()
        used_field = self._get_empty_field()
        mi = args[0]
        bin_indices_x = args[1].ravel().astype('int64')
        source_data = source_data[mi]
        weight_data = weight_data[mi]
        bin_profile1d(bin_indices_x, weight_data, source_data,
                      weight_field, binned_field,
                      m_field, q_field, used_field)
        # Fix for laziness, because at the *end* we will be
        # summing up all of the histograms and dividing by the
        # weights.  Accumulation likely doesn't work with weighted
        # average fields.
        if accumulation: 
            binned_field = np.add.accumulate(binned_field)
        return binned_field, weight_field, q_field, \
            used_field.astype("bool")

    @preserve_source_parameters
    def _get_bins(self, source, check_cut=False):
        source_data = self._get_field(source, self.bin_field, check_cut)
        if source_data.size == 0: # Nothing for us here.
            raise YTEmptyProfileData()
        # Truncate at boundaries.
        if self.end_collect:
            mi = np.ones_like(source_data).astype('bool')
        else:
            mi = ((source_data > self._bins.min())
               &  (source_data < self._bins.max()))
        sd = source_data[mi]
        if sd.size == 0:
            raise YTEmptyProfileData()
        # Stick the bins into our fixed bins, set at initialization
        bin_indices = np.digitize(sd, self._bins)
        if self.end_collect: #limit the range of values to 0 and n_bins-1
            bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        else: #throw away outside values
            bin_indices -= 1
          
        return (mi, bin_indices)

    def choose_bins(self, bin_style):
        # Depending on the bin_style, choose from bin edges 0...N either:
        # both: 0...N, left: 0...N-1, right: 1...N 
        # center: N bins that are the average (both in linear or log
        # space) of each pair of left/right edges
        x = self.field_data[self.bin_field]
        if bin_style is 'both': pass
        elif bin_style is 'left': x = x[:-1]
        elif bin_style is 'right': x = x[1:]
        elif bin_style is 'center':
            if self._x_log: x=np.log10(x)
            x = 0.5*(x[:-1] + x[1:])
            if self._x_log: x=10**x
        else:
            mylog.error('Did not recognize bin_style')
            raise ValueError
        return x

    def write_out(self, filename, format="%0.16e", bin_style='left'):
        ''' 
        Write out data in ascii file, using *format* and
        *bin_style* (left, right, center, both).
        '''
        fid = open(filename,"w")
        fields = [field for field in sorted(self.field_data.keys()) if field != "UsedBins"]
        fields.remove(self.bin_field)
        fid.write("\t".join(["#"] + [self.bin_field] + fields + ["\n"]))

        field_data = np.array(self.choose_bins(bin_style)) 
        if bin_style is 'both':
            field_data = np.append([field_data], np.array([self.field_data[field] for field in fields]), axis=0)
        else: 
            field_data = np.append([field_data], np.array([self.field_data[field][:-1] for field in fields]), axis=0)
        
        for line in range(field_data.shape[1]):
            field_data[:,line].tofile(fid, sep="\t", format=format)
            fid.write("\n")
        fid.close()

    def write_out_h5(self, filename, group_prefix=None, bin_style='left'):
        """
        Write out data in an hdf5 file *filename*.  Each profile is
        put into a group, named by the axis fields.  Optionally a
        *group_prefix* can be prepended to the group name.  If the
        group already exists, it will delete and replace.  However,
        due to hdf5 functionality, in only unlinks the data, so an
        h5repack may be necessary to conserve space.  Axes values are
        saved in group attributes.  Bins will be saved based on
        *bin_style* (left, right, center, both).
        """
        fid = h5py.File(filename)
        fields = [field for field in sorted(self.field_data.keys()) if (field != "UsedBins" and field != self.bin_field)]
        if group_prefix is None:
            name = "%s-1d" % (self.bin_field)
        else:
            name = "%s-%s-1d" % (group_prefix, self.bin_field)
            
        if name in fid: 
            mylog.info("Profile file is getting larger since you are attempting to overwrite a profile. You may want to repack")
            del fid[name] 
        group = fid.create_group(name)
        group.attrs["x-axis-%s" % self.bin_field] = self.choose_bins(bin_style)
        for field in fields:
            dset = group.create_dataset("%s" % field, data=self.field_data[field][:-1])
        fid.close()

    def _get_bin_fields(self):
        return [self.bin_field]

class BinnedProfile2D(BinnedProfile):
    """
    A 'Profile' produces either a weighted (or unweighted) average
    or a straight sum of a field in a bin defined by two other
    fields.  In the case of a weighted average, we have: p_i =
    sum( w_i * v_i ) / sum(w_i)

    We accept a *data_source*, which will be binned into
    *x_n_bins* by the field *x_bin_field* between the
    *x_lower_bound* and the *x_upper_bound* and then again binned
    into *y_n_bins* by the field *y_bin_field* between the
    *y_lower_bound* and the *y_upper_bound*.  These bins may or
    may not be equally divided in log-space as specified by
    *x_log* and *y_log*, and the *lazy_reader* flag controls
    whether we use a memory conservative approach. If
    *end_collect* is True, take all values outside the given
    bounds and store them in the 0 and *n_bins*-1 values.
    """
    def __init__(self, data_source,
                 x_n_bins, x_bin_field, x_lower_bound, x_upper_bound, x_log,
                 y_n_bins, y_bin_field, y_lower_bound, y_upper_bound, y_log,
                 lazy_reader=False, end_collect=False):
        BinnedProfile.__init__(self, data_source, lazy_reader)
        self.x_bin_field = x_bin_field
        self.y_bin_field = y_bin_field
        self._x_log = x_log
        self._y_log = y_log
        self.end_collect = end_collect
        self.x_n_bins = x_n_bins
        self.y_n_bins = y_n_bins

        func = {True:np.logspace, False:np.linspace}[x_log]
        bounds = fix_bounds(x_lower_bound, x_upper_bound, x_log)
        self._x_bins = func(bounds[0], bounds[1], x_n_bins + 1)
        self[x_bin_field] = self._x_bins

        func = {True:np.logspace, False:np.linspace}[y_log]
        bounds = fix_bounds(y_lower_bound, y_upper_bound, y_log)
        self._y_bins = func(bounds[0], bounds[1], y_n_bins + 1)
        self[y_bin_field] = self._y_bins

        if np.any(np.isnan(self[x_bin_field])) \
            or np.any(np.isnan(self[y_bin_field])):
            mylog.error("Your min/max values for x, y have given me a nan.")
            mylog.error("Usually this means you are asking for log, with a zero bound.")
            raise ValueError
        if not lazy_reader:
            self._args = self._get_bins(data_source)

    def _get_empty_field(self):
        return np.zeros((self[self.x_bin_field].size,
                         self[self.y_bin_field].size), dtype='float64')

    @preserve_source_parameters
    def _bin_field(self, source, field, weight, accumulation,
                   args, check_cut=False):
        source_data = self._get_field(source, field, check_cut)
        if weight: weight_data = self._get_field(source, weight, check_cut)
        else: weight_data = np.ones(source_data.shape, dtype='float64')
        self.total_stuff = source_data.sum()
        binned_field = self._get_empty_field()
        weight_field = self._get_empty_field()
        m_field = self._get_empty_field()
        q_field = self._get_empty_field()
        used_field = self._get_empty_field()
        mi = args[0]
        bin_indices_x = args[1].ravel().astype('int64')
        bin_indices_y = args[2].ravel().astype('int64')
        source_data = source_data[mi]
        weight_data = weight_data[mi]
        nx = bin_indices_x.size
        #mylog.debug("Binning %s / %s times", source_data.size, nx)
        bin_profile2d(bin_indices_x, bin_indices_y, weight_data, source_data,
                      weight_field, binned_field, m_field, q_field, used_field)
        if accumulation: # Fix for laziness
            if not iterable(accumulation):
                raise SyntaxError("Accumulation needs to have length 2")
            if accumulation[0]:
                binned_field = np.add.accumulate(binned_field, axis=0)
            if accumulation[1]:
                binned_field = np.add.accumulate(binned_field, axis=1)
        return binned_field, weight_field, q_field, \
            used_field.astype("bool")

    @preserve_source_parameters
    def _get_bins(self, source, check_cut=False):
        source_data_x = self._get_field(source, self.x_bin_field, check_cut)
        source_data_y = self._get_field(source, self.y_bin_field, check_cut)
        if source_data_x.size == 0:
            raise YTEmptyProfileData()

        if self.end_collect:
            mi = np.arange(source_data_x.size)
        else:
            mi = np.where( (source_data_x > self._x_bins.min())
                           & (source_data_x < self._x_bins.max())
                           & (source_data_y > self._y_bins.min())
                           & (source_data_y < self._y_bins.max()))
        sd_x = source_data_x[mi]
        sd_y = source_data_y[mi]
        if sd_x.size == 0 or sd_y.size == 0:
            raise YTEmptyProfileData()

        bin_indices_x = np.digitize(sd_x, self._x_bins) - 1
        bin_indices_y = np.digitize(sd_y, self._y_bins) - 1
        if self.end_collect:
            bin_indices_x = np.minimum(np.maximum(1, bin_indices_x), self.x_n_bins) - 1
            bin_indices_y = np.minimum(np.maximum(1, bin_indices_y), self.y_n_bins) - 1

        # Now we set up our inverse bin indices
        return (mi, bin_indices_x, bin_indices_y)

    def choose_bins(self, bin_style):
        # Depending on the bin_style, choose from bin edges 0...N either:
        # both: 0...N, left: 0...N-1, right: 1...N 
        # center: N bins that are the average (both in linear or log
        # space) of each pair of left/right edges

        x = self.field_data[self.x_bin_field]
        y = self.field_data[self.y_bin_field]
        if bin_style is 'both':
            pass
        elif bin_style is 'left':
            x = x[:-1]
            y = y[:-1]
        elif bin_style is 'right':
            x = x[1:]
            y = y[1:]
        elif bin_style is 'center':
            if self._x_log: x=np.log10(x)
            if self._y_log: y=np.log10(y)
            x = 0.5*(x[:-1] + x[1:])
            y = 0.5*(y[:-1] + y[1:])
            if self._x_log: x=10**x
            if self._y_log: y=10**y
        else:
            mylog.error('Did not recognize bin_style')
            raise ValueError

        return x,y

    def write_out(self, filename, format="%0.16e", bin_style='left'):
        """
        Write out the values of x,y,v in ascii to *filename* for every
        field in the profile.  Optionally a *format* can be specified.
        Bins will be saved based on *bin_style* (left, right, center,
        both).
        """
        fid = open(filename,"w")
        fields = [field for field in sorted(self.field_data.keys()) if field != "UsedBins"]
        fid.write("\t".join(["#"] + [self.x_bin_field, self.y_bin_field]
                          + fields + ["\n"]))
        x,y = self.choose_bins(bin_style)
        x,y = np.meshgrid(x,y)
        field_data = [x.ravel(), y.ravel()]
        if bin_style is not 'both':
            field_data += [self.field_data[field][:-1,:-1].ravel() for field in fields
                           if field not in [self.x_bin_field, self.y_bin_field]]
        else:
            field_data += [self.field_data[field].ravel() for field in fields
                           if field not in [self.x_bin_field, self.y_bin_field]]

        field_data = np.array(field_data)
        for line in range(field_data.shape[1]):
            field_data[:,line].tofile(fid, sep="\t", format=format)
            fid.write("\n")
        fid.close()

    def write_out_h5(self, filename, group_prefix=None, bin_style='left'):
        """
        Write out data in an hdf5 file.  Each profile is put into a
        group, named by the axis fields.  Optionally a group_prefix
        can be prepended to the group name.  If the group already
        exists, it will delete and replace.  However, due to hdf5
        functionality, in only unlinks the data, so an h5repack may be
        necessary to conserve space.  Axes values are saved in group
        attributes. Bins will be saved based on *bin_style* (left,
        right, center, both).
        """
        fid = h5py.File(filename)
        fields = [field for field in sorted(self.field_data.keys()) if (field != "UsedBins" and field != self.x_bin_field and field != self.y_bin_field)]
        if group_prefix is None:
            name = "%s-%s-2d" % (self.y_bin_field, self.x_bin_field)
        else:
            name = "%s-%s-%s-2d" % (group_prefix, self.y_bin_field, self.x_bin_field)
        if name in fid: 
            mylog.info("Profile file is getting larger since you are attempting to overwrite a profile. You may want to repack")
            del fid[name] 
        group = fid.create_group(name)

        xbins, ybins = self.choose_bins(bin_style)
        group.attrs["x-axis-%s" % self.x_bin_field] = xbins
        group.attrs["y-axis-%s" % self.y_bin_field] = ybins
        for field in fields:
            dset = group.create_dataset("%s" % field, data=self.field_data[field][:-1,:-1])
        fid.close()

    def _get_bin_fields(self):
        return [self.x_bin_field, self.y_bin_field]

def fix_bounds(upper, lower, logit):
    if logit: return np.log10(upper), np.log10(lower)
    return upper, lower

class BinnedProfile3D(BinnedProfile):
    """
    A 'Profile' produces either a weighted (or unweighted) average
    or a straight sum of a field in a bin defined by two other
    fields.  In the case of a weighted average, we have: p_i =
    sum( w_i * v_i ) / sum(w_i)
    
    We accept a *data_source*, which will be binned into
    *(x,y,z)_n_bins* by the field *(x,y,z)_bin_field* between the
    *(x,y,z)_lower_bound* and the *(x,y,z)_upper_bound*.  These bins may or
    may not be equally divided in log-space as specified by
    *(x,y,z)_log*, and the *lazy_reader* flag controls
    whether we use a memory conservative approach. If
    *end_collect* is True, take all values outside the given
    bounds and store them in the 0 and *n_bins*-1 values.
    """
    def __init__(self, data_source,
                 x_n_bins, x_bin_field, x_lower_bound, x_upper_bound, x_log,
                 y_n_bins, y_bin_field, y_lower_bound, y_upper_bound, y_log,
                 z_n_bins, z_bin_field, z_lower_bound, z_upper_bound, z_log,
                 lazy_reader=False, end_collect=False):
        BinnedProfile.__init__(self, data_source, lazy_reader)
        self.x_bin_field = x_bin_field
        self.y_bin_field = y_bin_field
        self.z_bin_field = z_bin_field
        self._x_log = x_log
        self._y_log = y_log
        self._z_log = z_log
        self.end_collect = end_collect
        self.x_n_bins = x_n_bins
        self.y_n_bins = y_n_bins
        self.z_n_bins = z_n_bins

        func = {True:np.logspace, False:np.linspace}[x_log]
        bounds = fix_bounds(x_lower_bound, x_upper_bound, x_log)
        self._x_bins = func(bounds[0], bounds[1], x_n_bins + 1)
        self[x_bin_field] = self._x_bins

        func = {True:np.logspace, False:np.linspace}[y_log]
        bounds = fix_bounds(y_lower_bound, y_upper_bound, y_log)
        self._y_bins = func(bounds[0], bounds[1], y_n_bins + 1)
        self[y_bin_field] = self._y_bins

        func = {True:np.logspace, False:np.linspace}[z_log]
        bounds = fix_bounds(z_lower_bound, z_upper_bound, z_log)
        self._z_bins = func(bounds[0], bounds[1], z_n_bins + 1)
        self[z_bin_field] = self._z_bins

        if np.any(np.isnan(self[x_bin_field])) \
            or np.any(np.isnan(self[y_bin_field])) \
            or np.any(np.isnan(self[z_bin_field])):
            mylog.error("Your min/max values for x, y or z have given me a nan.")
            mylog.error("Usually this means you are asking for log, with a zero bound.")
            raise ValueError
        if not lazy_reader:
            self._args = self._get_bins(data_source)

    def _get_empty_field(self):
        return np.zeros((self[self.x_bin_field].size,
                         self[self.y_bin_field].size,
                         self[self.z_bin_field].size), dtype='float64')

    @preserve_source_parameters
    def _bin_field(self, source, field, weight, accumulation,
                   args, check_cut=False):
        source_data = self._get_field(source, field, check_cut)
        weight_data = np.ones(source_data.shape).astype('float64')
        if weight: weight_data = self._get_field(source, weight, check_cut)
        else: weight_data = np.ones(source_data.shape).astype('float64')
        self.total_stuff = source_data.sum()
        binned_field = self._get_empty_field()
        weight_field = self._get_empty_field()
        m_field = self._get_empty_field()
        q_field = self._get_empty_field()
        used_field = self._get_empty_field()
        mi = args[0]
        bin_indices_x = args[1].ravel().astype('int64')
        bin_indices_y = args[2].ravel().astype('int64')
        bin_indices_z = args[3].ravel().astype('int64')
        source_data = source_data[mi]
        weight_data = weight_data[mi]
        bin_profile3d(bin_indices_x, bin_indices_y, bin_indices_z,
                      weight_data, source_data, weight_field, binned_field,
                      m_field, q_field, used_field)
        if accumulation: # Fix for laziness
            if not iterable(accumulation):
                raise SyntaxError("Accumulation needs to have length 2")
            if accumulation[0]:
                binned_field = np.add.accumulate(binned_field, axis=0)
            if accumulation[1]:
                binned_field = np.add.accumulate(binned_field, axis=1)
            if accumulation[2]:
                binned_field = np.add.accumulate(binned_field, axis=2)
        return binned_field, weight_field, q_field, \
            used_field.astype("bool")

    @preserve_source_parameters
    def _get_bins(self, source, check_cut=False):
        source_data_x = self._get_field(source, self.x_bin_field, check_cut)
        source_data_y = self._get_field(source, self.y_bin_field, check_cut)
        source_data_z = self._get_field(source, self.z_bin_field, check_cut)
        if source_data_x.size == 0:
            raise YTEmptyProfileData()
        if self.end_collect:
            mi = np.arange(source_data_x.size)
        else:
            mi = ( (source_data_x > self._x_bins.min())
                 & (source_data_x < self._x_bins.max())
                 & (source_data_y > self._y_bins.min())
                 & (source_data_y < self._y_bins.max())
                 & (source_data_z > self._z_bins.min())
                 & (source_data_z < self._z_bins.max()))
        sd_x = source_data_x[mi]
        sd_y = source_data_y[mi]
        sd_z = source_data_z[mi]
        if sd_x.size == 0 or sd_y.size == 0 or sd_z.size == 0:
            raise YTEmptyProfileData()

        bin_indices_x = np.digitize(sd_x, self._x_bins) - 1
        bin_indices_y = np.digitize(sd_y, self._y_bins) - 1
        bin_indices_z = np.digitize(sd_z, self._z_bins) - 1
        if self.end_collect:
            bin_indices_x = np.minimum(np.maximum(1, bin_indices_x), self.x_n_bins) - 1
            bin_indices_y = np.minimum(np.maximum(1, bin_indices_y), self.y_n_bins) - 1
            bin_indices_z = np.minimum(np.maximum(1, bin_indices_z), self.z_n_bins) - 1

        # Now we set up our inverse bin indices
        return (mi, bin_indices_x, bin_indices_y, bin_indices_z)

    def choose_bins(self, bin_style):
        # Depending on the bin_style, choose from bin edges 0...N either:
        # both: 0...N, left: 0...N-1, right: 1...N 
        # center: N bins that are the average (both in linear or log
        # space) of each pair of left/right edges

        x = self.field_data[self.x_bin_field]
        y = self.field_data[self.y_bin_field]
        z = self.field_data[self.z_bin_field]
        if bin_style is 'both':
            pass
        elif bin_style is 'left':
            x = x[:-1]
            y = y[:-1]
            z = z[:-1]
        elif bin_style is 'right':
            x = x[1:]
            y = y[1:]
            z = z[1:]
        elif bin_style is 'center':
            if self._x_log: x=np.log10(x)
            if self._y_log: y=np.log10(y)
            if self._z_log: z=np.log10(z)
            x = 0.5*(x[:-1] + x[1:])
            y = 0.5*(y[:-1] + y[1:])
            z = 0.5*(z[:-1] + z[1:])
            if self._x_log: x=10**x
            if self._y_log: y=10**y
            if self._z_log: y=10**z
        else:
            mylog.error('Did not recognize bin_style')
            raise ValueError

        return x,y,z

    def write_out(self, filename, format="%0.16e"):
        pass # Will eventually dump HDF5

    def write_out_h5(self, filename, group_prefix=None, bin_style='left'):
        """
        Write out data in an hdf5 file.  Each profile is put into a
        group, named by the axis fields.  Optionally a group_prefix
        can be prepended to the group name.  If the group already
        exists, it will delete and replace.  However, due to hdf5
        functionality, in only unlinks the data, so an h5repack may be
        necessary to conserve space.  Axes values are saved in group
        attributes.
        """
        fid = h5py.File(filename)
        fields = [field for field in sorted(self.field_data.keys()) 
                  if (field != "UsedBins" and field != self.x_bin_field and field != self.y_bin_field and field != self.z_bin_field)]
        if group_prefix is None:
            name = "%s-%s-%s-3d" % (self.z_bin_field, self.y_bin_field, self.x_bin_field)
        else:
            name = "%s-%s-%s-%s-3d" % (group_prefix,self.z_bin_field, self.y_bin_field, self.x_bin_field)

        if name in fid: 
            mylog.info("Profile file is getting larger since you are attempting to overwrite a profile. You may want to repack")
            del fid[name]
        group = fid.create_group(name)

        xbins, ybins, zbins= self.choose_bins(bin_style)
        group.attrs["x-axis-%s" % self.x_bin_field] = xbins
        group.attrs["y-axis-%s" % self.y_bin_field] = ybins
        group.attrs["z-axis-%s" % self.z_bin_field] = zbins
        
        for field in fields:
            dset = group.create_dataset("%s" % field, data=self.field_data[field][:-1,:-1,:-1])
        fid.close()


    def _get_bin_fields(self):
        return [self.x_bin_field, self.y_bin_field, self.z_bin_field]

    def store_profile(self, name, force=False):
        """
        By identifying the profile with a fixed, user-input *name* we can
        store it in the serialized data section of the hierarchy file.  *force*
        governs whether or not an existing profile with that name will be
        overwritten.
        """
        # First we get our data in order
        order = []
        set_attr = {'x_bin_field':self.x_bin_field,
                    'y_bin_field':self.y_bin_field,
                    'z_bin_field':self.z_bin_field,
                    'x_bin_values':self[self.x_bin_field],
                    'y_bin_values':self[self.y_bin_field],
                    'z_bin_values':self[self.z_bin_field],
                    '_x_log':self._x_log,
                    '_y_log':self._y_log,
                    '_z_log':self._z_log,
                    'shape': (self[self.x_bin_field].size,
                              self[self.y_bin_field].size,
                              self[self.z_bin_field].size),
                    'field_order':order }
        values = []
        for field in self.field_data:
            if field in set_attr.values(): continue
            order.append(field)
            values.append(self[field].ravel())
        values = np.array(values).transpose()
        self._data_source.hierarchy.save_data(values, "/Profiles", name,
                                              set_attr, force=force)

class ProfileFieldAccumulator(object):
    def __init__(self, n_fields, size):
        shape = size + (n_fields,)
        self.values = np.zeros(shape, dtype="float64")
        self.mvalues = np.zeros(shape, dtype="float64")
        self.qvalues = np.zeros(shape, dtype="float64")
        self.used = np.zeros(size, dtype='bool')
        self.weight_values = np.zeros(size, dtype="float64")

class ProfileND(ParallelAnalysisInterface):
    def __init__(self, data_source, weight_field = None):
        self.data_source = data_source
        self.pf = data_source.pf
        self.field_data = YTFieldData()
        self.weight_field = weight_field

    def add_fields(self, fields):
        fields = ensure_list(fields)
        temp_storage = ProfileFieldAccumulator(len(fields), self.size)
        for g in parallel_objects(self.data_source._grids):
            self._bin_grid(g, fields, temp_storage)
        self._finalize_storage(fields, temp_storage)

    def _finalize_storage(self, fields, temp_storage):
        # We use our main comm here
        # This also will fill _field_data
        # FIXME: Add parallelism and combining std stuff
        if self.weight_field is not None:
            temp_storage.values /= temp_storage.weight_values[...,None]
        blank = ~temp_storage.used
        for i, field in enumerate(fields):
            self.field_data[field] = temp_storage.values[...,i]
            self.field_data[field][blank] = 0.0
        
    def _bin_grid(self, grid, fields, storage):
        raise NotImplementedError

    def _filter(self, bin_fields, cut_points):
        # cut_points is initially just the points inside our region
        # we also want to apply a filtering based on min/max
        filter = np.zeros(bin_fields[0].shape, dtype='bool')
        filter[cut_points] = True
        for (mi, ma), data in zip(self.bounds, bin_fields):
            filter &= (data > mi)
            filter &= (data < ma)
        return filter, [data[filter] for data in bin_fields]
        
    def _get_data(self, grid, fields):
        # Save the values in the grid beforehand.
        old_params = grid.field_parameters
        old_keys = grid.field_data.keys()
        grid.field_parameters = self.data_source.field_parameters
        # Now we ask our source which values to include
        pointI = self.data_source._get_point_indices(grid)
        bin_fields = [grid[bf] for bf in self.bin_fields]
        # We want to make sure that our fields are within the bounds of the
        # binning
        filter, bin_fields = self._filter(bin_fields, pointI)
        if not np.any(filter): return None
        arr = np.zeros((bin_fields[0].size, len(fields)), dtype="float64")
        for i, field in enumerate(fields):
            arr[:,i] = grid[field][filter]
        if self.weight_field is not None:
            weight_data = grid[self.weight_field]
        else:
            weight_data = np.ones(grid.ActiveDimensions, dtype="float64")
        weight_data = weight_data[filter]
        # So that we can pass these into 
        grid.field_parameters = old_params
        grid.field_data = YTFieldData( [(k, grid.field_data[k]) for k in old_keys] )
        return arr, weight_data, bin_fields

    def __getitem__(self, key):
        return self.field_data[key]

    def __iter__(self):
        return sorted(self.field_data.items())

    def _get_bins(self, mi, ma, n, take_log):
        if take_log:
            return np.logspace(np.log10(mi), np.log10(ma), n+1)
        else:
            return np.linspace(mi, ma, n+1)

class Profile1D(ProfileND):
    def __init__(self, data_source, x_field, x_n, x_min, x_max, x_log,
                 weight_field = None):
        super(Profile1D, self).__init__(data_source, weight_field)
        self.x_field = x_field
        self.x_log = x_log
        self.x_bins = self._get_bins(x_min, x_max, x_n, x_log)

        self.size = (self.x_bins.size - 1,)
        self.bin_fields = (self.x_field,)
        self.bounds = ((self.x_bins[0], self.x_bins[-1]),)
        self.x = self.x_bins

    def _bin_grid(self, grid, fields, storage):
        gd = self._get_data(grid, fields)
        if gd is None: return
        fdata, wdata, (bf_x,) = gd
        bin_ind = np.digitize(bf_x, self.x_bins) - 1
        new_bin_profile1d(bin_ind, wdata, fdata,
                      storage.weight_values, storage.values,
                      storage.mvalues, storage.qvalues,
                      storage.used)
        # We've binned it!

class Profile2D(ProfileND):
    def __init__(self, data_source,
                 x_field, x_n, x_min, x_max, x_log,
                 y_field, y_n, y_min, y_max, y_log,
                 weight_field = None):
        super(Profile2D, self).__init__(data_source, weight_field)
        self.x_field = x_field
        self.x_log = x_log
        self.x_bins = self._get_bins(x_min, x_max, x_n, x_log)
        self.y_field = y_field
        self.y_log = y_log
        self.y_bins = self._get_bins(y_min, y_max, y_n, y_log)

        self.size = (self.x_bins.size - 1, self.y_bins.size - 1)

        self.bin_fields = (self.x_field, self.y_field)
        self.bounds = ((self.x_bins[0], self.x_bins[-1]),
                       (self.y_bins[0], self.y_bins[-1]))
        self.x = self.x_bins
        self.y = self.y_bins

    def _bin_grid(self, grid, fields, storage):
        rv = self._get_data(grid, fields)
        if rv is None: return
        fdata, wdata, (bf_x, bf_y) = rv
        bin_ind_x = np.digitize(bf_x, self.x_bins) - 1
        bin_ind_y = np.digitize(bf_y, self.y_bins) - 1
        new_bin_profile2d(bin_ind_x, bin_ind_y, wdata, fdata,
                      storage.weight_values, storage.values,
                      storage.mvalues, storage.qvalues,
                      storage.used)
        # We've binned it!

class Profile3D(ProfileND):
    def __init__(self, data_source,
                 x_field, x_n, x_min, x_max, x_log,
                 y_field, y_n, y_min, y_max, y_log,
                 z_field, z_n, z_min, z_max, z_log,
                 weight_field = None):
        super(Profile3D, self).__init__(data_source, weight_field)
        # X
        self.x_field = x_field
        self.x_log = x_log
        self.x_bins = self._get_bins(x_min, x_max, x_n, x_log)
        # Y
        self.y_field = y_field
        self.y_log = y_log
        self.y_bins = self._get_bins(y_min, y_max, y_n, y_log)
        # Z
        self.z_field = z_field
        self.z_log = z_log
        self.z_bins = self._get_bins(z_min, z_max, z_n, z_log)

        self.size = (self.x_bins.size - 1,
                     self.y_bins.size - 1,
                     self.z_bins.size - 1)

        self.bin_fields = (self.x_field, self.y_field, self.z_field)
        self.bounds = ((self.x_bins[0], self.x_bins[-1]),
                       (self.y_bins[0], self.y_bins[-1]),
                       (self.z_bins[0], self.z_bins[-1]))

        self.x = self.x_bins
        self.y = self.y_bins
        self.z = self.z_bins

    def _bin_grid(self, grid, fields, storage):
        rv = self._get_data(grid, fields)
        if rv is None: return
        fdata, wdata, (bf_x, bf_y, bf_z) = rv
        bin_ind_x = np.digitize(bf_x, self.x_bins) - 1
        bin_ind_y = np.digitize(bf_y, self.y_bins) - 1
        bin_ind_z = np.digitize(bf_z, self.z_bins) - 1
        new_bin_profile3d(bin_ind_x, bin_ind_y, bin_ind_z, wdata, fdata,
                      storage.weight_values, storage.values,
                      storage.mvalues, storage.qvalues,
                      storage.used)
        # We've binned it!

def create_profile(data_source, bin_fields, n = 64, 
                   weight_field = "CellMass", fields = None,
                   accumulation = False, fractional = False):
    r"""
    Create a 1, 2, or 3D profile object.

    The dimensionality of the profile object is chosen by the number of 
    fields given in the bin_fields argument.

    Parameters
    ----------
    data_source : AMR3DData Object
        The data object to be profiled.
    bin_fields : list of strings
        List of the binning fields for profiling.
    n : int or list of ints
        The number of bins in each dimension.  If None, 64 bins for 
        each bin are used for each bin field.
        Default: 64.
    weight_field : str
        The weight field for computing weighted average for the profile 
        values.  If None, the profile values are sums of the data in 
        each bin.
    fields : list of strings
        The fields to be profiled.
    accumulation : bool or list of bools
        If True, the profile values for a bin n are the cumulative sum of 
        all the values from bin 0 to n.  If -True, the sum is reversed so 
        that the value for bin n is the cumulative sum from bin N (total bins) 
        to n.  If the profile is 2D or 3D, a list of values can be given to 
        control the summation in each dimension independently.
        Default: False.
    fractional : If True the profile values are divided by the sum of all 
        the profile data such that the profile represents a probability 
        distribution function.

    Examples
    --------

    Create a 1d profile.  Access bin field from profile.x and field 
    data from profile.field_data.

    >>> pf = load("DD0046/DD0046")
    >>> ad = pf.h.all_data()
    >>> profile = create_profile(ad, ["Density"],
    ...                          fields=["Temperature", "x-velocity"]))
    >>> print profile.x
    >>> print profile.field_data["Temperature"]
    
    """
    if len(bin_fields) == 1:
        cls = Profile1D
    elif len(bin_fields) == 2:
        cls = Profile2D
    elif len(bin_fields) == 3:
        cls = Profile3D
    else:
        raise NotImplementedError
    if not iterable(n):
        n = [n] * len(bin_fields)
    if not iterable(accumulation):
        accumulation = [accumulation] * len(bin_fields)
    logs = [data_source.pf.field_info[f].take_log for f in bin_fields]
    ex = [data_source.quantities["Extrema"](f, non_zero=l)[0] \
          for f, l in zip(bin_fields, logs)]
    args = [data_source]
    for f, n, (mi, ma), l in zip(bin_fields, n, ex, logs):
        args += [f, n, mi, ma, l] 
    obj = cls(*args, weight_field = weight_field)
    setattr(obj, "accumulation", accumulation)
    setattr(obj, "fractional", fractional)
    if fields is not None:
        obj.add_fields(fields)
    for field in fields:
        if fractional:
            obj.field_data[field] /= obj.field_data[field].sum()
        for axis, acc in enumerate(accumulation):
            if not acc: continue
            temp = obj.field_data[field]
            temp = np.rollaxis(temp, axis)
            if acc < 0:
                temp = temp[::-1]
            temp = temp.cumsum(axis=0)
            if acc < 0:
                temp = temp[::-1]
            temp = np.rollaxis(temp, axis)
            obj.field_data[field] = temp
            
    return obj

