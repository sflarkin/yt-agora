"""
Geometry container base class.




"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import os
import cPickle
import weakref
import h5py
from exceptions import IOError, TypeError
from types import ClassType
import numpy as np
import abc
import copy

from yt.funcs import *
from yt.config import ytcfg
from yt.data_objects.data_containers import \
    data_object_registry
from yt.units.yt_array import \
    uconcatenate
from yt.fields.field_info_container import \
    NullFunc
from yt.fields.particle_fields import \
    particle_deposition_functions, \
    particle_scalar_functions
from yt.utilities.io_handler import io_registry
from yt.utilities.logger import ytLogger as mylog
from yt.utilities.parallel_tools.parallel_analysis_interface import \
    ParallelAnalysisInterface, parallel_splitter
from yt.utilities.exceptions import YTFieldNotFound

def _unsupported_object(pf, obj_name):
    def _raise_unsupp(*args, **kwargs):
        raise YTObjectNotImplemented(pf, obj_name)
    return _raise_unsupp

class GeometryHandler(ParallelAnalysisInterface):
    _global_mesh = True
    _unsupported_objects = ()

    def __init__(self, pf, data_style):
        self.filtered_particle_types = []
        ParallelAnalysisInterface.__init__(self)
        self.parameter_file = weakref.proxy(pf)
        self.pf = self.parameter_file

        self._initialize_state_variables()

        mylog.debug("Initializing data storage.")
        self._initialize_data_storage()

        # Must be defined in subclass
        mylog.debug("Setting up classes.")
        self._setup_classes()

        mylog.debug("Setting up domain geometry.")
        self._setup_geometry()

        mylog.debug("Initializing data grid data IO")
        self._setup_data_io()

        # Note that this falls under the "geometry" object since it's
        # potentially quite expensive, and should be done with the indexing.
        mylog.debug("Detecting fields.")
        self._detect_output_fields()

    def __del__(self):
        if self._data_file is not None:
            self._data_file.close()

    def _initialize_state_variables(self):
        self._parallel_locking = False
        self._data_file = None
        self._data_mode = None
        self._max_locations = {}
        self.num_grids = None

    def _setup_classes(self, dd):
        # Called by subclass
        self.object_types = []
        self.objects = []
        self.plots = []
        for name, cls in sorted(data_object_registry.items()):
            if name in self._unsupported_objects:
                setattr(self, name,
                    _unsupported_object(self.parameter_file, name))
                continue
            cname = cls.__name__
            if cname.endswith("Base"): cname = cname[:-4]
            self._add_object_class(name, cname, cls, dd)
        if self.pf.refine_by != 2 and hasattr(self, 'proj') and \
            hasattr(self, 'overlap_proj'):
            mylog.warning("Refine by something other than two: reverting to"
                        + " overlap_proj")
            self.proj = self.overlap_proj
        if self.pf.dimensionality < 3 and hasattr(self, 'proj') and \
            hasattr(self, 'overlap_proj'):
            mylog.warning("Dimensionality less than 3: reverting to"
                        + " overlap_proj")
            self.proj = self.overlap_proj
        self.object_types.sort()

    def _setup_particle_types(self, ptypes = None):
        mname = self.pf._particle_mass_name
        cname = self.pf._particle_coordinates_name
        vname = self.pf._particle_velocity_name
        # We require overriding if any of this is true
        df = []
        if ptypes is None: ptypes = self.pf.particle_types_raw
        if None in (mname, cname, vname): 
            # If we don't know what to do, then let's not.
            for ptype in set(ptypes):
                df += self.pf._setup_particle_type(ptype)
            # Now we have a bunch of new fields to add!
            # This is where the dependencies get calculated.
            #self._derived_fields_add(df)
            return
        fi = self.pf.field_info
        def _get_conv(cf):
            def _convert(data):
                return data.convert(cf)
            return _convert
        for ptype in ptypes:
            fi.add_field((ptype, vname), function=NullFunc,
                particle_type = True,
                convert_function=_get_conv("velocity"),
                units = r"\mathrm{cm}/\mathrm{s}")
            df.append((ptype, vname))
            fi.add_field((ptype, mname), function=NullFunc,
                particle_type = True,
                convert_function=_get_conv("mass"),
                units = r"\mathrm{g}")
            df.append((ptype, mname))
            df += particle_deposition_functions(ptype, cname, mname, fi)
            df += particle_scalar_functions(ptype, cname, vname, fi)
            fi.add_field((ptype, cname), function=NullFunc,
                         particle_type = True)
            df.append((ptype, cname))
            # Now we add some translations.
            df += self.pf._setup_particle_type(ptype)
        self._derived_fields_add(df)

    def _setup_field_registry(self):
        self.derived_field_list = []
        self.filtered_particle_types = []

    def _setup_filtered_type(self, filter):
        if not filter.available(self.derived_field_list):
            return False
        fi = self.parameter_file.field_info
        fd = self.parameter_file.field_dependencies
        available = False
        for fn in self.derived_field_list:
            if fn[0] == filter.filtered_type:
                # Now we can add this
                available = True
                self.derived_field_list.append(
                    (filter.name, fn[1]))
                fi[filter.name, fn[1]] = filter.wrap_func(fn, fi[fn])
                # Now we append the dependencies
                fd[filter.name, fn[1]] = fd[fn]
        if available:
            self.parameter_file.particle_types += (filter.name,)
            self.filtered_particle_types.append(filter.name)
            self._setup_particle_types([filter.name])
        return available

    # Now all the object related stuff
    def all_data(self, find_max=False):
        pf = self.parameter_file
        if find_max: c = self.find_max("Density")[1]
        else: c = (pf.domain_right_edge + pf.domain_left_edge)/2.0
        return self.region(c,
            pf.domain_left_edge, pf.domain_right_edge)

    def _initialize_data_storage(self):
        if not ytcfg.getboolean('yt','serialize'): return
        fn = self.pf.storage_filename
        if fn is None:
            if os.path.isfile(os.path.join(self.directory,
                                "%s.yt" % self.pf.unique_identifier)):
                fn = os.path.join(self.directory,"%s.yt" % self.pf.unique_identifier)
            else:
                fn = os.path.join(self.directory,
                        "%s.yt" % self.parameter_file.basename)
        dir_to_check = os.path.dirname(fn)
        if dir_to_check == '':
            dir_to_check = '.'
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
        writeable = writeable and not ytcfg.getboolean('yt','onlydeserialize')
        # We now have our conditional stuff
        self.comm.barrier()
        if not writeable and not exists: return
        if writeable:
            try:
                if not exists: self.__create_data_file(fn)
                self._data_mode = 'a'
            except IOError:
                self._data_mode = None
                return
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
        if getattr(self, "io", None) is not None: return
        self.io = io_registry[self.data_style](self.parameter_file)

    def _save_data(self, array, node, name, set_attr=None, force=False, passthrough = False):
        """
        Arbitrary numpy data will be saved to the region in the datafile
        described by *node* and *name*.  If data file does not exist, it throws
        no error and simply does not save.
        """

        if self._data_mode != 'a': return
        try:
            node_loc = self._data_file[node]
            if name in node_loc and force:
                mylog.info("Overwriting node %s/%s", node, name)
                del self._data_file[node][name]
            elif name in node_loc and passthrough:
                return
        except:
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

    save_data = parallel_splitter(_save_data, _reload_data_file)

    def _get_data_reader_dict(self):
        dd = { 'pf' : self.parameter_file, # Already weak
               'hierarchy': weakref.proxy(self) }
        return dd

    def _reset_save_data(self,round_robin=False):
        if round_robin:
            self.save_data = self._save_data
        else:
            self.save_data = parallel_splitter(self._save_data, self._reload_data_file)

    def save_object(self, obj, name):
        """
        Save an object (*obj*) to the data_file using the Pickle protocol,
        under the name *name* on the node /Objects.
        """
        s = cPickle.dumps(obj, protocol=-1)
        self.save_data(np.array(s, dtype='c'), "/Objects", name, force = True)

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
                if group not in myGroup:
                    return None
                myGroup = myGroup[group]
        if name not in myGroup:
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

    def find_max(self, field):
        """
        Returns (value, center) of location of maximum for a given field.
        """
        mylog.debug("Searching for maximum value of %s", field)
        source = self.all_data()
        max_val, maxi, mx, my, mz = \
            source.quantities["MaxLocation"](field)
        mylog.info("Max Value is %0.5e at %0.16f %0.16f %0.16f", 
              max_val, mx, my, mz)
        return max_val, np.array([mx, my, mz], dtype="float64")

    def _add_object_class(self, name, class_name, base, dd):
        self.object_types.append(name)
        obj = type(class_name, (base,), dd)
        setattr(self, name, obj)

    def _split_fields(self, fields):
        # This will split fields into either generated or read fields
        fields_to_read, fields_to_generate = [], []
        for ftype, fname in fields:
            if fname in self.field_list or (ftype, fname) in self.field_list:
                fields_to_read.append((ftype, fname))
            else:
                fields_to_generate.append((ftype, fname))
        return fields_to_read, fields_to_generate

    def _read_particle_fields(self, fields, dobj, chunk = None):
        if len(fields) == 0: return {}, []
        selector = dobj.selector
        if chunk is None:
            self._identify_base_chunk(dobj)
        fields_to_return = {}
        fields_to_read, fields_to_generate = self._split_fields(fields)
        if len(fields_to_read) == 0:
            return {}, fields_to_generate
        fields_to_return = self.io._read_particle_selection(
            self._chunk_io(dobj, cache = False),
            selector,
            fields_to_read)
        for field in fields_to_read:
            ftype, fname = field
            finfo = self.pf._get_field_info(*field)
        return fields_to_return, fields_to_generate

    def _read_fluid_fields(self, fields, dobj, chunk = None):
        if len(fields) == 0: return {}, []
        selector = dobj.selector
        if chunk is None:
            self._identify_base_chunk(dobj)
            chunk_size = dobj.size
        else:
            chunk_size = chunk.data_size
        fields_to_return = {}
        fields_to_read, fields_to_generate = self._split_fields(fields)
        if len(fields_to_read) == 0:
            return {}, fields_to_generate
        fields_to_return = self.io._read_fluid_selection(
            self._chunk_io(dobj),
            selector,
            fields_to_read,
            chunk_size)
        #mylog.debug("Don't know how to read %s", fields_to_generate)
        return fields_to_return, fields_to_generate


    def _chunk(self, dobj, chunking_style, ngz = 0, **kwargs):
        # A chunk is either None or (grids, size)
        if dobj._current_chunk is None:
            self._identify_base_chunk(dobj)
        if ngz != 0 and chunking_style != "spatial":
            raise NotImplementedError
        if chunking_style == "all":
            return self._chunk_all(dobj, **kwargs)
        elif chunking_style == "spatial":
            return self._chunk_spatial(dobj, ngz, **kwargs)
        elif chunking_style == "io":
            return self._chunk_io(dobj, **kwargs)
        else:
            raise NotImplementedError

def cached_property(func):
    n = '_%s' % func.func_name
    def cached_func(self):
        if self._cache and getattr(self, n, None) is not None:
            return getattr(self, n)
        if self.data_size is None:
            tr = self._accumulate_values(n[1:])
        else:
            tr = func(self)
        if self._cache:
            setattr(self, n, tr)
        return tr
    return property(cached_func)

class YTDataChunk(object):

    def __init__(self, dobj, chunk_type, objs, data_size = None,
                 field_type = None, cache = False):
        self.dobj = dobj
        self.chunk_type = chunk_type
        self.objs = objs
        self.data_size = data_size
        self._field_type = field_type
        self._cache = cache

    def _accumulate_values(self, method):
        # We call this generically.  It's somewhat slower, since we're doing
        # costly getattr functions, but this allows us to generalize.
        mname = "select_%s" % method
        arrs = []
        for obj in self.objs:
            f = getattr(obj, mname)
            arrs.append(f(self.dobj))
        arrs = uconcatenate(arrs)
        self.data_size = arrs.shape[0]
        return arrs

    @cached_property
    def fcoords(self):
        ci = np.empty((self.data_size, 3), dtype='float64')
        ci = YTArray(ci, input_units = "code_length",
                     registry = self.dobj.pf.unit_registry)
        if self.data_size == 0: return ci
        ind = 0
        for obj in self.objs:
            c = obj.select_fcoords(self.dobj)
            if c.shape[0] == 0: continue
            ci[ind:ind+c.shape[0], :] = c
            ind += c.shape[0]
        return ci

    @cached_property
    def icoords(self):
        ci = np.empty((self.data_size, 3), dtype='int64')
        if self.data_size == 0: return ci
        ind = 0
        for obj in self.objs:
            c = obj.select_icoords(self.dobj)
            if c.shape[0] == 0: continue
            ci[ind:ind+c.shape[0], :] = c
            ind += c.shape[0]
        return ci

    @cached_property
    def fwidth(self):
        ci = np.empty((self.data_size, 3), dtype='float64')
        ci = YTArray(ci, input_units = "code_length",
                     registry = self.dobj.pf.unit_registry)
        if self.data_size == 0: return ci
        ind = 0
        for obj in self.objs:
            c = obj.select_fwidth(self.dobj)
            if c.shape[0] == 0: continue
            ci[ind:ind+c.shape[0], :] = c
            ind += c.shape[0]
        return ci

    @cached_property
    def ires(self):
        ci = np.empty(self.data_size, dtype='int64')
        if self.data_size == 0: return ci
        ind = 0
        for obj in self.objs:
            c = obj.select_ires(self.dobj)
            if c.shape == 0: continue
            ci[ind:ind+c.size] = c
            ind += c.size
        return ci

    @cached_property
    def tcoords(self):
        self.dtcoords
        return self._tcoords

    @cached_property
    def dtcoords(self):
        ct = np.empty(self.data_size, dtype='float64')
        cdt = np.empty(self.data_size, dtype='float64')
        self._tcoords = ct # Se this for tcoords
        if self.data_size == 0: return cdt
        ind = 0
        for obj in self.objs:
            gdt, gt = obj.tcoords(self.dobj)
            if gt.shape == 0: continue
            ct[ind:ind+gt.size] = gt
            cdt[ind:ind+gdt.size] = gdt
            ind += gt.size
        return cdt

class ChunkDataCache(object):
    def __init__(self, base_iter, preload_fields, geometry_handler,
                 max_length = 256):
        # At some point, max_length should instead become a heuristic function,
        # potentially looking at estimated memory usage.  Note that this never
        # initializes the iterator; it assumes the iterator is already created,
        # and it calls next() on it.
        self.base_iter = base_iter.__iter__()
        self.queue = []
        self.max_length = max_length
        self.preload_fields = preload_fields
        self.geometry_handler = geometry_handler
        self.cache = {}

    def __iter__(self):
        return self
    
    def next(self):
        if len(self.queue) == 0:
            for i in range(self.max_length):
                try:
                    self.queue.append(self.base_iter.next())
                except StopIteration:
                    break
            # If it's still zero ...
            if len(self.queue) == 0: raise StopIteration
            chunk = YTDataChunk(None, "cache", self.queue, cache=False)
            self.cache = self.geometry_handler.io._read_chunk_data(
                chunk, self.preload_fields) or {}
        g = self.queue.pop(0)
        g._initialize_cache(self.cache.pop(g.id, {}))
        return g

