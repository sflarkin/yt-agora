"""
Subsets of octrees




"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np

from yt.data_objects.data_containers import \
    YTFieldData, \
    YTDataContainer, \
    YTSelectionContainer
from yt.fields.field_exceptions import \
    NeedsGridType, \
    NeedsOriginalGrid, \
    NeedsDataField, \
    NeedsProperty, \
    NeedsParameter
import yt.geometry.particle_deposit as particle_deposit
import yt.geometry.particle_smooth as particle_smooth
from yt.funcs import *

def cell_count_cache(func):
    def cc_cache_func(self, dobj):
        if hash(dobj.selector) != self._last_selector_id:
            self._cell_count = -1
        rv = func(self, dobj)
        self._cell_count = rv.shape[0]
        self._last_selector_id = hash(dobj.selector)
        return rv
    return cc_cache_func

class OctreeSubset(YTSelectionContainer):
    _spatial = True
    _num_ghost_zones = 0
    _type_name = 'octree_subset'
    _skip_add = True
    _con_args = ('base_region', 'domain', 'pf')
    _container_fields = (("index", "dx"),
                         ("index", "dy"),
                         ("index", "dz"),
                         ("index", "x"),
                         ("index", "y"),
                         ("index", "z"))
    _domain_offset = 0
    _cell_count = -1

    def __init__(self, base_region, domain, pf, over_refine_factor = 1):
        self._num_zones = 1 << (over_refine_factor)
        self.field_data = YTFieldData()
        self.field_parameters = {}
        self.domain = domain
        self.domain_id = domain.domain_id
        self.pf = domain.pf
        self.hierarchy = self.pf.hierarchy
        self.oct_handler = domain.oct_handler
        self._last_mask = None
        self._last_selector_id = None
        self._current_particle_type = 'all'
        self._current_fluid_type = self.pf.default_fluid_type
        self.base_region = base_region
        self.base_selector = base_region.selector

    def _generate_container_field(self, field):
        if self._current_chunk is None:
            self.hierarchy._identify_base_chunk(self)
        if isinstance(field, tuple): field = field[1]
        if field == "dx":
            return self._current_chunk.fwidth[:,0]
        elif field == "dy":
            return self._current_chunk.fwidth[:,1]
        elif field == "dz":
            return self._current_chunk.fwidth[:,2]
        else:
            raise RuntimeError

    def __getitem__(self, key):
        tr = super(OctreeSubset, self).__getitem__(key)
        try:
            fields = self._determine_fields(key)
        except YTFieldTypeNotFound:
            return tr
        finfo = self.pf._get_field_info(*fields[0])
        if not finfo.particle_type:
            # We may need to reshape the field, if it is being queried from
            # field_data.  If it's already cached, it just passes through.
            if len(tr.shape) < 4:
                tr = self._reshape_vals(tr)
            return tr
        return tr

    @property
    def nz(self):
        return self._num_zones + 2*self._num_ghost_zones

    def _reshape_vals(self, arr):
        if len(arr.shape) == 4: return arr
        nz = self.nz
        n_oct = arr.shape[0] / (nz**3.0)
        if arr.size == nz*nz*nz*n_oct:
            arr = arr.reshape((nz, nz, nz, n_oct), order="F")
        elif arr.size == nz*nz*nz*n_oct * 3:
            arr = arr.reshape((nz, nz, nz, n_oct, 3), order="F")
        else:
            raise RuntimeError
        arr = np.asfortranarray(arr)
        return arr

    _domain_ind = None

    def select_blocks(self, selector):
        mask = self.oct_handler.mask(selector, domain_id = self.domain_id)
        mask = self._reshape_vals(mask)
        slicer = OctreeSubsetBlockSlice(self)
        for i, sl in slicer:
            yield sl, mask[:,:,:,i]

    @property
    def domain_ind(self):
        if self._domain_ind is None:
            di = self.oct_handler.domain_ind(self.selector)
            self._domain_ind = di
        return self._domain_ind

    def deposit(self, positions, fields = None, method = None):
        # Here we perform our particle deposition.
        if fields is None: fields = []
        cls = getattr(particle_deposit, "deposit_%s" % method, None)
        if cls is None:
            raise YTParticleDepositionNotImplemented(method)
        nz = self.nz
        nvals = (nz, nz, nz, (self.domain_ind >= 0).sum())
        op = cls(nvals) # We allocate number of zones, not number of octs
        op.initialize()
        mylog.debug("Depositing %s (%s^3) particles into %s Octs",
            positions.shape[0], positions.shape[0]**0.3333333, nvals[-1])
        pos = np.array(positions, dtype="float64")
        # We should not need the following if we know in advance all our fields
        # need no casting.
        fields = [np.asarray(f, dtype="float64") for f in fields]
        op.process_octree(self.oct_handler, self.domain_ind, pos, fields,
            self.domain_id, self._domain_offset)
        vals = op.finalize()
        if vals is None: return
        return np.asfortranarray(vals)

    def smooth(self, positions, fields = None, method = None):
        # Here we perform our particle deposition.
        if fields is None: fields = []
        cls = getattr(particle_smooth, "%s_smooth" % method, None)
        if cls is None:
            raise YTParticleDepositionNotImplemented(method)
        nz = self.nz
        nvals = (nz, nz, nz, (self.domain_ind >= 0).sum())
        if fields is None: fields = []
        op = cls(nvals, len(fields), 64)
        op.initialize()
        mylog.debug("Smoothing %s particles into %s Octs",
            positions.shape[0], nvals[-1])
        op.process_octree(self.oct_handler, self.domain_ind, positions, fields,
            self.domain_id, self._domain_offset, self.pf.periodicity)
        vals = op.finalize()
        if vals is None: return
        if isinstance(vals, list):
            vals = [np.asfortranarray(v) for v in vals]
        else:
            vals = np.asfortranarray(vals)
        return vals

    @cell_count_cache
    def select_icoords(self, dobj):
        return self.oct_handler.icoords(dobj.selector, domain_id = self.domain_id,
                                     num_cells = self._cell_count)

    @cell_count_cache
    def select_fcoords(self, dobj):
        return self.oct_handler.fcoords(dobj.selector, domain_id = self.domain_id,
                                        num_cells = self._cell_count)

    @cell_count_cache
    def select_fwidth(self, dobj):
        return self.oct_handler.fwidth(dobj.selector, domain_id = self.domain_id,
                                       num_cells = self._cell_count)

    @cell_count_cache
    def select_ires(self, dobj):
        return self.oct_handler.ires(dobj.selector, domain_id = self.domain_id,
                                     num_cells = self._cell_count)

    def select(self, selector, source, dest, offset):
        n = self.oct_handler.selector_fill(selector, source, dest, offset,
                                           domain_id = self.domain_id)
        return n

    def count(self, selector):
        return -1

    def count_particles(self, selector, x, y, z):
        # We don't cache the selector results
        count = selector.count_points(x,y,z)
        return count

    def select_particles(self, selector, x, y, z):
        mask = selector.select_points(x,y,z)
        return mask

class ParticleOctreeSubset(OctreeSubset):
    # Subclassing OctreeSubset is somewhat dubious.
    # This is some subset of an octree.  Note that the sum of subsets of an
    # octree may multiply include data files.  While we can attempt to mitigate
    # this, it's unavoidable for many types of data storage on disk.
    _type_name = 'indexed_octree_subset'
    _con_args = ('data_files', 'pf', 'min_ind', 'max_ind')
    domain_id = -1
    def __init__(self, base_region, data_files, pf, min_ind = 0, max_ind = 0,
                 over_refine_factor = 1):
        # The first attempt at this will not work in parallel.
        self._num_zones = 1 << (over_refine_factor)
        self.data_files = data_files
        self.field_data = YTFieldData()
        self.field_parameters = {}
        self.pf = pf
        self.hierarchy = self.pf.hierarchy
        self.oct_handler = pf.h.oct_handler
        self.min_ind = min_ind
        if max_ind == 0: max_ind = (1 << 63)
        self.max_ind = max_ind
        self._last_mask = None
        self._last_selector_id = None
        self._current_particle_type = 'all'
        self._current_fluid_type = self.pf.default_fluid_type
        self.base_region = base_region
        self.base_selector = base_region.selector

class OctreeSubsetBlockSlice(object):
    def __init__(self, octree_subset):
        self.ind = None
        self.octree_subset = octree_subset
        # Cache some attributes
        nz = octree_subset.nz
        self.ActiveDimensions = np.array([nz,nz,nz], dtype="int64")
        for attr in ["ires", "icoords", "fcoords", "fwidth"]:
            v = getattr(octree_subset, attr)
            setattr(self, "_%s" % attr, octree_subset._reshape_vals(v))

    def __iter__(self):
        for i in range(self._ires.shape[-1]):
            self.ind = i
            yield i, self

    def clear_data(self):
        pass

    def __getitem__(self, key):
        return self.octree_subset[key][:,:,:,self.ind]

    def get_vertex_centered_data(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def id(self):
        return np.random.randint(1)

    @property
    def Level(self):
        return self._ires[0,0,0,self.ind]

    @property
    def LeftEdge(self):
        LE = (self._fcoords[0,0,0,self.ind,:]
            - self._fwidth[0,0,0,self.ind,:]*0.5)
        return LE

    @property
    def RightEdge(self):
        RE = (self._fcoords[-1,-1,-1,self.ind,:]
            + self._fwidth[-1,-1,-1,self.ind,:]*0.5)
        return RE

    @property
    def dds(self):
        return self._fwidth[0,0,0,self.ind,:]
