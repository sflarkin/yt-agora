"""
Enzo-specific IO functions



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

from collections import defaultdict

import exceptions
import os
import numpy as np

from yt.utilities.io_handler import \
    BaseIOHandler, _axis_ids
from yt.utilities.logger import ytLogger as mylog
from yt.utilities.lib.geometry_utils import compute_morton
from yt.utilities.exceptions import *

class IOHandlerStream(BaseIOHandler):

    _data_style = "stream"

    def __init__(self, pf):
        self.fields = pf.stream_handler.fields
        super(IOHandlerStream, self).__init__(pf)

    def _read_data_set(self, grid, field):
        # This is where we implement processor-locking
        #if grid.id not in self.grids_in_memory:
        #    mylog.error("Was asked for %s but I have %s", grid.id, self.grids_in_memory.keys())
        #    raise KeyError
        tr = self.fields[grid.id][field]
        # If it's particles, we copy.
        if len(tr.shape) == 1: return tr.copy()
        # New in-place unit conversion breaks if we don't copy first
        return tr

    def _read_fluid_selection(self, chunks, selector, fields, size):
        chunks = list(chunks)
        if any((ftype not in ("gas", "deposit") for ftype, fname in fields)):
            raise NotImplementedError
        rv = {}
        for field in fields:
            ftype, fname = field
            rv[field] = np.empty(size, dtype="float64")
        ng = sum(len(c.objs) for c in chunks)
        mylog.debug("Reading %s cells of %s fields in %s blocks",
                    size, [f2 for f1, f2 in fields], ng)
        for field in fields:
            ftype, fname = field
            if ftype == 'deposit':
                fname = field
            ind = 0
            for chunk in chunks:
                for g in chunk.objs:
                    ds = self.fields[g.id][fname]
                    ind += g.select(selector, ds, rv[field], ind) # caches
        return rv

    def _read_particle_coords(self, chunks, ptf):
        chunks = list(chunks)
        for chunk in chunks:
            for g in chunk.objs:
                if g.NumberOfParticles == 0: continue
                gf = self.fields[g.id]
                for ptype, field_list in sorted(ptf.items()):
                    x, y, z  = (gf[ptype, "particle_position_%s" % ax]
                                for ax in 'xyz')
                    yield ptype, (x, y, z)

    def _read_particle_fields(self, chunks, ptf, selector):
        chunks = list(chunks)
        for chunk in chunks:
            for g in chunk.objs:
                if g.NumberOfParticles == 0: continue
                gf = self.fields[g.id]
                for ptype, field_list in sorted(ptf.items()):
                    x, y, z  = (gf[ptype, "particle_position_%s" % ax]
                                for ax in 'xyz')
                    mask = selector.select_points(x, y, z)
                    if mask is None: continue
                    for field in field_list:
                        data = np.asarray(gf[ptype, field])
                        yield (ptype, field), data[mask]

    @property
    def _read_exception(self):
        return KeyError

class StreamParticleIOHandler(BaseIOHandler):

    _data_style = "stream_particles"

    def __init__(self, pf):
        self.fields = pf.stream_handler.fields
        super(StreamParticleIOHandler, self).__init__(pf)

    def _read_particle_coords(self, chunks, ptf):
        chunks = list(chunks)
        data_files = set([])
        for chunk in chunks:
            for obj in chunk.objs:
                data_files.update(obj.data_files)
        for data_file in data_files:
            f = self.fields[data_file.filename]
            # This double-reads
            for ptype, field_list in sorted(ptf.items()):
                yield ptype, (f[ptype, "particle_position_x"],
                              f[ptype, "particle_position_y"],
                              f[ptype, "particle_position_z"])
            
    def _read_particle_fields(self, chunks, ptf, selector):
        data_files = set([])
        for chunk in chunks:
            for obj in chunk.objs:
                data_files.update(obj.data_files)
        for data_file in data_files:
            f = self.fields[data_file.filename]
            for ptype, field_list in sorted(ptf.items()):
                x, y, z = (f[ptype, "particle_position_%s" % ax]
                           for ax in 'xyz')
                mask = selector.select_points(x, y, z)
                if mask is None: continue
                for field in field_list:
                    data = f[ptype, field][mask]
                    yield (ptype, field), data


    def _initialize_index(self, data_file, regions):
        # self.fields[g.id][fname] is the pattern here
        pos = np.column_stack(self.fields[data_file.filename][
                              ("io", "particle_position_%s" % ax)]
                              for ax in 'xyz')
        if np.any(pos.min(axis=0) < data_file.pf.domain_left_edge) or \
           np.any(pos.max(axis=0) > data_file.pf.domain_right_edge):
            raise YTDomainOverflow(pos.min(axis=0), pos.max(axis=0),
                                   data_file.pf.domain_left_edge,
                                   data_file.pf.domain_right_edge)
        regions.add_data_file(pos, data_file.file_id)
        morton = compute_morton(
                pos[:,0], pos[:,1], pos[:,2],
                data_file.pf.domain_left_edge,
                data_file.pf.domain_right_edge)
        return morton

    def _count_particles(self, data_file):
        npart = self.fields[data_file.filename]["io", "particle_position_x"].size
        return {'io': npart}

    def _identify_fields(self, data_file):
        return self.fields[data_file.filename].keys()

class IOHandlerStreamHexahedral(BaseIOHandler):
    _data_style = "stream_hexahedral"

    def __init__(self, pf):
        self.fields = pf.stream_handler.fields
        super(IOHandlerStreamHexahedral, self).__init__(pf)

    def _read_fluid_selection(self, chunks, selector, fields, size):
        chunks = list(chunks)
        assert(len(chunks) == 1)
        chunk = chunks[0]
        rv = {}
        for field in fields:
            ftype, fname = field
            rv[field] = np.empty(size, dtype="float64")
        ngrids = sum(len(chunk.objs) for chunk in chunks)
        mylog.debug("Reading %s cells of %s fields in %s blocks",
                    size, [fname for ftype, fname in fields], ngrids)
        for field in fields:
            ind = 0
            ftype, fname = field
            for chunk in chunks:
                for g in chunk.objs:
                    ds = self.fields[g.mesh_id].get(field, None)
                    if ds is None:
                        ds = self.fields[g.mesh_id][fname]
                    ind += g.select(selector, ds, rv[field], ind) # caches
        return rv

class IOHandlerStreamOctree(BaseIOHandler):
    _data_style = "stream_octree"

    def __init__(self, pf):
        self.fields = pf.stream_handler.fields
        super(IOHandlerStreamOctree, self).__init__(pf)

    def _read_fluid_selection(self, chunks, selector, fields, size):
        rv = {}
        ind = 0
        chunks = list(chunks)
        assert(len(chunks) == 1)
        for chunk in chunks:
            assert(len(chunk.objs) == 1)
            for subset in chunk.objs:
                field_vals = self.fields[subset.domain_id -
                                    subset._domain_offset]
                subset.fill(field_vals, rv, selector, ind)
        return rv
