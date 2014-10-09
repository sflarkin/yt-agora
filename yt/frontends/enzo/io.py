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

import os

from yt.utilities.io_handler import \
    BaseIOHandler, _axis_ids
from yt.utilities.logger import ytLogger as mylog
from yt.geometry.selection_routines import mask_fill, AlwaysSelector
import h5py

import numpy as np
from yt.funcs import *

_convert_mass = ("particle_mass","mass")

_particle_position_names = {}

class IOHandlerPackedHDF5(BaseIOHandler):

    _dataset_type = "enzo_packed_3d"
    _array_fields = {}
    _base = slice(None)

    def _read_field_names(self, grid):
        if grid.filename is None: return []
        f = h5py.File(grid.filename, "r")
        group = f["/Grid%08i" % grid.id]
        fields = []
        add_io = "io" in grid.ds.particle_types
        for name, v in group.iteritems():
            # NOTE: This won't work with 1D datasets or references.
            if not hasattr(v, "shape") or v.dtype == "O":
                continue
            elif len(v.dims) == 1:
                if grid.ds.dimensionality == 1:
                    fields.append( ("enzo", str(name)) )
                elif add_io:
                    fields.append( ("io", str(name)) )
            else:
                fields.append( ("enzo", str(name)) )
        for ptype, field_list in sorted(group['Particles/'].items()):
            pds = group['Particles/{0}'.format(ptype)]
            for field in field_list:
                if np.asarray(pds[field]).ndim > 1:
                    self._array_fields[field] = pds[field].shape

        f.close()
        return fields

    @property
    def _read_exception(self):
        return (KeyError,)

    def _read_particle_coords(self, chunks, ptf):
        chunks = list(chunks)
        for chunk in chunks: # These should be organized by grid filename
            f = None
            for g in chunk.objs:
                if g.filename is None: continue
                if f is None:
                    #print "Opening (count) %s" % g.filename
                    f = h5py.File(g.filename.encode('ascii'), "r")
                nap = sum(g.NumberOfActiveParticles.values())
                if g.NumberOfParticles == 0 and nap == 0:
                    continue
                ds = f.get("/Grid%08i" % g.id)
                for ptype, field_list in sorted(ptf.items()):
                    if ptype != "io":
                        if g.NumberOfActiveParticles[ptype] == 0: continue
                        pds = ds.get("Particles/%s" % ptype)
                    else:
                        pds = ds
                    pn = _particle_position_names.get(ptype,
                            r"particle_position_%s")
                    x, y, z = (np.asarray(pds.get(pn % ax).value, dtype="=f8")
                               for ax in 'xyz')
                    yield ptype, (x, y, z)
            if f: f.close()

    def _read_particle_fields(self, chunks, ptf, selector):
        chunks = list(chunks)
        for chunk in chunks: # These should be organized by grid filename
            f = None
            for g in chunk.objs:
                if g.filename is None: continue
                if f is None:
                    #print "Opening (read) %s" % g.filename
                    f = h5py.File(g.filename.encode('ascii'), "r")
                nap = sum(g.NumberOfActiveParticles.values())
                if g.NumberOfParticles == 0 and nap == 0:
                    continue
                ds = f.get("/Grid%08i" % g.id)
                for ptype, field_list in sorted(ptf.items()):
                    if ptype != "io":
                        if g.NumberOfActiveParticles[ptype] == 0: continue
                        pds = ds.get("Particles/%s" % ptype)
                    else:
                        pds = ds
                    pn = _particle_position_names.get(ptype,
                            r"particle_position_%s")
                    x, y, z = (np.asarray(pds.get(pn % ax).value, dtype="=f8")
                               for ax in 'xyz')
                    mask = selector.select_points(x, y, z, 0.0)
                    if mask is None: continue
                    for field in field_list:
                        data = np.asarray(pds.get(field).value, "=f8")
                        if field in _convert_mass:
                            data *= g.dds.prod(dtype="f8")
                        yield (ptype, field), data[mask]
            if f: f.close()

    def _read_fluid_selection(self, chunks, selector, fields, size):
        rv = {}
        # Now we have to do something unpleasant
        chunks = list(chunks)
        if selector.__class__.__name__ == "GridSelector":
            if not (len(chunks) == len(chunks[0].objs) == 1):
                raise RuntimeError
            g = chunks[0].objs[0]
            f = h5py.File(g.filename.encode('ascii'), 'r')
            gds = f.get("/Grid%08i" % g.id)
            for ftype, fname in fields:
                if fname in gds:
                    rv[(ftype, fname)] = gds.get(fname).value.swapaxes(0,2)
                else:
                    rv[(ftype, fname)] = np.zeros(g.ActiveDimensions)
            f.close()
            return rv
        if size is None:
            size = sum((g.count(selector) for chunk in chunks
                        for g in chunk.objs))
        for field in fields:
            ftype, fname = field
            fsize = size
            rv[field] = np.empty(fsize, dtype="float64")
        ng = sum(len(c.objs) for c in chunks)
        mylog.debug("Reading %s cells of %s fields in %s grids",
                   size, [f2 for f1, f2 in fields], ng)
        ind = 0
        for chunk in chunks:
            fid = None
            for g in chunk.objs:
                if g.filename is None: continue
                if fid is None:
                    fid = h5py.h5f.open(g.filename.encode('ascii'), h5py.h5f.ACC_RDONLY)
                data = np.empty(g.ActiveDimensions[::-1], dtype="float64")
                data_view = data.swapaxes(0,2)
                nd = 0
                for field in fields:
                    ftype, fname = field
                    try:
                        node = "/Grid%08i/%s" % (g.id, fname)
                        dg = h5py.h5d.open(fid, node.encode('ascii'))
                    except KeyError:
                        if fname == "Dark_Matter_Density": continue
                        raise
                    dg.read(h5py.h5s.ALL, h5py.h5s.ALL, data)
                    nd = g.select(selector, data_view, rv[field], ind) # caches
                ind += nd
            if fid: fid.close()
        return rv

    def _read_chunk_data(self, chunk, fields):
        fid = fn = None
        rv = {}
        mylog.debug("Preloading fields %s", fields)
        # Split into particles and non-particles
        fluid_fields, particle_fields = [], []
        for ftype, fname in fields:
            if ftype in self.ds.particle_types:
                particle_fields.append((ftype, fname))
            else:
                fluid_fields.append((ftype, fname))
        if len(particle_fields) > 0:
            selector = AlwaysSelector(self.ds)
            rv.update(self._read_particle_selection(
              [chunk], selector, particle_fields))
        if len(fluid_fields) == 0: return rv
        for g in chunk.objs:
            rv[g.id] = gf = {}
            if g.filename is None: continue
            elif g.filename != fn:
                if fid is not None: fid.close()
                fid = None
            if fid is None:
                fid = h5py.h5f.open(g.filename.encode('ascii'), h5py.h5f.ACC_RDONLY)
                fn = g.filename
            data = np.empty(g.ActiveDimensions[::-1], dtype="float64")
            data_view = data.swapaxes(0,2)
            for field in fluid_fields:
                ftype, fname = field
                try:
                    node = "/Grid%08i/%s" % (g.id, fname)
                    dg = h5py.h5d.open(fid, node.encode('ascii'))
                except KeyError:
                    if fname == "Dark_Matter_Density": continue
                    raise
                dg.read(h5py.h5s.ALL, h5py.h5s.ALL, data)
                gf[field] = data_view.copy()
        if fid: fid.close()
        return rv

class IOHandlerPackedHDF5GhostZones(IOHandlerPackedHDF5):
    _dataset_type = "enzo_packed_3d_gz"

    def __init__(self, *args, **kwargs):
        super(IOHandlerPackgedHDF5GhostZones, self).__init__(*args, **kwargs)
        NGZ = self.ds.parameters.get("NumberOfGhostZones", 3)
        self._base = (slice(NGZ, -NGZ),
                      slice(NGZ, -NGZ),
                      slice(NGZ, -NGZ))

    def _read_raw_data_set(self, grid, field):
        f = h5py.File(grid.filename, "r")
        ds = f["/Grid%08i/%s" % (grid.id, field)][:].swapaxes(0,2)
        f.close()
        return ds

class IOHandlerInMemory(BaseIOHandler):

    _dataset_type = "enzo_inline"

    def __init__(self, ds, ghost_zones=3):
        self.ds = ds
        import enzo
        self.enzo = enzo
        self.grids_in_memory = enzo.grid_data
        self.old_grids_in_memory = enzo.old_grid_data
        self.my_slice = (slice(ghost_zones,-ghost_zones),
                      slice(ghost_zones,-ghost_zones),
                      slice(ghost_zones,-ghost_zones))
        BaseIOHandler.__init__(self, ds)

    def _read_field_names(self, grid):
        fields = []
        add_io = "io" in grid.ds.particle_types
        for name, v in self.grids_in_memory[grid.id].items():
            # NOTE: This won't work with 1D datasets or references.
            if not hasattr(v, "shape") or v.dtype == "O":
                continue
            elif v.ndim == 1:
                if grid.ds.dimensionality == 1:
                    fields.append( ("enzo", str(name)) )
                elif add_io:
                    fields.append( ("io", str(name)) )
            else:
                fields.append( ("enzo", str(name)) )
        return fields

    def _read_fluid_selection(self, chunks, selector, fields, size):
        rv = {}
        # Now we have to do something unpleasant
        chunks = list(chunks)
        if selector.__class__.__name__ == "GridSelector":
            if not (len(chunks) == len(chunks[0].objs) == 1):
                raise RuntimeError
            g = chunks[0].objs[0]
            for ftype, fname in fields:
                rv[(ftype, fname)] = self.grids_in_memory[grid.id][fname].swapaxes(0,2)
            return rv
        if size is None:
            size = sum((g.count(selector) for chunk in chunks
                        for g in chunk.objs))
        for field in fields:
            ftype, fname = field
            fsize = size
            rv[field] = np.empty(fsize, dtype="float64")
        ng = sum(len(c.objs) for c in chunks)
        mylog.debug("Reading %s cells of %s fields in %s grids",
                   size, [f2 for f1, f2 in fields], ng)

        ind = 0
        for chunk in chunks:
            for g in chunk.objs:
                # We want a *hard error* here.
                #if g.id not in self.grids_in_memory: continue
                for field in fields:
                    ftype, fname = field
                    data_view = self.grids_in_memory[g.id][fname][self.my_slice].swapaxes(0,2)
                    nd = g.select(selector, data_view, rv[field], ind)
                ind += nd
        assert(ind == fsize)
        return rv

    def _read_particle_coords(self, chunks, ptf):
        chunks = list(chunks)
        for chunk in chunks: # These should be organized by grid filename
            for g in chunk.objs:
                if g.id not in self.grids_in_memory: continue
                nap = sum(g.NumberOfActiveParticles.values())
                if g.NumberOfParticles == 0 and nap == 0: continue
                for ptype, field_list in sorted(ptf.items()):
                    x, y, z = self.grids_in_memory[g.id]['particle_position_x'], \
                                        self.grids_in_memory[g.id]['particle_position_y'], \
                                        self.grids_in_memory[g.id]['particle_position_z']
                    yield ptype, (x, y, z)

    def _read_particle_fields(self, chunks, ptf, selector):
        chunks = list(chunks)
        for chunk in chunks: # These should be organized by grid filename
            for g in chunk.objs:
                if g.id not in self.grids_in_memory: continue
                nap = sum(g.NumberOfActiveParticles.values())
                if g.NumberOfParticles == 0 and nap == 0: continue
                for ptype, field_list in sorted(ptf.items()):
                    x, y, z = self.grids_in_memory[g.id]['particle_position_x'], \
                                        self.grids_in_memory[g.id]['particle_position_y'], \
                                        self.grids_in_memory[g.id]['particle_position_z']
                    mask = selector.select_points(x, y, z, 0.0)
                    if mask is None: continue
                    for field in field_list:
                        data = self.grids_in_memory[g.id][field]
                        if field in _convert_mass:
                            data = data * g.dds.prod(dtype="f8")
                        yield (ptype, field), data[mask]

class IOHandlerPacked2D(IOHandlerPackedHDF5):

    _dataset_type = "enzo_packed_2d"
    _particle_reader = False

    def _read_data_set(self, grid, field):
        f = h5py.File(grid.filename, "r")
        ds = f["/Grid%08i/%s" % (grid.id, field)][:]
        f.close()
        return ds.transpose()[:,:,None]

    def modify(self, field):
        pass

    def _read_fluid_selection(self, chunks, selector, fields, size):
        rv = {}
        # Now we have to do something unpleasant
        chunks = list(chunks)
        if selector.__class__.__name__ == "GridSelector":
            if not (len(chunks) == len(chunks[0].objs) == 1):
                raise RuntimeError
            g = chunks[0].objs[0]
            f = h5py.File(g.filename, 'r')
            gds = f.get("/Grid%08i" % g.id)
            for ftype, fname in fields:
                rv[(ftype, fname)] = np.atleast_3d(gds.get(fname).value)
            f.close()
            return rv
        if size is None:
            size = sum((g.count(selector) for chunk in chunks
                        for g in chunk.objs))
        for field in fields:
            ftype, fname = field
            fsize = size
            rv[field] = np.empty(fsize, dtype="float64")
        ng = sum(len(c.objs) for c in chunks)
        mylog.debug("Reading %s cells of %s fields in %s grids",
                   size, [f2 for f1, f2 in fields], ng)
        ind = 0
        for chunk in chunks:
            f = None
            for g in chunk.objs:
                if f is None:
                    #print "Opening (count) %s" % g.filename
                    f = h5py.File(g.filename, "r")
                gds = f.get("/Grid%08i" % g.id)
                for field in fields:
                    ftype, fname = field
                    ds = np.atleast_3d(gds.get(fname).value.transpose())
                    nd = g.select(selector, ds, rv[field], ind) # caches
                ind += nd
            f.close()
        return rv

class IOHandlerPacked1D(IOHandlerPackedHDF5):

    _dataset_type = "enzo_packed_1d"
    _particle_reader = False

    def _read_data_set(self, grid, field):
        f = h5py.File(grid.filename, "r")
        ds = f["/Grid%08i/%s" % (grid.id, field)][:]
        f.close()
        return ds.transpose()[:,None,None]

    def modify(self, field):
        pass
