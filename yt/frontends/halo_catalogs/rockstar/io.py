"""
Rockstar data-file handling function




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

from yt.utilities.exceptions import *
from yt.funcs import mylog

from yt.utilities.io_handler import \
    BaseIOHandler

import yt.utilities.fortran_utils as fpu
from .definitions import halo_dt
from yt.utilities.lib.geometry_utils import compute_morton

from yt.geometry.oct_container import _ORDER_MAX

class IOHandlerRockstarBinary(BaseIOHandler):
    _dataset_type = "rockstar_binary"

    def _read_fluid_selection(self, chunks, selector, fields, size):
        raise NotImplementedError

    def _read_particle_coords(self, chunks, ptf):
        # This will read chunks and yield the results.
        chunks = list(chunks)
        data_files = set([])
        # Only support halo reading for now.
        assert(len(ptf) == 1)
        assert(ptf.keys()[0] == "halos")
        for chunk in chunks:
            for obj in chunk.objs:
                data_files.update(obj.data_files)
        for data_file in data_files:
            pcount = data_file.header['num_halos']
            with open(data_file.filename, "rb") as f:
                f.seek(data_file._position_offset, os.SEEK_SET)
                halos = np.fromfile(f, dtype=halo_dt, count = pcount)
                x = halos['particle_position_x'].astype("float64")
                y = halos['particle_position_y'].astype("float64")
                z = halos['particle_position_z'].astype("float64")
                yield "halos", (x, y, z)

    def _read_particle_fields(self, chunks, ptf, selector):
        # Now we have all the sizes, and we can allocate
        chunks = list(chunks)
        data_files = set([])
        # Only support halo reading for now.
        assert(len(ptf) == 1)
        assert(ptf.keys()[0] == "halos")
        for chunk in chunks:
            for obj in chunk.objs:
                data_files.update(obj.data_files)
        for data_file in data_files:
            pcount = data_file.header['num_halos']
            with open(data_file.filename, "rb") as f:
                for ptype, field_list in sorted(ptf.items()):
                    f.seek(data_file._position_offset, os.SEEK_SET)
                    halos = np.fromfile(f, dtype=halo_dt, count = pcount)
                    x = halos['particle_position_x'].astype("float64")
                    y = halos['particle_position_y'].astype("float64")
                    z = halos['particle_position_z'].astype("float64")
                    mask = selector.select_points(x, y, z)
                    del x, y, z
                    if mask is None: continue
                    for field in field_list:
                        data = halos[field][mask].astype("float64")
                        yield (ptype, field), data

    def _initialize_index(self, data_file, regions):
        pcount = data_file.header["num_halos"]
        morton = np.empty(pcount, dtype='uint64')
        mylog.debug("Initializing index % 5i (% 7i particles)",
                    data_file.file_id, pcount)
        ind = 0
        with open(data_file.filename, "rb") as f:
            f.seek(data_file._position_offset, os.SEEK_SET)
            halos = np.fromfile(f, dtype=halo_dt, count = pcount)
            pos = np.empty((halos.size, 3), dtype="float64")
            # These positions are in Mpc, *not* "code" units
            pos = data_file.pf.arr(pos, "code_length")
            dx = np.finfo(halos['particle_position_x'].dtype).eps
            dx = 2.0*self.pf.quan(dx, "code_length")
            pos[:,0] = halos["particle_position_x"]
            pos[:,1] = halos["particle_position_y"]
            pos[:,2] = halos["particle_position_z"]
            # These are 32 bit numbers, so we give a little lee-way.
            # Otherwise, for big sets of particles, we often will bump into the
            # domain edges.  This helps alleviate that.
            np.clip(pos, self.pf.domain_left_edge + dx,
                         self.pf.domain_right_edge - dx, pos)
            #del halos
            if np.any(pos.min(axis=0) < self.pf.domain_left_edge) or \
               np.any(pos.max(axis=0) > self.pf.domain_right_edge):
                raise YTDomainOverflow(pos.min(axis=0),
                                       pos.max(axis=0),
                                       self.pf.domain_left_edge,
                                       self.pf.domain_right_edge)
            regions.add_data_file(pos, data_file.file_id)
            morton[ind:ind+pos.shape[0]] = compute_morton(
                pos[:,0], pos[:,1], pos[:,2],
                data_file.pf.domain_left_edge,
                data_file.pf.domain_right_edge)
        return morton

    def _count_particles(self, data_file):
        return {'halos': data_file.header['num_halos']}

    def _identify_fields(self, data_file):
        fields = [("halos", f) for f in halo_dt.fields if
                  "padding" not in f]
        return fields
