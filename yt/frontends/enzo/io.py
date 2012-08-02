"""
Enzo-specific IO functions

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
Homepage: http://yt-project.org/
License:
  Copyright (C) 2007-2011 Matthew Turk.  All Rights Reserved.

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

import exceptions
import os

from yt.utilities import hdf5_light_reader
from yt.utilities.io_handler import \
    BaseIOHandler, _axis_ids
from yt.utilities.logger import ytLogger as mylog
import h5py

import numpy as na
from yt.funcs import *

class IOHandlerPackedHDF5(BaseIOHandler):

    _data_style = "enzo_packed_3d"
    _base = slice(None)

    def _read_data_set(self, grid, field):
        handle = h5py.File(grid.filename)
        tr = handle["/Grid%08i/%s" % (grid.id, field)][:]
        return tr.swapaxes(0, 2)

    def _read_field_names(self, grid):
        return hdf5_light_reader.ReadListOfDatasets(
                    grid.filename, "/Grid%08i" % grid.id)

    @property
    def _read_exception(self):
        return (exceptions.KeyError, hdf5_light_reader.ReadingError)

    def _read_particle_selection(self, chunks, selector, fields):
        last = None
        rv = {}
        chunks = list(chunks)
        # Now we have to do something unpleasant
        dobj = chunks[0].dobj
        if "particle_type" not in dobj.pf.h.field_list and \
           any((ftype != "all" for ftype, fname in fields)):
            raise NotImplementedError("Sorry, but you'll have to manually " +
                    "discriminate without a particle_type field")
        mylog.debug("First pass: counting particles.")
        last = chunks[0].objs[0].filename
        handle = h5py.File(last)
        xn, yn, zn = ("particle_position_%s" % ax for ax in 'xyz')
        size = 0
        for chunk in chunks:
            for g in chunk.objs:
                if g.NumberOfParticles == 0: continue
                if last != g.filename:
                    last = g.filename
                    handle = h5py.File(last)
                base = handle["/Grid%08i" % g.id]
                x, y, z = (base["particle_position_%s" %
                            ax][:].astype("float64") for ax in 'xyz')
                size += g.count_particles(selector, x, y, z)
            handle.close()
        last = chunks[0].objs[0].filename
        handle = h5py.File(last)
        for field in fields:
            # TODO: figure out dataset types
            rv[field] = na.empty(size, dtype='float64')
        ng = sum(len(c.objs) for c in chunks)
        mylog.debug("Reading %s cells of %s fields in %s grids",
                   size, [f2 for f1, f2 in fields], ng)
        ind = 0
        for chunk in chunks:
            for g in chunk.objs:
                if g.NumberOfParticles == 0: continue
                if last != g.filename:
                    last = g.filename
                    handle = h5py.File(last)
                base = handle["/Grid%08i" % g.id]
                x, y, z = (base["particle_position_%s" %
                            ax][:].astype("float64") for ax in 'xyz')
                mask = g.select_particles(selector, x, y, z)
                if mask is None: continue
                for field in fields:
                    ftype, fname = field
                    ds = handle["/Grid%08i/%s" % (g.id, fname)]
                    data = ds[self._base][mask]
                    data 
                    rv[field][ind:ind+data.size] = data
                ind += data.size
            handle.close()
        return rv
        
    def _read_fluid_selection(self, chunks, selector, fields, size):
        last = None
        rv = {}
        if any((ftype != "gas" for ftype, fname in fields)):
            raise NotImplementedError

        # Now we have to do something unpleasant
        chunks = list(chunks)
        last = chunks[0].objs[0].filename
        handle = h5py.File(last)
        for field in fields:
            ftype, fname = field
            ds = handle["/Grid%08i/%s" % (chunks[0].objs[0].id, fname)]
            fsize = size
            rv[field] = na.empty(fsize, dtype=ds.dtype)
        ind = 0
        ng = sum(len(c.objs) for c in chunks)
        mylog.debug("Reading %s cells of %s fields in %s grids",
                   size, [f2 for f1, f2 in fields], ng)
        for chunk in chunks:
            for g in chunk.objs:
                if last != g.filename:
                    last = g.filename
                    handle = h5py.File(last)
                mask = g.select(selector)
                if mask is None: continue
                for field in fields:
                    ftype, fname = field
                    ds = handle["/Grid%08i/%s" % (g.id, fname)]
                    data = ds[self._base].swapaxes(0,2)[mask]
                    rv[field][ind:ind+data.size] = data
                ind += data.size
            handle.close()
        return rv

class IOHandlerPackedHDF5GhostZones(IOHandlerPackedHDF5):
    _data_style = "enzo_packed_3d_gz"
    _base = (slice(3, -3), slice(3, -3), slice(3, -3))

    def _read_raw_data_set(self, grid, field):
        return hdf5_light_reader.ReadData(grid.filename,
                "/Grid%08i/%s" % (grid.id, field))

class IOHandlerInMemory(BaseIOHandler):

    _data_style = "enzo_inline"

    def __init__(self, ghost_zones=3):
        import enzo
        self.enzo = enzo
        self.grids_in_memory = enzo.grid_data
        self.old_grids_in_memory = enzo.old_grid_data
        self.my_slice = (slice(ghost_zones,-ghost_zones),
                      slice(ghost_zones,-ghost_zones),
                      slice(ghost_zones,-ghost_zones))
        BaseIOHandler.__init__(self)

    def _read_data_set(self, grid, field):
        if grid.id not in self.grids_in_memory:
            mylog.error("Was asked for %s but I have %s", grid.id, self.grids_in_memory.keys())
            raise KeyError
        tr = self.grids_in_memory[grid.id][field]
        # If it's particles, we copy.
        if len(tr.shape) == 1: return tr.copy()
        # New in-place unit conversion breaks if we don't copy first
        return tr.swapaxes(0,2)[self.my_slice].copy()
        # We don't do this, because we currently do not interpolate
        coef1 = max((grid.Time - t1)/(grid.Time - t2), 0.0)
        coef2 = 1.0 - coef1
        t1 = enzo.yt_parameter_file["InitialTime"]
        t2 = enzo.hierarchy_information["GridOldTimes"][grid.id]
        return (coef1*self.grids_in_memory[grid.id][field] + \
                coef2*self.old_grids_in_memory[grid.id][field])\
                [self.my_slice]

    def modify(self, field):
        return field.swapaxes(0,2)

    def _read_field_names(self, grid):
        return self.grids_in_memory[grid.id].keys()

    def _read_data_slice(self, grid, field, axis, coord):
        sl = [slice(3,-3), slice(3,-3), slice(3,-3)]
        sl[axis] = slice(coord + 3, coord + 4)
        sl = tuple(reversed(sl))
        tr = self.grids_in_memory[grid.id][field][sl].swapaxes(0,2)
        # In-place unit conversion requires we return a copy
        return tr.copy()

    @property
    def _read_exception(self):
        return KeyError

class IOHandlerPacked2D(IOHandlerPackedHDF5):

    _data_style = "enzo_packed_2d"
    _particle_reader = False

    def _read_data_set(self, grid, field):
        return hdf5_light_reader.ReadData(grid.filename,
            "/Grid%08i/%s" % (grid.id, field)).transpose()[:,:,None]

    def modify(self, field):
        pass

    def _read_data_slice(self, grid, field, axis, coord):
        t = hdf5_light_reader.ReadData(grid.filename, "/Grid%08i/%s" %
                        (grid.id, field)).transpose()
        return t


class IOHandlerPacked1D(IOHandlerPackedHDF5):

    _data_style = "enzo_packed_1d"
    _particle_reader = False

    def _read_data_set(self, grid, field):
        return hdf5_light_reader.ReadData(grid.filename,
            "/Grid%08i/%s" % (grid.id, field)).transpose()[:,None,None]

    def modify(self, field):
        pass

    def _read_data_slice(self, grid, field, axis, coord):
        t = hdf5_light_reader.ReadData(grid.filename, "/Grid%08i/%s" %
                        (grid.id, field))
        return t

