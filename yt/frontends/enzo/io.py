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

from collections import defaultdict

try:
    from pyhdf_np import SD
except ImportError:
    pass

import exceptions
import os

from yt.utilities import hdf5_light_reader
from yt.utilities.io_handler import \
    BaseIOHandler, _axis_ids
from yt.utilities.logger import ytLogger as mylog
import h5py
from yt.geometry.selection_routines import \
    convert_mask_to_indices

import numpy as na
from yt.funcs import *

class IOHandlerEnzoHDF4(BaseIOHandler):

    _data_style = "enzo_hdf4"

    def modify(self, field):
        return field.swapaxes(0,2)

    def _read_field_names(self, grid):
        """
        Returns a list of fields associated with the filename
        Should *only* be called as EnzoGridInstance.getFields, never as getFields(object)
        """
        return SD.SD(grid.filename).datasets().keys()

    def _read_data_set(self, grid, field):
        """
        Returns after having obtained or generated a field.  Should throw an
        exception.  Should only be called as EnzoGridInstance.readData()

        @param field: field to read
        @type field: string
        """
        return SD.SD(grid.filename).select(field).get().swapaxes(0,2)

    def _read_data_slice(self, grid, field, axis, coord):
        """
        Reads a slice through the HDF4 data

        @param grid: Grid to slice
        @type grid: L{EnzoGrid<EnzoGrid>}
        @param field: field to get
        @type field: string
        @param sl: region to get
        @type sl: SliceType
        """
        sl = [slice(None), slice(None), slice(None)]
        sl[axis] = slice(coord, coord + 1)
        sl = tuple(reversed(sl))
        return SD.SD(grid.filename).select(field)[sl].swapaxes(0,2)

    @property
    def _read_exception(self):
        return SD.HDF4Error

class IOHandlerEnzoHDF4_2D(IOHandlerEnzoHDF4):

    _data_style = "enzo_hdf4_2d"

    def _read_data_set(self, grid, field):
        t = SD.SD(grid.filename).select(field).get()[:,:,None]
        return t.swapaxes(0,1)

    def _read_data_slice(self, grid, field, axis, coord):
        t = SD.SD(grid.filename).select(field).get()
        return t.transpose()

    def modify(self, field):
        return field

class IOHandlerEnzoHDF5(BaseIOHandler):

    _data_style = "enzo_hdf5"
    _particle_reader = True

    def _read_field_names(self, grid):
        """
        Returns a list of fields associated with the filename
        Should *only* be called as EnzoGridInstance.getFields, never as getFields(object)
        """
        return hdf5_light_reader.ReadListOfDatasets(grid.filename, "/")

    def _read_data_set(self, grid, field):
        return hdf5_light_reader.ReadData(grid.filename, "/%s" % field).swapaxes(0,2)

    def _read_data_slice(self, grid, field, axis, coord):
        """
        Reads a slice through the HDF5 data

        @param grid: Grid to slice
        @type grid: L{EnzoGrid<EnzoGrid>}
        @param field: field to get
        @type field: string
        @param axis: axis to slice along
        @param coord: coord to slice at
        """
        axis = {0:2,1:1,2:0}[axis]
        t = hdf5_light_reader.ReadDataSlice(grid.filename, "/%s" %
                        (field), axis, coord).transpose()
        return t

    def modify(self, field):
        return field.swapaxes(0,2)

    @property
    def _read_exception(self):
        return (exceptions.KeyError, hdf5_light_reader.ReadingError)

    def _read_particles(self, fields, rtype, args, grid_list, enclosed,
                        conv_factors):
        filenames = [g.filename for g in grid_list]
        ids = [g.id for g in grid_list]
        return hdf5_light_reader.ReadParticles(
            rtype, fields, filenames, ids, conv_factors, args, 0)

class IOHandlerPackedHDF5(BaseIOHandler):

    _data_style = "enzo_packed_3d"
    _particle_reader = True

    def _read_particles(self, fields, rtype, args, grid_list, enclosed,
                        conv_factors):
        filenames = [g.filename for g in grid_list]
        ids = [g.id for g in grid_list]
        filenames, ids, conv_factors = zip(*sorted(zip(filenames, ids, conv_factors)))
        return hdf5_light_reader.ReadParticles(
            rtype, fields, list(filenames), list(ids), conv_factors, args, 1)

    def modify(self, field):
        return field.swapaxes(0,2)

    def preload(self, grids, sets):
        if len(grids) == 0:
            data = None
            return
        # We need to deal with files first
        files_keys = defaultdict(lambda: [])
        pf_field_list = grids[0].pf.h.field_list
        sets = [dset for dset in list(sets) if dset in pf_field_list]
        for g in grids: files_keys[g.filename].append(g)
        exc = self._read_exception
        for file in files_keys:
            mylog.debug("Starting read %s (%s)", file, sets)
            nodes = [g.id for g in files_keys[file]]
            nodes.sort()
            # We want to pass on any error we might expect -- the preload
            # phase should be non-fatal in all cases, and instead dump back to
            # the grids.
            data = hdf5_light_reader.ReadMultipleGrids(file, nodes, sets)
            mylog.debug("Read %s items from %s", len(data), os.path.basename(file))
            for gid in data: self.queue[gid].update(data[gid])
        mylog.debug("Finished read of %s", sets)

    def _read_data_set(self, grid, field):
        return self.modify(hdf5_light_reader.ReadData(grid.filename,
                "/Grid%08i/%s" % (grid.id, field)))

    def _read_data_slice(self, grid, field, axis, coord):
        axis = _axis_ids[axis]
        return hdf5_light_reader.ReadDataSlice(grid.filename, "/Grid%08i/%s" %
                        (grid.id, field), axis, coord).transpose()

    def _read_field_names(self, grid):
        return hdf5_light_reader.ReadListOfDatasets(
                    grid.filename, "/Grid%08i" % grid.id)

    @property
    def _read_exception(self):
        return (exceptions.KeyError, hdf5_light_reader.ReadingError)

    def _read_selection(self, grids, selector, fields):
        last = None
        rv = {}
        counts = {}
        for g in grids:
            counts[g.id] = selector.count_cells(g)
        count = sum(counts.values())
        # Now we have to do something unpleasant
        grids = list(sorted(grids, key=lambda a: a.filename))
        last = grids[0].filename
        handle = h5py.File(last)
        for field in fields:
            ds = handle["/Grid%08i/%s" % (grids[0].id, field)]
            rv[field] = na.empty(count, dtype=ds.dtype)
        ind = 0
        mylog.info("Reading %s cells of %s fields in %s grids",
                   count, len(fields), len(grids))
        for i,g in enumerate(grids):
            if last != g.filename:
                handle.close()
                last = g.filename
                handle = h5py.File(last)
            mask = selector.fill_mask(g)
            c = counts[g.id]
            for field in fields:
                ds = handle["/Grid%08i/%s" % (g.id, field)]
                rv[field][ind:ind+c] = ds[:].transpose()[mask]
            ind += c
        handle.close()
        return rv

class IOHandlerPackedHDF5GhostZones(IOHandlerPackedHDF5):
    _data_style = "enzo_packed_3d_gz"

    def modify(self, field):
        tr = field[3:-3,3:-3,3:-3].swapaxes(0,2)
        return tr.copy() # To ensure contiguous

    def _read_data_slice(self, grid, field, axis, coord):
        axis = _axis_ids[axis]
        return hdf5_light_reader.ReadDataSlice(grid.filename, "/Grid%08i/%s" %
                        (grid.id, field), axis, coord)[3:-3,3:-3].transpose()

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

