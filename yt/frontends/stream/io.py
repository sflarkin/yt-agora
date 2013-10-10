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

import exceptions
import os
import numpy as np

from yt.utilities.io_handler import \
    BaseIOHandler, _axis_ids
from yt.utilities.logger import ytLogger as mylog
from yt.data_objects.yt_array import YTArray

class IOHandlerStream(BaseIOHandler):

    _data_style = "stream"

    def __init__(self, stream_handler):
        self.fields = stream_handler.fields
        self.field_units = stream_handler.field_units
        BaseIOHandler.__init__(self)

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
            rv[field] = YTArray(np.empty(size, dtype="float64"),
                                self.field_units[fname])
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
                    mask = g.select(selector) # caches
                    if mask is None: continue
                    ds = self.fields[g.id][fname]
                    data = ds[mask]
                    rv[field][ind:ind+data.size] = data
                    ind += data.size
        return rv

    def _read_particle_selection(self, chunks, selector, fields):
        chunks = list(chunks)
        if any((ftype != "all" for ftype, fname in fields)):
            raise NotImplementedError
        rv = {}
        # Now we have to do something unpleasant
        mylog.debug("First pass: counting particles.")
        size = 0
        pfields = [("all", "particle_position_%s" % ax) for ax in 'xyz']
        for chunk in chunks:
            for g in chunk.objs:
                if g.NumberOfParticles == 0: continue
                gf = self.fields[g.id]
                # Sometimes the stream operator won't have the 
                # ("all", "Something") fields, but instead just "Something".
                pns = []
                for pn in pfields:
                    if pn in gf: pns.append(pn)
                    else: pns.append(pn[1])
                size += g.count_particles(selector, 
                    gf[pns[0]], gf[pns[1]], gf[pns[2]])
        for field in fields:
            # TODO: figure out dataset types
            rv[field] = np.empty(size, dtype='float64')
        ng = sum(len(c.objs) for c in chunks)
        mylog.debug("Reading %s points of %s fields in %s grids",
                   size, [f2 for f1, f2 in fields], ng)
        ind = 0
        for chunk in chunks:
            for g in chunk.objs:
                if g.NumberOfParticles == 0: continue
                gf = self.fields[g.id]
                pns = []
                for pn in pfields:
                    if pn in gf: pns.append(pn)
                    else: pns.append(pn[1])
                mask = g.select_particles(selector,
                    gf[pns[0]], gf[pns[1]], gf[pns[2]])
                if mask is None: continue
                for field in set(fields):
                    if field in gf:
                        fn = field
                    else:
                        fn = field[1]
                    gdata = gf[fn][mask]
                    rv[field][ind:ind+gdata.size] = gdata
                ind += gdata.size
        return rv

    @property
    def _read_exception(self):
        return KeyError

