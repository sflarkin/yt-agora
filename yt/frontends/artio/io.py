"""
ARTIO-specific IO

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
import numpy as np

from yt.utilities.io_handler import \
    BaseIOHandler
from yt.utilities.logger import ytLogger as mylog
import yt.utilities.fortran_utils as fpu
import cStringIO

class IOHandlerARTIO(BaseIOHandler):
    _data_style = "artio"

    def _read_fluid_selection(self, chunks, selector, fields, size):
        tr = dict((f, np.empty(size, dtype='float64')) for f in fields)
        cp = 0
        for chunk in chunks:
            for subset in chunk.objs:
                print 'reading from ',fields, subset.domain.grid_fn
                rv = subset.fill(fields) 
                for ft, f in fields:
                    mylog.debug("Filling %s with %s (%0.3e %0.3e) (%s:%s)",
                        f, subset.cell_count, rv[f].min(), rv[f].max(),
                        cp, cp+subset.cell_count)
                    tr[(ft, f)][cp:cp+subset.cell_count] = rv.pop(f)
                cp += subset.cell_count
        return tr

    def _read_particle_selection(self, chunks, selector, fields):
        raise NotImplementedError #snl 

        size = 0
        masks = {}
        for chunk in chunks:
            for subset in chunk.objs:
                # We read the whole thing, then feed it back to the selector
                offsets = []
                f = open(subset.domain.part_fn, "rb")
                foffsets = subset.domain.particle_field_offsets
                selection = {}
                for ax in 'xyz':
                    field = "particle_position_%s" % ax
                    f.seek(foffsets[field])
                    selection[ax] = fpu.read_vector(f, 'd')
                mask = selector.select_points(selection['x'],
                            selection['y'], selection['z'])
                if mask is None: continue
                size += mask.sum()
                masks[id(subset)] = mask
        # Now our second pass
        tr = dict((f, np.empty(size, dtype="float64")) for f in fields)
        for chunk in chunks:
            for subset in chunk.objs:
                f = open(subset.domain.part_fn, "rb")
                mask = masks.pop(id(subset), None)
                if mask is None: continue
                for ftype, fname in fields:
                    offsets.append((foffsets[fname], (ftype,fname)))
                for offset, field in sorted(offsets):
                    f.seek(offset)
                    tr[field] = fpu.read_vector(f, 'd')[mask]
        return tr

