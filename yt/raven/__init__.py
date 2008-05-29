"""
Raven
=====

    Raven is the plotting interface, with support for several
    different engines.  Well, two for now, but maybe more later.
    Who knows?

G{packagetree}

@author: U{Matthew Turk<http://www.stanford.edu/~mturk/>}
@organization: U{KIPAC<http://www-group.slac.stanford.edu/KIPAC/>}
@contact: U{mturk@slac.stanford.edu<mailto:mturk@slac.stanford.edu>}
@license:
  Copyright (C) 2007 Matthew Turk.  All Rights Reserved.

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
from yt.config import ytcfg
from yt.logger import ravenLogger as mylog
from yt.arraytypes import *
import yt.lagos as lagos
try:
    import deliveration
except:
    mylog.warning("Deliverator import failed; all deliverator actions will fail!")

import matplotlib.image
import matplotlib.ticker
import matplotlib.axes
import matplotlib.figure
import matplotlib._image
import matplotlib.colors
import matplotlib.colorbar
import matplotlib.cm
import matplotlib.collections

import time, types, string, os

# @todo: Get rid of these
axis_labels = [('y','z'),('x','z'),('x','y')]
axis_names = {0: 'x', 1: 'y', 2: 'z'}

vm_axis_names = {0:'x', 1:'y', 2:'z', 3:'dx', 4:'dy'}

from ColorMaps import raven_colormaps, add_cmap

import PlotTypes
be = PlotTypes

from Callbacks import *

color_maps = matplotlib.cm.cmapnames + raven_colormaps.keys()

from PlotCollection import *
from PlotConfig import *
