"""
Raven
=====

Raven is the plotting interface, with support for several
different engines.  Well, two for now, but maybe more later.
Who knows?

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2007-2009 Matthew Turk.  All Rights Reserved.

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
from yt.funcs import *
import yt.lagos as lagos

import matplotlib
matplotlib.rc('contour', negative_linestyle='solid')

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

from ColorMaps import raven_colormaps, add_cmap, check_color

import PlotTypes
be = PlotTypes

from Callbacks import *
from FixedResolution import *

if 'cmapnames' in dir(matplotlib.cm):
    color_maps = matplotlib.cm.cmapnames + raven_colormaps.keys()
else:
    color_maps = matplotlib.cm._cmapnames + raven_colormaps.keys()
default_cmap = ytcfg.get("raven", "colormap")
if default_cmap != "jet":
    mylog.info("Setting default colormap to %s", default_cmap)
    matplotlib.rc('image', cmap=default_cmap)

from PlotCollection import *
try:
    from PlotConfig import *
except ImportError:
    mylog.warn("No automated plotting.  Thanks, elementtree!")

try:
    from Plot3DInterface import *
except ImportError:
    mylog.debug("S2PLOT interface not available")
