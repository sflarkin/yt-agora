"""
Very simple convenience function for importing all the modules, setting up
the namespace and getting the last argument on the command line.

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2008-2009 Matthew Turk.  All Rights Reserved.

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

from __future__ import absolute_import

#
# ALL IMPORTS GO HERE
#

# First module imports
import numpy as na
import sys, types, os, glob, cPickle

from yt.utilities.logger import ytLogger as mylog
from yt.utilities.performance_counters import yt_counters, time_function

from yt.data_objects.api import \
    BinnedProfile1D, BinnedProfile2D, BinnedProfile3D, \
    data_object_registry, \
    derived_field, add_field, FieldInfo, \
    ValidateParameter, ValidateDataField, ValidateProperty, \
    ValidateSpatial, ValidateGridType

from yt.frontends.enzo.api import \
    EnzoStaticOutput, EnzoStaticOutputInMemory, EnzoFieldInfo, \
    add_enzo_field, add_enzo_1d_field, add_enzo_2d_field

from yt.frontends.orion.api import \
    OrionStaticOutput, OrionFieldInfo, add_orion_field

from yt.frontends.flash.api import \
    FLASHStaticOutput, FLASHFieldInfo, add_flash_field

from yt.frontends.tiger.api import \
    TigerStaticOutput, TigerFieldInfo, add_tiger_field

from yt.frontends.ramses.api import \
    RAMSESStaticOutput, RAMSESFieldInfo, add_ramses_field

from yt.frontends.chombo.api import \
    ChomboStaticOutput, ChomboFieldInfo, add_chombo_field

# Import our analysis modules
from yt.analysis_modules.api import \
    Clump, write_clump_hierarchy, find_clumps, write_clumps, \
    get_lowest_clumps, \
    HaloFinder, HOPHaloFinder, FOFHaloFinder, parallelHF, \
    TwoPointFunctions, FcnSet

from yt.utilities.definitions import \
    axis_names, x_dict, y_dict

# Now individual component imports from raven
from yt.raven import PlotCollection, PlotCollectionInteractive, \
        get_multi_plot, FixedResolutionBuffer, ObliqueFixedResolutionBuffer, \
        AnnuliProfiler
from yt.raven.Callbacks import callback_registry
for name, cls in callback_registry.items():
    exec("%s = cls" % name)

# Optional component imports from raven
try:
    from yt.raven import VolumeRenderingDataCube, \
        VolumeRendering3DProfile, HaloMassesPositionPlot
except ImportError:
    pass

import yt.raven.PlotInterface as plots

# Individual imports from Fido
from yt.fido import GrabCollections, OutputCollection

import yt.funcs

from yt.convenience import all_pfs, max_spheres, load, projload

# Some convenience functions to ease our time running scripts
# from the command line

def get_pf():
    return EnzoStaticOutput(sys.argv[-1])

def get_pc():
    return PlotCollection(EnzoStaticOutput(sys.argv[-1]))

