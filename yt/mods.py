"""
Very simple convenience function for importing all the modules, setting up
the namespace and getting the last argument on the command line.



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

from __future__ import absolute_import

#
# ALL IMPORTS GO HERE
#

# First module imports
import sys, types, os, glob, cPickle, time
import numpy as na # For historical reasons
import numpy as np # For modern purposes
import numpy # In case anyone wishes to use it by name

# This next item will handle most of the actual startup procedures, but it will
# also attempt to parse the command line and set up the global state of various
# operations.  The variable unparsed_args is not used internally but is
# provided as a convenience for users who wish to parse arguments in scripts.
# See http://lists.spacepope.org/pipermail/yt-dev-spacepope.org/2011-December/
#     001727.html
import yt.startup_tasks as __startup_tasks
unparsed_args = __startup_tasks.unparsed_args

from yt.funcs import *
from yt.utilities.logger import ytLogger as mylog
from yt.utilities.performance_counters import yt_counters, time_function
from yt.config import ytcfg, ytcfg_defaults
import yt.utilities.physical_constants as physical_constants
import yt.units as units
from yt.units.yt_array import YTArray, YTQuantity

from yt.utilities.logger import level as __level
if __level >= int(ytcfg_defaults["loglevel"]):
    # This won't get displayed.
    mylog.debug("Turning off NumPy error reporting")
    np.seterr(all = 'ignore')

from yt.fields.api import \
    field_plugins, \
    DerivedField, \
    FieldDetector, \
    FieldInfoContainer, \
    ValidateParameter, \
    ValidateDataField, \
    ValidateProperty, \
    ValidateSpatial, \
    ValidateGridType, \
    add_field, \
    derived_field

from yt.data_objects.api import \
    BinnedProfile1D, BinnedProfile2D, BinnedProfile3D, \
    data_object_registry, \
    DatasetSeries, AnalysisTask, analysis_task, \
    ImageArray, particle_filter, create_profile, \
    Profile1D, Profile2D, Profile3D

from yt.frontends.enzo.api import \
    EnzoDataset, EnzoDatasetInMemory, \
    EnzoSimulation, EnzoFieldInfo, add_enzo_field

# Boxlib stuff
from yt.frontends.boxlib.api import \
    BoxlibDataset

# Orion stuff
#from yt.frontends.boxlib.api import \
#    OrionDataset, OrionFieldInfo, add_orion_field

# Maestro stuff
#from yt.frontends.boxlib.api import \
#    MaestroDataset

# Castro stuff
#from yt.frontends.boxlib.api import \
#    CastroDataset

from yt.frontends.flash.api import \
    FLASHDataset, FLASHFieldInfo

from yt.frontends.artio.api import \
    ARTIODataset, ARTIOFieldInfo

from yt.frontends.ramses.api import \
    RAMSESDataset, RAMSESFieldInfo

from yt.frontends.halo_catalogs.api import \
    HaloCatalogDataset, HaloCatalogFieldInfo, \
    RockstarDataset, RockstarFieldInfo

#from yt.frontends.chombo.api import \
#    ChomboDataset, ChomboFieldInfo, add_chombo_field

#from yt.frontends.gdf.api import \
#    GDFDataset, GDFFieldInfo, add_gdf_field

from yt.frontends.moab.api import \
    MoabHex8Dataset, MoabFieldInfo, \
    PyneMoabHex8Dataset, PyneFieldInfo

from yt.frontends.athena.api import \
    AthenaDataset, AthenaFieldInfo

from yt.frontends.art.api import \
    ARTDataset, ARTFieldInfo

#from yt.frontends.pluto.api import \
#     PlutoDataset, PlutoFieldInfo, add_pluto_field

from yt.frontends.stream.api import \
    StreamDataset, \
    StreamHandler, load_uniform_grid, load_amr_grids, \
    load_particles, load_hexahedral_mesh, load_octree

from yt.frontends.sph.api import \
    OWLSDataset, SPHFieldInfo, \
    GadgetDataset, GadgetHDF5Dataset, \
    TipsyDataset

# For backwards compatibility
GadgetStaticOutput = deprecated_class(GadgetDataset)
TipsyStaticOutput = deprecated_class(TipsyDataset)

#from yt.analysis_modules.list_modules import \
#    get_available_modules, amods
#available_analysis_modules = get_available_modules()

from yt.frontends.fits.api import \
    FITSDataset, FITSFieldInfo

# Import our analysis modules
from yt.analysis_modules.halo_finding.api import \
    HaloFinder

from yt.utilities.definitions import \
    axis_names, x_dict, y_dict, inv_axis_names

# Now individual component imports from the visualization API
from yt.visualization.api import \
    PlotCollection, PlotCollectionInteractive, \
    get_multi_plot, FixedResolutionBuffer, ObliqueFixedResolutionBuffer, \
    callback_registry, write_bitmap, write_image, \
    apply_colormap, scale_image, write_projection, \
    SlicePlot, AxisAlignedSlicePlot, OffAxisSlicePlot, \
    ProjectionPlot, OffAxisProjectionPlot, \
    show_colormaps, ProfilePlot, PhasePlot

from yt.visualization.volume_rendering.api import \
    ColorTransferFunction, PlanckTransferFunction, ProjectionTransferFunction, \
    HomogenizedVolume, Camera, off_axis_projection, MosaicFisheyeCamera

from yt.utilities.parallel_tools.parallel_analysis_interface import \
    parallel_objects

for name, cls in callback_registry.items():
    exec("%s = cls" % name)

from yt.convenience import \
    load, projload, simulation

# Import some helpful math utilities
from yt.utilities.math_utils import \
    ortho_find, quartiles, periodic_position


# We load plugins.  Keep in mind, this can be fairly dangerous -
# the primary purpose is to allow people to have a set of functions
# that get used every time that they don't have to *define* every time.
# This way, other command-line tools can be used very simply.
# Unfortunately, for now, I think the easiest and simplest way of doing
# this is also the most dangerous way.
if ytcfg.getboolean("yt","loadfieldplugins"):
    my_plugin_name = ytcfg.get("yt","pluginfilename")
    # We assume that it is with respect to the $HOME/.yt directory
    if os.path.isfile(my_plugin_name):
        _fn = my_plugin_name
    else:
        _fn = os.path.expanduser("~/.yt/%s" % my_plugin_name)
    if os.path.isfile(_fn):
        mylog.info("Loading plugins from %s", _fn)
        execfile(_fn)
