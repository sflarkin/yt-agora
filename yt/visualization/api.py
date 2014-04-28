"""
API for yt.visualization



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

from color_maps import \
    add_cmap, \
    show_colormaps

from plot_collection import \
    PlotCollection, \
    PlotCollectionInteractive, \
    concatenate_pdfs

from fixed_resolution import \
    FixedResolutionBuffer, \
    ObliqueFixedResolutionBuffer

from image_writer import \
    multi_image_composite, \
    write_bitmap, \
    write_image, \
    map_to_colors, \
    splat_points, \
    annotate_image, \
    apply_colormap, \
    scale_image, \
    write_projection, \
    write_fits

from plot_modifications import \
    PlotCallback, \
    callback_registry

from easy_plots import \
    plot_type_registry

from streamlines import \
    Streamlines

from plot_window import \
    SlicePlot, \
    OffAxisSlicePlot, \
    ProjectionPlot, \
    OffAxisProjectionPlot

from profile_plotter import \
     ProfilePlot, \
     PhasePlot
    
from base_plot_types import \
    get_multi_plot

