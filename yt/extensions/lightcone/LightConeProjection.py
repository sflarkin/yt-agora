"""
Create randomly centered, tiled projections to be used in light cones.

Author: Britton Smith <brittons@origins.colorado.edu>
Affiliation: CASA/University of CO, Boulder
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2008-2009 Britton Smith.  All Rights Reserved.

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

from yt.extensions.lightcone import *
from yt.logger import lagosLogger as mylog
from yt.config import ytcfg
import yt.lagos as lagos
import yt.raven as raven
import numpy as na
import copy

@lagos.parallel_blocking_call
def LightConeProjection(lightConeSlice, field, pixels, weight_field=None, save_image=False, name="", node=None, field_cuts=None,
                        add_redshift_label=False, **kwargs):
    "Create a single projection to be added into the light cone stack."

    # Use some projection parameters to seed random number generator to make unique node name.
    # We are just saving the projection object, so only the projection axis needs to be considered 
    # since the lateral shifting and tiling occurs after the projection object is made.
    # Likewise, only the DepthBoxFraction needs to be considered.

    # ProjectionAxis
    # ProjectionCenter[ProjectionAxis]
    # DepthBoxFraction

    # Name node with user specified keyword if given with 'node' keyword.
    node_name = "LightCone_%s_%d_%f_%f" % (node, lightConeSlice['ProjectionAxis'],
                                           lightConeSlice['ProjectionCenter'][lightConeSlice['ProjectionAxis']],
                                           lightConeSlice['DepthBoxFraction'])

    mylog.info("Making projection at z = %f from %s." % (lightConeSlice['redshift'], lightConeSlice['filename']))

    region_center = [0.5 * (lightConeSlice['object'].parameters['DomainRightEdge'][q] +
                            lightConeSlice['object'].parameters['DomainLeftEdge'][q]) \
                         for q in range(lightConeSlice['object'].parameters['TopGridRank'])]

    # Make the plot collection and put it in the slice so we can delete it cleanly in the same scope 
    # as where the frb will be deleted.
    lightConeSlice['pc'] = raven.PlotCollection(lightConeSlice['object'], center=region_center)

    # 1. The Depth Problem
    # Use coordinate field cut in line of sight to cut projection to proper depth.
    if field_cuts is None:
        these_field_cuts = []
    else:
        these_field_cuts = copy.deepcopy(field_cuts)

    if (lightConeSlice['DepthBoxFraction'] < 1):
        axis = ('x', 'y', 'z')[lightConeSlice['ProjectionAxis']]
        depthLeft = lightConeSlice['ProjectionCenter'][lightConeSlice['ProjectionAxis']] - 0.5 * lightConeSlice['DepthBoxFraction']
        depthRight = lightConeSlice['ProjectionCenter'][lightConeSlice['ProjectionAxis']] + 0.5 * lightConeSlice['DepthBoxFraction']
        if (depthLeft < 0):
            cut_mask = "((grid[\"%s\"] + 0.5*grid[\"d%s\"] >= 0) & (grid[\"%s\"] - 0.5*grid[\"d%s\"] <= %f)) | ((grid[\"%s\"] + 0.5*grid[\"d%s\"] >= %f) & (grid[\"%s\"] - 0.5*grid[\"d%s\"] <= 1))" % \
                (axis, axis, axis, axis, depthRight, axis, axis, (depthLeft+1), axis, axis)
        elif (depthRight > 1):
            cut_mask = "((grid[\"%s\"] + 0.5*grid[\"d%s\"] >= 0) & (grid[\"%s\"] - 0.5*grid[\"d%s\"] <= %f)) | ((grid[\"%s\"] + 0.5*grid[\"d%s\"] >= %f) & (grid[\"%s\"] - 0.5*grid[\"d%s\"] <= 1))" % \
                (axis, axis, axis, axis, (depthRight-1), axis, axis, depthLeft, axis, axis)
        else:
            cut_mask = "(grid[\"%s\"] + 0.5*grid[\"d%s\"] >= %f) & (grid[\"%s\"] - 0.5*grid[\"%s\"] <= %f)" % (axis, axis, depthLeft, axis, axis, depthRight)

        these_field_cuts.append(cut_mask)

    # Make projection.
    lightConeSlice['pc'].add_projection(field, lightConeSlice['ProjectionAxis'], weight_field=weight_field, field_cuts=these_field_cuts,
                                        node_name=node_name, **kwargs)

    # If parallel: all the processes have the whole projection object, but we only need to do the tiling, shifting, and cutting once.
    if ytcfg.getint("yt", "__parallel_rank") == 0:

        # 2. The Tile Problem
        # Tile projection to specified width.

        # Original projection data.
        original_px = copy.deepcopy(lightConeSlice['pc'].plots[0].data['px'])
        original_py = copy.deepcopy(lightConeSlice['pc'].plots[0].data['py'])
        original_pdx = copy.deepcopy(lightConeSlice['pc'].plots[0].data['pdx'])
        original_pdy = copy.deepcopy(lightConeSlice['pc'].plots[0].data['pdy'])
        original_field = copy.deepcopy(lightConeSlice['pc'].plots[0].data[field])
        original_weight_field = copy.deepcopy(lightConeSlice['pc'].plots[0].data['weight_field'])

        # Copy original into offset positions to make tiles.
        for x in range(int(na.ceil(lightConeSlice['WidthBoxFraction']))):
            for y in range(int(na.ceil(lightConeSlice['WidthBoxFraction']))):
                if ((x + y) > 0):
                    lightConeSlice['pc'].plots[0].data['px'] = na.concatenate([lightConeSlice['pc'].plots[0].data['px'], original_px+x])
                    lightConeSlice['pc'].plots[0].data['py'] = na.concatenate([lightConeSlice['pc'].plots[0].data['py'], original_py+y])
                    lightConeSlice['pc'].plots[0].data['pdx'] = na.concatenate([lightConeSlice['pc'].plots[0].data['pdx'], original_pdx])
                    lightConeSlice['pc'].plots[0].data['pdy'] = na.concatenate([lightConeSlice['pc'].plots[0].data['pdy'], original_pdy])
                    lightConeSlice['pc'].plots[0].data[field] = na.concatenate([lightConeSlice['pc'].plots[0].data[field], original_field])
                    lightConeSlice['pc'].plots[0].data['weight_field'] = na.concatenate([lightConeSlice['pc'].plots[0].data['weight_field'], original_weight_field])

        # Delete originals.
        del original_px
        del original_py
        del original_pdx
        del original_pdy
        del original_field
        del original_weight_field

        # 3. The Shift Problem
        # Shift projection by random x and y offsets.

        offset = copy.deepcopy(lightConeSlice['ProjectionCenter'])
        # Delete depth coordinate.
        del offset[lightConeSlice['ProjectionAxis']]

        # Shift x and y positions.
        lightConeSlice['pc'].plots[0]['px'] -= offset[0]
        lightConeSlice['pc'].plots[0]['py'] -= offset[1]

        # Wrap off-edge cells back around to other side (periodic boundary conditions).
        lightConeSlice['pc'].plots[0]['px'][lightConeSlice['pc'].plots[0]['px'] < 0] += na.ceil(lightConeSlice['WidthBoxFraction'])
        lightConeSlice['pc'].plots[0]['py'][lightConeSlice['pc'].plots[0]['py'] < 0] += na.ceil(lightConeSlice['WidthBoxFraction'])

        # After shifting, some cells have fractional coverage on both sides of the box.
        # Find those cells and make copies to be placed on the other side.

        # Cells hanging off the right edge.
        add_x_right = lightConeSlice['pc'].plots[0]['px'] + 0.5 * lightConeSlice['pc'].plots[0]['pdx'] > na.ceil(lightConeSlice['WidthBoxFraction'])
        add_x_px = lightConeSlice['pc'].plots[0]['px'][add_x_right]
        add_x_px -= na.ceil(lightConeSlice['WidthBoxFraction'])
        add_x_py = lightConeSlice['pc'].plots[0]['py'][add_x_right]
        add_x_pdx = lightConeSlice['pc'].plots[0]['pdx'][add_x_right]
        add_x_pdy = lightConeSlice['pc'].plots[0]['pdy'][add_x_right]
        add_x_field = lightConeSlice['pc'].plots[0][field][add_x_right]
        add_x_weight_field = lightConeSlice['pc'].plots[0]['weight_field'][add_x_right]
        del add_x_right

        # Cells hanging off the left edge.
        add_x_left = lightConeSlice['pc'].plots[0]['px'] - 0.5 * lightConeSlice['pc'].plots[0]['pdx'] < 0
        add2_x_px = lightConeSlice['pc'].plots[0]['px'][add_x_left]
        add2_x_px += na.ceil(lightConeSlice['WidthBoxFraction'])
        add2_x_py = lightConeSlice['pc'].plots[0]['py'][add_x_left]
        add2_x_pdx = lightConeSlice['pc'].plots[0]['pdx'][add_x_left]
        add2_x_pdy = lightConeSlice['pc'].plots[0]['pdy'][add_x_left]
        add2_x_field = lightConeSlice['pc'].plots[0][field][add_x_left]
        add2_x_weight_field = lightConeSlice['pc'].plots[0]['weight_field'][add_x_left]
        del add_x_left

        # Cells hanging off the top edge.
        add_y_right = lightConeSlice['pc'].plots[0]['py'] + 0.5 * lightConeSlice['pc'].plots[0]['pdy'] > na.ceil(lightConeSlice['WidthBoxFraction'])
        add_y_px = lightConeSlice['pc'].plots[0]['px'][add_y_right]
        add_y_py = lightConeSlice['pc'].plots[0]['py'][add_y_right]
        add_y_py -= na.ceil(lightConeSlice['WidthBoxFraction'])
        add_y_pdx = lightConeSlice['pc'].plots[0]['pdx'][add_y_right]
        add_y_pdy = lightConeSlice['pc'].plots[0]['pdy'][add_y_right]
        add_y_field = lightConeSlice['pc'].plots[0][field][add_y_right]
        add_y_weight_field = lightConeSlice['pc'].plots[0]['weight_field'][add_y_right]
        del add_y_right

        # Cells hanging off the bottom edge.
        add_y_left = lightConeSlice['pc'].plots[0]['py'] - 0.5 * lightConeSlice['pc'].plots[0]['pdy'] < 0
        add2_y_px = lightConeSlice['pc'].plots[0]['px'][add_y_left]
        add2_y_py = lightConeSlice['pc'].plots[0]['py'][add_y_left]
        add2_y_py += na.ceil(lightConeSlice['WidthBoxFraction'])
        add2_y_pdx = lightConeSlice['pc'].plots[0]['pdx'][add_y_left]
        add2_y_pdy = lightConeSlice['pc'].plots[0]['pdy'][add_y_left]
        add2_y_field = lightConeSlice['pc'].plots[0][field][add_y_left]
        add2_y_weight_field = lightConeSlice['pc'].plots[0]['weight_field'][add_y_left]
        del add_y_left

        # Add the hanging cells back to the projection data.
        lightConeSlice['pc'].plots[0].data['px'] = na.concatenate([lightConeSlice['pc'].plots[0]['px'], add_x_px, add_y_px, add2_x_px, add2_y_px])
        lightConeSlice['pc'].plots[0].data['py'] = na.concatenate([lightConeSlice['pc'].plots[0]['py'], add_x_py, add_y_py, add2_x_py, add2_y_py])
        lightConeSlice['pc'].plots[0].data['pdx'] = na.concatenate([lightConeSlice['pc'].plots[0]['pdx'], add_x_pdx, add_y_pdx, add2_x_pdx, add2_y_pdx])
        lightConeSlice['pc'].plots[0].data['pdy'] = na.concatenate([lightConeSlice['pc'].plots[0]['pdy'], add_x_pdy, add_y_pdy, add2_x_pdy, add2_y_pdy])
        lightConeSlice['pc'].plots[0].data[field] = na.concatenate([lightConeSlice['pc'].plots[0][field], add_x_field, add_y_field, add2_x_field, add2_y_field])
        lightConeSlice['pc'].plots[0].data['weight_field'] = na.concatenate([lightConeSlice['pc'].plots[0]['weight_field'], add_x_weight_field, add_y_weight_field,
                                                                             add2_x_weight_field, add2_y_weight_field])

        # Delete original copies of hanging cells.
        del add_x_px, add_y_px, add2_x_px, add2_y_px
        del add_x_py, add_y_py, add2_x_py, add2_y_py
        del add_x_pdx, add_y_pdx, add2_x_pdx, add2_y_pdx
        del add_x_pdy, add_y_pdy, add2_x_pdy, add2_y_pdy
        del add_x_field, add_y_field, add2_x_field, add2_y_field
        del add_x_weight_field, add_y_weight_field, add2_x_weight_field, add2_y_weight_field

        # Tiles were made rounding up the width to the nearest integer.
        # Cut off the edges to get the specified width.
        # Cut in the x direction.
        cut_x = lightConeSlice['pc'].plots[0].data['px'] - 0.5 * lightConeSlice['pc'].plots[0].data['pdx'] < lightConeSlice['WidthBoxFraction']
        lightConeSlice['pc'].plots[0].data['px'] = lightConeSlice['pc'].plots[0].data['px'][cut_x]
        lightConeSlice['pc'].plots[0].data['py'] = lightConeSlice['pc'].plots[0].data['py'][cut_x]
        lightConeSlice['pc'].plots[0].data['pdx'] = lightConeSlice['pc'].plots[0].data['pdx'][cut_x]
        lightConeSlice['pc'].plots[0].data['pdy'] = lightConeSlice['pc'].plots[0].data['pdy'][cut_x]
        lightConeSlice['pc'].plots[0].data[field] = lightConeSlice['pc'].plots[0].data[field][cut_x]
        lightConeSlice['pc'].plots[0].data['weight_field'] = lightConeSlice['pc'].plots[0].data['weight_field'][cut_x]
        del cut_x

        # Cut in the y direction.
        cut_y = lightConeSlice['pc'].plots[0].data['py'] - 0.5 * lightConeSlice['pc'].plots[0].data['pdy'] < lightConeSlice['WidthBoxFraction']
        lightConeSlice['pc'].plots[0].data['px'] = lightConeSlice['pc'].plots[0].data['px'][cut_y]
        lightConeSlice['pc'].plots[0].data['py'] = lightConeSlice['pc'].plots[0].data['py'][cut_y]
        lightConeSlice['pc'].plots[0].data['pdx'] = lightConeSlice['pc'].plots[0].data['pdx'][cut_y]
        lightConeSlice['pc'].plots[0].data['pdy'] = lightConeSlice['pc'].plots[0].data['pdy'][cut_y]
        lightConeSlice['pc'].plots[0].data[field] = lightConeSlice['pc'].plots[0].data[field][cut_y]
        lightConeSlice['pc'].plots[0].data['weight_field'] = lightConeSlice['pc'].plots[0].data['weight_field'][cut_y]
        del cut_y

        # Save an image if requested.
        if save_image:
            lightConeSlice['pc'].set_xlim(0, lightConeSlice['WidthBoxFraction'])
            lightConeSlice['pc'].set_ylim(0, lightConeSlice['WidthBoxFraction'])
            if add_redshift_label:
                lightConeSlice['pc'].plots[-1].modify['text']((0.5, 0.03), "z = %.3f" % lightConeSlice['redshift'], dict(color='black',size=50))
            lightConeSlice['pc'].save(name)

        # Create fixed resolution buffer to return back to the light cone object.
        # These buffers will be stacked together to make the light cone.
        frb = raven.FixedResolutionBuffer(lightConeSlice['pc'].plots[0].data, (0, lightConeSlice['WidthBoxFraction'], 0, lightConeSlice['WidthBoxFraction']),
                                          (pixels, pixels), antialias=False)

        return frb
    else:
        # If running in parallel and this is not the root process, return None.
        return None
