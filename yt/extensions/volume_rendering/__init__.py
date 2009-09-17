"""
Import the components of the volume rendering extension

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2009 Matthew Turk.  All Rights Reserved.

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

import numpy as na

from VolumeIntegrator import PartitionedGrid

def partition_grid(start_grid, field, log_field = True):
    if len(start_grid.Children) == 0:
        if log_field: d = na.log10(start_grid[field])
        else: d = start_grid[field]
        pg = PartitionedGrid(
                d.astype('float64'),
                na.array(start_grid.LeftEdge, dtype='float64'),
                na.array(start_grid.RightEdge, dtype='float64'),
                na.array(start_grid.ActiveDimensions, dtype='int64'))
        return [pg]

    x_vert = [0, start_grid.ActiveDimensions[0]]
    y_vert = [0, start_grid.ActiveDimensions[1]]
    z_vert = [0, start_grid.ActiveDimensions[2]]

    for grid in start_grid.Children:
        gi = start_grid.get_global_startindex()
        si = grid.get_global_startindex()/2 - gi
        ei = si + grid.ActiveDimensions/2 
        x_vert += [si[0], ei[0]]
        y_vert += [si[1], ei[1]]
        z_vert += [si[2], ei[2]]

    cim = start_grid.child_index_mask

    # Now we sort by our vertices, in axis order

    x_vert.sort()
    y_vert.sort()
    z_vert.sort()

    grids = []

    for xs, xe in zip(x_vert[:-1], x_vert[1:]):
        for ys, ye in zip(y_vert[:-1], y_vert[1:]):
            for zs, ze in zip(z_vert[:-1], z_vert[1:]):
                sl = (slice(xs, xe), slice(ys, ye), slice(zs, ze))
                dd = cim[sl]
                if dd.size == 0: continue
                uniq = na.unique(dd)
                if uniq.size > 1: continue
                if uniq[0] > -1: continue
                data = start_grid[field][xs:xe,ys:ye,zs:ze].astype('float64')
                if log_field: na.log10(data, data)
                dims = na.array(dd.shape, dtype='int64')
                start_index = na.array([xs,ys,zs], dtype='int64')
                left_edge = start_grid.LeftEdge + start_index * start_grid.dds
                right_edge = left_edge + (dims + 1) * start_grid.dds
                grids.append(PartitionedGrid(
                    data, left_edge, right_edge, dims))

    return grids

