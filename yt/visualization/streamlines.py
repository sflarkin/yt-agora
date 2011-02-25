"""
Import the components of the volume rendering extension

Author: Samuel Skillman <samskillman@gmail.com>
Affiliation: University of Colorado
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2010 Samuel Skillman.  All Rights Reserved.

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
from yt.funcs import *
from yt.utilities.parallel_tools.parallel_analysis_interface import \
    ParallelAnalysisInterface, parallel_passthrough
from yt.utilities.amr_kdtree.api import AMRKDTree

class Streamlines(ParallelAnalysisInterface):
    r"""A collection of streamlines that flow through the volume

    The Streamlines object contains a collection of streamlines
    defined as paths that are parallel to a specified vector field.  

    Parameters
    ----------
    pf : `~yt.lagos.StaticOutput`
        This is the parameter file to streamline
    pos : array_like
        An array of initial starting positions of the streamlines.
    xfield: field
        The x component of the vector field to be streamlined.
    yfield: field
        The y component of the vector field to be streamlined.
    zfield: field
        The z component of the vector field to be streamlined.
    volume : `yt.extensions.volume_rendering.HomogenizedVolume`, optional
        The volume to be streamlined.  Can be specified for
        finer-grained control, but otherwise will be automatically
        generated.  At this point it must use the AMRKDTree. 
        Default: None
    dx : float, optional
        Optionally specify the step size during the integration.
        Default: minimum dx
    length : float, optional
        Optionally specify the length of integration.  
        Default: na.max(self.pf.domain_right_edge-self.pf.domain_left_edge)
    direction : real, optional
        Specifies the direction of integration.  The magnitude of this
        value has no effect, only the sign.
    
    Examples
    --------
    >>> from yt.mods import *
    >>> from yt.visualization.api import Streamlines
    >>> pf = load('DD1701') # Load pf

    >>> c = na.array([0.5]*3)
    >>> N = 100
    >>> scale = 1.0
    >>> pos_dx = na.random.random((N,3))*scale-scale/2.
    >>> pos = c+pos_dx
    
    >>> streamlines = Streamlines(pf,pos,'x-velocity', 'y-velocity', 'z-velocity', length=1.0) 
    >>> streamlines.integrate_through_volume()
    
    >>> import matplotlib.pylab as pl
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> fig=pl.figure() 
    >>> ax = Axes3D(fig)
    >>> for stream in streamlines.streamlines:
    >>>     stream = stream[na.all(stream != 0.0, axis=1)]
    >>>     ax.plot3D(stream[:,0], stream[:,1], stream[:,2], alpha=0.1)
    >>> pl.savefig('streamlines.png')
    """
    def __init__(self, pf, positions, xfield, yfield, zfield, volume=None,
                 dx=None, length=None, direction=1):
        self.pf = pf
        self.start_positions = positions
        self.N = self.start_positions.shape[0]
        self.xfield = xfield
        self.yfield = yfield
        self.zfield = zfield
        self.direction = na.sign(direction)
        if volume is None:
            volume = AMRKDTree(self.pf, fields=[self.xfield,self.yfield,self.zfield],
                            log_fields=[False,False,False], merge_trees=True)
        self.volume = volume
        if dx is None:
            dx = self.pf.h.get_smallest_dx()
        self.dx = dx
        if length is None:
            length = na.max(self.pf.domain_right_edge-self.pf.domain_left_edge)
        self.length = length
        self.steps = int(length/dx)
        self.streamlines = na.zeros((self.N,self.steps,3), dtype='float64')

    def integrate_through_volume(self):
        nprocs = self._mpi_get_size()
        my_rank = self._mpi_get_rank()
        self.streamlines[my_rank::nprocs,0,:] = self.start_positions[my_rank::nprocs]

        pbar = get_pbar("Streamlining", self.N)
        for i,stream in enumerate(self.streamlines[my_rank::nprocs]):
            step = self.steps
            while (step > 1):
                this_brick = self.volume.locate_brick(stream[-step,:])
                step = self._integrate_through_brick(this_brick, stream, step)
            pbar.update(i)
        pbar.finish()
        
        self._finalize_parallel(None)

    @parallel_passthrough
    def _finalize_parallel(self,data):
        self.streamlines = self._mpi_allsum(self.streamlines)
        
    def _integrate_through_brick(self, node, stream, step, periodic=False):
        while (step > 1):
            self.volume.get_brick_data(node)
            brick = node.brick
            stream[-step+1] = stream[-step]
            brick.integrate_streamline(stream[-step+1], self.direction*self.dx)
            if na.any(stream[-step+1,:] <= self.pf.domain_left_edge) | \
                   na.any(stream[-step+1,:] >= self.pf.domain_right_edge):
                return 0

            if na.any(stream[-step+1,:] < node.l_corner) | \
                   na.any(stream[-step+1,:] >= node.r_corner):
                return step-1
            step -= 1
        return step

    
    

        
