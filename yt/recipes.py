"""
Helpful recipes to make tasty AMR desserts

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

# Named imports
import yt.lagos as lagos
import yt.raven as raven
from yt.funcs import *
import numpy as na
import os.path, inspect, types
from functools import wraps
from yt.logger import ytLogger as mylog

# These next couple functions are for 'fixing' arguments - accepting arguments
# of a couple different types and styles, and trying to coerce them into what
# we want.

def _fix_pf(pf):
    if isinstance(pf, lagos.StaticOutput): return pf
    if os.path.exists("%s.hierarchy" % pf):
        return lagos.EnzoStaticOutput(pf)
    elif os.path.isdir("%s" % pf) and \
         os.path.exists("%s/%s" % (pf,pf)):
        return lagos.EnzoStaticOutput("%s/%s" % (pf,pf))
    elif os.path.isdir("%s.dir" % pf) and \
         os.path.exists("%s.dir/%s" % (pf,pf)):
        return lagos.EnzoStaticOutput("%s.dir/%s" % (pf,pf))
    elif pf.endswith(".hierarchy"):
        return lagos.EnzoStaticOutput(pf[:-10])
    # JS will have to implement the Orion one
    else:
        raise IOError(pf)

__pf_centers = {}
def _fix_center(pf, center):
    if center is not None and iterable(center):
        center = na.array(center)
    else:
        if pf['CurrentTimeIdentifier'] in __pf_centers:
            center = __pf_centers[pf['CurrentTimeIdentifier']]
        else:
            center = pf.h.find_max("Density")[1]
            __pf_centers[pf['CurrentTimeIdentifier']] = center
    return center

def _fix_radius(pf, radius):
    if radius is not None:
        if iterable(radius):
            return radius[0] / pf[radius[1]]
        return radius
    mylog.info("Setting radius to be 0.1 of the box size")
    # yt-generalization : needs to be changed to 'unitary'
    return 0.1 / pf["1"]

def _fix_width(pf, width):
    if width is not None:
        if iterable(width):
            return width[0] / pf[width[1]]
        return width
    mylog.info("Setting width to be the full box")
    # yt-generalization : needs to be changed to 'unitary'
    return 1.0 / pf["1"]

def _fix_axis(pf, axis):
    if axis is None:
        raise ValueError("You need to specify an Axis!")
    elif isinstance(axis, types.IntType) and (0 <= x <= 2):
        return axis
    elif isinstance(axis, types.StringTypes) and \
         axis.upper() in 'XYZ':
        return 'XYZ'.find(axis.upper())
    else:
        raise ValueError("Invalid Axis specified.")


_arg_fixer = {
                'center' : (True, _fix_center),
                'radius' : (True, _fix_radius),
                'width' : (True, _fix_width),
                'axis' : (True, _fix_axis),
             }

#
# * Need to change filename to be an argument in the descriptor
# * Need to add cmap to the descriptor
#

class FakePlotCollection(object):
    def __init__(self, fig):
        self.fig = fig
    def save(self, filename):
        self.fig.savefig(filename)

def fix_plot_args(plot_func):
    @wraps(plot_func)
    def call_func(self, pf, **kwargs):
        pf = _fix_pf(pf)
        fkwargs = inspect.getargspec(plot_func)[0]
        for arg in fkwargs:
            if arg in _arg_fixer:
                needs, fixer = _arg_fixer[arg]
                if arg in kwargs:
                    kwargs[arg] = fixer(pf, kwargs[arg])
                elif needs:
                    kwargs[arg] = fixer(pf, None)
        retval = plot_func(self, pf, **kwargs)
        if 'filename' in kwargs and 'filename' in fkwargs:
            retval.save(kwargs['filename'])
        return retval
    return call_func

which_pc = raven.PlotCollection

# to add:
#   zoom movies

class _RecipeBook(object):

    @fix_plot_args
    def mass_enclosed_radius(self, pf, center=None, radius=None, radius_unit="Radiuspc",
                             filename = None):
        pc = which_pc(pf, center=center)
        p = pc.add_profile_sphere(radius, '1', 
                                  [radius_unit, "CellMassMsun"],
                                  accumulation=True, weight=None)
        return pc

    @fix_plot_args
    def mass_enclosed_field(self, pf, field="Density", center=None, radius=None,
                            weight="CellMassMsun", filename = None):
        pc = which_pc(pf, center=center)
        p = pc.add_profile_sphere(radius, '1', 
                                  ["Radius", "CellMassMsun"],
                                  accumulation=True, weight=None)
        p.switch_z(field, weight=weight)
        p.switch_x("CellMassMsun")
        return pc

    @fix_plot_args
    def slice_axis(self, pf, field="Density", center=None, width=None, filename=None, axis=None, coord=None):
        pc = which_pc(pf, center=center)
        p = pc.add_slice(field, axis, coord=coord)
        pc.set_width(width, '1')
        return pc

    @fix_plot_args
    def slice(self, pf, field="Density", center=None, width=None, filename=None):
        pc = which_pc(pf, center=center)
        for axis in range(3): pc.add_slice(field, axis, coord=center[axis])
        pc.set_width(width, '1')
        return pc

    def _multiplot(self, pc, func, pf, field="Density", center=None, width=None, filename=None):
        import pylab
        pylab.subplots_adjust(wspace=0.0, hspace=0.0,
                              top=1.0, bottom=0.0,
                              left=0.0, right=1.0)
        pylab.gcf().set_size_inches((12,12))
        pylab.clf()
        mi, ma = 1e30, -1e30
        for axis in range(3):
            axes = pylab.subplot(2,2,axis+1)
            func(field, axis, coord=center[axis], axes=axes)
        pc.set_width(width, '1')
        for axis in range(3):
            mi = min(mi, pc.plots[axis].image._A.min())
            ma = max(ma, pc.plots[axis].image._A.max())
        pc.set_zlim(mi,ma)
        cba = pylab.axes([0.55, 0.05, 0.02, 0.40])
        cb = pylab.colorbar(cax=cba, mappable=pc.plots[-1].image)
        pc.plots[-1].colorbar = cb
        pc.plots[-1].autoset_label()
        return FakePlotCollection(pylab.gcf())

    @fix_plot_args
    def multiplot_slice(self, pf, field="Density", center=None, width=None, filename=None):
        pc = which_pc(pf, center=center)
        func = pc.add_slice
        return self._multiplot(pc, func, pf, field, center, width, filename)

    @fix_plot_args
    def multiplot_proj(self, pf, field="Density", center=None, width=None, filename=None):
        pc = which_pc(pf, center=center)
        func = pc.add_projection
        return self._multiplot(pc, func, pf, field, center, width, filename)

    def dispatch(self):
        pass

RecipeBook = _RecipeBook()

if __name__ == "__main__":
    RecipeBook.dispatch()
