"""
Callbacks to add additional functionality on to plots.



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np

from yt.funcs import *
from _mpl_imports import *
from yt.utilities.definitions import \
    x_dict, x_names, \
    y_dict, y_names, \
    axis_names, \
    axis_labels
from yt.utilities.physical_constants import \
    sec_per_Gyr, sec_per_Myr, \
    sec_per_kyr, sec_per_year, \
    sec_per_day, sec_per_hr
from yt.visualization.image_writer import apply_colormap

import _MPL

callback_registry = {}

class PlotCallback(object):
    class __metaclass__(type):
        def __init__(cls, name, b, d):
            type.__init__(cls, name, b, d)
            callback_registry[name] = cls

    def __init__(self, *args, **kwargs):
        pass

    def convert_to_plot(self, plot, coord, offset = True):
        # coord should be a 2 x ncoord array-like datatype.
        try:
            ncoord = np.array(coord).shape[1]
        except IndexError:
            ncoord = 1

        # Convert the data and plot limits to tiled numpy arrays so that
        # convert_to_plot is automatically vectorized.

        x0 = np.tile(plot.xlim[0],ncoord)
        x1 = np.tile(plot.xlim[1],ncoord)
        xx0 = np.tile(plot._axes.get_xlim()[0],ncoord)
        xx1 = np.tile(plot._axes.get_xlim()[1],ncoord)
        
        y0 = np.tile(plot.ylim[0],ncoord)
        y1 = np.tile(plot.ylim[1],ncoord)
        yy0 = np.tile(plot._axes.get_ylim()[0],ncoord)
        yy1 = np.tile(plot._axes.get_ylim()[1],ncoord)
        
        # We need a special case for when we are only given one coordinate.
        if np.array(coord).shape == (2,):
            return ((coord[0]-x0)/(x1-x0)*(xx1-xx0) + xx0,
                    (coord[1]-y0)/(y1-y0)*(yy1-yy0) + yy0)
        else:
            return ((coord[0][:]-x0)/(x1-x0)*(xx1-xx0) + xx0,
                    (coord[1][:]-y0)/(y1-y0)*(yy1-yy0) + yy0)

    def pixel_scale(self,plot):
        x0, x1 = plot.xlim
        xx0, xx1 = plot._axes.get_xlim()
        dx = (xx1 - xx0)/(x1 - x0)
        
        y0, y1 = plot.ylim
        yy0, yy1 = plot._axes.get_ylim()
        dy = (yy1 - yy0)/(y1 - y0)

        return (dx,dy)


class VelocityCallback(PlotCallback):
    """
    annotate_velocity(factor=16, scale=None, scale_units=None, normalize=False):
    
    Adds a 'quiver' plot of velocity to the plot, skipping all but
    every *factor* datapoint. *scale* is the data units per arrow
    length unit using *scale_units* (see
    matplotlib.axes.Axes.quiver for more info). if *normalize* is
    True, the velocity fields will be scaled by their local
    (in-plane) length, allowing morphological features to be more
    clearly seen for fields with substantial variation in field
    strength (normalize is not implemented and thus ignored for
    Cutting Planes).
    """
    _type_name = "velocity"
    def __init__(self, factor=16, scale=None, scale_units=None, normalize=False):
        PlotCallback.__init__(self)
        self.factor = factor
        self.scale  = scale
        self.scale_units = scale_units
        self.normalize = normalize

    def __call__(self, plot):
        # Instantiation of these is cheap
        if plot._type_name == "CuttingPlane":
            qcb = CuttingQuiverCallback("CuttingPlaneVelocityX",
                                        "CuttingPlaneVelocityY",
                                        self.factor)
        else:
            xv = "%s-velocity" % (x_names[plot.data.axis])
            yv = "%s-velocity" % (y_names[plot.data.axis])

            bv = plot.data.get_field_parameter("bulk_velocity")
            if bv is not None:
                bv_x = bv[x_dict[plot.data.axis]]
                bv_y = bv[y_dict[plot.data.axis]]
            else: bv_x = bv_y = 0

            qcb = QuiverCallback(xv, yv, self.factor, scale=self.scale, 
                                 scale_units=self.scale_units, 
                                 normalize=self.normalize, bv_x=bv_x, bv_y=bv_y)
        return qcb(plot)

class MagFieldCallback(PlotCallback):
    """
    annotate_magnetic_field(factor=16, scale=None, scale_units=None, normalize=False):

    Adds a 'quiver' plot of magnetic field to the plot, skipping all but
    every *factor* datapoint. *scale* is the data units per arrow
    length unit using *scale_units* (see
    matplotlib.axes.Axes.quiver for more info). if *normalize* is
    True, the magnetic fields will be scaled by their local
    (in-plane) length, allowing morphological features to be more
    clearly seen for fields with substantial variation in field strength.
    """
    _type_name = "magnetic_field"
    def __init__(self, factor=16, scale=None, scale_units=None, normalize=False):
        PlotCallback.__init__(self)
        self.factor = factor
        self.scale  = scale
        self.scale_units = scale_units
        self.normalize = normalize

    def __call__(self, plot):
        # Instantiation of these is cheap
        if plot._type_name == "CuttingPlane":
            qcb = CuttingQuiverCallback("CuttingPlaneBx",
                                        "CuttingPlaneBy",
                                        self.factor)
        else:
            xv = "B%s" % (x_names[plot.data.axis])
            yv = "B%s" % (y_names[plot.data.axis])
            qcb = QuiverCallback(xv, yv, self.factor, scale=self.scale, scale_units=self.scale_units, normalize=self.normalize)
        return qcb(plot)

class QuiverCallback(PlotCallback):
    """
    annotate_quiver(field_x, field_y, factor=16, scale=None, scale_units=None, 
                    normalize=False, bv_x=0, bv_y=0):

    Adds a 'quiver' plot to any plot, using the *field_x* and *field_y*
    from the associated data, skipping every *factor* datapoints
    *scale* is the data units per arrow length unit using *scale_units* 
    (see matplotlib.axes.Axes.quiver for more info)
    """
    _type_name = "quiver"
    def __init__(self, field_x, field_y, factor=16, scale=None, scale_units=None, normalize=False, bv_x=0, bv_y=0):
        PlotCallback.__init__(self)
        self.field_x = field_x
        self.field_y = field_y
        self.bv_x = bv_x
        self.bv_y = bv_y
        self.factor = factor
        self.scale = scale
        self.scale_units = scale_units
        self.normalize = normalize

    def __call__(self, plot):
        x0, x1 = plot.xlim
        y0, y1 = plot.ylim
        xx0, xx1 = plot._axes.get_xlim()
        yy0, yy1 = plot._axes.get_ylim()
        plot._axes.hold(True)
        nx = plot.image._A.shape[0] / self.factor
        ny = plot.image._A.shape[1] / self.factor
        # periodicity
        ax = plot.data.axis
        pf = plot.data.pf
        period_x = pf.domain_width[x_dict[ax]]
        period_y = pf.domain_width[y_dict[ax]]
        periodic = int(any(pf.periodicity))
        pixX = _MPL.Pixelize(plot.data['px'],
                             plot.data['py'],
                             plot.data['pdx'],
                             plot.data['pdy'],
                             plot.data[self.field_x] - self.bv_x,
                             int(nx), int(ny),
                             (x0, x1, y0, y1), 0, # bounds, antialias
                             (period_x, period_y), periodic,
                           ).transpose()
        pixY = _MPL.Pixelize(plot.data['px'],
                             plot.data['py'],
                             plot.data['pdx'],
                             plot.data['pdy'],
                             plot.data[self.field_y] - self.bv_y,
                             int(nx), int(ny),
                             (x0, x1, y0, y1), 0, # bounds, antialias
                             (period_x, period_y), periodic,
                           ).transpose()
        X,Y = np.meshgrid(np.linspace(xx0,xx1,nx,endpoint=True),
                          np.linspace(yy0,yy1,ny,endpoint=True))
        if self.normalize:
            nn = np.sqrt(pixX**2 + pixY**2)
            pixX /= nn
            pixY /= nn
        plot._axes.quiver(X,Y, pixX, pixY, scale=self.scale, scale_units=self.scale_units)
        plot._axes.set_xlim(xx0,xx1)
        plot._axes.set_ylim(yy0,yy1)
        plot._axes.hold(False)

class ContourCallback(PlotCallback):
    """
    annotate_contour(field, ncont=5, factor=4, take_log=None, clim=None,
                     plot_args=None, label=False, label_args=None,
                     data_source=None):

    Add contours in *field* to the plot.  *ncont* governs the number of
    contours generated, *factor* governs the number of points used in the
    interpolation, *take_log* governs how it is contoured and *clim* gives
    the (upper, lower) limits for contouring.  An alternate data source can be
    specified with *data_source*, but by default the plot's data source will be
    queried.
    """
    _type_name = "contour"
    def __init__(self, field, ncont=5, factor=4, clim=None,
                 plot_args = None, label = False, take_log = None, 
                 label_args = None, data_source = None):
        PlotCallback.__init__(self)
        self.ncont = ncont
        self.field = field
        self.factor = factor
        from matplotlib.delaunay.triangulate import Triangulation as triang
        self.triang = triang
        self.clim = clim
        self.take_log = take_log
        if plot_args is None: plot_args = {'colors':'k'}
        self.plot_args = plot_args
        self.label = label
        if label_args is None:
            label_args = {}
        self.label_args = label_args
        self.data_source = data_source

    def __call__(self, plot):
        x0, x1 = plot.xlim
        y0, y1 = plot.ylim
        
        xx0, xx1 = plot._axes.get_xlim()
        yy0, yy1 = plot._axes.get_ylim()
        
        plot._axes.hold(True)
        
        numPoints_x = plot.image._A.shape[0]
        numPoints_y = plot.image._A.shape[1]
        
        # Multiply by dx and dy to go from data->plot
        dx = (xx1 - xx0) / (x1-x0)
        dy = (yy1 - yy0) / (y1-y0)

        # We want xi, yi in plot coordinates
        xi, yi = np.mgrid[xx0:xx1:numPoints_x/(self.factor*1j),
                          yy0:yy1:numPoints_y/(self.factor*1j)]
        data = self.data_source or plot.data

        if plot._type_name in ['CuttingPlane','Projection','Slice']:
            if plot._type_name == 'CuttingPlane':
                x = data["px"]*dx
                y = data["py"]*dy
                z = data[self.field]
            elif plot._type_name in ['Projection','Slice']:
                #Makes a copy of the position fields "px" and "py" and adds the
                #appropriate shift to the copied field.  

                AllX = np.zeros(data["px"].size, dtype='bool')
                AllY = np.zeros(data["py"].size, dtype='bool')
                XShifted = data["px"].copy()
                YShifted = data["py"].copy()
                dom_x, dom_y = plot._period
                for shift in np.mgrid[-1:1:3j]:
                    xlim = ((data["px"] + shift*dom_x >= x0) &
                            (data["px"] + shift*dom_x <= x1))
                    ylim = ((data["py"] + shift*dom_y >= y0) &
                            (data["py"] + shift*dom_y <= y1))
                    XShifted[xlim] += shift * dom_x
                    YShifted[ylim] += shift * dom_y
                    AllX |= xlim
                    AllY |= ylim
            
                # At this point XShifted and YShifted are the shifted arrays of
                # position data in data coordinates
                wI = (AllX & AllY)

                # This converts XShifted and YShifted into plot coordinates
                x = (XShifted[wI]-x0)*dx + xx0
                y = (YShifted[wI]-y0)*dy + yy0
                z = data[self.field][wI]
        
            # Both the input and output from the triangulator are in plot
            # coordinates
            zi = self.triang(x,y).nn_interpolator(z)(xi,yi)
        elif plot._type_name == 'OffAxisProjection':
            zi = plot.frb[self.field][::self.factor,::self.factor].transpose()
        
        if self.take_log is None:
            self.take_log = plot.pf.field_info[self.field].take_log

        if self.take_log: zi=np.log10(zi)

        if self.take_log and self.clim is not None: 
            self.clim = (np.log10(self.clim[0]), np.log10(self.clim[1]))
        
        if self.clim is not None: 
            self.ncont = np.linspace(self.clim[0], self.clim[1], self.ncont)
        
        cset = plot._axes.contour(xi,yi,zi,self.ncont, **self.plot_args)
        plot._axes.set_xlim(xx0,xx1)
        plot._axes.set_ylim(yy0,yy1)
        plot._axes.hold(False)
        
        if self.label:
            plot._axes.clabel(cset, **self.label_args)
        

class GridBoundaryCallback(PlotCallback):
    """
    annotate_grids(alpha=0.7, min_pix=1, min_pix_ids=20, draw_ids=False, periodic=True, 
                 min_level=None, max_level=None, cmap='B-W LINEAR_r'):

    Draws grids on an existing PlotWindow object.
    Adds grid boundaries to a plot, optionally with alpha-blending. By default, 
    colors different levels of grids with different colors going from white to
    black, but you can change to any arbitrary colormap with cmap keyword 
    (or all black cells for all levels with cmap=None).  Cuttoff for display is at 
    min_pix wide. draw_ids puts the grid id in the corner of the grid. 
    (Not so great in projections...).  One can set min and maximum level of
    grids to display.
    """
    _type_name = "grids"
    def __init__(self, alpha=0.7, min_pix=1, min_pix_ids=20, draw_ids=False, periodic=True, 
                 min_level=None, max_level=None, cmap='B-W LINEAR_r'):
        PlotCallback.__init__(self)
        self.alpha = alpha
        self.min_pix = min_pix
        self.min_pix_ids = min_pix_ids
        self.draw_ids = draw_ids # put grid numbers in the corner.
        self.periodic = periodic
        self.min_level = min_level
        self.max_level = max_level
        self.cmap = cmap

    def __call__(self, plot):
        x0, x1 = plot.xlim
        y0, y1 = plot.ylim
        xx0, xx1 = plot._axes.get_xlim()
        yy0, yy1 = plot._axes.get_ylim()
        xi = x_dict[plot.data.axis]
        yi = y_dict[plot.data.axis]
        (dx, dy) = self.pixel_scale(plot)
        (xpix, ypix) = plot.image._A.shape
        px_index = x_dict[plot.data.axis]
        py_index = y_dict[plot.data.axis]
        dom = plot.data.pf.domain_right_edge - plot.data.pf.domain_left_edge
        if self.periodic:
            pxs, pys = np.mgrid[-1:1:3j,-1:1:3j]
        else:
            pxs, pys = np.mgrid[0:0:1j,0:0:1j]
        GLE = plot.data.grid_left_edge
        GRE = plot.data.grid_right_edge
        levels = plot.data.grid_levels[:,0]
        min_level = self.min_level
        max_level = self.max_level
        if max_level is None:
            max_level = plot.data.pf.h.max_level
        if min_level is None:
            min_level = 0

        for px_off, py_off in zip(pxs.ravel(), pys.ravel()):
            pxo = px_off * dom[px_index]
            pyo = py_off * dom[py_index]
            left_edge_x = (GLE[:,px_index]+pxo-x0)*dx + xx0
            left_edge_y = (GLE[:,py_index]+pyo-y0)*dy + yy0
            right_edge_x = (GRE[:,px_index]+pxo-x0)*dx + xx0
            right_edge_y = (GRE[:,py_index]+pyo-y0)*dy + yy0
            visible =  ( xpix * (right_edge_x - left_edge_x) / (xx1 - xx0) > self.min_pix ) & \
                       ( ypix * (right_edge_y - left_edge_y) / (yy1 - yy0) > self.min_pix ) & \
                       ( levels >= min_level) & \
                       ( levels <= max_level)

            if self.cmap is not None: 
                edgecolors = apply_colormap(levels[(levels <= max_level) & (levels >= min_level)]*1.0,
                                  color_bounds=[0,plot.data.pf.h.max_level],
                                  cmap_name=self.cmap)[0,:,:]*1.0/255.
                edgecolors[:,3] = self.alpha
            else:
                edgecolors = (0.0,0.0,0.0,self.alpha)

            if visible.nonzero()[0].size == 0: continue
            verts = np.array(
                [(left_edge_x, left_edge_x, right_edge_x, right_edge_x),
                 (left_edge_y, right_edge_y, right_edge_y, left_edge_y)])
            verts=verts.transpose()[visible,:,:]
            grid_collection = matplotlib.collections.PolyCollection(
                verts, facecolors="none",
                edgecolors=edgecolors)
            plot._axes.hold(True)
            plot._axes.add_collection(grid_collection)

            if self.draw_ids:
                visible_ids =  ( xpix * (right_edge_x - left_edge_x) / (xx1 - xx0) > self.min_pix_ids ) & \
                               ( ypix * (right_edge_y - left_edge_y) / (yy1 - yy0) > self.min_pix_ids )
                active_ids = np.unique(plot.data['GridIndices'])
                for i in np.where(visible_ids)[0]:
                    plot._axes.text(
                        left_edge_x[i] + (2 * (xx1 - xx0) / xpix),
                        left_edge_y[i] + (2 * (yy1 - yy0) / ypix),
                        "%d" % active_ids[i], clip_on=True)
            plot._axes.hold(False)

class StreamlineCallback(PlotCallback):
    """
    annotate_streamlines(field_x, field_y, factor = 16,
                         density = 1, plot_args=None):

    Add streamlines to any plot, using the *field_x* and *field_y*
    from the associated data, skipping every *factor* datapoints like
    'quiver'. *density* is the index of the amount of the streamlines.
    """
    _type_name = "streamlines"
    def __init__(self, field_x, field_y, factor = 16,
                 density = 1, plot_args=None):
        PlotCallback.__init__(self)
        self.field_x = field_x
        self.field_y = field_y
        self.bv_x = self.bv_y = 0
        self.factor = factor
        self.dens = density
        if plot_args is None: plot_args = {}
        self.plot_args = plot_args
        
    def __call__(self, plot):
        x0, x1 = plot.xlim
        y0, y1 = plot.ylim
        xx0, xx1 = plot._axes.get_xlim()
        yy0, yy1 = plot._axes.get_ylim()
        plot._axes.hold(True)
        nx = plot.image._A.shape[0] / self.factor
        ny = plot.image._A.shape[1] / self.factor
        pixX = _MPL.Pixelize(plot.data['px'],
                             plot.data['py'],
                             plot.data['pdx'],
                             plot.data['pdy'],
                             plot.data[self.field_x] - self.bv_x,
                             int(nx), int(ny),
                             (x0, x1, y0, y1),).transpose()
        pixY = _MPL.Pixelize(plot.data['px'],
                             plot.data['py'],
                             plot.data['pdx'],
                             plot.data['pdy'],
                             plot.data[self.field_y] - self.bv_y,
                             int(nx), int(ny),
                             (x0, x1, y0, y1),).transpose()
        X,Y = (np.linspace(xx0,xx1,nx,endpoint=True),
               np.linspace(yy0,yy1,ny,endpoint=True))
        plot._axes.streamplot(X,Y, pixX, pixY, density = self.dens,
                              **self.plot_args)
        plot._axes.set_xlim(xx0,xx1)
        plot._axes.set_ylim(yy0,yy1)
        plot._axes.hold(False)

class LabelCallback(PlotCallback):
    """
    This adds a label to the plot.
    """
    _type_name = "axis_label"
    def __init__(self, label):
        PlotCallback.__init__(self)
        self.label = label

    def __call__(self, plot):
        plot._figure.subplots_adjust(hspace=0, wspace=0,
                                     bottom=0.1, top=0.9,
                                     left=0.0, right=1.0)
        plot._axes.set_xlabel(self.label)
        plot._axes.set_ylabel(self.label)

def get_smallest_appropriate_unit(v, pf):
    max_nu = 1e30
    good_u = None
    for unit in ['mpc', 'kpc', 'pc', 'au', 'rsun', 'km', 'cm']:
        vv = v*pf[unit]
        if vv < max_nu and vv > 1.0:
            good_u = unit
            max_nu = v*pf[unit]
    if good_u is None : good_u = 'cm'
    return good_u

class UnitBoundaryCallback(PlotCallback):
    """
    Add on a plot indicating where *factor*s of *unit* are shown.
    Optionally *text_annotate* on the *text_which*-indexed box on display.
    """
    _type_name = "units"
    def __init__(self, unit = "au", factor=4, text_annotate=True, text_which=-2):
        PlotCallback.__init__(self)
        self.unit = unit
        self.factor = factor
        self.text_annotate = text_annotate
        self.text_which = -2

    def __call__(self, plot):
        x0, x1 = plot.xlim
        y0, y1 = plot.ylim
        l, b, width, height = mpl_get_bounds(plot._axes.bbox)
        xi = x_dict[plot.data.axis]
        yi = y_dict[plot.data.axis]
        dx = plot.image._A.shape[0] / (x1-x0)
        dy = plot.image._A.shape[1] / (y1-y0)
        center = plot.data.center
        min_dx = plot.data['pdx'].min()
        max_dx = plot.data['pdx'].max()
        w_min_x = 250.0 * min_dx
        w_max_x = 1.0 / self.factor
        min_exp_x = np.ceil(np.log10(w_min_x*plot.data.pf[self.unit])
                           /np.log10(self.factor))
        max_exp_x = np.floor(np.log10(w_max_x*plot.data.pf[self.unit])
                            /np.log10(self.factor))
        n_x = max_exp_x - min_exp_x + 1
        widths = np.logspace(min_exp_x, max_exp_x, num = n_x, base=self.factor)
        widths /= plot.data.pf[self.unit]
        left_edge_px = (center[xi] - widths/2.0 - x0)*dx
        left_edge_py = (center[yi] - widths/2.0 - y0)*dy
        right_edge_px = (center[xi] + widths/2.0 - x0)*dx
        right_edge_py = (center[yi] + widths/2.0 - y0)*dy
        verts = np.array(
                [(left_edge_px, left_edge_px, right_edge_px, right_edge_px),
                 (left_edge_py, right_edge_py, right_edge_py, left_edge_py)])
        visible =  ( right_edge_px - left_edge_px > 25 ) & \
                   ( right_edge_px - left_edge_px > 25 ) & \
                   ( (right_edge_px < width) & (left_edge_px > 0) ) & \
                   ( (right_edge_py < height) & (left_edge_py > 0) )
        verts=verts.transpose()[visible,:,:]
        grid_collection = matplotlib.collections.PolyCollection(
                verts, facecolors="none",
                       edgecolors=(0.0,0.0,0.0,1.0),
                       linewidths=2.5)
        plot._axes.hold(True)
        plot._axes.add_collection(grid_collection)
        if self.text_annotate:
            ti = max(self.text_which, -1*len(widths[visible]))
            if ti < len(widths[visible]): 
                w = widths[visible][ti]
                good_u = get_smallest_appropriate_unit(w, plot.data.pf)
                w *= plot.data.pf[good_u]
                plot._axes.annotate("%0.3e %s" % (w,good_u), verts[ti,1,:]+5)
        plot._axes.hold(False)

class LinePlotCallback(PlotCallback):
    """
    annotate_line(x, y, plot_args = None)

    Over plot *x* and *y* with *plot_args* fed into the plot.
    """
    _type_name = "line"
    def __init__(self, x, y, plot_args = None):
        PlotCallback.__init__(self)
        self.x = x
        self.y = y
        if plot_args is None: plot_args = {}
        self.plot_args = plot_args

    def __call__(self, plot):
        plot._axes.hold(True)
        plot._axes.plot(self.x, self.y, **self.plot_args)
        plot._axes.hold(False)

class ImageLineCallback(LinePlotCallback):
    """
    annotate_image_line(p1, p2, data_coords=False, plot_args = None)

    Plot from *p1* to *p2* (image plane coordinates)
    with *plot_args* fed into the plot.
    """
    _type_name = "image_line"
    def __init__(self, p1, p2, data_coords=False, plot_args = None):
        PlotCallback.__init__(self)
        self.p1 = p1
        self.p2 = p2
        if plot_args is None: plot_args = {}
        self.plot_args = plot_args
        self._ids = []
        self.data_coords = data_coords

    def __call__(self, plot):
        # We manually clear out any previous calls to this callback:
        plot._axes.lines = [l for l in plot._axes.lines if id(l) not in self._ids]
        kwargs = self.plot_args.copy()
        if self.data_coords and len(plot.image._A.shape) == 2:
            p1 = self.convert_to_plot(plot, self.p1)
            p2 = self.convert_to_plot(plot, self.p2)
        else:
            p1, p2 = self.p1, self.p2
            if not self.data_coords:
                kwargs["transform"] = plot._axes.transAxes

        px, py = (p1[0], p2[0]), (p1[1], p2[1])

        # Save state
        xx0, xx1 = plot._axes.get_xlim()
        yy0, yy1 = plot._axes.get_ylim()
        plot._axes.hold(True)
        ii = plot._axes.plot(px, py, **kwargs)
        self._ids.append(id(ii[0]))
        # Reset state
        plot._axes.set_xlim(xx0,xx1)
        plot._axes.set_ylim(yy0,yy1)
        plot._axes.hold(False)

class CuttingQuiverCallback(PlotCallback):
    """
    annotate_cquiver(field_x, field_y, factor)

    Get a quiver plot on top of a cutting plane, using *field_x* and
    *field_y*, skipping every *factor* datapoint in the discretization.
    """
    _type_name = "cquiver"
    def __init__(self, field_x, field_y, factor):
        PlotCallback.__init__(self)
        self.field_x = field_x
        self.field_y = field_y
        self.factor = factor

    def __call__(self, plot):
        x0, x1 = plot.xlim
        y0, y1 = plot.ylim
        xx0, xx1 = plot._axes.get_xlim()
        yy0, yy1 = plot._axes.get_ylim()
        plot._axes.hold(True)
        nx = plot.image._A.shape[0] / self.factor
        ny = plot.image._A.shape[1] / self.factor
        indices = np.argsort(plot.data['dx'])[::-1]
        pixX = _MPL.CPixelize( plot.data['x'], plot.data['y'], plot.data['z'],
                               plot.data['px'], plot.data['py'],
                               plot.data['pdx'], plot.data['pdy'], plot.data['pdz'],
                               plot.data.center, plot.data._inv_mat, indices,
                               plot.data[self.field_x],
                               int(nx), int(ny),
                               (x0, x1, y0, y1),).transpose()
        pixY = _MPL.CPixelize( plot.data['x'], plot.data['y'], plot.data['z'],
                               plot.data['px'], plot.data['py'],
                               plot.data['pdx'], plot.data['pdy'], plot.data['pdz'],
                               plot.data.center, plot.data._inv_mat, indices,
                               plot.data[self.field_y],
                               int(nx), int(ny),
                               (x0, x1, y0, y1),).transpose()
        X,Y = np.meshgrid(np.linspace(xx0,xx1,nx,endpoint=True),
                          np.linspace(yy0,yy1,ny,endpoint=True))
        plot._axes.quiver(X,Y, pixX, pixY)
        plot._axes.set_xlim(xx0,xx1)
        plot._axes.set_ylim(yy0,yy1)
        plot._axes.hold(False)

class ClumpContourCallback(PlotCallback):
    """
    annotate_clumps(clumps, plot_args = None)

    Take a list of *clumps* and plot them as a set of contours.
    """
    _type_name = "clumps"
    def __init__(self, clumps, plot_args = None):
        self.clumps = clumps
        if plot_args is None: plot_args = {}
        self.plot_args = plot_args

    def __call__(self, plot):
        x0, x1 = plot.xlim
        y0, y1 = plot.ylim
        xx0, xx1 = plot._axes.get_xlim()
        yy0, yy1 = plot._axes.get_ylim()

        extent = [xx0,xx1,yy0,yy1]

        plot._axes.hold(True)

        px_index = x_dict[plot.data.axis]
        py_index = y_dict[plot.data.axis]

        xf = axis_names[px_index]
        yf = axis_names[py_index]
        dxf = "d%s" % xf
        dyf = "d%s" % yf

        DomainRight = plot.data.pf.domain_right_edge
        DomainLeft = plot.data.pf.domain_left_edge
        DomainWidth = DomainRight - DomainLeft

        nx, ny = plot.image._A.shape
        buff = np.zeros((nx,ny),dtype='float64')
        for i,clump in enumerate(reversed(self.clumps)):
            mylog.debug("Pixelizing contour %s", i)

            xf_copy = clump[xf].copy()
            yf_copy = clump[yf].copy()

            temp = _MPL.Pixelize(xf_copy, yf_copy,
                                 clump[dxf]/2.0,
                                 clump[dyf]/2.0,
                                 clump[dxf]*0.0+i+1, # inits inside Pixelize
                                 int(nx), int(ny),
                             (x0, x1, y0, y1), 0).transpose()
            buff = np.maximum(temp, buff)
        self.rv = plot._axes.contour(buff, np.unique(buff),
                                     extent=extent,**self.plot_args)
        plot._axes.hold(False)

class ArrowCallback(PlotCallback):
    """
    annotate_arrow(pos, code_size, plot_args = None)

    This adds an arrow pointing at *pos* with size *code_size* in code
    units.  *plot_args* is a dict fed to matplotlib with arrow properties.
    """
    _type_name = "arrow"
    def __init__(self, pos, code_size, plot_args = None):
        self.pos = pos
        if not iterable(code_size):
            code_size = (code_size, code_size)
        self.code_size = code_size
        if plot_args is None: plot_args = {}
        self.plot_args = plot_args

    def __call__(self, plot):
        if len(self.pos) == 3:
            pos = (self.pos[x_dict[plot.data.axis]],
                   self.pos[y_dict[plot.data.axis]])
        else: pos = self.pos
        from matplotlib.patches import Arrow
        # Now convert the pixels to code information
        x, y = self.convert_to_plot(plot, pos)
        x1, y1 = pos[0]+self.code_size[0], pos[1]+self.code_size[1]
        x1, y1 = self.convert_to_plot(plot, (x1, y1), False)
        dx, dy = x1 - x, y1 - y
        arrow = Arrow(x-dx, y-dy, dx, dy, **self.plot_args)
        plot._axes.add_patch(arrow)

class PointAnnotateCallback(PlotCallback):
    """
    annotate_point(pos, text, text_args = None)

    This adds *text* at position *pos*, where *pos* is in code-space.
    *text_args* is a dict fed to the text placement code.
    """
    _type_name = "point"
    def __init__(self, pos, text, text_args = None):
        self.pos = pos
        self.text = text
        if text_args is None: text_args = {}
        self.text_args = text_args

    def __call__(self, plot):
        if len(self.pos) == 3:
            pos = (self.pos[x_dict[plot.data.axis]],
                   self.pos[y_dict[plot.data.axis]])
        else: pos = self.pos
        width,height = plot.image._A.shape
        x,y = self.convert_to_plot(plot, pos)
        
        plot._axes.text(x, y, self.text, **self.text_args)

class MarkerAnnotateCallback(PlotCallback):
    """
    annotate_marker(pos, marker='x', plot_args=None)

    Adds text *marker* at *pos* in code units.  *plot_args* is a dict
    that will be forwarded to the plot command.
    """
    _type_name = "marker"
    def __init__(self, pos, marker='x', plot_args=None):
        self.pos = pos
        self.marker = marker
        if plot_args is None: plot_args = {}
        self.plot_args = plot_args

    def __call__(self, plot):
        xx0, xx1 = plot._axes.get_xlim()
        yy0, yy1 = plot._axes.get_ylim()
        if len(self.pos) == 3:
            pos = (self.pos[x_dict[plot.data.axis]],
                   self.pos[y_dict[plot.data.axis]])
        elif len(self.pos) == 2:
            pos = self.pos
        x,y = self.convert_to_plot(plot, pos)
        plot._axes.hold(True)
        plot._axes.scatter(x,y, marker = self.marker, **self.plot_args)
        plot._axes.set_xlim(xx0,xx1)
        plot._axes.set_ylim(yy0,yy1)
        plot._axes.hold(False)

class SphereCallback(PlotCallback):
    """
    annotate_sphere(center, radius, circle_args = None,
                    text = None, text_args = None)
    
    A sphere centered at *center* in code units with radius *radius* in
    code units will be created, with optional *circle_args*, *text*, and
    *text_args*.
    """
    _type_name = "sphere"
    def __init__(self, center, radius, circle_args = None,
                 text = None, text_args = None):
        self.center = center
        self.radius = radius
        if circle_args is None: circle_args = {}
        if 'fill' not in circle_args: circle_args['fill'] = False
        self.circle_args = circle_args
        self.text = text
        self.text_args = text_args
        if self.text_args is None: self.text_args = {}

    def __call__(self, plot):
        from matplotlib.patches import Circle
        
        radius = self.radius * self.pixel_scale(plot)[0]

        if plot.data.axis == 4:
            (xi, yi) = (0, 1)
        else:
            (xi, yi) = (x_dict[plot.data.axis], y_dict[plot.data.axis])

        (center_x,center_y) = self.convert_to_plot(plot,(self.center[xi], self.center[yi]))
        
        cir = Circle((center_x, center_y), radius, **self.circle_args)
        plot._axes.add_patch(cir)
        if self.text is not None:
            plot._axes.text(center_x, center_y, self.text,
                            **self.text_args)

class HopCircleCallback(PlotCallback):
    """
    annotate_hop_circles(hop_output, max_number=None,
                         annotate=False, min_size=20, max_size=10000000,
                         font_size=8, print_halo_size=False,
                         print_halo_mass=False, width=None)

    Accepts a :class:`yt.HopList` *hop_output* and plots up to
    *max_number* (None for unlimited) halos as circles.
    """
    _type_name = "hop_circles"
    def __init__(self, hop_output, max_number=None,
                 annotate=False, min_size=20, max_size=10000000,
                 font_size=8, print_halo_size=False,
                 print_halo_mass=False, width=None):
        self.hop_output = hop_output
        self.max_number = max_number
        self.annotate = annotate
        self.min_size = min_size
        self.max_size = max_size
        self.font_size = font_size
        self.print_halo_size = print_halo_size
        self.print_halo_mass = print_halo_mass
        self.width = width

    def __call__(self, plot):
        from matplotlib.patches import Circle
        num = len(self.hop_output[:self.max_number])
        for halo in self.hop_output[:self.max_number]:
            size = halo.get_size()
            if size < self.min_size or size > self.max_size: continue
            # This could use halo.maximum_radius() instead of width
            if self.width is not None and \
                np.abs(halo.center_of_mass() - 
                       plot.data.center)[plot.data.axis] > \
                   self.width:
                continue
            
            radius = halo.maximum_radius() * self.pixel_scale(plot)[0]
            center = halo.center_of_mass()
            
            (xi, yi) = (x_dict[plot.data.axis], y_dict[plot.data.axis])

            (center_x,center_y) = self.convert_to_plot(plot,(center[xi], center[yi]))
            color = np.ones(3) * (0.4 * (num - halo.id)/ num) + 0.6
            cir = Circle((center_x, center_y), radius, fill=False, color=color)
            plot._axes.add_patch(cir)
            if self.annotate:
                if self.print_halo_size:
                    plot._axes.text(center_x+radius, center_y+radius, "%s" % size,
                    fontsize=self.font_size, color=color)
                elif self.print_halo_mass:
                    plot._axes.text(center_x+radius, center_y+radius, "%s" % halo.total_mass(),
                    fontsize=self.font_size, color=color)
                else:
                    plot._axes.text(center_x+radius, center_y+radius, "%s" % halo.id,
                    fontsize=self.font_size, color=color)

class HopParticleCallback(PlotCallback):
    """
    annotate_hop_particles(hop_output, max_number, p_size=1.0,
                           min_size=20, alpha=0.2):

    Adds particle positions for the members of each halo as identified
    by HOP. Along *axis* up to *max_number* groups in *hop_output* that are
    larger than *min_size* are plotted with *p_size* pixels per particle; 
    *alpha* determines the opacity of each particle.
    """
    _type_name = "hop_particles"
    def __init__(self, hop_output, max_number=None, p_size=1.0,
                min_size=20, alpha=0.2):
        self.hop_output = hop_output
        self.p_size = p_size
        self.max_number = max_number
        self.min_size = min_size
        self.alpha = alpha
    
    def __call__(self,plot):
        (dx,dy) = self.pixel_scale(plot)

        (xi, yi) = (x_names[plot.data.axis], y_names[plot.data.axis])

        # now we loop over the haloes
        for halo in self.hop_output[:self.max_number]:
            size = halo.get_size()

            if size < self.min_size: continue

            (px,py) = self.convert_to_plot(plot,(halo["particle_position_%s" % xi],
                                                 halo["particle_position_%s" % yi]))
            
            # Need to get the plot limits and set the hold state before scatter
            # and then restore the limits and turn off the hold state afterwards
            # because scatter will automatically adjust the plot window which we
            # do not want
            
            xlim = plot._axes.get_xlim()
            ylim = plot._axes.get_ylim()
            plot._axes.hold(True)

            plot._axes.scatter(px, py, edgecolors="None",
                s=self.p_size, c='black', alpha=self.alpha)
            
            plot._axes.set_xlim(xlim)
            plot._axes.set_ylim(ylim)
            plot._axes.hold(False)


class CoordAxesCallback(PlotCallback):
    """
    Creates x and y axes for a VMPlot. In the future, it will
    attempt to guess the proper units to use.
    """
    _type_name = "coord_axes"
    def __init__(self,unit=None,coords=False):
        PlotCallback.__init__(self)
        self.unit = unit
        self.coords = coords

    def __call__(self,plot):
        # 1. find out what the domain is
        # 2. pick a unit for it
        # 3. run self._axes.set_xlabel & self._axes.set_ylabel to actually lay things down.
        # 4. adjust extent information to make sure labels are visable.

        # put the plot into data coordinates
        nx,ny = plot.image._A.shape
        dx = (plot.xlim[1] - plot.xlim[0])/nx
        dy = (plot.ylim[1] - plot.ylim[0])/ny

        unit_conversion = plot.pf[plot.im["Unit"]]
        aspect = (plot.xlim[1]-plot.xlim[0])/(plot.ylim[1]-plot.ylim[0])

        print "aspect ratio = ", aspect

        # if coords is False, label axes relative to the center of the
        # display. if coords is True, label axes with the absolute
        # coordinates of the region.
        xcenter = 0.
        ycenter = 0.
        if not self.coords:
            center = plot.data.center
            if plot.data.axis == 0:
                xcenter = center[1]
                ycenter = center[2]
            elif plot.data.axis == 1:
                xcenter = center[0]
                ycenter = center[2]
            else:
                xcenter = center[0]
                ycenter = center[1]


            xformat_function = lambda a,b: '%7.1e' %((a*dx + plot.xlim[0] - xcenter)*unit_conversion)
            yformat_function = lambda a,b: '%7.1e' %((a*dy + plot.ylim[0] - ycenter)*unit_conversion)
        else:
            xformat_function = lambda a,b: '%7.1e' %((a*dx + plot.xlim[0])*unit_conversion)
            yformat_function = lambda a,b: '%7.1e' %((a*dy + plot.ylim[0])*unit_conversion)
            
        xticker = matplotlib.ticker.FuncFormatter(xformat_function)
        yticker = matplotlib.ticker.FuncFormatter(yformat_function)
        plot._axes.xaxis.set_major_formatter(xticker)
        plot._axes.yaxis.set_major_formatter(yticker)
        
        xlabel = '%s (%s)' % (axis_labels[plot.data.axis][0],plot.im["Unit"])
        ylabel = '%s (%s)' % (axis_labels[plot.data.axis][1],plot.im["Unit"])
        xticksize = nx/4.
        yticksize = ny/4.
        plot._axes.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([i*xticksize for i in range(0,5)]))
        plot._axes.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([i*yticksize for i in range(0,5)]))
        
        plot._axes.set_xlabel(xlabel,visible=True)
        plot._axes.set_ylabel(ylabel,visible=True)
        plot._figure.subplots_adjust(left=0.1,right=0.8)

class TextLabelCallback(PlotCallback):
    """
    annotate_text(pos, text, data_coords=False, text_args = None)

    Accepts a position in (0..1, 0..1) of the image, some text and
    optionally some text arguments. If data_coords is True,
    position will be in code units instead of image coordinates.
    """
    _type_name = "text"
    def __init__(self, pos, text, data_coords=False, text_args = None):
        self.pos = pos
        self.text = text
        self.data_coords = data_coords
        if text_args is None: text_args = {}
        self.text_args = text_args

    def __call__(self, plot):
        kwargs = self.text_args.copy()
        if self.data_coords and len(plot.image._A.shape) == 2:
            if len(self.pos) == 3:
                pos = (self.pos[x_dict[plot.data.axis]],
                       self.pos[y_dict[plot.data.axis]])
            else: pos = self.pos
            x,y = self.convert_to_plot(plot, pos)
        else:
            x, y = self.pos
            if not self.data_coords:
                kwargs["transform"] = plot._axes.transAxes
        plot._axes.text(x, y, self.text, **kwargs)

class ParticleCallback(PlotCallback):
    """
    annotate_particles(width, p_size=1.0, col='k', marker='o', stride=1.0,
                       ptype=None, stars_only=False, dm_only=False,
                       minimum_mass=None, alpha=1.0)

    Adds particle positions, based on a thick slab along *axis* with a
    *width* along the line of sight.  *p_size* controls the number of
    pixels per particle, and *col* governs the color.  *ptype* will
    restrict plotted particles to only those that are of a given type.
    *minimum_mass* will require that the particles be of a given mass,
    calculated via ParticleMassMsun, to be plotted. *alpha* determines
    each particle's opacity.
    """
    _type_name = "particles"
    region = None
    _descriptor = None
    def __init__(self, width, p_size=1.0, col='k', marker='o', stride=1.0,
                 ptype=None, stars_only=False, dm_only=False,
                 minimum_mass=None, alpha=1.0):
        PlotCallback.__init__(self)
        self.width = width
        self.p_size = p_size
        self.color = col
        self.marker = marker
        self.stride = stride
        self.ptype = ptype
        self.stars_only = stars_only
        self.dm_only = dm_only
        self.minimum_mass = minimum_mass
        self.alpha = alpha

    def __call__(self, plot):
        data = plot.data
        # we construct a recantangular prism
        x0, x1 = plot.xlim
        y0, y1 = plot.ylim
        xx0, xx1 = plot._axes.get_xlim()
        yy0, yy1 = plot._axes.get_ylim()
        reg = self._get_region((x0,x1), (y0,y1), plot.data.axis, data)
        field_x = "particle_position_%s" % axis_names[x_dict[data.axis]]
        field_y = "particle_position_%s" % axis_names[y_dict[data.axis]]
        gg = ( ( reg[field_x] >= x0 ) & ( reg[field_x] <= x1 )
           &   ( reg[field_y] >= y0 ) & ( reg[field_y] <= y1 ) )
        if self.ptype is not None:
            gg &= (reg["particle_type"] == self.ptype)
            if gg.sum() == 0: return
        if self.stars_only:
            gg &= (reg["creation_time"] > 0.0)
            if gg.sum() == 0: return
        if self.dm_only:
            gg &= (reg["creation_time"] <= 0.0)
            if gg.sum() == 0: return
        if self.minimum_mass is not None:
            gg &= (reg["ParticleMassMsun"] >= self.minimum_mass)
            if gg.sum() == 0: return
        plot._axes.hold(True)
        px, py = self.convert_to_plot(plot,
                    [reg[field_x][gg][::self.stride],
                     reg[field_y][gg][::self.stride]])
        plot._axes.scatter(px, py, edgecolors='None', marker=self.marker,
                           s=self.p_size, c=self.color,alpha=self.alpha)
        plot._axes.set_xlim(xx0,xx1)
        plot._axes.set_ylim(yy0,yy1)
        plot._axes.hold(False)


    def _get_region(self, xlim, ylim, axis, data):
        LE, RE = [None]*3, [None]*3
        xax = x_dict[axis]
        yax = y_dict[axis]
        zax = axis
        LE[xax], RE[xax] = xlim
        LE[yax], RE[yax] = ylim
        LE[zax] = data.center[zax] - self.width*0.5
        RE[zax] = data.center[zax] + self.width*0.5
        if self.region is not None \
            and np.all(self.region.left_edge <= LE) \
            and np.all(self.region.right_edge >= RE):
            return self.region
        self.region = data.pf.h.periodic_region(
            data.center, LE, RE)
        return self.region

class TitleCallback(PlotCallback):
    """
    annotate_title(title)

    Accepts a *title* and adds it to the plot
    """
    _type_name = "title"
    def __init__(self, title):
        PlotCallback.__init__(self)
        self.title = title

    def __call__(self,plot):
        plot._axes.set_title(self.title)

class FlashRayDataCallback(PlotCallback):
    """ 
    annotate_flash_ray_data(cmap_name='bone', sample=None)

    Adds ray trace data to the plot.  *cmap_name* is the name of the color map 
    ('bone', 'jet', 'hot', etc).  *sample* dictates the amount of down sampling 
    to do to prevent all of the rays from being  plotted.  This may be None 
    (plot all rays, default), an integer (step size), or a slice object.
    """
    _type_name = "flash_ray_data"
    def __init__(self, cmap_name='bone', sample=None):
        self.cmap_name = cmap_name
        self.sample = sample if isinstance(sample, slice) else slice(None, None, sample)

    def __call__(self, plot):
        ray_data = plot.data.pf._handle["RayData"][:]
        idx = ray_data[:,0].argsort(kind="mergesort")
        ray_data = ray_data[idx]

        tags = ray_data[:,0]
        coords = ray_data[:,1:3]
        power = ray_data[:,4]
        power /= power.max()
        cx, cy = self.convert_to_plot(plot, coords.T)
        coords[:,0], coords[:,1] = cx, cy
        splitidx = np.argwhere(0 < (tags[1:] - tags[:-1])) + 1
        coords = np.split(coords, splitidx.flat)[self.sample]
        power = np.split(power, splitidx.flat)[self.sample]
        cmap = matplotlib.cm.get_cmap(self.cmap_name)

        plot._axes.hold(True)
        colors = [cmap(p.max()) for p in power]
        lc = matplotlib.collections.LineCollection(coords, colors=colors)
        plot._axes.add_collection(lc)
        plot._axes.hold(False)


class TimestampCallback(PlotCallback):
    """ 
    annotate_timestamp(x, y, units=None, format="{time:.3G} {units}", **kwargs,
                       normalized=False, bbox_dict=None)

    Adds the current time to the plot at point given by *x* and *y*.  If *units* 
    is given ('s', 'ms', 'ns', etc), it will covert the time to this basis.  If 
    *units* is None, it will attempt to figure out the correct value by which to 
    scale.  The *format* keyword is a template string that will be evaluated and 
    displayed on the plot.  If *normalized* is true, *x* and *y* are interpreted 
    as normalized plot coordinates (0,0 is lower-left and 1,1 is upper-right) 
    otherwise *x* and *y* are assumed to be in plot coordinates. The *bbox_dict* 
    is an optional dict of arguments for the bbox that frames the timestamp, see 
    matplotlib's text annotation guide for more details. All other *kwargs* will 
    be passed to the text() method on the plot axes.  See matplotlib's text() 
    functions for more information.
    """
    _type_name = "timestamp"
    _time_conv = {
          'as': 1e-18,
          'attosec': 1e-18,
          'attosecond': 1e-18,
          'attoseconds': 1e-18,
          'fs': 1e-15,
          'femtosec': 1e-15,
          'femtosecond': 1e-15,
          'femtoseconds': 1e-15,
          'ps': 1e-12,
          'picosec': 1e-12,
          'picosecond': 1e-12,
          'picoseconds': 1e-12,
          'ns': 1e-9,
          'nanosec': 1e-9,
          'nanosecond':1e-9,
          'nanoseconds' : 1e-9,
          'us': 1e-6,
          'microsec': 1e-6,
          'microsecond': 1e-6,
          'microseconds': 1e-6,
          'ms': 1e-3,
          'millisec': 1e-3,
          'millisecond': 1e-3,
          'milliseconds': 1e-3,
          's': 1.0,
          'sec': 1.0,
          'second':1.0,
          'seconds': 1.0,
          'm': 60.0,
          'min': 60.0,
          'minute': 60.0,
          'minutes': 60.0,
          'h': sec_per_hr,
          'hour': sec_per_hr,
          'hours': sec_per_hr,
          'd': sec_per_day,
          'day': sec_per_day,
          'days': sec_per_day,
          'y': sec_per_year,
          'year': sec_per_year,
          'years': sec_per_year,
          'kyr': sec_per_kyr,
          'myr': sec_per_Myr,
          'gyr': sec_per_Gyr,
          'ev': 1e-9 * 7.6e-8 / 6.03,
          'kev': 1e-12 * 7.6e-8 / 6.03,
          'mev': 1e-15 * 7.6e-8 / 6.03,
          }
    _bbox_dict = {'boxstyle': 'square,pad=0.6', 'fc': 'white', 'ec': 'black', 'alpha': 1.0}

    def __init__(self, x, y, units=None, format="{time:.3G} {units}", normalized=False, 
                 bbox_dict=None, **kwargs):
        self.x = x
        self.y = y
        self.format = format
        self.units = units
        self.normalized = normalized
        if bbox_dict is not None:
            self.bbox_dict = bbox_dict
        else:
            self.bbox_dict = self._bbox_dict
        self.kwargs = {'color': 'k'}
        self.kwargs.update(kwargs)

    def __call__(self, plot):
        if self.units is None:
            t = plot.data.pf.current_time * plot.data.pf['Time']
            scale_keys = ['as', 'fs', 'ps', 'ns', 'us', 'ms', 's', 
                          'hour', 'day', 'year', 'kyr', 'myr', 'gyr']
            self.units = 's'
            for k in scale_keys:
                if t < self._time_conv[k]:
                    break
                self.units = k
        t = plot.data.pf.current_time * plot.data.pf['Time'] 
        t /= self._time_conv[self.units.lower()]
        if self.units == 'us':
            self.units = '$\\mu s$'
        s = self.format.format(time=t, units=self.units)
        plot._axes.hold(True)
        if self.normalized:
            plot._axes.text(self.x, self.y, s, horizontalalignment='center',
                            verticalalignment='center', 
                            transform = plot._axes.transAxes, bbox=self.bbox_dict)
        else:
            plot._axes.text(self.x, self.y, s, bbox=self.bbox_dict, **self.kwargs)
        plot._axes.hold(False)


class MaterialBoundaryCallback(ContourCallback):
    """ 
    annotate_material_boundary(self, field='targ', ncont=1, factor=4, 
                               clim=(0.9, 1.0), **kwargs):

    Add the limiting contours of *field* to the plot.  Nominally, *field* is 
    the target material but may be any other field present in the hierarchy.
    The number of contours generated is given by *ncount*, *factor* governs 
    the number of points used in the interpolation, and *clim* gives the 
    (upper, lower) limits for contouring.  For this to truly be the boundary
    *clim* should be close to the edge.  For example the default is (0.9, 1.0)
    for 'targ' which is defined on the range [0.0, 1.0].  All other *kwargs* 
    will be passed to the contour() method on the plot axes.  See matplotlib
    for more information.
    """
    _type_name = "material_boundary"
    def __init__(self, field='targ', ncont=1, factor=4, clim=(0.9, 1.0), **kwargs):
        plot_args = {'colors': 'w'}
        plot_args.update(kwargs)
        super(MaterialBoundaryCallback, self).__init__(field=field, ncont=ncont,
                                                       factor=factor, clim=clim,
                                                       plot_args=plot_args)

    def __call__(self, plot):
        super(MaterialBoundaryCallback, self).__call__(plot)

