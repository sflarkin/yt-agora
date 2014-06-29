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
import h5py

from yt.funcs import *
from yt.extern.six import add_metaclass
from _mpl_imports import *
from yt.utilities.physical_constants import \
    sec_per_Gyr, sec_per_Myr, \
    sec_per_kyr, sec_per_year, \
    sec_per_day, sec_per_hr
from yt.units.yt_array import YTQuantity, YTArray
from yt.visualization.image_writer import apply_colormap
from yt.utilities.lib.geometry_utils import triangle_plane_intersect

from . import _MPL

callback_registry = {}

class RegisteredCallback(type):
    def __init__(cls, name, b, d):
        type.__init__(cls, name, b, d)
        callback_registry[name] = cls

@add_metaclass(RegisteredCallback)
class PlotCallback(object):
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

        x0 = np.array(np.tile(plot.xlim[0],ncoord))
        x1 = np.array(np.tile(plot.xlim[1],ncoord))
        xx0 = np.tile(plot._axes.get_xlim()[0],ncoord)
        xx1 = np.tile(plot._axes.get_xlim()[1],ncoord)

        y0 = np.array(np.tile(plot.ylim[0],ncoord))
        y1 = np.array(np.tile(plot.ylim[1],ncoord))
        yy0 = np.tile(plot._axes.get_ylim()[0],ncoord)
        yy1 = np.tile(plot._axes.get_ylim()[1],ncoord)

        ccoord = np.array(coord)

        # We need a special case for when we are only given one coordinate.
        if ccoord.shape == (2,):
            return ((ccoord[0]-x0)/(x1-x0)*(xx1-xx0) + xx0,
                    (ccoord[1]-y0)/(y1-y0)*(yy1-yy0) + yy0)
        else:
            return ((ccoord[0][:]-x0)/(x1-x0)*(xx1-xx0) + xx0,
                    (ccoord[1][:]-y0)/(y1-y0)*(yy1-yy0) + yy0)

    def pixel_scale(self,plot):
        x0, x1 = np.array(plot.xlim)
        xx0, xx1 = plot._axes.get_xlim()
        dx = (xx1 - xx0)/(x1 - x0)

        y0, y1 = np.array(plot.ylim)
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
            qcb = CuttingQuiverCallback("cutting_plane_velocity_x",
                                        "cutting_plane_velocity_y",
                                        self.factor)
        else:
            ax = plot.data.axis
            (xi, yi) = (plot.data.pf.coordinates.x_axis[ax],
                        plot.data.pf.coordinates.y_axis[ax])
            axis_names = plot.data.pf.coordinates.axis_name
            xv = "velocity_%s" % (axis_names[xi])
            yv = "velocity_%s" % (axis_names[yi])

            bv = plot.data.get_field_parameter("bulk_velocity")
            if bv is not None:
                bv_x = bv[xi]
                bv_y = bv[yi]
            else: bv_x = bv_y = YTQuantity(0, 'cm/s')

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
            qcb = CuttingQuiverCallback("cutting_plane_bx",
                                        "cutting_plane_by",
                                        self.factor)
        else:
            xax = plot.data.pf.coordinates.x_axis[plot.data.axis]
            yax = plot.data.pf.coordinates.y_axis[plot.data.axis]
            axis_names = plot.data.pf.coordinates.axis_name
            xv = "magnetic_field_%s" % (axis_names[xax])
            yv = "magnetic_field_%s" % (axis_names[yax])
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
        (xi, yi) = (pf.coordinates.x_axis[ax],
                    pf.coordinates.y_axis[ax])
        period_x = pf.domain_width[xi]
        period_y = pf.domain_width[yi]
        periodic = int(any(pf.periodicity))
        fv_x = plot.data[self.field_x]
        if self.bv_x != 0.0:
            # Workaround for 0.0 without units
            fv_x -= self.bv_x
        fv_y = plot.data[self.field_y]
        if self.bv_y != 0.0:
            # Workaround for 0.0 without units
            fv_y -= self.bv_y
        pixX = _MPL.Pixelize(plot.data['px'],
                             plot.data['py'],
                             plot.data['pdx'],
                             plot.data['pdy'],
                             fv_x,
                             int(nx), int(ny),
                             (x0, x1, y0, y1), 0, # bounds, antialias
                             (period_x, period_y), periodic,
                           ).transpose()
        pixY = _MPL.Pixelize(plot.data['px'],
                             plot.data['py'],
                             plot.data['pdx'],
                             plot.data['pdy'],
                             fv_y,
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
        # These need to be in code_length
        x0, x1 = (v.in_units("code_length") for v in plot.xlim)
        y0, y1 = (v.in_units("code_length") for v in plot.ylim)

        # These are in plot coordinates, which may not be code coordinates.
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
                x = ((XShifted[wI]-x0)*dx).ndarray_view() + xx0
                y = ((YShifted[wI]-y0)*dy).ndarray_view() + yy0
                z = data[self.field][wI]
        
            # Both the input and output from the triangulator are in plot
            # coordinates
            zi = self.triang(x,y).nn_interpolator(z)(xi,yi)
        elif plot._type_name == 'OffAxisProjection':
            zi = plot.frb[self.field][::self.factor,::self.factor].transpose()
        
        if self.take_log is None:
            field = data._determine_fields([self.field])[0]
            self.take_log = plot.pf._get_field_info(*field).take_log

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
        (dx, dy) = self.pixel_scale(plot)
        (xpix, ypix) = plot.image._A.shape
        ax = plot.data.axis
        px_index = plot.data.pf.coordinates.x_axis[ax]
        py_index = plot.data.pf.coordinates.y_axis[ax]
        DW = plot.data.pf.domain_width
        if self.periodic:
            pxs, pys = np.mgrid[-1:1:3j,-1:1:3j]
        else:
            pxs, pys = np.mgrid[0:0:1j,0:0:1j]
        GLE, GRE, levels = [], [], []
        for block, mask in plot.data.blocks:
            GLE.append(block.LeftEdge.in_units("code_length"))
            GRE.append(block.RightEdge.in_units("code_length"))
            levels.append(block.Level)
        if len(GLE) == 0: return
        # Retain both units and registry
        GLE = YTArray(GLE, input_units = GLE[0].units)
        GRE = YTArray(GRE, input_units = GRE[0].units)
        levels = np.array(levels)
        min_level = self.min_level or 0
        max_level = self.max_level or levels.max()

        for px_off, py_off in zip(pxs.ravel(), pys.ravel()):
            pxo = px_off * DW[px_index]
            pyo = py_off * DW[py_index]
            left_edge_x = np.array((GLE[:,px_index]+pxo-x0)*dx) + xx0
            left_edge_y = np.array((GLE[:,py_index]+pyo-y0)*dy) + yy0
            right_edge_x = np.array((GRE[:,px_index]+pxo-x0)*dx) + xx0
            right_edge_y = np.array((GRE[:,py_index]+pyo-y0)*dy) + yy0
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
                active_ids = np.unique(plot.data['grid_indices'])
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
                             plot.data[self.field_x],
                             int(nx), int(ny),
                             (x0, x1, y0, y1),).transpose()
        pixY = _MPL.Pixelize(plot.data['px'],
                             plot.data['py'],
                             plot.data['pdx'],
                             plot.data['pdy'],
                             plot.data[self.field_y],
                             int(nx), int(ny),
                             (x0, x1, y0, y1),).transpose()
        X,Y = (np.linspace(xx0,xx1,nx,endpoint=True),
               np.linspace(yy0,yy1,ny,endpoint=True))
        streamplot_args = {'x': X, 'y': Y, 'u':pixX, 'v': pixY,
                           'density': self.dens}
        streamplot_args.update(self.plot_args)
        plot._axes.streamplot(**streamplot_args)
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
    for unit in ['Mpc', 'kpc', 'pc', 'au', 'rsun', 'km', 'cm']:
        uq = YTQuantity(1.0, unit)
        if uq < v:
            good_u = unit
            break
    if good_u is None : good_u = 'cm'
    return good_u

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
        xx0, xx1 = plot._axes.get_xlim()
        yy0, yy1 = plot._axes.get_ylim()
        plot._axes.hold(True)
        plot._axes.plot(self.x, self.y, **self.plot_args)
        plot._axes.set_xlim(xx0,xx1)
        plot._axes.set_ylim(yy0,yy1)
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

        ax = plot.data.axis
        px_index = plot.data.pf.coordinates.x_axis[ax]
        py_index = plot.data.pf.coordinates.y_axis[ax]

        xf = plot.data.pf.coordinates.axis_name[px_index]
        yf = plot.data.pf.coordinates.axis_name[py_index]
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
        if isinstance(code_size, YTArray):
            code_size = code_size.in_units('code_length')
        if not iterable(code_size):
            code_size = (code_size, code_size)
        self.code_size = code_size
        if plot_args is None: plot_args = {}
        self.plot_args = plot_args

    def __call__(self, plot):
        if len(self.pos) == 3:
            ax = plot.data.axis
            (xi, yi) = (plot.data.pf.coordinates.x_axis[ax],
                        plot.data.pf.coordinates.y_axis[ax])
            pos = self.pos[xi], self.pos[yi]
        else: pos = self.pos
        if isinstance(self.code_size[1], basestring):
            code_size = plot.data.pf.quan(*self.code_size)
            code_size = code_size.in_units('code_length').value
            self.code_size = (code_size, code_size)
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
            ax = plot.data.axis
            (xi, yi) = (plot.data.pf.coordinates.x_axis[ax],
                        plot.data.pf.coordinates.y_axis[ax])
            pos = self.pos[xi], self.pos[yi]
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
            ax = plot.data.axis
            (xi, yi) = (plot.data.pf.coordinates.x_axis[ax],
                        plot.data.pf.coordinates.y_axis[ax])
            pos = self.pos[xi], self.pos[yi]
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

        if iterable(self.radius):
            self.radius = plot.data.pf.quan(self.radius[0], self.radius[1])
            self.radius = np.float64(self.radius.in_units(plot.xlim[0].units))

        radius = self.radius * self.pixel_scale(plot)[0]

        if plot.data.axis == 4:
            (xi, yi) = (0, 1)
        else:
            ax = plot.data.axis
            (xi, yi) = (plot.data.pf.coordinates.x_axis[ax],
                        plot.data.pf.coordinates.y_axis[ax])

        (center_x,center_y) = self.convert_to_plot(plot,(self.center[xi], self.center[yi]))
        
        cir = Circle((center_x, center_y), radius, **self.circle_args)
        plot._axes.add_patch(cir)
        if self.text is not None:
            plot._axes.text(center_x, center_y, self.text,
                            **self.text_args)


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
                ax = plot.data.axis
                (xi, yi) = (plot.data.pf.coordinates.x_axis[ax],
                            plot.data.pf.coordinates.y_axis[ax])
                pos = self.pos[xi], self.pos[yi]
            else: pos = self.pos
            x,y = self.convert_to_plot(plot, pos)
        else:
            x, y = self.pos
            if not self.data_coords:
                kwargs["transform"] = plot._axes.transAxes
        plot._axes.text(x, y, self.text, **kwargs)

class HaloCatalogCallback(PlotCallback):

    _type_name = 'halos'
    region = None
    _descriptor = None

    def __init__(self, halo_catalog, col='white', alpha =1, 
            width = None, annotate_field = False, font_kwargs = None):

        PlotCallback.__init__(self)
        self.halo_catalog = halo_catalog
        self.color = col
        self.alpha = alpha
        self.width = width
        self.annotate_field = annotate_field
        self.font_kwargs = font_kwargs

    def __call__(self, plot):
        data = plot.data
        x0, x1 = plot.xlim
        y0, y1 = plot.ylim
        xx0, xx1 = plot._axes.get_xlim()
        yy0, yy1 = plot._axes.get_ylim()
        
        halo_data= self.halo_catalog.halos_pf.all_data()
        axis_names = plot.data.pf.coordinates.axis_name
        xax = plot.data.pf.coordinates.x_axis[data.axis]
        yax = plot.data.pf.coordinates.y_axis[data.axis]
        field_x = "particle_position_%s" % axis_names[xax]
        field_y = "particle_position_%s" % axis_names[yax]
        field_z = "particle_position_%s" % axis_names[data.axis]
        plot._axes.hold(True)

        # Set up scales for pixel size and original data
        units = 'Mpccm'
        pixel_scale = self.pixel_scale(plot)[0]
        data_scale = data.pf.length_unit

        # Convert halo positions to code units of the plotted data
        # and then to units of the plotted window
        px = halo_data[field_x][:].in_units(units) / data_scale
        py = halo_data[field_y][:].in_units(units) / data_scale
        px, py = self.convert_to_plot(plot,[px,py])
        
        # Convert halo radii to a radius in pixels
        radius = halo_data['radius'][:].in_units(units)
        radius = radius*pixel_scale/data_scale

        if self.width:
            pz = halo_data[field_z][:].in_units(units)/data_scale
            pz = data.pf.arr(pz, 'code_length')
            c = data.center[data.axis]

            # I should catch an error here if width isn't in this form
            # but I dont really want to reimplement get_sanitized_width...
            width = data.pf.arr(self.width[0], self.width[1]).in_units('code_length')

            indices = np.where((pz > c-width) & (pz < c+width))

            px = px[indices]
            py = py[indices]
            radius = radius[indices]

        plot._axes.scatter(px, py, edgecolors='None', marker='o',
                           s=radius, c=self.color,alpha=self.alpha)
        plot._axes.set_xlim(xx0,xx1)
        plot._axes.set_ylim(yy0,yy1)
        plot._axes.hold(False)

        if self.annotate_field:
            annotate_dat = halo_data[self.annotate_field]
            texts = ['{0}'.format(dat) for dat in annotate_dat]
            for pos_x, pos_y, t in zip(px, py, texts): 
                plot._axes.text(pos_x, pos_y, t, **self.font_kwargs)
 

class ParticleCallback(PlotCallback):
    """
    annotate_particles(width, p_size=1.0, col='k', marker='o', stride=1.0,
                       ptype=None, stars_only=False, dm_only=False,
                       minimum_mass=None, alpha=1.0)

    Adds particle positions, based on a thick slab along *axis* with a
    *width* along the line of sight.  *p_size* controls the number of
    pixels per particle, and *col* governs the color.  *ptype* will
    restrict plotted particles to only those that are of a given type.
    Particles with masses below *minimum_mass* will not be plotted.
    *alpha* determines the opacity of the marker symbol used in the scatter
    plot.
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
        if iterable(self.width):
            self.width = np.float64(plot.data.pf.quan(self.width[0], self.width[1]))
        # we construct a recantangular prism
        x0, x1 = plot.xlim
        y0, y1 = plot.ylim
        xx0, xx1 = plot._axes.get_xlim()
        yy0, yy1 = plot._axes.get_ylim()
        reg = self._get_region((x0,x1), (y0,y1), plot.data.axis, data)
        ax = data.axis
        xax = plot.data.pf.coordinates.x_axis[ax]
        yax = plot.data.pf.coordinates.y_axis[ax]
        axis_names = plot.data.pf.coordinates.axis_name
        field_x = "particle_position_%s" % axis_names[xax]
        field_y = "particle_position_%s" % axis_names[yax]
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
            gg &= (reg["particle_mass"] >= self.minimum_mass)
            if gg.sum() == 0: return
        plot._axes.hold(True)
        px, py = self.convert_to_plot(plot,
                    [np.array(reg[field_x][gg][::self.stride]),
                     np.array(reg[field_y][gg][::self.stride])])
        plot._axes.scatter(px, py, edgecolors='None', marker=self.marker,
                           s=self.p_size, c=self.color,alpha=self.alpha)
        plot._axes.set_xlim(xx0,xx1)
        plot._axes.set_ylim(yy0,yy1)
        plot._axes.hold(False)


    def _get_region(self, xlim, ylim, axis, data):
        LE, RE = [None]*3, [None]*3
        pf = data.pf
        xax = pf.coordinates.x_axis[axis]
        yax = pf.coordinates.y_axis[axis]
        zax = axis
        LE[xax], RE[xax] = xlim
        LE[yax], RE[yax] = ylim
        LE[zax] = data.center[zax].ndarray_view() - self.width*0.5
        RE[zax] = data.center[zax].ndarray_view() + self.width*0.5
        if self.region is not None \
            and np.all(self.region.left_edge <= LE) \
            and np.all(self.region.right_edge >= RE):
            return self.region
        self.region = data.pf.region(data.center, LE, RE)
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
    the target material but may be any other field present in the index.
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

class TriangleFacetsCallback(PlotCallback):
    """ 
    annotate_triangle_facets(triangle_vertices, plot_args=None )

    Intended for representing a slice of a triangular faceted 
    geometry in a slice plot. 

    Uses a set of *triangle_vertices* to find all trangles the plane of a 
    SlicePlot intersects with. The lines between the intersection points 
    of the triangles are then added to the plot to create an outline
    of the geometry represented by the triangles. 
    """
    _type_name = "triangle_facets"
    def __init__(self, triangle_vertices, plot_args=None):
        super(TriangleFacetsCallback, self).__init__()
        self.plot_args = {} if plot_args is None else plot_args
        self.vertices = triangle_vertices

    def __call__(self, plot):
        plot._axes.hold(True)
        ax = plot.data.axis
        xax = plot.data.pf.coordinates.x_axis[ax]
        yax = plot.data.pf.coordinates.y_axis[ax]
        l_cy = triangle_plane_intersect(plot.data.axis, plot.data.coord, self.vertices)[:,:,(xax, yax)]
        lc = matplotlib.collections.LineCollection(l_cy, **self.plot_args)
        plot._axes.add_collection(lc)
        plot._axes.hold(False)

