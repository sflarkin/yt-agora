"""
A plotting mechanism based on the idea of a "window" into the data.



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import base64
import numpy as np
import matplotlib
import cStringIO
import types

from matplotlib.delaunay.triangulate import Triangulation as triang
from matplotlib.mathtext import MathTextParser
from distutils import version
from numbers import Number

from ._mpl_imports import FigureCanvasAgg
from .image_writer import apply_colormap
from .base_plot_types import ImagePlotMPL
from .fixed_resolution import \
    FixedResolutionBuffer, \
    ObliqueFixedResolutionBuffer, \
    OffAxisProjectionFixedResolutionBuffer
from .plot_modifications import get_smallest_appropriate_unit, \
    callback_registry
from .plot_container import \
    ImagePlotContainer, log_transform, linear_transform, \
    invalidate_data, invalidate_plot, apply_callback

from yt.funcs import \
    mylog, iterable, ensure_list, \
    fix_axis, assert_valid_width_tuple
from yt.units.unit_object import Unit
from yt.utilities.png_writer import \
    write_png_to_string
from yt.utilities.definitions import \
    x_dict, y_dict, \
    axis_names, axis_labels, \
    formatted_length_unit_names
from yt.utilities.math_utils import \
    ortho_find
from yt.utilities.exceptions import \
    YTUnitNotRecognized, YTInvalidWidthError, YTCannotParseUnitDisplayName

from yt.data_objects.time_series import \
    DatasetSeries
from yt.units.yt_array import YTArray, YTQuantity

# Some magic for dealing with pyparsing being included or not
# included in matplotlib (not in gentoo, yes in everything else)
# Also accounting for the fact that in 1.2.0, pyparsing got renamed.
try:
    if version.LooseVersion(matplotlib.__version__) < \
        version.LooseVersion("1.2.0"):
        from matplotlib.pyparsing import ParseFatalException
    else:
        from matplotlib.pyparsing_py2 import ParseFatalException
except ImportError:
    from pyparsing import ParseFatalException

def fix_unitary(u):
    if u is '1':
        return 'code_length'
    else:
        return u

def assert_valid_width_tuple(width):
    if not iterable(width) or len(width) != 2:
        raise YTInvalidWidthError("width (%s) is not a two element tuple" % width)
    if not isinstance(width[0], Number) and isinstance(width[1], basestring):
        msg = "width (%s) is invalid. " % str(width)
        msg += "Valid widths look like this: (12, 'au')"
        raise YTInvalidWidthError(msg)

def validate_iterable_width(width, pf, unit=None):
    if isinstance(width[0], tuple) and isinstance(width[1], tuple):
        assert_valid_width_tuple(width[0])
        assert_valid_width_tuple(width[1])
        return (pf.quan(width[0][0], fix_unitary(width[0][1])),
                pf.quan(width[1][0], fix_unitary(width[1][1])))
    elif isinstance(width[0], Number) and isinstance(width[1], Number):
        return (pf.quan(width[0], 'code_length'),
                pf.quan(width[1], 'code_length'))
    elif isinstance(width[0], YTQuantity) and isinstance(width[1], YTQuantity):
        return width
    else:
        assert_valid_width_tuple(width)
        # If width and unit are both valid width tuples, we
        # assume width controls x and unit controls y
        try:
            assert_valid_width_tuple(unit)
            return (pf.quan(width[0], fix_unitary(width[1])),
                    pf.quan(unit[0], fix_unitary(unit[1])))
        except YTInvalidWidthError:
            return (pf.quan(width[0], fix_unitary(width[1])),
                    pf.quan(width[0], fix_unitary(width[1])))

def get_sanitized_width(axis, width, depth, pf):
    if width is None:
        # Default to code units
        if not iterable(axis):
            w = pf.domain_width[[x_dict[axis], y_dict[axis]]]
        else:
            # axis is actually the normal vector
            # for an off-axis data object.
            mi = np.argmin(pf.domain_width)
            w = pf.domain_width[[mi,mi]]
        width = (w[0], w[1])
    elif iterable(width):
        width = validate_iterable_width(width, pf)
    else:
        try:
            assert isinstance(width, Number), \
              "width (%s) is invalid" % str(width)
        except AssertionError, e:
            raise YTInvalidWidthError(e)
        width = (pf.arr(width, 'code_length'),
                 pf.arr(width, 'code_length'))
    if depth is not None:
        if iterable(depth):
            assert_valid_width_tuple(depth)
            depth = (pf.quan(depth[0], fix_unitary(depth[1])),)
        else:
            try:
                assert isinstance(depth, Number), \
                  "width (%s) is invalid" % str(depth)
            except AssertionError, e:
                raise YTInvalidWidthError(e)
            depth = (pf.quan(depth, 'code_length',
                     registry = pf.unit_registry),)
        return width + depth
    return width

def get_sanitized_center(center, pf):
    if isinstance(center, basestring):
        if center.lower() == "m" or center.lower() == "max":
            v, center = pf.h.find_max("density")
            center = pf.arr(center, 'code_length')
        elif center.lower() == "c" or center.lower() == "center":
            center = (pf.domain_left_edge + pf.domain_right_edge) / 2
        else:
            raise RuntimeError('center keyword \"%s\" not recognized' % center)
    elif isinstance(center, YTArray):
        pass
    elif iterable(center):
        if iterable(center[0]) and isinstance(center[1], basestring):
            center = pf.arr(center[0], center[1])
        else:
            center = pf.arr(center, 'code_length')
    else:
        raise RuntimeError("center keyword \"%s\" not recognized" % center)
    return center

def get_window_parameters(axis, center, width, pf):
    if pf.geometry == "cartesian":
        width = get_sanitized_width(axis, width, None, pf)
        center = get_sanitized_center(center, pf)
    elif pf.geometry in ("polar", "cylindrical"):
        # Set our default width to be the full domain
        width = [pf.domain_right_edge[0]*2.0, pf.domain_right_edge[0]*2.0]
        center = pf.arr([0.0, 0.0, 0.0], "code_length")
    elif pf.geometry == "spherical":
        if axis == 0:
            width = pf.domain_width[1], pf.domain_width[2]
            center = 0.5*(pf.domain_left_edge +
                pf.domain_right_edge).in_units("code_length")
        else:
            # Our default width here is the full domain
            width = [pf.domain_right_edge[0]*2.0, pf.domain_right_edge[0]*2.0]
            center = pf.arr([0.0, 0.0, 0.0], "code_length")
    else:
        raise NotImplementedError
    bounds = (center[x_dict[axis]]-width[0] / 2,
              center[x_dict[axis]]+width[0] / 2,
              center[y_dict[axis]]-width[1] / 2,
              center[y_dict[axis]]+width[1] / 2)
    return (bounds, center)

def get_oblique_window_parameters(normal, center, width, pf, depth=None):
    width = get_sanitized_width(normal, width, depth, pf)
    center = get_sanitized_center(center, pf)

    if len(width) == 2:
        # Transforming to the cutting plane coordinate system
        center = (center - pf.domain_left_edge)/pf.domain_width - 0.5
        (normal,perp1,perp2) = ortho_find(normal)
        mat = np.transpose(np.column_stack((perp1,perp2,normal)))
        center = np.dot(mat,center)

    bounds = tuple( ( (2*(i%2))-1)*width[i//2]/2 for i in range(len(width)*2))

    return (bounds, center)

def get_axes_unit(width, pf):
    r"""
    Infers the axes unit names from the input width specification
    """
    if iterable(width):
        if isinstance(width[1], basestring):
            axes_unit = (width[1], width[1])
        elif iterable(width[1]):
            axes_unit = (width[0][1], width[1][1])
        elif isinstance(width[0], YTArray):
            axes_unit = (str(width[0].units), str(width[1].units))
        else:
            axes_unit = None
    else:
        if isinstance(width, YTArray):
            axes_unit = (str(width.units), str(width.units))
        else:
            axes_unit = None
    return axes_unit

class PlotWindow(ImagePlotContainer):
    r"""
    A ploting mechanism based around the concept of a window into a
    data source. It can have arbitrary fields, each of which will be
    centered on the same viewpoint, but will have individual zlimits.

    The data and plot are updated separately, and each can be
    invalidated as the object is modified.

    Data is handled by a FixedResolutionBuffer object.

    Parameters
    ----------
    data_source : :class:`yt.data_objects.data_containers.AMRProjBase` or
                  :class:`yt.data_objects.data_containers.AMRSliceBase`
        This is the source to be pixelized, which can be a projection or a
        slice.  (For cutting planes, see
        `yt.visualization.fixed_resolution.ObliqueFixedResolutionBuffer`.)
    bounds : sequence of floats
        Bounds are the min and max in the image plane that we want our
        image to cover.  It's in the order of (xmin, xmax, ymin, ymax),
        where the coordinates are all in the appropriate code units.
    buff_size : sequence of ints
        The size of the image to generate.
    antialias : boolean
        This can be true or false.  It determines whether or not sub-pixel
        rendering is used during data deposition.
    window_size : float
        The size of the window on the longest axis (in units of inches),
        including the margins but not the colorbar.

    """
    _frb = None
    def __init__(self, data_source, bounds, buff_size=(800,800), antialias=True,
                 periodic=True, origin='center-window', oblique=False,
                 window_size=8.0, fields=None, fontsize=18, setup=False):
        if not hasattr(self, "pf"):
            self.pf = data_source.pf
            ts = self._initialize_dataset(self.pf)
            self.ts = ts
        self._initfinished = False
        self._axes_unit_names = None
        self.center = None
        self._periodic = periodic
        self.oblique = oblique
        self.buff_size = buff_size
        self.antialias = antialias
        skip = list(FixedResolutionBuffer._exclude_fields) + data_source._key_fields
        if fields is None:
            fields = []
        else:
            fields = ensure_list(fields)
        self.override_fields = list(set(fields).intersection(set(skip)))
        self.fields = fields
        super(PlotWindow, self).__init__(data_source, window_size, fontsize)
        self._set_window(bounds) # this automatically updates the data and plot
        self.origin = origin
        if self.data_source.center is not None and oblique == False:
            center = [self.data_source.center[i] for i in
                      range(len(self.data_source.center))
                      if i != self.data_source.axis]
            self.set_center(center)
        for field in self._frb.data.keys():
            finfo = self.data_source.pf._get_field_info(*field)
            if finfo.take_log:
                self._field_transform[field] = log_transform
            else:
                self._field_transform[field] = linear_transform
        self.setup_callbacks()
        self._initfinished = True

    def _initialize_dataset(self, ts):
        if not isinstance(ts, DatasetSeries):
            if not iterable(ts): ts = [ts]
            ts = DatasetSeries(ts)
        return ts

    def __iter__(self):
        for pf in self.ts:
            mylog.warning("Switching to %s", pf)
            self._switch_pf(pf)
            yield self

    def piter(self, *args, **kwargs):
        for pf in self.ts.piter(*args, **kwargs):
            self._switch_pf(pf)
            yield self

    def _recreate_frb(self):
        old_fields = None
        if self._frb is not None:
            old_fields = self._frb.keys()
            old_units = [str(self._frb[of].units) for of in old_fields]
        if hasattr(self,'zlim'):
            bounds = self.xlim+self.ylim+self.zlim
        else:
            bounds = self.xlim+self.ylim
        if self._frb_generator is ObliqueFixedResolutionBuffer:
            bounds = np.array(bounds)
        self._frb = self._frb_generator(self.data_source,
                                        bounds, self.buff_size,
                                        self.antialias,
                                        periodic=self._periodic)
        if old_fields is None:
            self._frb._get_data_source_fields()
        else:
            for key, unit in zip(old_fields, old_units):
                self._frb[key]
                self._frb[key].convert_to_units(unit)
        for key in self.override_fields:
            self._frb[key]
        self._data_valid = True

    @property
    def width(self):
        Wx = self.xlim[1] - self.xlim[0]
        Wy = self.ylim[1] - self.ylim[0]
        return (Wx, Wy)

    @property
    def bounds(self):
        return self.xlim+self.ylim

    @invalidate_data
    def zoom(self, factor):
        r"""This zooms the window by *factor*.

        Parameters
        ----------
        factor : float
            multiplier for the current width

        """
        Wx, Wy = self.width
        centerx = self.xlim[0] + Wx*0.5
        centery = self.ylim[0] + Wy*0.5
        nWx, nWy = Wx/factor, Wy/factor
        self.xlim = (centerx - nWx*0.5, centerx + nWx*0.5)
        self.ylim = (centery - nWy*0.5, centery + nWy*0.5)
        return self

    @invalidate_data
    def pan(self, deltas):
        r"""Pan the image by specifying absolute code unit coordinate deltas.

        Parameters
        ----------
        deltas : sequence of floats
            (delta_x, delta_y) in *absolute* code unit coordinates

        """
        self.xlim = (self.xlim[0] + deltas[0], self.xlim[1] + deltas[0])
        self.ylim = (self.ylim[0] + deltas[1], self.ylim[1] + deltas[1])
        return self

    @invalidate_data
    def pan_rel(self, deltas):
        r"""Pan the image by specifying relative deltas, to the FOV.

        Parameters
        ----------
        deltas : sequence of floats
            (delta_x, delta_y) in *relative* code unit coordinates

        """
        Wx, Wy = self.width
        self.xlim = (self.xlim[0] + Wx*deltas[0], self.xlim[1] + Wx*deltas[0])
        self.ylim = (self.ylim[0] + Wy*deltas[1], self.ylim[1] + Wy*deltas[1])
        return self

    @invalidate_plot
    def set_unit(self, field, new_unit):
        """Sets a new unit for the requested field

        parameters
        ----------
        field : string or field tuple
           The name of the field that is to be changed.

        new_unit : string or Unit object
           The name of the new unit.
        """
        field = self.data_source._determine_fields(field)[0]
        field = ensure_list(field)
        new_unit = ensure_list(new_unit)
        if len(field) > 1 and len(new_unit) != len(field):
            raise RuntimeError(
                "Field list {} and unit "
                "list {} are incompatible".format(field, new_unit))
        for f, u in zip(field, new_unit):
            self._frb[f].convert_to_units(u)
        return self

    @invalidate_data
    def _set_window(self, bounds):
        """Set the bounds of the plot window.
        This is normally only called internally, see set_width.


        Parameters
        ----------

        bounds : a four element sequence of floats
            The x and y bounds, in the format (x0, x1, y0, y1)

        """
        if self.center is not None:
            dx = bounds[1] - bounds[0]
            dy = bounds[3] - bounds[2]
            self.xlim = (self.center[0] - dx/2., self.center[0] + dx/2.)
            self.ylim = (self.center[1] - dy/2., self.center[1] + dy/2.)
        else:
            self.xlim = tuple(bounds[0:2])
            self.ylim = tuple(bounds[2:4])
            if len(bounds) == 6:
                self.zlim = tuple(bounds[4:6])
        mylog.info("xlim = %f %f" %self.xlim)
        mylog.info("ylim = %f %f" %self.ylim)
        if hasattr(self,'zlim'):
            mylog.info("zlim = %f %f" %self.zlim)

    @invalidate_data
    def set_width(self, width, unit = None):
        """set the width of the plot window

        parameters
        ----------
        width : float, array of floats, (float, unit) tuple, or tuple of
                (float, unit) tuples.
             Width can have four different formats to support windows with
             variable x and y widths.  They are:

             ==================================     =======================
             format                                 example
             ==================================     =======================
             (float, string)                        (10,'kpc')
             ((float, string), (float, string))     ((10,'kpc'),(15,'kpc'))
             float                                  0.2
             (float, float)                         (0.2, 0.3)
             ==================================     =======================

             For example, (10, 'kpc') requests a plot window that is 10
             kiloparsecs wide in the x and y directions,
             ((10,'kpc'),(15,'kpc')) requests a window that is 10 kiloparsecs
             wide along the x axis and 15 kiloparsecs wide along the y axis.
             In the other two examples, code units are assumed, for example
             (0.2, 0.3) requests a plot that has an x width of 0.2 and a y
             width of 0.3 in code units.  If units are provided the resulting
             plot axis labels will use the supplied units.
        unit : str
             the unit the width has been specified in. If width is a tuple, this
             argument is ignored. Defaults to code units.
        """
        if isinstance(width, Number):
            if unit is None:
                width = (width, 'code_length')
            else:
                width = (width, unit)

        axes_unit = get_axes_unit(width, self.pf)

        width = get_sanitized_width(self._frb.axis, width, None, self.pf)

        centerx = (self.xlim[1] + self.xlim[0])/2.
        centery = (self.ylim[1] + self.ylim[0])/2.

        self.xlim = (centerx - width[0]/2, centerx + width[0]/2)
        self.ylim = (centery - width[1]/2, centery + width[1]/2)

        if hasattr(self,'zlim'):
            centerz = (self.zlim[1] + self.zlim[0])/2.
            mw = np.max([width[0], width[1]])
            self.zlim = (centerz - mw/2.,
                         centerz + mw/2.)

        self.set_axes_unit(axes_unit)

        return self

    @invalidate_data
    def set_center(self, new_center, unit = 'code_length'):
        """Sets a new center for the plot window

        parameters
        ----------
        new_center : two element sequence of floats
            The coordinates of the new center of the image in the
            coordinate system defined by the plot axes. If the unit
            keyword is not specified, the coordinates are assumed to
            be in code units.

        unit : string
            The name of the unit new_center is given in.  If new_center is a
            YTArray or tuple of YTQuantities, this keyword is ignored.

        """
        error = RuntimeError(
            "\n"
            "new_center must be a two-element list or tuple of floats \n"
            "corresponding to a coordinate in the plot relative to \n"
            "the plot coordinate system.\n"
        )
        if new_center is None:
            self.center = None
        elif iterable(new_center):
            if len(new_center) != 2:
                raise error
            for el in new_center:
                if not isinstance(el, Number) and not isinstance(el, YTQuantity):
                    raise error
            if isinstance(new_center[0], Number):
                new_center = [self.pf.quan(c, unit) for c in new_center]
            self.center = new_center
        else:
            raise error
        self._set_window(self.bounds)
        return self

    @invalidate_data
    def set_antialias(self,aa):
        self.antialias = aa

    @invalidate_data
    def set_buff_size(self, size):
        """Sets a new buffer size for the fixed resolution buffer

        parameters
        ----------
        size : int or two element sequence of ints
            The number of data elements in the buffer on the x and y axes.
            If a scalar is provided,  then the buffer is assumed to be square.
        """
        if iterable(size):
            self.buff_size = size
        else:
            self.buff_size = (size, size)
        return self

    def set_window_size(self, size):
        """This calls set_figure_size to adjust the size of the plot window.

        This is equivalent to set_figure_size but it still available to maintain
        backwards compatibility.
        """
        self.set_figure_size(size)
        return self

    @invalidate_plot
    def set_axes_unit(self, unit_name):
        r"""Set the unit for display on the x and y axes of the image.

        Parameters
        ----------
        unit_name : string or two element tuple of strings
            A unit, available for conversion in the parameter file, that the
            image extents will be displayed in.  If set to None, any previous
            units will be reset.  If the unit is None, the default is chosen.
            If unit_name is '1', 'u', or 'unitary', it will not display the
            units, and only show the axes name. If unit_name is a tuple, the
            first element is assumed to be the unit for the x axis and the
            second element the unit for the y axis.

        Raises
        ------
        YTUnitNotRecognized
            If the unit is not known, this will be raised.

        Examples
        --------

        >>> p = ProjectionPlot(pf, "y", "Density")
        >>> p.show()
        >>> p.set_axes_unit("kpc")
        >>> p.show()
        >>> p.set_axes_unit(None)
        >>> p.show()
        """
        # blind except because it could be in conversion_factors or units
        if unit_name is not None:
            if isinstance(unit_name, basestring):
                unit_name = (unit_name, unit_name)
            for un in unit_name:
                try:
                    self.pf.length_unit.in_units(un)
                except (YTUnitConversionError, UnitParseError):
                    raise YTUnitNotRecognized(un)
        self._axes_unit_names = unit_name
        return self

class PWViewerMPL(PlotWindow):
    """Viewer using matplotlib as a backend via the WindowPlotMPL.

    """
    _current_field = None
    _frb_generator = None
    _plot_type = None

    def __init__(self, *args, **kwargs):
        if self._frb_generator is None:
            self._frb_generator = kwargs.pop("frb_generator")
        if self._plot_type is None:
            self._plot_type = kwargs.pop("plot_type")
        PlotWindow.__init__(self, *args, **kwargs)

    def _setup_origin(self):
        origin = self.origin
        axis_index = self.data_source.axis
        if isinstance(origin, basestring):
            origin = tuple(origin.split('-'))[:3]
        if 1 == len(origin):
            origin = ('lower', 'left') + origin
        elif 2 == len(origin) and origin[0] in set(['left','right','center']):
            o0map = {'left': 'lower', 'right': 'upper', 'center': 'center'}
            origin = (o0map[origin[0]],) + origin
        elif 2 == len(origin) and origin[0] in set(['lower','upper','center']):
            origin = (origin[0], 'center', origin[-1])
        assert origin[-1] in ['window', 'domain', 'native']

        if origin[2] == 'window':
            xllim, xrlim = self.xlim
            yllim, yrlim = self.ylim
        elif origin[2] == 'domain':
            xllim = self.pf.domain_left_edge[x_dict[axis_index]]
            xrlim = self.pf.domain_right_edge[x_dict[axis_index]]
            yllim = self.pf.domain_left_edge[y_dict[axis_index]]
            yrlim = self.pf.domain_right_edge[y_dict[axis_index]]
        elif origin[2] == 'native':
            return (self.pf.quan(0.0, 'code_length'),
                    self.pf.quan(0.0, 'code_length'))
        else:
            mylog.warn("origin = {0}".format(origin))
            msg = \
              ('origin keyword "{0}" not recognized, must declare "domain" '
               'or "center" as the last term in origin.').format(self.origin)
            raise RuntimeError(msg)

        if origin[0] == 'lower':
            yc = yllim
        elif origin[0] == 'upper':
            yc = yrlim
        elif origin[0] == 'center':
            yc = (yllim + yrlim)/2.0
        else:
            mylog.warn("origin = {0}".format(origin))
            msg = ('origin keyword "{0}" not recognized, must declare "lower" '
                   '"upper" or "center" as the first term in origin.')
            msg = msg.format(self.origin)
            raise RuntimeError(msg)

        if origin[1] == 'left':
            xc = xllim
        elif origin[1] == 'right':
            xc = xrlim
        elif origin[1] == 'center':
            xc = (xllim + xrlim)/2.0
        else:
            mylog.warn("origin = {0}".format(origin))
            msg = ('origin keyword "{0}" not recognized, must declare "left" '
                   '"right" or "center" as the second term in origin.')
            msg = msg.format(self.origin)
            raise RuntimeError(msg)

        return xc, yc

    def _setup_plots(self):
        self._colorbar_valid = True
        for f in self.data_source._determine_fields(self.fields):
            axis_index = self.data_source.axis

            xc, yc = self._setup_origin()

            if self._axes_unit_names is None:
                unit = get_smallest_appropriate_unit(
                    self.xlim[1] - self.xlim[0], self.pf)
                (unit_x, unit_y) = (unit, unit)
            else:
                (unit_x, unit_y) = self._axes_unit_names

            aspect = np.float64(self.pf.quan(1.0, unit_y)/self.pf.quan(1.0, unit_x))

            extentx = [(self.xlim[i] - xc).in_units(unit_x) for i in (0, 1)]
            extenty = [(self.ylim[i] - yc).in_units(unit_y) for i in (0, 1)]

            extent = extentx + extenty

            if f in self.plots.keys():
                zlim = (self.plots[f].zmin, self.plots[f].zmax)
            else:
                zlim = (None, None)

            image = self._frb[f]

            if image.max() == image.min():
              if self._field_transform[f] == log_transform:
                mylog.warning("Plot image for field %s has zero dynamic " \
                              "range. Min = Max = %d." % \
                              (f, image.max()))
                mylog.warning("Switching to linear colorbar scaling.")
                self._field_transform[f] = linear_transform

            fp = self._font_properties

            fig = None
            axes = None
            cax = None
            draw_colorbar = True
            draw_axes = True
            if f in self.plots:
                draw_colorbar = self.plots[f]._draw_colorbar
                draw_axes = self.plots[f]._draw_axes
                if self.plots[f].figure is not None:
                    fig = self.plots[f].figure
                    axes = self.plots[f].axes
                    cax = self.plots[f].cax

            self.plots[f] = WindowPlotMPL(
                image, self._field_transform[f].name,
                self._colormaps[f], extent, zlim,
                self.figure_size, fp.get_size(),
                aspect, fig, axes, cax)

            axes_unit_labels = ['', '']
            comoving = False
            hinv = False
            for i, un in enumerate((unit_x, unit_y)):
                # Use sympy to factor h out of the unit.  In this context 'un'
                # is a string, so we call the Unit constructor.
                expr = Unit(un, registry=self.pf.unit_registry).expr
                h_expr = Unit('h', registry=self.pf.unit_registry).expr
                # See http://docs.sympy.org/latest/modules/core.html#sympy.core.expr.Expr
                h_power = expr.as_coeff_exponent(h_expr)[1]
                # un is now the original unit, but with h factored out.
                un = str(expr*h_expr**(-1*h_power))
                if str(un).endswith('cm') and un != 'cm':
                    comoving = True
                    un = un[:-2]
                # no length units besides code_length end in h so this is safe
                if h_power == -1:
                    hinv = True
                elif h_power != 0:
                    # It doesn't make sense to scale a position by anything
                    # other than h**-1
                    raise RuntimeError
                if un in formatted_length_unit_names:
                    un = formatted_length_unit_names[un]
                if un not in ['1', 'u', 'unitary']:
                    if hinv:
                        un = un + '\,h^{-1}'
                    if comoving:
                        un = un + '\,(1+z)^{-1}'
                    axes_unit_labels[i] = '\/\/('+un+')'

            if self.oblique:
                labels = [r'$\rm{Image\/x'+axes_unit_labels[0]+'}$',
                          r'$\rm{Image\/y'+axes_unit_labels[1]+'}$']
            else:
                labels = [r'$\rm{'+axis_labels[axis_index][i]+
                          axes_unit_labels[i] + r'}$' for i in (0,1)]

            self.plots[f].axes.set_xlabel(labels[0],fontproperties=fp)
            self.plots[f].axes.set_ylabel(labels[1],fontproperties=fp)

            for label in (self.plots[f].axes.get_xticklabels() +
                          self.plots[f].axes.get_yticklabels() +
                          [self.plots[f].axes.xaxis.get_offset_text(),
                           self.plots[f].axes.yaxis.get_offset_text()]):
                label.set_fontproperties(fp)

            colorbar_label = image.info['label']

            # Determine the units of the data
            units = Unit(self._frb[f].units).latex_representation()

            if units is None or units == '':
                pass
            else:
                colorbar_label += r'$\/\/('+units+r')$'

            parser = MathTextParser('Agg')
            try:
                parser.parse(colorbar_label)
            except ParseFatalException, err:
                raise YTCannotParseUnitDisplayName(f, colorbar_label, str(err))

            self.plots[f].cb.set_label(colorbar_label, fontproperties=fp)

            for label in (self.plots[f].cb.ax.get_xticklabels() +
                          self.plots[f].cb.ax.get_yticklabels() +
                          [self.plots[f].cb.ax.axes.xaxis.get_offset_text(),
                           self.plots[f].cb.ax.axes.yaxis.get_offset_text()]):
                label.set_fontproperties(fp)

            self.run_callbacks(f)

            if draw_axes is False:
                self.plots[f]._toggle_axes(draw_axes)

            if draw_colorbar is False:
                self.plots[f]._toggle_colorbar(draw_colorbar)

            if self._font_color is not None:
                ax = self.plots[f].axes
                cbax = self.plots[f].cb.ax
                labels = \
                  ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels() + \
                  cbax.yaxis.get_ticklabels() + \
                  [ax.xaxis.label, ax.yaxis.label, cbax.yaxis.label]
                for label in labels:
                    label.set_color(self._font_color)

        self._plot_valid = True

    def setup_callbacks(self):
        for key in callback_registry:
            ignored = ['PlotCallback','CoordAxesCallback','LabelCallback',
                       'UnitBoundaryCallback']
            if self._plot_type.startswith('OffAxis'):
                ignored += ['HopCirclesCallback','HopParticleCallback',
                            'ParticleCallback','ClumpContourCallback',
                            'GridBoundaryCallback']
            if self._plot_type == 'OffAxisProjection':
                ignored += ['VelocityCallback','MagFieldCallback',
                            'QuiverCallback','CuttingQuiverCallback',
                            'StreamlineCallback']
            if key in ignored:
                continue
            cbname = callback_registry[key]._type_name
            CallbackMaker = callback_registry[key]
            callback = invalidate_plot(apply_callback(CallbackMaker))
            callback.__doc__ = CallbackMaker.__doc__
            self.__dict__['annotate_'+cbname] = types.MethodType(callback,self)

class AxisAlignedSlicePlot(PWViewerMPL):
    r"""Creates a slice plot from a parameter file

    Given a pf object, an axis to slice along, and a field name
    string, this will return a PWViewrMPL object containing
    the plot.

    The plot can be updated using one of the many helper functions
    defined in PlotWindow.

    Parameters
    ----------
    pf : `Dataset`
         This is the parameter file object corresponding to the
         simulation output to be plotted.
    axis : int or one of 'x', 'y', 'z'
         An int corresponding to the axis to slice along (0=x, 1=y, 2=z)
         or the axis name itself
    fields : string
         The name of the field(s) to be plotted.
    center : A sequence floats, a string, or a tuple.
         The coordinate of the center of the image. If set to 'c', 'center' or
         left blank, the plot is centered on the middle of the domain. If set to
         'max' or 'm', the center will be located at the maximum of the
         ('gas', 'density') field. Units can be specified by passing in center
         as a tuple containing a coordinate and string unit name or by passing
         in a YTArray.  If a list or unitless array is supplied, code units are
         assumed.
    width : tuple or a float.
         Width can have four different formats to support windows with variable
         x and y widths.  They are:

         ==================================     =======================
         format                                 example
         ==================================     =======================
         (float, string)                        (10,'kpc')
         ((float, string), (float, string))     ((10,'kpc'),(15,'kpc'))
         float                                  0.2
         (float, float)                         (0.2, 0.3)
         ==================================     =======================

         For example, (10, 'kpc') requests a plot window that is 10 kiloparsecs
         wide in the x and y directions, ((10,'kpc'),(15,'kpc')) requests a
         window that is 10 kiloparsecs wide along the x axis and 15
         kiloparsecs wide along the y axis.  In the other two examples, code
         units are assumed, for example (0.2, 0.3) requests a plot that has an
         x width of 0.2 and a y width of 0.3 in code units.  If units are
         provided the resulting plot axis labels will use the supplied units.
    axes_unit : A string
         The name of the unit for the tick labels on the x and y axes.
         Defaults to None, which automatically picks an appropriate unit.
         If axes_unit is '1', 'u', or 'unitary', it will not display the
         units, and only show the axes name.
    origin : string or length 1, 2, or 3 sequence of strings
         The location of the origin of the plot coordinate system.  This is
         represented by '-' separated string or a tuple of strings.  In the
         first index the y-location is given by 'lower', 'upper', or 'center'.
         The second index is the x-location, given as 'left', 'right', or
         'center'.  Finally, the whether the origin is applied in 'domain'
         space, plot 'window' space or 'native' simulation coordinate system
         is given. For example, both 'upper-right-domain' and ['upper',
         'right', 'domain'] both place the origin in the upper right hand
         corner of domain space. If x or y are not given, a value is inffered.
         For instance, 'left-domain' corresponds to the lower-left hand corner
         of the simulation domain, 'center-domain' corresponds to the center
         of the simulation domain, or 'center-window' for the center of the
         plot window. Further examples:

         ==================================     ============================
         format                                 example
         ==================================     ============================
         '{space}'                              'domain'
         '{xloc}-{space}'                       'left-window'
         '{yloc}-{space}'                       'upper-domain'
         '{yloc}-{xloc}-{space}'                'lower-right-window'
         ('{space}',)                           ('window',)
         ('{xloc}', '{space}')                  ('right', 'domain')
         ('{yloc}', '{space}')                  ('lower', 'window')
         ('{yloc}', '{xloc}', '{space}')        ('lower', 'right', 'window')
         ==================================     ============================
    fontsize : integer
         The size of the fonts for the axis, colorbar, and tick labels.
    field_parameters : dictionary
         A dictionary of field parameters than can be accessed by derived
         fields.

    Examples
    --------

    This will save an image the the file 'sliceplot_Density

    >>> pf = load('galaxy0030/galaxy0030')
    >>> p = SlicePlot(pf,2,'Density','c',(20,'kpc'))
    >>> p.save('sliceplot')

    """
    _plot_type = 'Slice'
    _frb_generator = FixedResolutionBuffer

    def __init__(self, pf, axis, fields, center='c', width=None, axes_unit=None,
                 origin='center-window', fontsize=18, field_parameters=None):
        # this will handle time series data and controllers
        ts = self._initialize_dataset(pf)
        self.ts = ts
        pf = self.pf = ts[0]
        axis = fix_axis(axis)
        (bounds, center) = get_window_parameters(axis, center, width, pf)
        if field_parameters is None: field_parameters = {}
        slc = pf.slice(axis, center[axis],
            field_parameters = field_parameters, center=center)
        slc.get_data(fields)
        PWViewerMPL.__init__(self, slc, bounds, origin=origin,
                             fontsize=fontsize, fields=fields)
        if axes_unit is None:
            axes_unit = get_axes_unit(width, pf)
        self.set_axes_unit(axes_unit)

class ProjectionPlot(PWViewerMPL):
    r"""Creates a projection plot from a parameter file

    Given a pf object, an axis to project along, and a field name
    string, this will return a PWViewrMPL object containing
    the plot.

    The plot can be updated using one of the many helper functions
    defined in PlotWindow.

    Parameters
    ----------
    pf : `Dataset`
        This is the parameter file object corresponding to the
        simulation output to be plotted.
    axis : int or one of 'x', 'y', 'z'
         An int corresponding to the axis to slice along (0=x, 1=y, 2=z)
         or the axis name itself
    fields : string
         The name of the field(s) to be plotted.
    center : A sequence floats, a string, or a tuple.
         The coordinate of the center of the image. If set to 'c', 'center' or
         left blank, the plot is centered on the middle of the domain. If set to
         'max' or 'm', the center will be located at the maximum of the
         ('gas', 'density') field. Units can be specified by passing in center
         as a tuple containing a coordinate and string unit name or by passing
         in a YTArray.  If a list or unitless array is supplied, code units are
         assumed.
    width : tuple or a float.
         Width can have four different formats to support windows with variable
         x and y widths.  They are:

         ==================================     =======================
         format                                 example
         ==================================     =======================
         (float, string)                        (10,'kpc')
         ((float, string), (float, string))     ((10,'kpc'),(15,'kpc'))
         float                                  0.2
         (float, float)                         (0.2, 0.3)
         ==================================     =======================

         For example, (10, 'kpc') requests a plot window that is 10 kiloparsecs
         wide in the x and y directions, ((10,'kpc'),(15,'kpc')) requests a
         window that is 10 kiloparsecs wide along the x axis and 15
         kiloparsecs wide along the y axis.  In the other two examples, code
         units are assumed, for example (0.2, 0.3) requests a plot that has an
         x width of 0.2 and a y width of 0.3 in code units.  If units are
         provided the resulting plot axis labels will use the supplied units.
    axes_unit : A string
         The name of the unit for the tick labels on the x and y axes.
         Defaults to None, which automatically picks an appropriate unit.
         If axes_unit is '1', 'u', or 'unitary', it will not display the
         units, and only show the axes name.
    origin : string or length 1, 2, or 3 sequence of strings
         The location of the origin of the plot coordinate system.  This is
         represented by '-' separated string or a tuple of strings.  In the
         first index the y-location is given by 'lower', 'upper', or 'center'.
         The second index is the x-location, given as 'left', 'right', or
         'center'.  Finally, the whether the origin is applied in 'domain'
         space, plot 'window' space or 'native' simulation coordinate system
         is given. For example, both 'upper-right-domain' and ['upper',
         'right', 'domain'] both place the origin in the upper right hand
         corner of domain space. If x or y are not given, a value is inffered.
         For instance, 'left-domain' corresponds to the lower-left hand corner
         of the simulation domain, 'center-domain' corresponds to the center
         of the simulation domain, or 'center-window' for the center of the
         plot window. Further examples:

         ==================================     ============================
         format                                 example
         ==================================     ============================
         '{space}'                              'domain'
         '{xloc}-{space}'                       'left-window'
         '{yloc}-{space}'                       'upper-domain'
         '{yloc}-{xloc}-{space}'                'lower-right-window'
         ('{space}',)                           ('window',)
         ('{xloc}', '{space}')                  ('right', 'domain')
         ('{yloc}', '{space}')                  ('lower', 'window')
         ('{yloc}', '{xloc}', '{space}')        ('lower', 'right', 'window')
         ==================================     ============================

    data_source : AMR3DData Object
         Object to be used for data selection.  Defaults to a region covering
         the entire simulation.
    weight_field : string
         The name of the weighting field.  Set to None for no weight.
    max_level: int
         The maximum level to project to.
    fontsize : integer
         The size of the fonts for the axis, colorbar, and tick labels.
    field_parameters : dictionary
         A dictionary of field parameters than can be accessed by derived
         fields.

    Examples
    --------

    This is a very simple way of creating a projection plot.

    >>> pf = load('galaxy0030/galaxy0030')
    >>> p = ProjectionPlot(pf,2,'Density','c',(20,'kpc'))
    >>> p.save('sliceplot')

    """
    _plot_type = 'Projection'
    _frb_generator = FixedResolutionBuffer

    def __init__(self, pf, axis, fields, center='c', width=None, axes_unit=None,
                 weight_field=None, max_level=None, origin='center-window',
                 fontsize=18, field_parameters=None, data_source=None,
                 proj_style = "integrate"):
        ts = self._initialize_dataset(pf)
        self.ts = ts
        pf = self.pf = ts[0]
        axis = fix_axis(axis)
        (bounds, center) = get_window_parameters(axis, center, width, pf)
        if field_parameters is None: field_parameters = {}
        proj = pf.proj(fields, axis, weight_field=weight_field,
                         center=center, data_source=data_source,
                         field_parameters = field_parameters, style = proj_style)
        PWViewerMPL.__init__(self, proj, bounds, fields=fields, origin=origin,
                             fontsize=fontsize)
        if axes_unit is None:
            axes_unit = get_axes_unit(width, pf)
        self.set_axes_unit(axes_unit)

class OffAxisSlicePlot(PWViewerMPL):
    r"""Creates an off axis slice plot from a parameter file

    Given a pf object, a normal vector defining a slicing plane, and
    a field name string, this will return a PWViewrMPL object
    containing the plot.

    The plot can be updated using one of the many helper functions
    defined in PlotWindow.

    Parameters
    ----------
    pf : :class:`yt.data_objects.api.Dataset`
         This is the parameter file object corresponding to the
         simulation output to be plotted.
    normal : a sequence of floats
         The vector normal to the slicing plane.
    fields : string
         The name of the field(s) to be plotted.
    center : A sequence floats, a string, or a tuple.
         The coordinate of the center of the image. If set to 'c', 'center' or
         left blank, the plot is centered on the middle of the domain. If set to
         'max' or 'm', the center will be located at the maximum of the
         ('gas', 'density') field. Units can be specified by passing in center
         as a tuple containing a coordinate and string unit name or by passing
         in a YTArray.  If a list or unitless array is supplied, code units are
         assumed.
    width : tuple or a float.
         Width can have four different formats to support windows with variable
         x and y widths.  They are:

         ==================================     =======================
         format                                 example
         ==================================     =======================
         (float, string)                        (10,'kpc')
         ((float, string), (float, string))     ((10,'kpc'),(15,'kpc'))
         float                                  0.2
         (float, float)                         (0.2, 0.3)
         ==================================     =======================

         For example, (10, 'kpc') requests a plot window that is 10 kiloparsecs
         wide in the x and y directions, ((10,'kpc'),(15,'kpc')) requests a
         window that is 10 kiloparsecs wide along the x axis and 15
         kiloparsecs wide along the y axis.  In the other two examples, code
         units are assumed, for example (0.2, 0.3) requests a plot that has an
         x width of 0.2 and a y width of 0.3 in code units.  If units are
         provided the resulting plot axis labels will use the supplied units.
    axes_unit : A string
         The name of the unit for the tick labels on the x and y axes.
         Defaults to None, which automatically picks an appropriate unit.
         If axes_unit is '1', 'u', or 'unitary', it will not display the
         units, and only show the axes name.
    north-vector : a sequence of floats
         A vector defining the 'up' direction in the plot.  This
         option sets the orientation of the slicing plane.  If not
         set, an arbitrary grid-aligned north-vector is chosen.
    fontsize : integer
         The size of the fonts for the axis, colorbar, and tick labels.
    field_parameters : dictionary
         A dictionary of field parameters than can be accessed by derived
         fields.
    """

    _plot_type = 'OffAxisSlice'
    _frb_generator = ObliqueFixedResolutionBuffer

    def __init__(self, pf, normal, fields, center='c', width=None,
                 axes_unit=None, north_vector=None, fontsize=18,
                 field_parameters=None):
        (bounds, center_rot) = get_oblique_window_parameters(normal,center,width,pf)
        if field_parameters is None: field_parameters = {}
        cutting = pf.cutting(normal, center, north_vector = north_vector,
                              field_parameters = field_parameters)
        cutting.get_data(fields)
        # Hard-coding the origin keyword since the other two options
        # aren't well-defined for off-axis data objects
        PWViewerMPL.__init__(self, cutting, bounds, fields=fields,
                             origin='center-window',periodic=False,
                             oblique=True, fontsize=fontsize)
        if axes_unit is None:
            axes_unit = get_axes_unit(width, pf)
        self.set_axes_unit(axes_unit)

class OffAxisProjectionDummyDataSource(object):
    _type_name = 'proj'
    proj_style = 'integrate'
    _key_fields = []
    def __init__(self, center, pf, normal_vector, width, fields,
                 interpolated, resolution = (800,800), weight=None,
                 volume=None, no_ghost=False, le=None, re=None,
                 north_vector=None):
        self.center = center
        self.pf = pf
        self.axis = 4 # always true for oblique data objects
        self.normal_vector = normal_vector
        self.width = width
        self.dd = pf.h.all_data()
        fields = self.dd._determine_fields(fields)
        self.fields = fields
        self.interpolated = interpolated
        self.resolution = resolution
        self.weight_field = weight
        self.volume = volume
        self.no_ghost = no_ghost
        self.le = le
        self.re = re
        self.north_vector = north_vector

    def _determine_fields(self, *args):
        return self.dd._determine_fields(*args)

class OffAxisProjectionPlot(PWViewerMPL):
    r"""Creates an off axis projection plot from a parameter file

    Given a pf object, a normal vector to project along, and
    a field name string, this will return a PWViewrMPL object
    containing the plot.

    The plot can be updated using one of the many helper functions
    defined in PlotWindow.

    Parameters
    ----------
    pf : :class:`yt.data_objects.api.Dataset`
        This is the parameter file object corresponding to the
        simulation output to be plotted.
    normal : a sequence of floats
        The vector normal to the slicing plane.
    fields : string
        The name of the field(s) to be plotted.
    center : A sequence floats, a string, or a tuple.
         The coordinate of the center of the image. If set to 'c', 'center' or
         left blank, the plot is centered on the middle of the domain. If set to
         'max' or 'm', the center will be located at the maximum of the
         ('gas', 'density') field. Units can be specified by passing in center
         as a tuple containing a coordinate and string unit name or by passing
         in a YTArray.  If a list or unitless array is supplied, code units are
         assumed.
    width : tuple or a float.
         Width can have four different formats to support windows with variable
         x and y widths.  They are:

         ==================================     =======================
         format                                 example
         ==================================     =======================
         (float, string)                        (10,'kpc')
         ((float, string), (float, string))     ((10,'kpc'),(15,'kpc'))
         float                                  0.2
         (float, float)                         (0.2, 0.3)
         ==================================     =======================

         For example, (10, 'kpc') requests a plot window that is 10 kiloparsecs
         wide in the x and y directions, ((10,'kpc'),(15,'kpc')) requests a
         window that is 10 kiloparsecs wide along the x axis and 15
         kiloparsecs wide along the y axis.  In the other two examples, code
         units are assumed, for example (0.2, 0.3) requests a plot that has an
         x width of 0.2 and a y width of 0.3 in code units.  If units are
         provided the resulting plot axis labels will use the supplied units.
    depth : A tuple or a float
         A tuple containing the depth to project through and the string
         key of the unit: (width, 'unit').  If set to a float, code units
         are assumed
    weight_field : string
         The name of the weighting field.  Set to None for no weight.
    max_level: int
         The maximum level to project to.
    axes_unit : A string
         The name of the unit for the tick labels on the x and y axes.
         Defaults to None, which automatically picks an appropriate unit.
         If axes_unit is '1', 'u', or 'unitary', it will not display the
         units, and only show the axes name.
    north-vector : a sequence of floats
         A vector defining the 'up' direction in the plot.  This
         option sets the orientation of the slicing plane.  If not
         set, an arbitrary grid-aligned north-vector is chosen.
    fontsize : integer
         The size of the fonts for the axis, colorbar, and tick labels.

    """
    _plot_type = 'OffAxisProjection'
    _frb_generator = OffAxisProjectionFixedResolutionBuffer

    def __init__(self, pf, normal, fields, center='c', width=None,
                 depth=(1, '1'), axes_unit=None, weight_field=None,
                 max_level=None, north_vector=None, volume=None, no_ghost=False,
                 le=None, re=None, interpolated=False, fontsize=18):
        (bounds, center_rot) = \
          get_oblique_window_parameters(normal,center,width,pf,depth=depth)
        fields = ensure_list(fields)[:]
        oap_width = pf.arr((bounds[1] - bounds[0],
                            bounds[3] - bounds[2],
                            bounds[5] - bounds[4]))
        OffAxisProj = OffAxisProjectionDummyDataSource(
            center_rot, pf, normal, oap_width, fields, interpolated,
            weight=weight_field,  volume=volume, no_ghost=no_ghost,
            le=le, re=re, north_vector=north_vector)
        # Hard-coding the origin keyword since the other two options
        # aren't well-defined for off-axis data objects
        PWViewerMPL.__init__(
            self, OffAxisProj, bounds, fields=fields, origin='center-window',
            periodic=False, oblique=True, fontsize=fontsize)
        if axes_unit is None:
            axes_unit = get_axes_unit(width, pf)
        self.set_axes_unit(axes_unit)

    def _recreate_frb(self):
        if self._frb is not None:
            raise NotImplementedError
        super(OffAxisProjectionPlot, self)._recreate_frb()

_metadata_template = """
%(pf)s<br>
<br>
Field of View:  %(x_width)0.3f %(axes_unit_names)s<br>
Minimum Value:  %(mi)0.3e %(colorbar_unit)s<br>
Maximum Value:  %(ma)0.3e %(colorbar_unit)s<br>
Central Point:  (data coords)<br>
&nbsp;&nbsp;&nbsp;%(xc)0.14f<br>
&nbsp;&nbsp;&nbsp;%(yc)0.14f<br>
&nbsp;&nbsp;&nbsp;%(zc)0.14f
"""

class PWViewerExtJS(PlotWindow):
    """A viewer for the web interface.

    """
    _ext_widget_id = None
    _current_field = None
    _widget_name = "plot_window"
    _frb_generator = FixedResolutionBuffer
    _contour_info = None
    _vector_info = None

    def _setup_plots(self):
        from yt.gui.reason.bottle_mods import PayloadHandler
        ph = PayloadHandler()
        if self._current_field is not None \
           and self._ext_widget_id is not None:
            fields = [self._current_field]
            addl_keys = {'type': 'widget_payload',
                         'widget_id': self._ext_widget_id}
        else:
            fields = self._frb.data.keys()
            addl_keys = {}
        if self._colorbar_valid == False:
            addl_keys['colorbar_image'] = self._get_cbar_image()
            self._colorbar_valid = True
        min_zoom = 200*self.pf.index.get_smallest_dx() * self.pf['unitary']
        for field in fields:
            to_plot = apply_colormap(self._frb[field],
                func = self._field_transform[field],
                cmap_name = self._colormaps[field])
            pngs = self._apply_modifications(to_plot)
            img_data = base64.b64encode(pngs)
            # We scale the width between 200*min_dx and 1.0
            x_width = self.xlim[1] - self.xlim[0]
            zoom_fac = np.log10(x_width*self.pf['unitary'])/np.log10(min_zoom)
            zoom_fac = 100.0*max(0.0, zoom_fac)
            ticks = self.get_ticks(field)
            payload = {'type':'png_string',
                       'image_data':img_data,
                       'metadata_string': self.get_metadata(field),
                       'zoom': zoom_fac,
                       'ticks': ticks}
            payload.update(addl_keys)
            ph.add_payload(payload)

    def _apply_modifications(self, img):
        if self._contour_info is None and self._vector_info is None:
            return write_png_to_string(img)
        from matplotlib.figure import Figure

        vi, vj, vn = img.shape

        # Now we need to get our field values
        fig = Figure((vi/100.0, vj/100.0), dpi = 100)
        fig.figimage(img)
        # Add our contour
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
        ax.patch.set_alpha(0.0)

        # Now apply our modifications
        self._apply_contours(ax, vi, vj)
        self._apply_vectors(ax, vi, vj)

        canvas = FigureCanvasAgg(fig)
        f = cStringIO.StringIO()
        canvas.print_figure(f)
        f.seek(0)
        img = f.read()
        return img

    def _apply_contours(self, ax, vi, vj):
        if self._contour_info is None: return
        plot_args = {}
        field, number, colors, logit = self._contour_info
        if colors is not None: plot_args['colors'] = colors

        raw_data = self._frb.data_source
        b = self._frb.bounds
        xi, yi = np.mgrid[b[0]:b[1]:(vi / 8) * 1j,
                          b[2]:b[3]:(vj / 8) * 1j]
        x = raw_data['px']
        y = raw_data['py']
        z = raw_data[field]
        if logit: z = np.log10(z)
        fvals = triang(x,y).nn_interpolator(z)(xi,yi).transpose()[::-1,:]

        ax.contour(fvals, number, colors='w')

    def _apply_vectors(self, ax, vi, vj):
        if self._vector_info is None: return
        skip, scale = self._vector_info

        nx = self._frb.buff_size[0]/skip
        ny = self._frb.buff_size[1]/skip
        new_frb = FixedResolutionBuffer(self._frb.data_source,
                        self._frb.bounds, (nx,ny))

        axis = self._frb.data_source.axis
        fx = "%s-velocity" % (axis_names[x_dict[axis]])
        fy = "%s-velocity" % (axis_names[y_dict[axis]])
        px = new_frb[fx][::-1,:]
        py = new_frb[fy][::-1,:]
        x = np.mgrid[0:vi-1:ny*1j]
        y = np.mgrid[0:vj-1:nx*1j]
        # Always normalize, then we scale
        nn = ((px**2.0 + py**2.0)**0.5).max()
        px /= nn
        py /= nn
        print scale, px.min(), px.max(), py.min(), py.max()
        ax.quiver(x, y, px, py, scale=float(vi)/skip)

    def get_ticks(self, field, height = 400):
        # This will eventually change to work with non-logged fields
        ticks = []
        transform = self._field_transform[field]
        mi, ma = self._frb[field].min(), self._frb[field].max()
        tick_locs = transform.ticks(mi, ma)
        mi, ma = transform((mi, ma))
        for v1,v2 in zip(tick_locs, transform(tick_locs)):
            if v2 < mi or v2 > ma: continue
            p = height - height * (v2 - mi)/(ma - mi)
            ticks.append((p,v1,v2))
        return ticks

    def _get_cbar_image(self, height = 400, width = 40, field = None):
        if field is None: field = self._current_field
        cmap_name = self._colormaps[field]
        vals = np.mgrid[1:0:height * 1j] * np.ones(width)[:,None]
        vals = vals.transpose()
        to_plot = apply_colormap(vals, cmap_name = cmap_name)
        pngs = write_png_to_string(to_plot)
        img_data = base64.b64encode(pngs)
        return img_data

    # This calls an invalidation routine from within
    def scroll_zoom(self, value):
        # We accept value from 0..100, and assume it has been set from the
        # scroll bar.  In that case, we undo the logic for calcualting
        # 'zoom_fac' from above.
        min_val = 200*self.pf.index.get_smallest_dx()
        unit = self.pf['unitary']
        width = (min_val**(value/100.0))/unit
        self.set_width(width)

    def image_recenter(self, img_x, img_y, img_size_x, img_size_y):
        dx = (self.xlim[1] - self.xlim[0]) / img_size_x
        dy = (self.ylim[1] - self.ylim[0]) / img_size_y
        new_x = img_x * dx + self.xlim[0]
        new_y = img_y * dy + self.ylim[0]
        print img_x, img_y, dx, dy, new_x, new_y
        self.set_center((new_x, new_y))

    def get_field_units(self, field, strip_mathml = True):
        ds = self._frb.data_source
        pf = self.pf
        field = self._check_field(field)
        finfo = self.data_source.pf._get_field_info(*field)
        if ds._type_name in ("slice", "cutting"):
            units = finfo.get_units()
        elif ds._type_name == "proj" and (ds.weight_field is not None or 
                                        ds.proj_style == "mip"):
            units = finfo.get_units()
        elif ds._type_name == "proj":
            units = finfo.get_projected_units()
        else:
            units = ""
        if strip_mathml:
            units = units.replace(r"\rm{", "").replace("}","")
        return units

    def get_metadata(self, field, strip_mathml = True, return_string = True):
        fval = self._frb[field]
        mi = fval.min()
        ma = fval.max()
        x_width = self.xlim[1] - self.xlim[0]
        y_width = self.ylim[1] - self.ylim[0]
        if self._axes_unit_names is None:
            unit = get_smallest_appropriate_unit(x_width, self.pf)
            unit = (unit, unit)
        else:
            unit = self._axes_unit_names
        units = self.get_field_units(field, strip_mathml)
        center = getattr(self._frb.data_source, "center", None)
        if center is None or self._frb.axis == 4:
            xc, yc, zc = -999, -999, -999
        else:
            center[x_dict[self._frb.axis]] = 0.5 * (
                self.xlim[0] + self.xlim[1])
            center[y_dict[self._frb.axis]] = 0.5 * (
                self.ylim[0] + self.ylim[1])
            xc, yc, zc = center
        if return_string:
            md = _metadata_template % dict(
                pf = self.pf,
                x_width = x_width*self.pf[unit[0]],
                y_width = y_width*self.pf[unit[1]],
                axes_unit_names = unit[0], colorbar_unit = units,
                mi = mi, ma = ma, xc = xc, yc = yc, zc = zc)
        else:
            md = dict(pf = self.pf,
                      x_width = x_width*self.pf[unit[0]],
                      y_width = y_width*self.pf[unit[1]],
                      axes_unit_names = unit, colorbar_unit = units,
                      mi = mi, ma = ma, xc = xc, yc = yc, zc = zc)
        return md

    @invalidate_plot
    def set_contour_info(self, field_name, n_cont = 8, colors = None,
                         logit = True):
        if field_name == "None" or n_cont == 0:
            self._contour_info = None
            return
        self._contour_info = (field_name, n_cont, colors, logit)

    @invalidate_plot
    def set_vector_info(self, skip, scale = 1):
        self._vector_info = (skip, scale)

    @invalidate_data
    def set_current_field(self, field):
        field = self._check_field(field)
        self._current_field = field
        self._frb[field]
        finfo = self.data_source.pf._get_field_info(*field)
        if finfo.take_log:
            self._field_transform[field] = log_transform
        else:
            self._field_transform[field] = linear_transform


class WindowPlotMPL(ImagePlotMPL):
    def __init__(self, data, cbname, cmap, extent, zlim, figure_size, fontsize,
                 unit_aspect, figure, axes, cax):
        self._draw_colorbar = True
        self._draw_axes = True
        self._fontsize = fontsize
        self._figure_size = figure_size

        # Compute layout
        fontscale = float(fontsize) / 18.0
        if fontscale < 1.0:
            fontscale = np.sqrt(fontscale)

        self._cb_size = 0.0375*figure_size
        self._ax_text_size = [0.9*fontscale, 0.7*fontscale]
        self._top_buff_size = 0.30*fontscale
        self._aspect = ((extent[1] - extent[0])/(extent[3] - extent[2]))

        size, axrect, caxrect = self._get_best_layout()

        super(WindowPlotMPL, self).__init__(
            size, axrect, caxrect, zlim, figure, axes, cax)

        self._init_image(data, cbname, cmap, extent, unit_aspect)

        self.image.axes.ticklabel_format(scilimits=(-2, 3))
        if cbname == 'linear':
            self.cb.formatter.set_scientific(True)
            self.cb.formatter.set_powerlimits((-2, 3))
            self.cb.update_ticks()


def SlicePlot(pf, normal=None, fields=None, axis=None, *args, **kwargs):
    r"""
    A factory function for
    :class:`yt.visualization.plot_window.AxisAlignedSlicePlot`
    and :class:`yt.visualization.plot_window.OffAxisSlicePlot` objects.  This
    essentially allows for a single entry point to both types of slice plots,
    the distinction being determined by the specified normal vector to the
    slice.

        The returned plot object can be updated using one of the many helper
    functions defined in PlotWindow.

    Parameters
    ----------
    pf : :class:`yt.data_objects.api.Dataset`
        This is the parameter file object corresponding to the
        simulation output to be plotted.
    normal : int or one of 'x', 'y', 'z', or sequence of floats
        This specifies the normal vector to the slice.  If given as an integer
        or a coordinate string (0=x, 1=y, 2=z), this function will return an
        :class:`AxisAlignedSlicePlot` object.  If given as a sequence of floats,
        this is interpretted as an off-axis vector and an
        :class:`OffAxisSlicePlot` object is returned.
    fields : string
         The name of the field(s) to be plotted.
    axis : int or one of 'x', 'y', 'z'
         An int corresponding to the axis to slice along (0=x, 1=y, 2=z)
         or the axis name itself.  If specified, this will replace normal.
         
    The following are nominally keyword arguments passed onto the respective
    slice plot objects generated by this function.

    center : A sequence floats, a string, or a tuple.
         The coordinate of the center of the image. If set to 'c', 'center' or
         left blank, the plot is centered on the middle of the domain. If set to
         'max' or 'm', the center will be located at the maximum of the
         ('gas', 'density') field. Units can be specified by passing in center
         as a tuple containing a coordinate and string unit name or by passing
         in a YTArray.  If a list or unitless array is supplied, code units are
         assumed.
    width : tuple or a float.
         Width can have four different formats to support windows with variable
         x and y widths.  They are:

         ==================================     =======================
         format                                 example
         ==================================     =======================
         (float, string)                        (10,'kpc')
         ((float, string), (float, string))     ((10,'kpc'),(15,'kpc'))
         float                                  0.2
         (float, float)                         (0.2, 0.3)
         ==================================     =======================

         For example, (10, 'kpc') requests a plot window that is 10 kiloparsecs
         wide in the x and y directions, ((10,'kpc'),(15,'kpc')) requests a
         window that is 10 kiloparsecs wide along the x axis and 15
         kiloparsecs wide along the y axis.  In the other two examples, code
         units are assumed, for example (0.2, 0.3) requests a plot that has an
         x width of 0.2 and a y width of 0.3 in code units.  If units are
         provided the resulting plot axis labels will use the supplied units.
    axes_unit : A string
         The name of the unit for the tick labels on the x and y axes.
         Defaults to None, which automatically picks an appropriate unit.
         If axes_unit is '1', 'u', or 'unitary', it will not display the
         units, and only show the axes name.
    origin : string or length 1, 2, or 3 sequence of strings
         The location of the origin of the plot coordinate system for
         `AxisAlignedSlicePlot` objects; for `OffAxisSlicePlot` objects,
         this parameter is discarded.  This is represented by '-' separated
         string or a tuple of strings.  In the first index the y-location is
         given by 'lower', 'upper', or 'center'.  The second index is the
         x-location, given as 'left', 'right', or 'center'.  Finally, the
         whether the origin is applied in 'domain' space, plot 'window' space
         or 'native' simulation coordinate system is given. For example, both
         'upper-right-domain' and ['upper', 'right', 'domain'] both place the
         origin in the upper right hand corner of domain space. If x or y are
         not given, a value is inffered.  For instance, 'left-domain'
         corresponds to the lower-left hand corner of the simulation domain,
         'center-domain' corresponds to the center of the simulation domain,
         or 'center-window' for the center of the plot window. Further
         examples:

         ==================================     ============================
         format                                 example
         ==================================     ============================
         '{space}'                              'domain'
         '{xloc}-{space}'                       'left-window'
         '{yloc}-{space}'                       'upper-domain'
         '{yloc}-{xloc}-{space}'                'lower-right-window'
         ('{space}',)                           ('window',)
         ('{xloc}', '{space}')                  ('right', 'domain')
         ('{yloc}', '{space}')                  ('lower', 'window')
         ('{yloc}', '{xloc}', '{space}')        ('lower', 'right', 'window')
         ==================================     ============================
    north-vector : a sequence of floats
        A vector defining the 'up' direction in the `OffAxisSlicePlot`; not
        used in `AxisAlignedSlicePlot`.  This option sets the orientation of the
        slicing plane.  If not set, an arbitrary grid-aligned north-vector is
        chosen.
    fontsize : integer
         The size of the fonts for the axis, colorbar, and tick labels.
    field_parameters : dictionary
         A dictionary of field parameters than can be accessed by derived
         fields.

    Raises
    ------
    AssertionError
        If a proper normal axis is not specified via the normal or axis
        keywords, and/or if a field to plot is not specified.

    Examples
    --------

    >>> slc = SlicePlot(pf, "x", "Density", center=[0.2,0.3,0.4])
    >>> slc = SlicePlot(pf, 2, "Temperature")
    >>> slc = SlicePlot(pf, [0.4,0.2,-0.1], "Pressure",
                        north_vector=[0.2,-0.3,0.1])

    """
    # Make sure we are passed a normal
    # we check the axis keyword for backwards compatability
    if normal is None: normal = axis
    if normal is None:
        raise AssertionError("Must pass a normal vector to the slice!")

    # to keep positional ordering we had to make fields a keyword; make sure
    # it is present
    if fields is None:
        raise AssertionError("Must pass field(s) to plot!")

    # use an AxisAlignedSlicePlot where possible, e.g.:
    # maybe someone passed normal=[0,0,0.2] when they should have just used "z"
    if iterable(normal) and not isinstance(normal, basestring):
        if np.count_nonzero(normal) == 1:
            normal = ("x","y","z")[np.nonzero(normal)[0][0]]
        else:
            normal = np.array(normal)
            np.divide(normal, np.dot(normal,normal), normal)
        
    # by now the normal should be properly set to get either a On/Off Axis plot
    if iterable(normal) and not isinstance(normal, basestring):
        # OffAxisSlicePlot has hardcoded origin; remove it if in kwargs
        if 'origin' in kwargs: 
            msg = "Ignoring 'origin' keyword as it is ill-defined for " \
                  "an OffAxisSlicePlot object."
            mylog.warn(msg)
            del kwargs['origin']
        
        return OffAxisSlicePlot(pf, normal, fields, *args, **kwargs)
    else:
        # north_vector not used in AxisAlignedSlicePlots; remove it if in kwargs
        if 'north_vector' in kwargs: 
            msg = "Ignoring 'north_vector' keyword as it is ill-defined for " \
                  "an AxisAlignedSlicePlot object."
            mylog.warn(msg)
            del kwargs['north_vector']
        
        return AxisAlignedSlicePlot(pf, normal, fields, *args, **kwargs)
