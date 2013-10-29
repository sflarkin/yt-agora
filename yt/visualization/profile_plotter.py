"""
This is a simple mechanism for interfacing with Profile and Phase plots



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------


import base64
import types

from functools import wraps
from itertools import izip, repeat
import numpy as np
import cStringIO

from .image_writer import \
    write_image, apply_colormap
from yt.data_objects.profiles import \
     create_profile
from yt.utilities.lib import \
    write_png_to_string
from yt.data_objects.profiles import \
    BinnedProfile1D, \
    BinnedProfile2D
from .tick_locators import LogLocator, LinearLocator
from yt.utilities.logger import ytLogger as mylog
import _mpl_imports as mpl
from yt.funcs import \
     ensure_list, \
     get_image_suffix

def get_canvas(name):
    suffix = get_image_suffix(name)
    
    if suffix == '':
        suffix = '.png'
    if suffix == ".png":
        canvas_cls = mpl.FigureCanvasAgg
    elif suffix == ".pdf":
        canvas_cls = mpl.FigureCanvasPdf
    elif suffix in (".eps", ".ps"):
        canvas_cls = mpl.FigureCanvasPS
    else:
        mylog.warning("Unknown suffix %s, defaulting to Agg", suffix)
        canvas_cls = FigureCanvasAgg
    return canvas_cls

def invalidate_plot(f):
    @wraps(f)
    def newfunc(*args, **kwargs):
        rv = f(*args, **kwargs)
        args[0]._plot_valid = False
        return rv
    return newfunc

class FigureContainer(dict):
    def __init__(self):
        super(dict, self).__init__()

    def __missing__(self, key):
        figure = mpl.matplotlib.figure.Figure((10, 8))
        self[key] = figure
        return self[key]

class AxesContainer(dict):
    def __init__(self, fig_container):
        self.fig_container = fig_container
        super(dict, self).__init__()

    def __missing__(self, key):
        figure = self.fig_container[key]
        self[key] = figure.add_subplot(111)
        return self[key]

class ProfilePlot(object):
    r"""
    Create a 1d profile plot from a data source or from a list 
    of profile objects.

    Given a data object (all_data, region, sphere, etc.), an x field, 
    and a y field (or fields), this will create a one-dimensional profile 
    of the average (or total) value of the y field in bins of the x field.

    This can be used to create profiles from given fields or to plot 
    multiple profiles created from 
    `yt.data_objects.profiles.create_profile`.
    
    Parameters
    ----------
    data_source : AMR4DData Object
        The data object to be profiled, such as all_data, region, or 
        sphere.
    x_field : str
        The binning field for the profile.
    y_fields : str or list
        The field or fields to be profiled.
    weight_field : str
        The weight field for calculating weighted averages.  If None, 
        the profile values are the sum of the field values within the bin.
        Otherwise, the values are a weighted average.
        Default : "CellMass".
    n_bins : int
        The number of bins in the profile.
        Default: 64.
    accumulation : bool
        If True, the profile values for a bin N are the cumulative sum of 
        all the values from bin 0 to N.
        Default: False.
    label : str or list of strings
        If a string, the label to be put on the line plotted.  If a list, 
        this should be a list of labels for each profile to be overplotted.
        Default: None.
    plot_spec : dict or list of dicts
        A dictionary or list of dictionaries containing plot keyword 
        arguments.  For example, dict(color="red", linestyle=":").
        Default: None.
    profiles : list of profiles
        If not None, a list of profile objects created with 
        `yt.data_objects.profiles.create_profile`.
        Default: None.

    Examples
    --------

    This creates profiles of a single dataset.

    >>> pf = load("DD0046/DD0046")
    >>> ad = pf.h.all_data()
    >>> plot = ProfilePlot(ad, "Density", ["Temperature", "x-velocity"], 
                           weight_field="CellMass",
                           plot_spec=dict(color='red', linestyle="--"))
    >>> plot.save()

    This creates profiles from a time series object.
    
    >>> es = simulation("AMRCosmology.enzo", "Enzo")
    >>> es.get_time_series()

    >>> profiles = []
    >>> labels = []
    >>> plot_specs = []
    >>> for pf in es[-4:]:
    ...     ad = pf.h.all_data()
    ...     profiles.append(create_profile(ad, ["Density"],
    ...                                    fields=["Temperature",
    ...                                            "x-velocity"]))
    ...     labels.append(pf.current_redshift)
    ...     plot_specs.append(dict(linestyle="--", alpha=0.7))
    >>>
    >>> plot = ProfilePlot.from_profiles(profiles, labels=labels,
    ...                                  plot_specs=plot_specs)
    >>> plot.save()

    Use plot_line_property to change line properties of one or all profiles.
    
    """
    plot_spec = None
    x_log = None
    y_log = None
    x_title = None
    y_title = None

    _plot_valid = False

    def __init__(self, data_source, x_field, y_fields, 
                 weight_field="CellMass", n_bins=64, accumulation=False,
                 label=None, plot_spec=None, profiles=None):
        self.y_log = {}
        self.y_title = {}
        if profiles is None:
            self.profiles = [create_profile(data_source, [x_field], n_bins,
                                            fields=ensure_list(y_fields),
                                            weight_field=weight_field)]
        else:
            self.profiles = ensure_list(profiles)

        if accumulation:
            for profile in self.profiles:
                for field in profile.field_data:
                    profile.field_data[field] = \
                      profile.field_data[field].cumsum()
        
        self.label = label
        if not isinstance(self.label, list):
            self.label = [self.label] * len(self.profiles)

        self.plot_spec = plot_spec
        if self.plot_spec is None:
            self.plot_spec = [{}] * len(self.profiles)
        if not isinstance(self.plot_spec, list):
            self.plot_spec = [self.plot_spec] * len(self.profiles)
        
        self._setup_plots()
        
    def save(self, name=None):
        r"""
        Saves a 1d profile plot.

        Parameters
        ----------
        name : str
            The output file keyword.
        
        """
        if not self._plot_valid: self._setup_plots()
        unique = set(self.figures.values())
        if len(unique) < len(self.figures):
            figiter = izip(xrange(len(unique)), sorted(unique))
        else:
            iters = self.figures.iteritems()
        if name is None:
            if len(self.profiles) == 1:
                prefix = self.profiles[0].pf
            else:
                prefix = "Multi-data"
            name = "%s.png" % prefix
        suffix = get_image_suffix(name)
        prefix = name[:name.rfind(suffix)]
        if not suffix:
            suffix = ".png"
        canvas_cls = get_canvas(name)
        for uid, fig in iters:
            canvas = canvas_cls(fig)
            fn = "%s_1d-Profile_%s_%s%s" % \
              (prefix, self.profiles[0].x_field, uid, suffix)
            mylog.info("Saving %s", fn)
            canvas.print_figure(fn)
        return self

    def show(self):
        r"""This will send any existing plots to the IPython notebook.
        function name.

        If yt is being run from within an IPython session, and it is able to
        determine this, this function will send any existing plots to the
        notebook for display.

        If yt can't determine if it's inside an IPython session, it will raise
        YTNotInsideNotebook.

        Examples
        --------

        >>> slc = SlicePlot(pf, "x", ["Density", "VelocityMagnitude"])
        >>> slc.show()

        """
        if "__IPYTHON__" in dir(__builtin__):
            api_version = get_ipython_api_version()
            if api_version in ('0.10', '0.11'):
                self._send_zmq()
            else:
                from IPython.display import display
                display(self)
        else:
            raise YTNotInsideNotebook


    def _repr_html_(self):
        """Return an html representation of the plot object. Will display as a
        png for each WindowPlotMPL instance in self.plots"""
        ret = ''
        unique = set(self.figures.values())
        if len(unique) < len(self.figures):
            figiter = izip(xrange(len(unique)), sorted(unique))
        else:
            iters = self.figures.iteritems()
        for uid, fig in iters:
            canvas = mpl.FigureCanvasAgg(fig)
            f = cStringIO.StringIO()
            canvas.print_figure(f)
            f.seek(0)
            img = base64.b64encode(f.read())
            ret += '<img src="data:image/png;base64,%s"><br>' % img
        return ret

    def _setup_plots(self):
        self.figures = FigureContainer()
        self.axes = AxesContainer(self.figures)
        for i, profile in enumerate(self.profiles):
            for field, field_data in profile.field_data.items():
                self.axes[field].plot(profile.x[:-1], field_data, 
                                      label=self.label[i],
                                      **self.plot_spec[i])
        
        # This relies on 'profile' leaking
        for fname, axes in self.axes.items():
            xscale, yscale = self._get_field_log(fname, profile)
            xtitle, ytitle = self._get_field_title(fname, profile)
            axes.set_xscale(xscale)
            axes.set_yscale(yscale)
            axes.set_xlabel(xtitle)
            axes.set_ylabel(ytitle)
            axes.legend(loc="best")
        self._plot_valid = True

    @classmethod
    def from_profiles(cls, profiles, labels=None, plot_specs=None):
        r"""
        Instantiate a ProfilePlot object from a list of profiles 
        created with `yt.data_objects.profiles.create_profile`.

        Parameters
        ----------
        profiles : list of profiles
            If not None, a list of profile objects created with 
            `yt.data_objects.profiles.create_profile`.
        labels : list of strings
            A list of labels for each profile to be overplotted.
            Default: None.
        plot_specs : list of dicts
            A list of dictionaries containing plot keyword 
            arguments.  For example, [dict(color="red", linestyle=":")].
            Default: None.

        Examples
        --------

        >>> es = simulation("AMRCosmology.enzo", "Enzo")
        >>> es.get_time_series()

        >>> profiles = []
        >>> labels = []
        >>> plot_specs = []
        >>> for pf in es[-4:]:
        ...     ad = pf.h.all_data()
        ...     profiles.append(create_profile(ad, ["Density"],
        ...                                    fields=["Temperature",
        ...                                            "x-velocity"]))
        ...     labels.append(pf.current_redshift)
        ...     plot_specs.append(dict(linestyle="--", alpha=0.7))
        >>>
        >>> plot = ProfilePlot.from_profiles(profiles, labels=labels,
        ...                                  plot_specs=plot_specs)
        >>> plot.save()
        
        """
        if labels is not None and len(profiles) != len(labels):
            raise RuntimeError("Profiles list and labels list must be the same size.")
        if plot_specs is not None and len(plot_specs) != len(profiles):
            raise RuntimeError("Profiles list and plot_specs list must be the same size.")
        obj = cls(None, None, None, profiles=profiles, label=labels,
                  plot_spec=plot_specs)
        return obj

    @invalidate_plot
    def set_line_property(self, property, value, index=None):
        r"""
        Set properties for one or all lines to be plotted.

        Parameters
        ----------
        property : str
            The line property to be set.
        value : str, int, float
            The value to set for the line property.
        index : int
            The index of the profile in the list of profiles to be 
            changed.  If None, change all plotted lines.
            Default : None.

        Examples
        --------

        Change all the lines in a plot
        plot.set_line_property("linestyle", "-")

        Change a single line.
        plot.set_line_property("linewidth", 4, index=0)
        
        """
        if index is None:
            specs = self.plot_spec
        else:
            specs = [self.plot_spec[index]]
        for spec in specs:
            spec[property] = value
        return self
            
    def _get_field_log(self, field_y, profile):
        pf = profile.data_source.pf
        yfi = pf.field_info[field_y]
        if self.x_log is None:
            x_log = profile.x_log
        else:
            x_log = self.x_log
        if field_y in self.y_log:
            y_log = self.y_log[field_y]
        else:
            y_log = yfi.take_log
        scales = {True: 'log', False: 'linear'}
        return scales[x_log], scales[y_log]

    def _get_field_label(self, field, field_info):
        units = field_info.get_units()
        field_name = field_info.display_name
        if field_name is None:
            field_name = r'$\rm{'+field+r'}$'
        elif field_name.find('$') == -1:
            field_name = r'$\rm{'+field+r'}$'
        if units is None or units == '':
            label = field_name
        else:
            label = field_name+r'$\/\/('+units+r')$'
        return label

    def _get_field_title(self, field_y, profile):
        pf = profile.data_source.pf
        field_x = profile.x_field
        xfi = pf.field_info[field_x]
        yfi = pf.field_info[field_y]
        x_title = self.x_title or self._get_field_label(field_x, xfi)
        y_title = self.y_title.get(field_y, None) or \
                    self._get_field_label(field_y, yfi)
        return (x_title, y_title)
            

class PhasePlotter(object):
    scale = None
    _current_field = None

    def __init__(self, data_source, field_x, field_y, field_z,
                 weight="CellMassMsun", accumulation=False,
                 x_bins=128, x_log=True, x_bounds=None,
                 y_bins=128, y_log=True, y_bounds=None,
                 lazy_reader=True, fractional=False):
        r"""From an existing object, create a 2D, binned profile.

        This function will accept an existing `AMRData` source and from that,
        it will generate a `Binned2DProfile`, based on the specified options.
        This is useful if you have extracted a region, or if you wish to bin
        some set of massages data -- or even if you wish to bin anything other
        than a sphere.  The profile will be 2D, which means while it can have
        an arbitrary number of fields, those fields will all be binned based on
        two fields.

        Parameters
        ----------
        data_source : `yt.data_objects.api.AMRData`
            This is a data source respecting the `AMRData` protocol (i.e., it
            has grids and so forth) that will be used as input to the profile
            generation.
        fields : list of strings
            The first element of this list is the field by which we will bin
            into the x-axis, the second is the field by which we will bin onto
            the y-axis.  All subsequent fields will be binned and their
            profiles added to the underlying `BinnedProfile2D`.
        cmap : string, optional
            An acceptable colormap.  See either yt.visualization.color_maps or
            http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps .
        weight : string, default "CellMassMsun"
            The weighting field for an average.  This defaults to mass-weighted
            averaging.
        accumulation : list of booleans, optional
            If true, from the low-value to the high-value the values in all
            binned fields will be accumulated.  This is useful for instance
            when adding an unweighted CellMassMsun to a radial plot, as it will
            show mass interior to that radius.  The first value is for the
            x-axis, the second value for the y-axis.  Note that accumulation
            will only be along each row or column.
        x_bins : int, optional
            How many bins should there be in the x-axis variable?
        x_log : boolean, optional
            Should the bin edges be log-spaced?
        x_bounds : tuple of floats, optional
            If specified, the boundary values for the binning.  If unspecified,
            the min/max from the data_source will be used.  (Non-zero min/max
            in case of log-spacing.)
        y_bins : int, optional
            How many bins should there be in the y-axis variable?
        y_log : boolean, optional
            Should the bin edges be log-spaced?
        y_bounds : tuple of floats, optional
            If specified, the boundary values for the binning.  If unspecified,
            the min/max from the data_source will be used.  (Non-zero min/max
            in case of log-spacing.)
        lazy_reader : boolean, optional
            If this is false, all of the data will be read into memory before
            any processing occurs.  It defaults to true, and grids are binned
            on a one-by-one basis.  Note that parallel computation requires
            this to be true.
        fractional : boolean
            If true, the plot will be normalized to the sum of all the binned
            values.

        Returns
        -------
        plot : `yt.visualization.plot_types.PlotTypes.PhasePlot`
            The plot that has been added to the PlotCollection.

        See Also
        --------
        yt.data_objects.profiles.BinnedProfile2D : This is the object that does the
                                   transformation of raw data into a 1D
                                   profile.
        
        Examples
        --------
        This will show the mass-distribution in the Density-Temperature plane.
        
        >>> pf = load("RD0005-mine/RedshiftOutput0005")
        >>> reg = pf.h.region([0.1, 0.2, 0.3], [0.0, 0.1, 0.2],
        ...                   [0.2, 0.3, 0.4])
        >>> pc.add_phase_object(reg, ["Density", "Temperature", "CellMassMsun"],
        ...                     weight = None)
        """
        if x_bounds is None:
            x_min, x_max = data_source.quantities["Extrema"](
                                    field_x, non_zero = x_log,
                                    lazy_reader=lazy_reader)[0]
        else:
            x_min, x_max = x_bounds
        if y_bounds is None:
            y_min, y_max = data_source.quantities["Extrema"](
                                    field_y, non_zero = y_log,
                                    lazy_reader=lazy_reader)[0]
        else:
            y_min, y_max = y_bounds
        profile = BinnedProfile2D(data_source,
                                  x_bins, field_x, x_min, x_max, x_log,
                                  y_bins, field_y, y_min, y_max, y_log,
                                  lazy_reader)
        # This is a fallback, in case we forget.
        if field_z.startswith("CellMass") or \
           field_z.startswith("CellVolume"):
            mylog.warning("Setting weight to None")
            weight = None
        self._initial_weight = weight
        profile.add_fields(field_z, weight=weight, accumulation=accumulation, fractional=fractional)
        self._current_field = field_z
        self.profile = profile
        self.scale = {True:'log', False:'linear'}.get(
                data_source.pf.field_info[field_z].take_log, "log")
        self._setup_plot()

    def _setup_plot(self):
        xax = AxisSpec()
        xax.title = self.profile.x_bin_field
        xax.bounds = (self.profile._x_bins[0],
                      self.profile._x_bins[-1])
        xax.scale = {True: 'log', False: 'linear'}[self.profile._x_log]
        xax.calculate_ticks()

        yax = AxisSpec()
        yax.title = self.profile.y_bin_field
        yax.bounds = (self.profile._y_bins[0],
                      self.profile._y_bins[-1])
        yax.scale = {True: 'log', False: 'linear'}[self.profile._y_log]
        yax.calculate_ticks()

        cbar = ColorbarSpec()
        cbar.title = self._current_field
        if self.scale == 'log':
            nz = (self.profile[self._current_field] > 0)
            mi = self.profile[self._current_field][nz].min()
        else:
            mi = self.profile[self._current_field].min()
        ma = self.profile[self._current_field].max()
        cbar.bounds = (mi, ma)
        cbar.cmap = 'algae'
        cbar.scale = self.scale
        cbar.calculate_ticks()

        self.plot = ImagePlotContainer()
        self.plot.image = self.profile[self._current_field].transpose()
        self.plot.x_spec = xax
        self.plot.y_spec = yax
        self.plot.cbar = cbar

    def to_mpl(self, place = None):
        import _mpl_imports as mpl
        if isinstance(place, mpl.matplotlib.figure.Figure):
            figure, place = place, None
            place = None
        else:
            figure = mpl.matplotlib.figure.Figure((10,8))
        if isinstance(place, mpl.matplotlib.axes.Axes):
            axes, place = place, None
        else:
            axes = figure.add_subplot(1,1,1)
        # We'll go with a mesh here, even if it's inappropriate
        use_mesh = False
        xmi, xma = self.x_spec.bounds
        if self.x_spec.scale == 'log':
            x_bins = np.logspace(np.log10(xmi), np.log10(xma),
                                 self.image.shape[0]+1)
            use_mesh = True
        else:
            x_bins = np.logspace(xmi, xma, self.image.shape[0]+1)

        ymi, yma = self.y_spec.bounds
        if self.y_spec.scale == 'log':
            y_bins = np.logspace(np.log10(ymi), np.log10(yma),
                                 self.image.shape[0]+1)
            use_mesh = True
        else:
            y_bins = np.logspace(ymi, yma, self.image.shape[0]+1)

        im = self.image
        if self.cbar.scale == 'log':
            norm = mpl.matplotlib.colors.LogNorm()
        else:
            norm = mpl.matplotlib.colors.Normalize()
        if use_mesh:
            pcm = axes.pcolormesh(x_bins, y_bins, self.image, norm=norm,
                                  shading='flat', cmap = self.cbar.cmap,
                                  rasterized=True)
            if self.x_spec.scale == 'log': axes.set_xscale("log")
            if self.y_spec.scale == 'log': axes.set_yscale("log")
        else:
            axes.imshow(self.image, origin='lower', interpolation='nearest',
                        cmap = self.cbar.cmap, extent = [xmi,xma,ymi,yma],
                        norm = norm)
        if self.x_spec.title is not None:
            axes.set_xlabel(self.x_spec.title)
        if self.y_spec.title is not None:
            axes.set_ylabel(self.y_spec.title)
        if isinstance(place, types.StringTypes):
            canvas = mpl.FigureCanvasAgg(figure)
            canvas.print_figure(place)
        return figure, axes

