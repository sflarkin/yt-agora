"""
Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2010 Matthew Turk.  All Rights Reserved.

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

# Major library imports
from vm_panner import VariableMeshPanner
from numpy import linspace, meshgrid, pi, sin, mgrid, zeros

# Enthought library imports
from enthought.enable.api import Component, ComponentEditor, Window
from enthought.traits.api import HasTraits, Instance, Button, Any, Callable, \
        on_trait_change, Bool, DelegatesTo, List, Enum
from enthought.traits.ui.api import Item, Group, View

# Chaco imports
from enthought.chaco.api import ArrayPlotData, jet, Plot, HPlotContainer, \
        ColorBar, DataRange1D, DataRange2D, LinearMapper, ImageData, \
        CMapImagePlot, OverlayPlotContainer
from enthought.chaco.tools.api import PanTool, ZoomTool, RangeSelection, \
        RangeSelectionOverlay, RangeSelection
from zoom_overlay import ZoomOverlay
from enthought.chaco.tools.image_inspector_tool import ImageInspectorTool, \
     ImageInspectorOverlay

if not hasattr(DataRange2D, "_subranges_updated"):
    print "You'll need to add _subranges updated to enthought/chaco/data_range_2d.py"
    print 'Add this at the correct indentation level:'
    print
    print '    @on_trait_change("_xrange.updated,_yrange.updated")'
    print '    def _subranges_updated(self):'
    print '        self.updated = True'
    print
    raise RuntimeError

class FunctionImageData(ImageData):
    # The function to call with the low and high values of the range.
    # It should return an array of values.
    func = Callable

    # A reference to a datarange
    data_range = Instance(DataRange2D)

    def __init__(self, **kw):
        # Explicitly call the AbstractDataSource constructor because
        # the ArrayDataSource ctor wants a data array
        ImageData.__init__(self, **kw)
        self.recalculate()

    @on_trait_change('data_range.updated')
    def recalculate(self):
        if self.func is not None and self.data_range is not None:
            newarray = self.func(self.data_range.low, self.data_range.high)
            ImageData.set_data(self, newarray)
        else:
            self._data = zeros((512,512),dtype=float)

    def set_data(self, *args, **kw):
        raise RuntimeError("Cannot set numerical data on a FunctionDataSource")

    def set_mask(self, mask):
        raise NotImplementedError

    def remove_mask(self):
        raise NotImplementedError

class ImagePixelizerHelper(object):
    index = None
    def __init__(self, panner, run_callbacks = False):
        self.panner = panner
        self.run_callbacks = run_callbacks

    def __call__(self, low, high):
        b = self.panner.set_low_high(low, high)
        if self.run_callbacks:
            self.panner._run_callbacks()
        if self.index is not None:
            num_x_ticks = b.shape[0] + 1
            num_y_ticks = b.shape[1] + 1
            xs = mgrid[low[0]:high[0]:num_x_ticks*1j]
            ys = mgrid[low[1]:high[1]:num_y_ticks*1j]
            self.index.set_data( xs, ys )
        return b

class ZoomedPlotUpdater(object):
    fid = None
    def __init__(self, panner, zoom_factor=4):
        """
        Supply this an a viewport_callback argument to a panner if you want to
        update a second panner in a smaller portion at higher resolution.  If
        you then set the *fid* property, you can also have it update a
        FunctionImageData datarange.  *panner* is the panner to update (not the
        one this is a callback to) and *zoom_factor* is how much to zoom in by.
        """
        self.panner = panner
        self.zoom_factor = zoom_factor

    def __call__(self, xlim, ylim):
        self.panner.xlim = xlim
        self.panner.ylim = ylim
        self.panner.zoom(self.zoom_factor)
        nxlim = self.panner.xlim
        nylim = self.panner.ylim
        if self.fid is not None:
            self.fid.data_range.set_bounds(
                (nxlim[0], nylim[0]), (nxlim[1], nylim[1]))

class VMImagePlot(HasTraits):
    plot = Instance(Plot)
    fid = Instance(FunctionImageData)
    img_plot = Instance(CMapImagePlot)
    panner = Instance(VariableMeshPanner)
    helper = Instance(ImagePixelizerHelper)
    fields = List

    def __init__(self, *args, **kwargs):
        super(VMImagePlot, self).__init__(**kwargs)
        self.add_trait("field", Enum(*self.fields))
        self.field = self.panner.field

    def _plot_default(self):
        pd = ArrayPlotData()
        plot = Plot(pd, padding = 0)
        self.fid._data = self.panner.buffer

        pd.set_data("imagedata", self.fid)

        img_plot = plot.img_plot("imagedata", colormap=jet,
                                 interpolation='nearest',
                                 xbounds=(0.0, 1.0),
                                 ybounds=(0.0, 1.0))[0]
        self.fid.data_range = plot.range2d
        self.helper.index = img_plot.index
        self.img_plot = img_plot
        return plot

    def _field_changed(self, old, new):
        self.panner.field = new
        self.fid.recalculate()

    def _fid_default(self):
        return FunctionImageData(func = self.helper)

    def _helper_default(self):
        return ImagePixelizerHelper(self.panner)

    def _fields_default(self):
        keys = []
        for field in self.panner.source.data:
            if field not in ['px','py','pdx','pdy',
                             'pz','pdz','weight_field']:
                keys.append(field)
        return keys

class VariableMeshPannerView(HasTraits):

    plot = Instance(Plot)
    spawn_zoom = Button
    vm_plot = Instance(VMImagePlot)
    use_tools = Bool(True)
    full_container = Instance(HPlotContainer)
    container = Instance(OverlayPlotContainer)
    
    traits_view = View(
                    Group(
                        Item('full_container',
                             editor=ComponentEditor(size=(512,512)), 
                             show_label=False),
                        Item('field', show_label=False),
                        orientation = "vertical"),
                    width = 800, height=800,
                    resizable=True, title="Pan and Scan",
                    )

    def _vm_plot_default(self):
        return VMImagePlot(panner=self.panner)
    
    def __init__(self, **kwargs):
        super(VariableMeshPannerView, self).__init__(**kwargs)
        # Create the plot
        self.add_trait("field", DelegatesTo("vm_plot"))

        plot = self.vm_plot.plot
        img_plot = self.vm_plot.img_plot

        if self.use_tools:
            plot.tools.append(PanTool(img_plot))
            zoom = ZoomTool(component=img_plot, tool_mode="box", always_on=False)
            plot.overlays.append(zoom)
            imgtool = ImageInspectorTool(img_plot)
            img_plot.tools.append(imgtool)
            overlay = ImageInspectorOverlay(component=img_plot, image_inspector=imgtool,
                                            bgcolor="white", border_visible=True)
            img_plot.overlays.append(overlay)


        image_value_range = DataRange1D(self.vm_plot.fid)
        cbar_index_mapper = LinearMapper(range=image_value_range)
        self.colorbar = ColorBar(index_mapper=cbar_index_mapper,
                                 plot=img_plot,
                                 padding_right=40,
                                 resizable='v',
                                 width=30)

        self.colorbar.tools.append(
            PanTool(self.colorbar, constrain_direction="y", constrain=True))
        zoom_overlay = ZoomTool(self.colorbar, axis="index", tool_mode="range",
                                always_on=True, drag_button="right")
        self.colorbar.overlays.append(zoom_overlay)

        # create a range selection for the colorbar
        range_selection = RangeSelection(component=self.colorbar)
        self.colorbar.tools.append(range_selection)
        self.colorbar.overlays.append(
                RangeSelectionOverlay(component=self.colorbar,
                                      border_color="white",
                                      alpha=0.8, fill_color="lightgray"))

        # we also want to the range selection to inform the cmap plot of
        # the selection, so set that up as well
        range_selection.listeners.append(img_plot)

        self.full_container = HPlotContainer(padding=30)
        self.container = OverlayPlotContainer(padding=0)
        self.full_container.add(self.colorbar)
        self.full_container.add(self.container)
        self.container.add(self.vm_plot.plot)

    def _spawn_zoom_fired(self):
        np = self.panner.source.pf.h.image_panner(
                self.panner.source, self.panner.size, self.panner.field)
        new_window = VariableMeshPannerView(panner = np)
