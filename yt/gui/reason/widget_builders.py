"""
These widget objects provide the interaction to yt tasks.



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
from yt.mods import *
import weakref

class RenderingScene(object):
    _camera = None
    _tf = None
    
    def __init__(self, pf, camera=None, tf=None):
        self.pf = weakref.proxy(pf)
        self._camera = camera
        self._tf = tf

        self.center = self.pf.domain_center
        self.normal_vector = np.array([0.7,1.0,0.3])
        self.north_vector = [0.,0.,1.]
        self.steady_north = True
        self.fields = ['Density']
        self.log_fields = [True]
        self.l_max = 0
        self.mi = None
        self.ma = None
        if self._tf is None:
            self._new_tf(self.pf)

        if self._camera is None:
            self._new_camera(self.pf)

    def _new_tf(self, pf, mi=None, ma=None, nbins=1024):
        if mi is None or ma is None:
            roi = self.pf.h.region(self.center, self.center-self.width, self.center+self.width)
            self.mi, self.ma = roi.quantities['Extrema'](self.fields[0])[0]
            if self.log_fields[0]:
                self.mi, self.ma = np.log10(self.mi), np.log10(self.ma)

        self._tf = ColorTransferFunction((self.mi-2, self.ma+2), nbins=nbins)

    def add_contours(self, n_contours=7, contour_width=0.05, colormap='kamae'):
        self._tf.add_layers(n_contours=n_contours ,w=contour_width,
                                  col_bounds = (self.mi,self.ma), 
                                  colormap=colormap)

    def _new_camera(self, pf):
        del self._camera
        self._camera = self.pf.camera(self.center, self.normal_vector, 
                                      self.width, self.resolution, self._tf,
                                      north_vector=self.north_vector,
                                      steady_north=self.steady_north,
                                      fields=self.fields, log_fields=self.log_fields,
                                      l_max=self.l_max)
    def snapshot(self):
        return self._camera.snapshot()


def get_corners(pf, max_level=None):
    DL = pf.domain_left_edge[None,:,None]
    DW = pf.domain_width[None,:,None]/100.0
    corners = ((pf.h.grid_corners-DL)/DW)
    levels = pf.h.grid_levels
    return corners, levels

def get_isocontour(pf, field, value=None, rel_val = False):

    dd = pf.h.all_data()
    if value is None or rel_val:
        if value is None: value = 0.5
        mi, ma = np.log10(dd.quantities["Extrema"]("Density")[0])
        value = 10.0**(value*(ma - mi) + mi)
    vert = dd.extract_isocontours("Density", value)
    np.multiply(vert, 100, vert)
    return vert

def get_streamlines(pf):
    from yt.visualization.api import Streamlines
    streamlines = Streamlines(pf, pf.domain_center) 
    streamlines.integrate_through_volume()
    stream = streamlines.path(0)
    matplotlib.pylab.semilogy(stream['t'], stream['Density'], '-x')


