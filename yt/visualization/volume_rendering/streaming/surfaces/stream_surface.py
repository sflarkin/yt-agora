import pycuda.driver as drv
import numpy as np
from   yt.visualization.volume_rendering.surfaces.array_surface import ArraySurface
import matplotlib.pyplot as plot

class StreamSurface(ArraySurface):
	def __init__(self, array=None, size=None, copy=True):
		ArraySurface.__init__(self)

		self.local_array = None
		self.bounds = size
            self.data_type = np.uint8
		if (array != None):
		    self.set_array(array, copy=copy)
		else:
		    if size != None:
	                self.set_array(np.zeros(size, dtype=np.uint8), copy=copy)
