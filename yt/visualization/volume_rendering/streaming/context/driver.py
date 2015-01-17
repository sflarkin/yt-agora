import threading
from multiprocessing import Pipe
from Queue import Queue

from yt.visualization.volume_rendering.streaming.context.render import Render
import yt.visualization.volume_rendering.streaming.helpers.type_conversion as tc
from yt.visualization.volume_rendering.theia.scene import TheiaScene

import pycuda.driver as drv
import sys

class Driver(threading.Thread):
      def __init__(self, port = None, size = (864, 480), grid = None, callback = None,  wrapper = None): 
            threading.Thread.__init__(self)
            self.pipe1, self.pipe2 = Pipe()
            self.contexts = []

            self.grid   = grid
            self.grid_path = None
            self.size  = size 
            self.gpu_post    = None
            
            self.options = None

            # When passing as object via json, all get type unicode,
            #  specify the variable's type here.
            self.options_types   = {
                  "menu_updated"   : tc.BOOL,
                  "mouseDX"        : tc.FLOAT,
                  "mouseDY"        : tc.FLOAT,
                  "width"          : tc.INT,
                  "height"         : tc.INT,
                  "mouse_button"   : tc.STR,
                  "density"        : tc.FLOAT,
                  "brightness"     : tc.FLOAT,
                  "tfl"            : tc.FLOAT,
                  "tfu"            : tc.FLOAT,
                  "colormap"       : tc.STR,
                  "tf_func"        : tc.STR,
                  "grid"           : tc.STR,
                  "tfmean"         : tc.FLOAT,
                  "tfstddev"       : tc.FLOAT,
                  "nclip"          : tc.FLOAT,
                  "fclip"          : tc.FLOAT,
                  "samplesize"     : tc.FLOAT,
                  "maxsamples"     : tc.FLOAT,
                  "x_persp"        : tc.FLOAT,
                  "y_persp"        : tc.FLOAT,
                  "z_persp"        : tc.FLOAT,
                  "x_axis"         : tc.BOOL,
                  "y_axis"         : tc.BOOL,
                  "z_axis"         : tc.BOOL,
                  "grid_dir"       : tc.STR,
                  "img_save"       : tc.BOOL,
                  "grid"           : tc.STR,
                  "grid_path"      : tc.STR,
                  "xml_save"       : tc.BOOL,
                  "xml_load"       : tc.BOOL,
                  "xml_path"       : tc.STR,
                  "img_path"       : tc.STR,
                  "video_save"     : tc.BOOL,
                  "rot"            : tc.NPLIST,
                  "pos"            : tc.NPLIST,
                  "scale"          : tc.FLOAT,
                  "rev_tfbounds"   : tc.BOOL,
                  "video_arm"      : tc.BOOL,
                  "video_path"     : tc.STR,
                  "video_rec"      : tc.BOOL,
            }
            
            self.theiascene = None

            self.image_queue = Queue()

            s = ImageServer(self.image_queue, callback, wrapper)
            s.start()

      def run(self):
            drv.init()
            self.num_gpus   = drv.Device.count()

            self.theiascene = TheiaScene()

            for i in range(self.num_gpus):
                  p1, p2 = Pipe()
                  t = Render(ts = self.theiascene, pipe = p1, image_queue = self.image_queue, device_num = i, grid = self.grid, size = self.size)
                  t.start()
                  self.contexts.append(p2)
                  
            self.frame = 0

            while True:
                  data = self.pipe2.recv()
                  if data == None:
                        break
                  data['frame'] = self.frame
                  gpu = self.contexts[ self.frame % self.num_gpus ]
                  gpu.send(data)
                  if gpu.poll():
                        self.gpu_post = gpu.recv()
                  self.frame += 1

            # Close Render contexts
            for i in range(self.num_gpus):
                  self.contexts[i].send(None)

      def update(self):
            self.theiascene.camera.perspective(x= self.options['x_persp'], y= self.options['y_persp'], z= self.options['z_persp'])
            self.options['matrix'] = self.theiascene.camera.get_matrix()
            self.pipe1.send(self.options)

      def stop(self):
            self.pipe1.send(None)
            
class ImageServer(threading.Thread):
      def __init__(self, queue, callback, wrapper = None):
            threading.Thread.__init__(self)
            self.queue = queue
            self.callback = callback
            if wrapper == None:
                  wrapper = lambda f, d, id: f(d, id)
            self.wrapper = wrapper

      def run(self):
            while True:
                  try:
                        image_data = self.queue.get()
                  except Queue.Empty:
                        continue
                  self.wrapper(self.callback, image_data[0], image_data[1])

"""
No print statements, doesn't initialize a gpu that could exist with server thread
"""
from subprocess import Popen, PIPE
class DriverServer(threading.Thread):
      def __init__(self, camera = None, callback = None, wrapper = None): 
            threading.Thread.__init__(self)
            self.pipe1, self.pipe2 = Pipe()
            self.camera = camera
            self.drivers = []
            self.image_queue = Queue()
            self.num_drivers = 1
            s = ImageServer(self.image_queue, callback, wrapper)
            s.start()

      def run(self):
            for i in range(self.num_drivers):
                  p = Popen(["/home/nihasmit/yt-conda/bin/python","-u","./pyRGBA/ts.py","-driver"], stdin=PIPE, stdout= PIPE)
                  self.drivers.append(p)
                  
            self.frame = 0

            while True:
                  data = self.pipe2.recv()
                  if data == None:
                        break
                  data['frame'] = self.frame
                  self.drivers[ self.frame % self.num_gpus ].communicate(data)
                  self.frame += 1

            # Close Drivers
            for i in range(self.num_drivers):
                  self.drivers[i].communicate(None)

      def update(self):
            self.pipe1.send({ 'matrix': self.camera.matrix(), 
                              'brightness': self.camera.brightness, 
                              'density': self.camera.density, 
                              'size': (self.camera.width, self.camera.height) })

      def stop(self):
            self.pipe1.send(None)

      def update_func(f):
            if self.driver == False:
                  f()
