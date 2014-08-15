import threading
from yt.visualization.volume_rendering.video.ffmpeg import FFMpegJpegStream, FFMpegMp4VideoStream

import pycuda.driver as drv

from yt.visualization.volume_rendering.theia.algorithms.front_to_back import FrontToBackRaycaster

#yt transfer functions
from yt.visualization.volume_rendering.transfer_functions import ColorTransferFunction
from yt.visualization.color_maps import *

#cuda transfer functions
from yt.visualization.volume_rendering.theia.transfer.linear_transfer import LinearTransferFunction
from yt.visualization.volume_rendering.theia.transfer.helper import *


class Render(threading.Thread):
      """Render()
      Create a rending thread so that all rendering / image processing occurs on
      a seperate thread.  The pipe variable is shared with the websocket class
      which serves it new mouse position updates along with other information 
      regarding framerate and transfer function variables.  The final variable
      passed through the pipe is an asynchronous callback that invokes sending
      the binary jpeg options to the browser.
      """
      sll = 0 
      slm = 1000 
      def __init__(self, pipe = None, ts = None, image_queue = None, device_num  = None, grid = None, size = (864, 480)):
            threading.Thread.__init__(self)
            self.pipe = pipe

            self.device_num  = device_num
            self.image_queue = image_queue
            self.grid = grid
            self.size = size
            self.grid_path = None
            self.prev_grid_path = None
            self.tfmin = 0.0
            self.tfmax = 1.0

            self.tf_func = {
                  'gaussian'    : self.gaussian,
                  'exponential' : self.exponential,
                  'linear'      : self.linear,
            }
 
            self.ts          = ts
            self.video_arm = False
            self.video_out = None

      def run(self):
            self.ctx = drv.Device(self.device_num).make_context()
            self.ctx.push()

            self.ts.source.set_volume(volume = self.grid)
            self.ts.source.set_raycaster(raycaster = FrontToBackRaycaster(size = self.size))

            lower = 0.6
            upper = 4.6
            bins = 5000
            self.tfl = lower
            self.tfu = upper
            self.tfmean      = 0.0
            self.tfstddev    = 1.0
            colormap = "Accent"
            tf = ColorTransferFunction( (lower, upper), bins)
            tf.map_to_colormap(lower, upper, colormap=colormap, scale_func = self.gaussian)
            self.ts.source.raycaster.set_transfer(tf)#LinearTransferFunction(arrays = yt_to_rgba_transfer(tf), range = tf.x_bounds))

            while True:
                  options = self.pipe.recv()
                  if options == None:
                        break

                  if self.grid_path != options['grid_path'] and options['grid_path'] != None:
                        self.pipe.send({ 'grid_change': None, })
                        import numpy as np
                        self.grid_path = options['grid_path']
                        try:
                              vol = np.load(self.grid_path)
                              self.tfmin = np.amin(vol)
                              self.tfmax = np.amax(vol)
                              self.ts.source.set_volume(volume = np.load(self.grid_path))
                              self.pipe.send({ 'grid_change': (self.tfmin, self.tfmax), })
                        except IOError:
                              print "Invalid file path:", self.grid_path
                        self.pipe.send(None)

                  (mi, ma) = tf.x_bounds
                  if colormap != options['colormap'] or mi != options['tfl'] or ma != options['tfu'] or self.tfmean != options['tfmean'] or self.tfstddev != options['tfstddev']:
                        if options['rev_tfbounds']:
                              tfl = options['tfu']
                              tfu = options['tfl']
                        else:
                              tfl = options['tfl']
                              tfu = options['tfu']
                        colormap = options['colormap']
                        self.tfstddev    = options['tfstddev']
                        self.tfmean      = options['tfmean']
                        if options['tfstddev'] != 0.0:
                              tf = ColorTransferFunction( (tfl, tfu), bins)
                              tf.map_to_colormap(tfl, tfu, colormap=options['colormap'], scale_func = self.tf_func[options['tf_func']])
                              #TODO: does tf.x_bounds need to be changed when reversing the colormap?
                              self.ts.source.raycaster.set_transfer(tf)#(LinearTransferFunction(arrays = yt_to_rgba_transfer(tf), range = tf.x_bounds))

                  # Update raycaster
                  self.ts.source.raycaster.set_opacity(options['density'])
                  self.ts.source.raycaster.set_brightness(options['brightness'])
                  self.ts.source.raycaster.set_matrix(options['matrix'])

                  self.ts.source.raycaster.set_max_samples(options['maxsamples'])
                  self.ts.source.raycaster.set_sample_size(options['samplesize'])

                  self.ts.source.update()
                  
                  if options['img_save']:
                        f = FFMpegJpegStream(size = (options['width'], options['height']))
                        f.write(self.ts.source.get_results(), options['frame'], options['img_path'])

                  if options['video_arm']:
                        if not self.video_arm:
                              self.video_out = FFMpegMp4VideoStream(filename = options['video_path'], size = (options['width'], options['height']))
                        self.video_arm = True
                        if options['video_rec']:
                              self.video_out.write(self.ts.source.get_results())
                  else:
                        if self.video_arm:
                              self.video_arm = False
                              self.video_out.close()

                  # Put it in the queue for the server to deal with
                  self.image_queue.put(((self.ts.source.get_results()), options['frame']))

            self.ctx.pop()

      def scale_func(self, v, mi, ma):
          #mu = (500) * ( 2 * 1000 *  mi  +  ma * (self.sll + self.slu) ) 
          mu = 3.5 
          sigma = 0.5
          #mi = ((self.sll/1000)*mi) 
          #ma = ((self.slu/1000)*ma)
          #_sll =  (self.sll / 1000)*(self.tfu - self.tfl)
          #_slm =  (self.slm / 1000)*(self.tfu - self.tfl)
          #return np.minimum( np.abs(v - _slm) / np.abs(_slm - _sll), 1.0)
          #return np.minimum(( 1 / (sigma * np.sqrt(2 * np.pi)) ) * np.exp ( (- (v - mu)**2) / (2*sigma**2)), 1.0)
          return np.minimum(np.exp(v), 1.0)

      def gaussian(self, v, mi, ma):
            tfmean = (self.tfmean / 5.0) + ((mi + ma) / 2)
            tfstddev = (self.tfstddev / 5.0) * ((mi + ma) / 2)
            return (1.0 / (tfstddev * np.sqrt(2.0 * np.pi))) * np.exp(- (v - tfmean)**2.0 / (2.0 * tfstddev**2.0))

      def exponential(self, v, mi, ma):
            base = self.tfstddev
            exp  = self.tfmean
            return np.exp(v * base, exp)

      def linear(self, v, mi, ma):
            slope = self.tfstddev
            yint  = self.tfmean
            return v * slope + yint

