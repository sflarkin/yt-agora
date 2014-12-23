"""Stream Handler

This module creates a tornado server that is responsible 
for serving the HTML5 canvas with a jpeg stream and an
interactive user interface.

""" 
from IPython.display import HTML
import tornado.ioloop
import tornado.web
import tornado.websocket
import signal
import functools
import multiprocessing
from yt.visualization.volume_rendering.streaming.context.driver import Driver
import yt.visualization.volume_rendering.streaming.streamserver.jpeg_server_html as jpeg_server_html
import yt.visualization.volume_rendering.streaming.helpers.type_conversion as tc
from yt.visualization.volume_rendering.video.ffmpeg import FFMpegFlashVideoStream
import numpy as np
import json

#Catch CTL-C and properly shut down the server
def sig_handler(sig, frame):
      print "Going down..."
      tornado.ioloop.IOLoop.instance().stop()

signal.signal(signal.SIGTERM, sig_handler)
signal.signal(signal.SIGINT, sig_handler)

class EchoWebSocket(tornado.websocket.WebSocketHandler):
      """EchoWebSocket()
      Communication with browser via websocket.
      """

      def open(self):
            self.files = []
            self.directory = self.application.directory
            self.grid_path = ""
            self.screen = ""
            self.colormap_images = {}
            import os
            f = open(os.path.dirname(__file__) + "/static/colormap_images/color_maps")
            colormap_lines = f.readlines()
            f.close()
            for line in colormap_lines:
                  key_val = line.split("|")
                  self.colormap_images[key_val[0]] = os.path.dirname(__file__) + "/static/colormap_images/" + key_val[1].strip('\n')
            self.start_server()
            print "WebSocket opened"

      @tornado.web.asynchronous
      def on_message(self, message):
            """on_message()
            On message event, grab the mouse / framerate / etc. data
            and enqueue for the render thread.  Wait for the render
            thread until it has rendered a new image for us to send.
            """
            self.ctxdrv.options = tc.json2dict(json.loads(message), self.ctxdrv.options_types)

            if self.ctxdrv.options['xml_save']:
                  self.ctxdrv.options['rot'] = self.ctxdrv.theiascene.camera.rot
                  self.ctxdrv.options['scale'] = self.ctxdrv.theiascene.camera.scale
                  self.ctxdrv.options['pos'] = self.ctxdrv.theiascene.camera.pos
                  try:
                        xml_file = open(self.ctxdrv.options['xml_path'], "w")
                  except IOError:
                        self.ctxdrv.options['is_serv_error'] = True
                        self.ctxdrv.options['serv_error'] = "File not found!"
                        self.send_message()
                        return
                  xml_file.write(tc.dict2xml(self.ctxdrv.options, self.ctxdrv.options_types))
                  xml_file.close()

            if self.ctxdrv.options['xml_load']:
                  try:
                        xml_file = open(self.ctxdrv.options['xml_path'], "r")
                  except IOError:
                        self.ctxdrv.options['is_serv_error'] = True
                        self.ctxdrv.options['serv_error'] = "File not found!"
                        self.send_message()
                        return
                  self.ctxdrv.options = tc.xml2dict(xml_file, self.ctxdrv.options_types)
                  xml_file.close()
                  self.ctxdrv.options['xml_load'] = False
                  self.ctxdrv.theiascene.camera.rot = self.ctxdrv.options['rot']
                  self.ctxdrv.theiascene.camera.scale = self.ctxdrv.options['scale']
                  self.ctxdrv.theiascene.camera.pos = self.ctxdrv.options['pos']
                  self.ctxdrv.update()
                  self.ctxdrv.update()
                  self.ctxdrv.update()
                  self.ctxdrv.update()
                  self.send_message()
                  return

            if self.ctxdrv.gpu_post != None and self.ctxdrv.options['grid'] != None:
                  if self.ctxdrv.gpu_post['grid_change'] == None:
                        self.ctxdrv.options['grid_change'] = True
                        self.send_message()
                        self.ctxdrv.update()
                        return
                  else:
                        self.ctxdrv.options['tfmin'], self.ctxdrv.options['tfmax'] = self.ctxdrv.gpu_post['grid_change']
                        self.ctxdrv.options['grid_change'] = False
                        self.send_message()
            self.ctxdrv.options['grid_change'] = False

            if (self.directory != self.ctxdrv.options['grid_dir']  or self.ctxdrv.options['grid'] == '' or self.ctxdrv.options['grid'] == 'None') and len(self.ctxdrv.options['grid_dir']) > 0:
                  self.directory = self.ctxdrv.options['grid_dir']
                  if self.directory[-1] != '/':
                        self.directory += '/'
                  import os
                  import re
                  dir = os.listdir(self.directory)
                  self.files = []
                  for file in dir:
                        if re.match(".*\.npy|.*\.xml", file) != None:
                              self.files.append(file)
                  self.send_message()

            if (self.directory + self.ctxdrv.options['grid']) != self.grid_path and self.ctxdrv.options['grid'] != None and self.ctxdrv.options['grid'] != '':
                  self.ctxdrv.options['grid_path'] = self.directory + self.ctxdrv.options['grid']

            if self.ctxdrv.options['menu_updated']:
                  if self.ctxdrv.options['x_axis']:
                        self.ctxdrv.theiascene.camera.rot = np.array([0.0, - np.pi / 2.0, 0.0])
                        self.ctxdrv.update()
                  if self.ctxdrv.options['y_axis']:
                        self.ctxdrv.theiascene.camera.rot = np.array([- np.pi / 2.0, 0.0, 0.0])
                        self.ctxdrv.update()
                  if self.ctxdrv.options['z_axis']:
                        self.ctxdrv.theiascene.camera.rot = np.array([0.0, 0.0, 0.0])
                        self.ctxdrv.update()
                  self.send_message()
            else:
                  if self.ctxdrv.options['mouse_button'] == "L":
                        self.ctxdrv.theiascene.camera.rotateX(2 * float(self.ctxdrv.options['mouseDY']) / self.ctxdrv.options['width'])
                        self.ctxdrv.theiascene.camera.rotateY(2 * float(self.ctxdrv.options['mouseDX']) / self.ctxdrv.options['width'])
                  elif self.ctxdrv.options['mouse_button'] == "M":
                        self.ctxdrv.theiascene.camera.translateX(4 * float(self.ctxdrv.options['mouseDX']) / self.ctxdrv.options['width'])
                        self.ctxdrv.theiascene.camera.translateY(4 * -float(self.ctxdrv.options['mouseDY']) / self.ctxdrv.options['height'])
                  elif self.ctxdrv.options['mouse_button'] == "R":
                        self.ctxdrv.theiascene.camera.zoom(200.0 * (float(self.ctxdrv.options['mouseDY']) / self.ctxdrv.options['width']))

            self.ctxdrv.update()

      @tornado.web.asynchronous
      def send_message(self):
            if self.videostream.has_sent_frame():
                  import base64
                  self.ctxdrv.options['matrix'] = None
                  self.write_message("|".join([
                                               base64.b64encode(open(self.colormap_images[str(self.ctxdrv.options['colormap'])], 'rb').read()),
                                               " ".join(self.files),
                                               str(self.directory),
                                               str(tc.dict2json(self.ctxdrv.options)),
                                              ]))

      def start_server(self):
            self.videostream = FFMpegFlashVideoStream(port = self.application.video_port, size = (self.application.width, self.application.height))
            self.ctxdrv = Driver(port = self.application.port, callback = self.videostream.write, size = (self.application.width, self.application.height))
            self.ctxdrv.start()

      def close_server(self):
            self.ctxdrv.stop()
            self.videostream.close()

      def on_close(self):
            self.close_server()
            print "WebSocket closed"

class MainHandler(tornado.web.RequestHandler):
      """MainHandler()
      Tornado requires HTML to establish a server connection when using
      websockets.  This serves the html and javascript to the client. This
      function is extraneous when using IPython Notebook; then Ipython
      Notebook serves the html and javascript.
      """
      def get(self):
            import os
            f = open(os.path.dirname(__file__) + "/static/index.html")
            html = "<script>var websocket_port=%i; var video_port=%i; var width=%i; var height=%i;</script>%s" % \
                        (self.application.port, self.application.video_port, self.application.width, self.application.height, f.read())
            self.write(html)
            self.finish()

class Application(tornado.web.Application):
      """Application
      Tornado class that initializes handlers and other shared variables,
      most importantly the pipe.
      """
      def __init__(self, port = 8899, video_port = 8090, resolution = "480p", directory = None, quality = None):
            handlers = [
                  (r"/", MainHandler),
                  (r"/websocket", EchoWebSocket),
                  ]
            tornado.web.Application.__init__(self, handlers)
            if resolution == "240p":
                  self.width = 320
                  self.height = 240 # Rounded to nearest multiple of 16 for ffserver
            elif resolution == "360p":
                  self.width = 640
                  self.height = 352 # Rounded to nearest multiple of 16 for ffserver
            elif resolution == "480p":
                  self.width = 864
                  self.height = 480
            elif resolution == "720p":
                  self.width = 1280
                  self.height = 720
            elif resolution == "1080p":
                  self.width = 1920
                  self.height = 1088 # Rounded to nearest multiple of 16 for ffserver
            elif resolution == "400p":
                  self.width = 400
                  self.height = 400
            else:
                  self.width = 864
                  self.height = 480

            self.quality = quality
            self.port    = port
            self.video_port = video_port
            self.directory = directory
            print "Waiting for WebSocket..."

class StartServer(multiprocessing.Process):
      """StartServer()
      Start a new server on a new Process for the time being.  Have trouble
      closing the server when it only exists on a seperate thread.
      """
      def __init__(self, port = 8899, video_port = 8090, resolution = "480p", directory = None):
            multiprocessing.Process.__init__(self)
            self.port       = port
            self.video_port = video_port
            self.resolution = resolution
            self.directory  = directory

      def run(self):
            application = Application(port = self.port, video_port = self.video_port, resolution = self.resolution, directory = self.directory)
            application.listen(self.port)
            tornado.ioloop.IOLoop.instance().start()

