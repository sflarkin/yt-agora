"""StreamServer

This module provides users with a tool to live stream
interactive renderings into an html5 canvas. 

"""

from IPython.display import HTML
from yt.visualization.volume_rendering.streaming.streamserver.streamhandler import StartServer
import numpy as np
import yt.visualization.volume_rendering.streaming.streamserver.jpeg_server_html as jpeg_server_html

class StreamServer():
      r"""
      This is a stream server that will connect a tornado websocket to a
      gpu raycaster.  The input data is expected to be either a numpy array
      or any data structure that yt supports.

      Parameters
      ----------
      port   : int
            Specify websocket port.  (Requires ssh tunnel on this port)
      resolution  : string
            Specifies screen resolution returned by streamer.
      data : numpy array
            If data is set ds should be empty.  Contains numpy array with 
            unigrid volume data to be rendered.
      ds     : yt data container
            Any data set loaded using yt's load functions.
      """
      def __init__(self, port = 8899, video_port = 8090, resolution = None, data = None, ds = None, directory = None):
            self.resolution = resolution
            self.port   = port
            self.video_port = video_port
            self.directory = directory

      def stream(self):
            t = StartServer(port = self.port, video_port = self.video_port, resolution = self.resolution, directory = self.directory)
            t.start()
