import subprocess as sp
import numpy as np

class FFMpegOutputFile:
  def __init__(self, filename, size, bin="ffmpeg", debug=False):
    sx, sy = size
  
    self.pipe = sp.Popen([ bin,
      '-y', # (optional) overwrite the output file if it already exists
      '-f', 'rawvideo',
      '-vcodec','rawvideo',
      '-s', str(sx) + 'x' + str(sy), # size of one frame
      '-pix_fmt', 'bgr32',
      '-r', '29.97', # frames per second
      '-i', '-', # The input comes from a pipe
      '-an', # Tells FFMPEG not to expect any audio
      '-vcodec', 'dnxhd', '-b:v', '220M',
      '-threads', '4',
      str(filename) ],
      stdin=sp.PIPE,stdout=sp.PIPE)
    
  def write(self, array):
    array.tofile(self.pipe.stdin)
    
  def close(self):
    self.pipe.terminate()
    
    
class FFMpegOutputStream:
  def __init__(self, port, size, bin="ffmpeg", debug=False):
    sx, sy = size
  
    self.pipe = sp.Popen([ bin,
      '-y', # (optional) overwrite the output file if it already exists
      '-f', 'rawvideo',
      '-vcodec','rawvideo',
      '-s', str(sx) + 'x' + str(sy), # size of one frame
      '-pix_fmt', 'bgr32',
      '-r', '29.97', # frames per second
      '-i', '-', # The input comes from a pipe
      '-s', str(sx) + 'x' + str(sy),
      '-an', # Tells FFMPEG not to expect any audio
      '-vcodec', 'libx264',
      '-tune', 'zerolatency',
      '-b:v', '900k',
      '-f', 'rtp', 'rtp://127.0.0.1:' + str(port)],
      stdin=sp.PIPE,stdout=sp.PIPE)
  
  def write(self, array):
    array.tofile(self.pipe.stdin)
    
  def close(self):
    self.pipe.terminate()
    
class FFMpegMp4VideoStream:
      def __init__(self, filename, size, bin="ffmpeg", debug=False):
            self.debug = debug
            self.size = size
            sw, sh = size
            self.pipe = sp.Popen([ bin, '-y', # (optional) overwrite the output file if it already exists 
                              '-re', # Read frames as fast as possible
                              #'-loglevel', 'quiet', # Print nothing
                              '-f', 'rawvideo', '-vcodec','rawvideo', '-s', str(sw) + 'x' + str(sh), # size of one frame 
                              '-pix_fmt', 'rgba', #'-r', '60',
                              '-i', '-', # The imput comes from a pipe 
                              filename # Stream feed to ffserver
                              ], stdin=sp.PIPE,stdout=sp.PIPE)

      def write(self, array):
            array.tofile(self.pipe.stdin)

      def close(self):
            self.pipe.terminate()

class FFMpegJpegStream:
      def __init__(self, size, bin="ffmpeg", debug=False):
            self.debug = debug
            self.sw, self.sh = size
            self.bin = bin
            """
            self.pipe = sp.Popen([ bin, '-y', # (optional) overwrite the output file if it already exists 
                              '-s', str(self.sw) + 'x' + str(self.sh), # size of one frame 
                              '-pix_fmt', 'rgba',
                              '-i', '-', # The imput comes from a pipe 
                              '/home/nihasmit/test.jpg',
                              ], stdin=sp.PIPE,stdout=sp.PIPE)
            self.pipe = sp.Popen([ bin, '-y',
                              '-vcodec', 'rawvideo',
                              '-f', 'rawvideo',
                              '-pix_fmt', 'rgba',
                              '-s', str(sw) + 'x' + str(sh), # size of one frame 
                              '-i', '-', # The imput comes from a pipe 
                              '-f', 'image2',
                              '-vcodec', 'mjpeg',
                              '/home/nihasmit/asdf.jpg',
                              ], stdin=sp.PIPE,stdout=sp.PIPE)
            """

      def write(self, array, frame, name):
            # TODO: Temp way of writing to jpg
            array.tofile('/tmp/theia-curr-scene.rgb')
            sp.call([ self.bin, '-y', # (optional) overwrite the output file if it already exists 
                              '-s', str(self.sw) + 'x' + str(self.sh), # size of one frame 
                              '-pix_fmt', 'rgba',
                              '-i', '/tmp/theia-curr-scene.rgb', # The imput comes from a pipe 
                              name,
                              ])

class FFMpegFlashVideoStream:
      def __init__(self, port, size, bin="ffmpeg", debug=False):
            bitrate = 100000
            self.server = FFServer(port, size, bitrate = bitrate, debug=debug)
            self.debug = debug
            self.frame_sent = False
            self.size = size
            sw, sh = size
            self.pipe = sp.Popen([ bin, '-y', # (optional) overwrite the output file if it already exists 
                              '-re', # Read frames as fast as possible
                              #'-loglevel', 'quiet', # Print nothing
                              '-f', 'rawvideo', '-vcodec','rawvideo', '-s', str(sw) + 'x' + str(sh), # size of one frame 
                              '-pix_fmt', 'rgba', #'-r', '60',
                              '-i', '-', # The imput comes from a pipe 
                              '-c:v', 'flv', # Video codec
                              #'-c:v', 'libx264', # Video codec
                              '-pix_fmt', 'yuv420p',
                              '-b:v', str(bitrate) ,
                              '-an', # Tells FFMPEG not to expect any audio 
                              'http://localhost:' + str(port) + '/feed1.ffm' # Stream feed to ffserver
                              ], stdin=sp.PIPE,stdout=sp.PIPE)

      def write(self, array, frame):
            self.frame_sent = True
            array.tofile(self.pipe.stdin)

      def has_sent_frame(self):
            return self.frame_sent

      def close(self):
            self.pipe.terminate()
            self.server.close()


class FFMpegHTML5VideoStream:
      def __init__(self, port, size, bin="ffmpeg", debug=False):
            self.server = FFServer(port, size, debug=False)
            self.debug = debug
            self.frame_sent = False
            sw, sh = size
#ffmpeg -i "concat:/home/user/Video/VTS_01_1.VOB|/home/user/Video/VTS_01_2.VOB" -vcodec libx264 -vprofile high -preset slow -b:v 500k -maxrate 500k -bufsize 1000k -vf scale=-1:480 -threads 0 -pass 2 -acodec libvo_aacenc -b:a 128k -f mp4 output_file.mp4

            self.pipe = sp.Popen([ bin, '-y', # (optional) overwrite the output file if it already exists 
                              #'-loglevel', 'quiet', # Print nothing
                              '-f', 'rawvideo', '-vcodec','rawvideo', '-s', str(sw) + 'x' + str(sh), # size of one frame 
                              '-pix_fmt', 'rgba', #'-r', '60',
                              '-i', '-', # The imput comes from a pipe 
                              '-c:v', 'libx264', # Video codec
                              #'-preset', 'ultrafast', # Encoding speed (consequence lack of quality)
                              #'-tune', 'zerolatency',
                              #'-crf', '30',
                              #'-pix_fmt', 'yuv420p',
                              #'-b:v', '555k',
                              #'-bufsize', '3000k',
                              #'-pass', '1',
                              '-an', # Tells FFMPEG not to expect any audio 
                              #'-movflags', '+faststart',
                              #'-r', '60',
                              #'-f', 'mp4',
                              'rtmp://localhost:' + str(port) + '/feed1.ffm' # Stream feed to ffserver
                              ], stdin=sp.PIPE,stdout=sp.PIPE) # was 'o.mp4'

      def write(self, array, frame):
            self.frame_sent = True
            array.tofile(self.pipe.stdin)

      def has_sent_frame(self):
            return self.frame_sent

      def close(self):
            self.pipe.terminate()
            self.server.terminate()


class FFServer:
      def __init__(self, port, size, bitrate = 555, bin="ffserver", debug="False"):
            import os
            file_path = os.path.dirname(__file__) + "/tmp.conf"
            w, h = size
            str_size = str(w) + 'x' + str(h)
            header = """
                        Port """ + str(port) + """
                        BindAddress 0.0.0.0
                        MaxHTTPConnections 2000
                        MaxClients 1000
                        MaxBandwidth 1200000
                        CustomLog -
                        NoDaemon
                        """
            feed = """
                        <Feed feed1.ffm>
                        File ./feed1.ffm #when remarked, no file is beeing created and the stream keeps working!!
                        FileMaxSize 200M
                        ACL allow 127.0.0.1
                        </Feed>
                        """
            swf = """
                        <Stream test.swf>
                        Feed feed1.ffm
                        Format swf
                        VideoCodec flv
                        VideoFrameRate 30
                        VideoBitRate """ + str(bitrate) + """
                        VideoQMin 1
                        VideoQMax 3
                        VideoSize """ + str_size + """
                        NoAudio
                        </Stream>
                        """
            flv = """
                        <Stream test.flv>
                        Feed feed1.ffm
                        Format flv
                        VideoCodec flv
                        VideoFrameRate 30
                        VideoBitRate """ + str(bitrate) + """
                        VideoQMin 1
                        VideoQMax 3
                        VideoSize """ + str_size + """
                        NoAudio
                        </Stream>
                        """
            mp4 = """
                        <Stream test.mp4>
                        Feed feed1.ffm
                        Format mp4
                        VideoCodec libx264
                        VideoFrameRate 30
                        VideoBitRate """ + str(bitrate) + """
                        VideoQMin 1
                        VideoQMax 3
                        VideoSize """ + str_size + """
                        NoAudio
                        </Stream>
                        """
            stat = """
                        <Stream stat.html>
                        Format status
                        </Stream>
                        """
            redirect = """
                        <Redirect index.html>
                        URL http://ffmpeg.sourceforge.net/
                        </Redirect>
                        """
            tmp = open(file_path, 'w')
            tmp.write(header + feed + swf + flv + stat + redirect)
            tmp.close()
            f = os.path.dirname(__file__) + "/flash.conf"
            self.server = sp.Popen([ "ffserver", "-f", file_path])
            
      def close(self):
            self.server.terminate()

