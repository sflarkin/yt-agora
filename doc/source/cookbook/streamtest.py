#import the stream server from yt
#the server is dependent on theia which depends on pyCUDA
#this means an Nvidia card is required
from yt.visualization.volume_rendering.streaming.stream import StreamServer

#create a streaming object
#port specifies the forwarded connection to the local web browser
#video_port specifies the port that ffserver will use to stream the flash video
#resolution defines the size of the video to be streamed
#directory specifies where the numpy array (.npy) files are that will be 
#visualized in the streraming
scam = StreamServer(port = 8899, video_port = 8090, resolution = "360p", directory="/data/")

#start the stream server
scam.stream()

#In this example the user can open a web browser with HTML5 and flash support
# connect to the server that will perform the streaming forwarding ports through ssh
# eg. ssh -L8899:localhost:8899 -L8090:localhost:8090
# Then in the web browser url enter : localhost:8899
