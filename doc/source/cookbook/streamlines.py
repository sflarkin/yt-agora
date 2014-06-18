import yt
import numpy as np
from yt.visualization.api import Streamlines
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset 
ds = yt.load("IsolatedGalaxy/galaxy0030/galaxy0030")

# Define c: the center of the box, N: the number of streamlines, 
# scale: the spatial scale of the streamlines relative to the boxsize,
# and then pos: the random positions of the streamlines.
c = np.array([0.5]*3)
N = 100
scale = 1.0
pos_dx = np.random.random((N,3))*scale-scale/2.
pos = c+pos_dx

# Create the streamlines from these positions with the velocity fields as the 
# fields to be traced
streamlines = Streamlines(ds, pos, 'velocity_x', 'velocity_y', 'velocity_z', length=1.0) 
streamlines.integrate_through_volume()

# Create a 3D matplotlib figure for visualizing the streamlines
fig = pl.figure() 
ax = Axes3D(fig)

# Trace the streamlines through the volume of the 3D figure
for stream in streamlines.streamlines:
    stream = stream[ np.all(stream != 0.0, axis=1)]

    # Make the colors of each stream vary continuously from blue to red
    # from low-x to high-x of the stream start position (each color is R, G, B)
    color = (stream[0,0], 0, 1-stream[0,0])

    # Plot the stream in 3D
    ax.plot3D(stream[:,0], stream[:,1], stream[:,2], alpha=0.3, color=color)

# Save the figure
pl.savefig('streamlines.png')
