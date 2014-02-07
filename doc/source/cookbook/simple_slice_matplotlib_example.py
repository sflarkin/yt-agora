from yt.mods import *

# Load the dataset.
pf = load("GasSloshing/sloshing_nomag2_hdf5_plt_cnt_0150")

# Create a slice object
slc = SlicePlot(pf,'x','Density',width=(800.0,'kpc'))

# Get a reference to the matplotlib axes object for the plot
ax = slc.plots['Density'].axes

# Let's adjust the x axis tick labels
for label in ax.xaxis.get_ticklabels():
    label.set_color('red')
    label.set_rotation(45)
    label.set_fontsize(16)

# Get a reference to the matplotlib figure object for the plot
fig = slc.plots['Density'].figure

rect = (0.2,0.2,0.2,0.2)
new_ax = fig.add_axes(rect)

n, bins, patches = new_ax.hist(na.random.randn(1000)+20, 50,
    facecolor='yellow', edgecolor='yellow')
new_ax.set_xlabel('Dinosaurs per furlong')

slc.save()
