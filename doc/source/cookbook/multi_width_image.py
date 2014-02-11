from yt.mods import *

# Load the dataset.
pf = load("IsolatedGalaxy/galaxy0030/galaxy0030")

# Create a slice plot for the dataset.  With no additional arguments,
# the width will be the size of the domain and the center will be the
# center of the simulation box
slc = SlicePlot(pf,2,'Density')

# Create a list of a couple of widths and units.
widths = [(1, 'mpc'),
          (15, 'kpc')]

# Loop through the list of widths and units.
for width, unit in widths:

    # Set the width.
    slc.set_width(width, unit)

    # Write out the image with a unique name.
    slc.save("%s_%010d_%s" % (pf, width, unit))

zoomFactors = [2,4,5]

# recreate the original slice
slc = SlicePlot(pf,2,'Density')

for zoomFactor in zoomFactors:

    # zoom in
    slc.zoom(zoomFactor)

    # Write out the image with a unique name.
    slc.save("%s_%i" % (pf, zoomFactor))
