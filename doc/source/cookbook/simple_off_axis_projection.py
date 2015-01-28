import yt

# Load the dataset.
ds = yt.load("IsolatedGalaxy/galaxy0030/galaxy0030")

# Create a 15 kpc radius sphere, centered on the center of the sim volume
sp = ds.sphere("center", (15.0, "kpc"))

# Get the angular momentum vector for the sphere.
L = sp.quantities.angular_momentum_vector()

print "Angular momentum vector: {0}".format(L)

# Create an OffAxisProjectionPlot of density centered on the object with the L 
# vector as its normal and a width of 25 kpc on a side
p = yt.OffAxisProjectionPlot(ds, L, "density", sp.center, (25, "kpc"))
p.save()
