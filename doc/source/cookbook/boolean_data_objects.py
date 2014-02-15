from yt.mods import * # set up our namespace

pf = load("Enzo_64/DD0043/data0043") # load data
# Make a few data ojbects to start.
re1 = pf.h.region([0.5, 0.5, 0.5], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6])
re2 = pf.h.region([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.6, 0.6, 0.6])
sp1 = pf.h.sphere([0.5, 0.5, 0.5], 0.05)
sp2 = pf.h.sphere([0.1, 0.2, 0.3], 0.1)
# The "AND" operator. This will make a region identical to re2.
bool1 = pf.h.boolean([re1, "AND", re2])
xp = bool1["particle_position_x"]
# The "OR" operator. This will make a region identical to re1.
bool2 = pf.h.boolean([re1, "OR", re2])
# The "NOT" operator. This will make a region like re1, but with the corner
# that re2 covers cut out.
bool3 = pf.h.boolean([re1, "NOT", re2])
# Disjoint regions can be combined with the "OR" operator.
bool4 = pf.h.boolean([sp1, "OR", sp2])
# Find oddly-shaped overlapping regions.
bool5 = pf.h.boolean([re2, "AND", sp1])
# Nested logic with parentheses.
# This is re1 with the oddly-shaped region cut out.
bool6 = pf.h.boolean([re1, "NOT", "(", re1, "AND", sp1, ")"])
