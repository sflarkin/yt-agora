from yt.mods import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

fn = "IsolatedGalaxy/galaxy0030/galaxy0030"
pf = load(fn) # load data

fig = plt.figure()

# See http://matplotlib.org/mpl_toolkits/axes_grid/api/axes_grid_api.html
# These choices of keyword arguments produce two colorbars, both drawn on the
# right hand side.  This means there are only two colorbar axes, one for Density
# and another for Temperature.  In addition, axes labels will be drawn for all
# plots.
grid = AxesGrid(fig, (0.075,0.075,0.85,0.85),
                nrows_ncols = (2, 2),
                axes_pad = 1.0,
                label_mode = "all",
                share_all = True,
                cbar_location="right",
                cbar_mode="edge",
                cbar_size="5%",
                cbar_pad="0%")

cuts = ['x', 'y', 'z', 'z']
fields = ['Density', 'Density', 'Density', 'Temperature']

for i, (direction, field) in enumerate(zip(cuts, fields)):
    # Load the data and create a single plot
    p = SlicePlot(pf, direction, field)
    p.zoom(40)

    # This forces the ProjectionPlot to redraw itself on the AxesGrid axes.
    plot = p.plots[field]
    plot.figure = fig
    plot.axes = grid[i].axes

    # Since there are only two colorbar axes, we need to make sure we don't try
    # to set the temperature colorbar to cbar_axes[4], which would if we used i
    # to index cbar_axes, yielding a plot without a Temperature colorbar.
    # This unecessarily redraws the Density colorbar three times, but that has
    # no effect on the final plot.
    if field == 'Density':
        plot.cax = grid.cbar_axes[0]
    elif field == 'Temperature':
        plot.cax = grid.cbar_axes[1]

    # Finally, redraw the plot.
    p._setup_plots()

plt.savefig('multiplot_2x2_coordaxes_slice.png')
