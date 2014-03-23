from yt.mods import * # set up our namespace
import matplotlib.colorbar as cb
from matplotlib.colors import LogNorm

fn = "GasSloshing/sloshing_nomag2_hdf5_plt_cnt_0150" # parameter file to load
orient = 'horizontal'

pf = load(fn) # load data

# There's a lot in here:
#   From this we get a containing figure, a list-of-lists of axes into which we
#   can place plots, and some axes that we'll put colorbars.
# We feed it:
#   Number of plots on the x-axis, number of plots on the y-axis, and how we
#   want our colorbars oriented.  (This governs where they will go, too.
#   bw is the base-width in inches, but 4 is about right for most cases.
fig, axes, colorbars = get_multi_plot(3, 2, colorbar=orient, bw = 4)

slc = pf.slice(2, 0.0, fields=["density","temperature","velocity_magnitude"], 
                 center=pf.domain_center)
proj = pf.proj(2, "density", weight_field="density", center=pf.domain_center)

slc_frb = slc.to_frb((1.0, "mpc"), 512)
proj_frb = proj.to_frb((1.0, "mpc"), 512)

dens_axes = [axes[0][0], axes[1][0]]
temp_axes = [axes[0][1], axes[1][1]]
vels_axes = [axes[0][2], axes[1][2]]

for dax, tax, vax in zip(dens_axes, temp_axes, vels_axes) :

    dax.xaxis.set_visible(False)
    dax.yaxis.set_visible(False)
    tax.xaxis.set_visible(False)
    tax.yaxis.set_visible(False)
    vax.xaxis.set_visible(False)
    vax.yaxis.set_visible(False)

plots = [dens_axes[0].imshow(slc_frb["density"], origin='lower', norm=LogNorm()),
         dens_axes[1].imshow(proj_frb["density"], origin='lower', norm=LogNorm()),
         temp_axes[0].imshow(slc_frb["temperature"], origin='lower'),    
         temp_axes[1].imshow(proj_frb["temperature"], origin='lower'),
         vels_axes[0].imshow(slc_frb["velocity_magnitude"], origin='lower', norm=LogNorm()),
         vels_axes[1].imshow(proj_frb["velocity_magnitude"], origin='lower', norm=LogNorm())]
         
plots[0].set_clim((1.0e-27,1.0e-25))
plots[0].set_cmap("bds_highcontrast")
plots[1].set_clim((1.0e-27,1.0e-25))
plots[1].set_cmap("bds_highcontrast")
plots[2].set_clim((1.0e7,1.0e8))
plots[2].set_cmap("hot")
plots[3].set_clim((1.0e7,1.0e8))
plots[3].set_cmap("hot")
plots[4].set_clim((1e6, 1e8))
plots[4].set_cmap("gist_rainbow")
plots[5].set_clim((1e6, 1e8))
plots[5].set_cmap("gist_rainbow")

titles=[r'$\mathrm{Density}\ (\mathrm{g\ cm^{-3}})$', 
        r'$\mathrm{temperature}\ (\mathrm{K})$',
        r'$\mathrm{VelocityMagnitude}\ (\mathrm{cm\ s^{-1}})$']

for p, cax, t in zip(plots[0:6:2], colorbars, titles):
    cbar = fig.colorbar(p, cax=cax, orientation=orient)
    cbar.set_label(t)

# And now we're done! 
fig.savefig("%s_3x2" % pf)
