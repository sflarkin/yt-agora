.. _changelog:

ChangeLog
=========


This is a non-comprehensive log of changes to the code.

Contributors
------------

Here are all of the contributors to the code base, in alphabetical order.

 * Tom Abel
 * David Collins
 * Brian Crosby
 * Andrew Cunningham
 * Hilary Egan
 * John Forbes
 * Nathan Goldbaum
 * Markus Haider
 * Cameron Hummels
 * Christian Karch
 * Ji-hoon Kim
 * Steffen Klemer
 * Kacper Kowalik
 * Michael Kuhlen
 * Eve Lee
 * Sam Leitner
 * Yuan Li
 * Chris Malone
 * Josh Moloney
 * Chris Moody
 * Andrew Myers
 * Jill Naiman
 * Kaylea Nelson
 * Jeﬀ Oishi
 * Jean-Claude Passy
 * Mark Richardson
 * Thomass Robitaille
 * Anna Rosen
 * Douglas Rudd
 * Anthony Scopatz
 * Noel Scudder
 * Devin Silvia
 * Sam Skillman
 * Stephen Skory
 * Britton Smith
 * Geoffrey So
 * Casey Stark
 * Elizabeth Tasker
 * Stephanie Tonnesen
 * Matthew Turk
 * Rick Wagner
 * Andrew Wetzel
 * John Wise
 * John ZuHone

Version 2.6
-----------

This is a scheduled release, bringing to a close the development in the 2.5
series.  Below are the itemized, aggregate changes since version 2.5.

Major changes:

  * yt is now licensed under the 3-clause BSD license.
  * HEALPix has been removed for the time being, as a result of licensing
    incompatibility.
  * The addition of a frontend for the Pluto code
  * The addition of an OBJ exporter to enable transparent and multi-surface
    exports of surfaces to Blender and Sketchfab
  * New absorption spectrum analysis module with documentation
  * Adding ability to draw lines with Grey Opacity in volume rendering
  * Updated physical constants to reflect 2010 CODATA data
  * Dependency updates (including IPython 1.0)
  * Better notebook support for yt plots
  * Considerably (10x+) faster kD-tree building for volume rendering
  * yt can now export to RADMC3D
  * Athena frontend now supports Static Mesh Refinement and units (
    http://hub.yt-project.org/nb/7l1zua )
  * Fix long-standing bug for plotting arrays with range of zero
  * Adding option to have interpolation based on non-uniform bins in
    interpolator code
  * Upgrades to most of the dependencies in the install script
  * ProjectionPlot now accepts a data_source keyword argument

Minor or bugfix changes:

  * Fix for volume rendering on the command line
  * map_to_colormap will no longer return out-of-bounds errors
  * Fixes for dds in covering grid calculations
  * Library searching for build process is now more reliable
  * Unit fix for "VorticityGrowthTimescale" field
  * Pyflakes stylistic fixes
  * Number density added to FLASH
  * Many fixes for Athena frontend
  * Radius and ParticleRadius now work for reduced-dimensionality datasets
  * Source distributions now work again!
  * Athena data now 64 bits everywhere
  * Grids displays on plots are now shaded to reflect the level of refinement
  * show_colormaps() is a new function for displaying all known colormaps
  * PhasePlotter by default now adds a colormap.
  * System build fix for POSIX systems
  * Fixing domain offsets for halo centers-of-mass
  * Removing some Enzo-specific terminology in the Halo Mass Function
  * Addition of coordinate vectors on volume render
  * Pickling fix for extracted regions
  * Addition of some tracer particle annotation functions
  * Better error message for "yt" command
  * Fix for radial vs poloidal fields
  * Piernik 2D data handling fix
  * Fixes for FLASH current redshift
  * PlotWindows now have a set_font function and a new default font setting
  * Colorbars less likely to extend off the edge of a PlotWindow
  * Clumps overplotted on PlotWindows are now correctly contoured
  * Many fixes to light ray and profiles for integrated cosmological analysis
  * Improvements to OpenMP compilation
  * Typo in value for km_per_pc (not used elsewhere in the code base) has been
    fixed
  * Enable parallel IPython notebook sessions (
    http://hub.yt-project.org/nb/qgn19h )
  * Change (~1e-6) to particle_density deposition, enabling it to be used by
    FLASH and other frontends
  * Addition of is_root function for convenience in parallel analysis sessions
  * Additions to Orion particle reader
  * Fixing TotalMass for case when particles not present
  * Fixing the density threshold or HOP and pHOP to match the merger tree
  * Reason can now plot with latest plot window
  * Issues with VelocityMagnitude and aliases with velo have been corrected in
    the FLASH frontend
  * Halo radii are calculated correctly for domains that do not start at 0,0,0.
  * Halo mass function now works for non-Enzo frontends.
  * Bug fixes for directory creation, typos in docstrings
  * Speed improvements to ellipsoidal particle detection
  * Updates to FLASH fields
  * CASTRO frontend bug fixes
  * Fisheye camera bug fixes
  * Answer testing now includes plot window answer testing
  * Athena data serialization
  * load_uniform_grid can now decompose dims >= 1024.  (#537)
  * Axis unit setting works correctly for unit names  (#534)
  * ThermalEnergy is now calculated correctly for Enzo MHD simulations (#535)
  * Radius fields had an asymmetry in periodicity calculation (#531)
  * Boolean regions can now be pickled (#517)

Version 2.5
-----------

Many below-the-surface changes happened in yt 2.5 to improve reliability,
fidelity of the answers, and streamlined user interface.  The major change in
this release has been the immense expansion in testing of yt.  We now have over
2000 unit tests (run on every commit, thanks to both Kacper Kowalik and Shining
Panda) as well as answer testing for FLASH, Enzo, Chombo and Orion data.

The Stream frontend, which can construct datasets in memory, has been improved
considerably.  It's now easier than ever to load data from disk.  If you know
how to get volumetric data into Python, you can use either the
``load_uniform_grid`` function or the ``load_amr_grid`` function to create an
in-memory dataset that yt can analyze.

yt now supports the Athena code.

yt is now focusing on providing first class support for the IPython notebook.
In this release, plots can be displayed inline.  The Reason HTML5 GUI will be
merged with the IPython notebook in a future release.

Install Script Changes:
~~~~~~~~~~~~~~~~~~~~~~~

 * SciPy can now be installed
 * Rockstar can now be installed
 * Dependencies can be updated with "yt update --all"
 * Cython has been upgraded to 0.17.1
 * Python has been upgraded to 2.7.3
 * h5py has been upgraded to 2.1.0
 * hdf5 has been upgraded to 1.8.9
 * matplotlib has been upgraded to 1.2.0
 * IPython has been upgraded to 0.13.1
 * Forthon has been upgraded to 0.8.10
 * nose has been added
 * sympy has been added
 * python-hglib has been added

We've also improved support for installing on OSX, Ubuntu and OpenSUSE.

Most Visible Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~

 * Nearly 200 pull requests and over 1000 changesets have been merged since yt
   2.4 was release on August 2nd, 2012.
 * numpy is now imported as np, not na.  na will continue to work for the
   foreseeable future.
 * You can now get a `yt cheat sheet <http://yt-project.org/docs/2.5/cheatsheet.pdf>`!
 * yt can now load simulation data created by Athena.
 * The Rockstar halo finder can now be installed by the install script
 * SciPy can now be installed by the install script
 * Data can now be written out in two ways:

   * Sidecar files containing expensive derived fields can be written and
     implicitly loaded from.
   * GDF files, which are portable yt-specific representations of full
     simulations, can be created from any dataset.  Work is underway on
     a pure C library that can be linked against to load these files into
     simulations.

 * The "Stream" frontend, for loading raw data in memory, has been greatly
   expanded and now includes initial conditions generation functionality,
   particle fields, and simple loading of AMR grids with ``load_amr_grids``.
 * Spherical and Cylindrical fields have been sped up and made to have a
   uniform interface.  These fields can be the building blocks of more advanced
   fields.
 * Coordinate transformations have been sped up and streamlined. It is now
   possible to convert any scalar or vector field to a new cartesian, spherical,
   or cylindrical coordinate system with an arbitrary orientation. This makes it
   possible to do novel analyses like profiling the toroidal and poloidal
   velocity as a function of radius in an inclined disk.
 * Many improvements to the EnzoSimulation class, which can now find many
   different types of data.
 * Image data is now encapsulated in an ImageArray class, which carries with it
   provenance information about its trajectory through yt.
 * Streamlines now query at every step along the streamline, not just at every
   cell.
 * Surfaces can now be extracted and examined, as well as uploaded to
   Sketchfab.com for interactive visualization in a web browser.
 * allsky_projection can now accept a datasource, making it easier to cut out
   regions to examine.
 * Many, many improvements to PlotWindow.  If you're still using
   PlotCollection, check out ``ProjectionPlot``, ``SlicePlot``,
   ``OffAxisProjectionPlot`` and ``OffAxisSlicePlot``.
 * PlotWindow can now accept a timeseries instead of a dataset.
 * Many fixes for 1D and 2D data, especially in FLASH datasets.
 * Vast improvements to the particle file handling for FLASH datasets.
 * Particles can now be created ex nihilo with CICSample_3.
 * Rockstar halo finding is now a targeted goal.  Support for using Rockstar
   has improved dramatically.
 * Increased support for tracking halos across time using the FOF halo finder.
 * The command ``yt notebook`` has been added to spawn an IPython notebook
   server, and the ``yt.imods`` module can replace ``yt.mods`` in the IPython
   Notebook to enable better integration.
 * Metallicity-dependent X-ray fields have now been added.
 * Grid lines can now be added to volume renderings.
 * Volume rendering backend has been updated to use an alpha channel, fixing
   parallel opaque volume renderings.  This also enables easier blending of 
   multiple images and annotations to the rendering. Users are encouraged
   to look at the capabilities of the ``ImageArray`` for writing out renders,
   as updated in the cookbook examples. Volume renders can now be saved with
   an arbitrary background color.
 * Periodicity, or alternately non-periodicity, is now a part of radius
   calculations.
 * The AMRKDTree has been rewritten.  This allows parallelism with other than 
   power-of-2 MPI processes, arbitrary sets of grids, and splitting of
   unigrids. 
 * Fixed Resolution Buffers and volume rendering images now utilize a new 
   ImageArray class that stores information such as data source, field names,
   and other information in a .info dictionary. See the ``ImageArray``
   docstrings for more information on how they can be used to save to a bitmap
   or hdf5 file.

Version 2.4
-----------

The 2.4 release was particularly large, encompassing nearly a thousand
changesets and a number of new features.

To help you get up to speed, we've made an IPython notebook file demonstrating
a few of the changes to the scripting API.  You can
`download it here <http://yt-project.org/files/yt24.ipynb>`_.

Most Visible Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~

 * Threaded volume renderer, completely refactored from the ground up for
   speed and parallelism.
 * The Plot Window (see :ref:`simple-inspection`) is now fully functional!  No
   more PlotCollections, and full, easy access to Matplotlib axes objects.
 * Many improvements to Time Series analysis:
    * EnzoSimulation now integrates with TimeSeries analysis!
    * Auto-parallelization of analysis and parallel iteration
    * Memory usage when iterating over datasets reduced substantially
 * Many improvements to Reason, the yt GUI
    * Addition of "yt reason" as a startup command
    * Keyboard shortcuts in projection & slice mode: z, Z, x, X for zooms,
      hjkl, HJKL for motion
    * Drag to move in projection & slice mode
    * Contours and vector fields in projection & slice mode
    * Color map selection in projection & slice mode
    * 3D Scene
 * Integration with the all new yt Hub ( http://hub.yt-project.org/ ): upload
   variable resolution projections, slices, project information, vertices and
   plot collections right from the yt command line!

Other Changes
~~~~~~~~~~~~~

 * :class:`~yt.visualization.plot_window.ProjectionPlot` and 
   :class:`~yt.visualization.plot_window.SlicePlot` supplant the functionality
   of PlotCollection.
 * Camera path creation from keyframes and splines
 * Ellipsoidal data containers and ellipsoidal parameter calculation for halos
 * PyX and ZeroMQ now available in the install script
 * Consolidation of unit handling
 * HDF5 updated to 1.8.7, Mercurial updated to 2.2, IPython updated to 0.12
 * Preview of integration with Rockstar halo finder
 * Improvements to merger tree speed and memory usage
 * Sunrise exporter now compatible with Sunrise 4.0
 * Particle trajectory calculator now available!
 * Speed and parallel scalability improvements in projections, profiles and HOP
 * New Vorticity-related fields
 * Vast improvements to the ART frontend
 * Many improvements to the FLASH frontend, including full parameter reads,
   speedups, and support for more corner cases of FLASH 2, 2.5 and 3 data.
 * Integration of the Grid Data Format frontend, and a converter for Athena
   data to this format.
 * Improvements to command line parsing
 * Parallel import improvements on parallel filesystems
   (``from yt.pmods import *``)
 * proj_style keyword for projections, for Maximum Intensity Projections
   (``proj_style = "mip"``)
 * Fisheye rendering for planetarium rendering
 * Profiles now provide \*_std fields for standard deviation of values
 * Generalized Orientation class, providing 6DOF motion control
 * parallel_objects iteration now more robust, provides optional barrier.
   (Also now being used as underlying iteration mechanism in many internal
   routines.)
 * Dynamic load balancing in parallel_objects iteration.
 * Parallel-aware objects can now be pickled.
 * Many new colormaps included
 * Numerous improvements to the PyX-based eps_writer module
 * FixedResolutionBuffer to FITS export.
 * Generic image to FITS export.
 * Multi-level parallelism for extremely large cameras in volume rendering
 * Light cone and light ray updates to fit with current best practices for
   parallelism

Version 2.3 
-----------

`(yt 2.3 docs) <http://yt-project.org/docs/2.3>`_
 * Multi-level parallelism
 * Real, extensive answer tests
 * Boolean data regions (see :ref:`boolean_data_objects`)
 * Isocontours / flux calculations (see :ref:`extracting-isocontour-information`)
 * Field reorganization (see :ref:`types_of_fields`)
 * PHOP memory improvements
 * Bug fixes for tests
 * Parallel data loading for RAMSES, along with other speedups and improvements
   there
 * WebGL interface for isocontours and a pannable map widget added to Reason
 * Performance improvements for volume rendering
 * Adaptive HEALPix support
 * Column density calculations (see :ref:`radial-column-density`)
 * Massive speedup for 1D profiles
 * Lots more, bug fixes etc.
 * Substantial improvements to the documentation, including
   :ref:`manual-plotting` and a revamped orientation.

Version 2.2
-----------

`(yt 2.2 docs) <http://yt-project.org/docs/2.2>`_
 * Command-line submission to the yt Hub (http://hub.yt-project.org/)
 * Initial release of the web-based GUI Reason, designed for efficient remote
   usage over SSH tunnels
 * Absorption line spectrum generator for cosmological simulations (see
   :ref:`absorption_spectrum`)
 * Interoperability with ParaView for volume rendering, slicing, and so forth
 * Support for the Nyx code
 * An order of magnitude speed improvement in the RAMSES support
 * Quad-tree projections, speeding up the process of projecting by up to an
   order of magnitude and providing better load balancing
 * “mapserver” for in-browser, Google Maps-style slice and projection
   visualization (see :ref:`mapserver`)
 * Many bug fixes and performance improvements
 * Halo loader (see :ref:`load_haloes`)

Version 2.1
-----------

`(yt 2.1 docs) <http://yt-project.org/docs/2.1>`_
 * HEALPix-based volume rendering for 4pi, allsky volume rendering
 * libconfig is now included
 * SQLite3 and Forthon now included by default in the install script
 * Development guide has been lengthened substantially and a development
   bootstrap script (:ref:`bootstrap-dev`) is now included.
 * Installation script now installs Python 2.7 and HDF5 1.8.6
 * iyt now tab-completes field names
 * Halos can now be stored on-disk much more easily between HaloFinding runs.
 * Halos found inline in Enzo can be loaded and merger trees calculated
 * Support for CASTRO particles has been added
 * Chombo support updated and fixed
 * New code contributions 
 * Contour finder has been sped up by a factor of a few
 * Constrained two-point functions are now possible, for LOS power spectra
 * Time series analysis (:ref:`time-series-analysis`) now much easier
 * Stream Lines now a supported 1D data type (:class:`AMRStreamlineBase`)
 * Stream Lines now able to be calculated and plotted (:ref:`streamlines`)
 * In situ Enzo visualization now much faster
 * "gui" source directory reorganized and cleaned up
 * Cython now a compile-time dependency, reducing the size of source tree
   updates substantially
 * ``yt-supplemental`` repository now checked out by default, containing
   cookbook, documentation, handy mercurial extensions, and advanced plotting
   examples and helper scripts.
 * Pasteboards now supported and available 
 * Parallel yt efficiency improved by removal of barriers and improvement of
   collective operations

Version 2.0
-----------

 * Major reorganization of the codebase for speed, ease of modification, and maintainability
 * Re-organization of documentation and addition of Orientation Session
 * Support for FLASH code
 * Preliminary support for MAESTRO, CASTRO, ART, and RAMSES (contributions welcome!)
 * Perspective projection for volume rendering
 * Exporting to Sunrise
 * Preliminary particle rendering in volume rendering visualization
 * Drastically improved parallel volume rendering, via kD-tree decomposition
 * Simple merger tree calculation for FOF catalogs
 * New and greatly expanded documentation, with a "source" button

Version 1.7
-----------

 * Direct writing of PNGs
 * Multi-band image writing
 * Parallel halo merger tree (see :ref:`merger_tree`)
 * Parallel structure function generator (see :ref:`two_point_functions`)
 * Image pan and zoom object and display widget.
 * Parallel volume rendering (see :ref:`volume_rendering`)
 * Multivariate volume rendering, allowing for multiple forms of emission and
   absorption, including approximate scattering and Planck emissions. (see
   :ref:`volume_rendering`)
 * Added Camera interface to volume rendering (See :ref:`volume_rendering`)
 * Off-axis projection (See :ref:`volume_rendering`)
 * Stereo (toe-in) volume rendering (See :ref:`volume_rendering`)
 * DualEPS extension for better EPS construction
 * yt now uses Distribute instead of SetupTools
 * Better ``iyt`` initialization for GUI support
 * Rewritten, memory conservative and speed-improved contour finding algorithm
 * Speed improvements to volume rendering
 * Preliminary support for the Tiger code
 * Default colormap is now ``algae``
 * Lightweight projection loading with ``projload``
 * Improvements to `yt.data_objects.time_series`
 * Improvements to :class:`yt.extensions.EnzoSimulation` (See
   :ref:`analyzing-an-entire-simulation`)
 * Removed ``direct_ray_cast``
 * Fixed bug causing double data-read in projections
 * Added Cylinder support to ParticleIO
 * Fixes for 1- and 2-D Enzo datasets
 * Preliminary, largely non-functional Gadget support
 * Speed improvements to basic HOP
 * Added physical constants module
 * Beginning to standardize and enforce docstring requirements, changing to
   ``autosummary``-based API documentation.

Version 1.6.1
-------------

 * Critical fixes to ParticleIO
 * Halo mass function fixes for comoving coordinates
 * Fixes to halo finding
 * Fixes to the installation script
 * "yt instinfo" command to report current installation information as well as
   auto-update some types of installations
 * Optimizations to the volume renderer (2x-26x reported speedups)

Version 1.6
-----------

Version 1.6 is a point release, primarily notable for the new parallel halo
finder (see :ref:`halo_finding`)

 * (New) Parallel HOP ( http://arxiv.org/abs/1001.3411 , :ref:`halo_finding` )
 * (Beta) Software ray casting and volume rendering
   (see :ref:`volume_rendering`)
 * Rewritten, faster and better contouring engine for clump identification
 * Spectral Energy Distribution calculation for stellar populations
   (see :ref:`synthetic_spectrum`)
 * Optimized data structures such as the index
 * Star particle analysis routines
   (see :ref:`star_analysis`)
 * Halo mass function routines (see :ref:`hmf_howto`)
 * Completely rewritten, massively faster and more memory efficient Particle IO
 * Fixes for plots, including normalized phase plots
 * Better collective communication in parallel routines
 * Consolidation of optimized C routines into ``amr_utils``
 * Many bug fixes and minor optimizations 

Version 1.5
-----------

Version 1.5 features many new improvements, most prominently that of the
addition of parallel computing abilities (see :ref:`parallel-computation`) and
generalization for multiple AMR data formats, specifically both Enzo and Orion.

 * Rewritten documentation
 * Fully parallel slices, projections, cutting planes, profiles,
   quantities
 * Parallel HOP
 * Friends-of-friends halo finder
 * Object storage and serialization
 * Major performance improvements to the clump finder (factor of five)
 * Generalized domain sizes
 * Generalized field info containers
 * Dark Matter-only simulations
 * 1D and 2D simulations
 * Better IO for HDF5 sets
 * Support for the Orion AMR code
 * Spherical re-gridding
 * Halo profiler
 * Disk image stacker
 * Light cone generator
 * Callback interface improved
 * Several new callbacks
 * New data objects -- ortho and non-ortho rays, limited ray-tracing
 * Fixed resolution buffers
 * Spectral integrator for CLOUDY data
 * Substantially better interactive interface
 * Performance improvements *everywhere*
 * Command-line interface to *many* common tasks
 * Isolated plot handling, independent of PlotCollections

Version 1.0
-----------

 * Initial release!
