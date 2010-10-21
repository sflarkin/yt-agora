"""
A means of running standalone commands with a shared set of options.

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2008-2009 Matthew Turk.  All Rights Reserved.

  This file is part of yt.

  yt is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from yt.mods import *
from yt.funcs import *
import cmdln as cmdln
import optparse, os, os.path, math, sys, time, subprocess

def _fix_pf(arg):
    if os.path.isdir("%s" % arg) and \
        os.path.exists("%s/%s" % (arg,arg)):
        pf = load("%s/%s" % (arg,arg))
    elif os.path.isdir("%s.dir" % arg) and \
        os.path.exists("%s.dir/%s" % (arg,arg)):
        pf = load("%s.dir/%s" % (arg,arg))
    elif arg.endswith(".hierarchy"):
        pf = load(arg[:-10])
    else:
        pf = load(arg)
    if pf is None:
        raise IOError
    return pf

_common_options = dict(
    axis    = dict(short="-a", long="--axis",
                   action="store", type="int",
                   dest="axis", default=4,
                   help="Axis (4 for all three)"),
    log     = dict(short="-l", long="--log",
                   action="store_true",
                   dest="takelog", default=True,
                   help="Take the log of the field?"),
    text    = dict(short="-t", long="--text",
                   action="store", type="string",
                   dest="text", default=None,
                   help="Textual annotation"),
    field   = dict(short="-f", long="--field",
                   action="store", type="string",
                   dest="field", default="Density",
                   help="Field to color by"),
    weight  = dict(short="-g", long="--weight",
                   action="store", type="string",
                   dest="weight", default=None,
                   help="Field to weight projections with"),
    cmap    = dict(short="", long="--colormap",
                   action="store", type="string",
                   dest="cmap", default="jet",
                   help="Colormap name"),
    zlim    = dict(short="-z", long="--zlim",
                   action="store", type="float",
                   dest="zlim", default=None,
                   nargs=2,
                   help="Color limits (min, max)"),
    dex     = dict(short="", long="--dex",
                   action="store", type="float",
                   dest="dex", default=None,
                   nargs=1,
                   help="Number of dex above min to display"),
    width   = dict(short="-w", long="--width",
                   action="store", type="float",
                   dest="width", default=1.0,
                   help="Width in specified units"),
    unit    = dict(short="-u", long="--unit",
                   action="store", type="string",
                   dest="unit", default='1',
                   help="Desired units"),
    center  = dict(short="-c", long="--center",
                   action="store", type="float",
                   dest="center", default=[0.5, 0.5, 0.5],
                   nargs=3,
                   help="Center (-1,-1,-1 for max)"),
    bn      = dict(short="-b", long="--basename",
                   action="store", type="string",
                   dest="basename", default=None,
                   help="Basename of parameter files"),
    output  = dict(short="-o", long="--output",
                   action="store", type="string",
                   dest="output", default="frames/",
                   help="Folder in which to place output images"),
    outputfn= dict(short="-o", long="--output",
                   action="store", type="string",
                   dest="output", default=None,
                   help="File in which to place output"),
    skip    = dict(short="-s", long="--skip",
                   action="store", type="int",
                   dest="skip", default=1,
                   help="Skip factor for outputs"),
    proj    = dict(short="-p", long="--projection",
                   action="store_true", 
                   dest="projection", default=False,
                   help="Use a projection rather than a slice"),
    maxw    = dict(short="", long="--max-width",
                   action="store", type="float",
                   dest="max_width", default=1.0,
                   help="Maximum width in code units"),
    minw    = dict(short="", long="--min-width",
                   action="store", type="float",
                   dest="min_width", default=50,
                   help="Minimum width in units of smallest dx (default: 50)"),
    nframes = dict(short="-n", long="--nframes",
                   action="store", type="int",
                   dest="nframes", default=100,
                   help="Number of frames to generate"),
    slabw   = dict(short="", long="--slab-width",
                   action="store", type="float",
                   dest="slab_width", default=1.0,
                   help="Slab width in specified units"),
    slabu   = dict(short="-g", long="--slab-unit",
                   action="store", type="string",
                   dest="slab_unit", default='1',
                   help="Desired units for the slab"),
    ptype   = dict(short="", long="--particle-type",
                   action="store", type="int",
                   dest="ptype", default=2,
                   help="Particle type to select"),
    agecut  = dict(short="", long="--age-cut",
                   action="store", type="float",
                   dest="age_filter", default=None,
                   nargs=2,
                   help="Bounds for the field to select"),
    uboxes  = dict(short="", long="--unit-boxes",
                   action="store_true",
                   dest="unit_boxes",
                   help="Display helpful unit boxes"),
    thresh  = dict(short="", long="--threshold",
                   action="store", type="float",
                   dest="threshold", default=None,
                   help="Density threshold"),
    dm_only = dict(short="", long="--all-particles",
                   action="store_false", 
                   dest="dm_only", default=True,
                   help="Use all particles"),
    grids   = dict(short="", long="--show-grids",
                   action="store_true",
                   dest="grids", default=False,
                   help="Show the grid boundaries"),
    time    = dict(short="", long="--time",
                   action="store_true",
                   dest="time", default=False,
                   help="Print time in years on image"),
    halos   = dict(short="", long="--halos",
                   action="store", type="string",
                   dest="halos",default="multiple",
                   help="Run halo profiler on a 'single' halo or 'multiple' halos."),
    halo_radius = dict(short="", long="--halo_radius",
                       action="store", type="float",
                       dest="halo_radius",default=0.1,
                       help="Constant radius for profiling halos if using hop output files with no radius entry. Default: 0.1."),
    halo_radius_units = dict(short="", long="--halo_radius_units",
                             action="store", type="string",
                             dest="halo_radius_units",default="1",
                             help="Units for radius used with --halo_radius flag. Default: '1' (code units)."),
    halo_hop_style = dict(short="", long="--halo_hop_style",
                          action="store", type="string",
                          dest="halo_hop_style",default="new",
                          help="Style of hop output file.  'new' for yt_hop files and 'old' for enzo_hop files."),
    halo_parameter_file = dict(short="", long="--halo_parameter_file",
                               action="store", type="string",
                               dest="halo_parameter_file",default=None,
                               help="HaloProfiler parameter file."),
    make_profiles = dict(short="", long="--make_profiles",
                         action="store_true", default=False,
                         help="Make profiles with halo profiler."),
    make_projections = dict(short="", long="--make_projections",
                            action="store_true", default=False,
                            help="Make projections with halo profiler.")

    )

def _add_options(parser, *options):
    for opt in options:
        oo = _common_options[opt].copy()
        parser.add_option(oo.pop("short"), oo.pop("long"), **oo)

def _get_parser(*options):
    parser = optparse.OptionParser()
    _add_options(parser, *options)
    return parser

def add_cmd_options(options):
    opts = []
    for option in options:
        vals = _common_options[option].copy()
        opts.append(([vals.pop("short"), vals.pop("long")],
                      vals))
    def apply_options(func):
        for args, kwargs in opts:
            func = cmdln.option(*args, **kwargs)(func)
        return func
    return apply_options

def check_args(func):
    @wraps(func)
    def arg_iterate(self, subcmd, opts, *args):
        if len(args) == 1:
            pfs = args
        elif len(args) == 2 and opts.basename is not None:
            pfs = ["%s%04i" % (opts.basename, r)
                   for r in range(int(args[0]), int(args[1]), opts.skip) ]
        else: pfs = args
        for arg in pfs:
            func(self, subcmd, opts, arg)
    return arg_iterate

def _get_vcs_type(path):
    if os.path.exists(os.path.join(path, ".hg")):
        return "hg"
    if os.path.exists(os.path.join(path, ".svn")):
        return "svn"
    return None

def _get_svn_version(path):
    p = subprocess.Popen(["svn", "info", path], stdout = subprocess.PIPE,
                                               stderr = subprocess.STDOUT)
    stdout, stderr = p.communicate()
    return stdout

def _update_svn(path):
    f = open(os.path.join(path, "yt_updater.log"), "a")
    f.write("\n\nUPDATE PROCESS: %s\n\n" % (time.asctime()))
    p = subprocess.Popen(["svn", "up", path], stdout = subprocess.PIPE,
                                              stderr = subprocess.STDOUT)
    stdout, stderr = p.communicate()
    f.write(stdout)
    f.write("\n\n")
    if p.returncode:
        print "BROKEN: See %s" % (os.path.join(path, "yt_updater.log"))
        sys.exit(1)
    f.write("Rebuilding modules\n\n")
    p = subprocess.Popen([sys.executable, "setup.py", "build_ext", "-i"], cwd=path,
                        stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    stdout, stderr = p.communicate()
    f.write(stdout)
    f.write("\n\n")
    if p.returncode:
        print "BROKEN: See %s" % (os.path.join(path, "yt_updater.log"))
        sys.exit(1)
    f.write("Successful!\n")

def _update_hg(path):
    from mercurial import hg, ui, commands
    f = open(os.path.join(path, "yt_updater.log"), "a")
    u = ui.ui()
    u.pushbuffer()
    config_fn = os.path.join(path, ".hg", "hgrc")
    print "Reading configuration from ", config_fn
    u.readconfig(config_fn)
    repo = hg.repository(u, path)
    commands.pull(u, repo)
    f.write(u.popbuffer())
    f.write("\n\n")
    u.pushbuffer()
    commands.identify(u, repo)
    if "+" in u.popbuffer():
        print "Can't rebuild modules by myself."
        print "You will have to do this yourself.  Here's a sample commands:"
        print
        print "    $ cd %s" % (path)
        print "    $ hg up"
        print "    $ %s setup.py develop" % (sys.executable)
        sys.exit(1)
    f.write("Rebuilding modules\n\n")
    p = subprocess.Popen([sys.executable, "setup.py", "build_ext", "-i"], cwd=path,
                        stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    stdout, stderr = p.communicate()
    f.write(stdout)
    f.write("\n\n")
    if p.returncode:
        print "BROKEN: See %s" % (os.path.join(path, "yt_updater.log"))
        sys.exit(1)
    f.write("Successful!\n")

def _get_hg_version(path):
    from mercurial import hg, ui, commands
    u = ui.ui()
    u.pushbuffer()
    repo = hg.repository(u, path)
    commands.identify(u, repo)
    return u.popbuffer()

_vcs_identifier = dict(svn = _get_svn_version,
                        hg = _get_hg_version)
_vcs_updater = dict(svn = _update_svn,
                     hg = _update_hg)

class YTCommands(cmdln.Cmdln):
    name="yt"

    def __init__(self, *args, **kwargs):
        cmdln.Cmdln.__init__(self, *args, **kwargs)
        cmdln.Cmdln.do_help.aliases.append("h")

    def do_loop(self, subcmd, opts, *args):
        """
        Interactive loop

        ${cmd_option_list}
        """
        self.cmdloop()

    @cmdln.option("-u", "--update-source", action="store_true",
                  default = False,
                  help="Update the yt installation, if able")
    @cmdln.option("-o", "--output-version", action="store",
                  default = None, dest="outputfile",
                  help="File into which the current revision number will be stored")
    def do_instinfo(self, subcmd, opts):
        """
        ${cmd_name}: Get some information about the yt installation

        ${cmd_usage}
        ${cmd_option_list}
        """
        import pkg_resources
        yt_provider = pkg_resources.get_provider("yt")
        path = os.path.dirname(yt_provider.module_path)
        print
        print "yt module located at:"
        print "    %s" % (path)
        if "site-packages" not in path:
            vc_type = _get_vcs_type(path)
            vstring = _vcs_identifier[vc_type](path)
            print
            print "The current version of the code is:"
            print
            print "---"
            print vstring
            print "---"
            print
            print "This installation CAN be automatically updated."
            if opts.update_source:  
                _vcs_updater[vc_type](path)
            print "Updated successfully."
        if vstring is not None and opts.outputfile is not None:
            open(opts.outputfile, "w").write(vstring)

    def do_load(self, subcmd, opts, arg):
        """
        Load a single dataset into an IPython instance.

        ${cmd_option_list}
        """
        try:
            pf = _fix_pf(arg)
        except IOError:
            print "Could not load file."
            sys.exit()
        import yt.mods
        from IPython.Shell import IPShellEmbed
        local_ns = yt.mods.__dict__.copy()
        local_ns['pf'] = pf
        shell = IPShellEmbed()
        shell(local_ns = local_ns,
              header =
            "\nHi there!  Welcome to yt.\n\nWe've loaded your parameter file as 'pf'.  Enjoy!"
             )

    @add_cmd_options(['outputfn','bn','thresh','dm_only','skip'])
    @check_args
    def do_hop(self, subcmd, opts, arg):
        """
        Run HOP on one or more datasets

        ${cmd_option_list}
        """
        pf = _fix_pf(arg)
        kwargs = {'dm_only' : opts.dm_only}
        if opts.threshold is not None: kwargs['threshold'] = opts.threshold
        hop_list = HaloFinder(pf, **kwargs)
        if opts.output is None: fn = "%s.hop" % pf
        else: fn = opts.output
        hop_list.write_out(fn)

    @add_cmd_options(['make_profiles','make_projections','halo_parameter_file',
                      'halos','halo_hop_style','halo_radius','halo_radius_units'])
    def do_halos(self, subcmd, opts, arg):
        """
        Run HaloProfiler on one dataset.

        ${cmd_option_list}
        """
        import yt.extensions.HaloProfiler as HP
        kwargs = {'halos': opts.halos,
                  'hop_style': opts.halo_hop_style,
                  'radius': opts.halo_radius,
                  'radius_units': opts.halo_radius_units}

        hp = HP.HaloProfiler(arg,opts.halo_parameter_file,**kwargs)
        if opts.make_profiles:
            hp.makeProfiles()
        if opts.make_projections:
            hp.makeProjections()

    @add_cmd_options(["maxw", "minw", "proj", "axis", "field", "weight",
                      "zlim", "nframes", "output", "cmap", "uboxes", "dex",
                      "text"])
    def do_zoomin(self, subcmd, opts, arg):
        """
        Create a set of zoomin frames

        ${cmd_option_list}
        """
        pf = _fix_pf(arg)
        min_width = opts.min_width * pf.h.get_smallest_dx()
        if opts.axis == 4:
            axes = range(3)
        else:
            axes = [opts.axis]
        pc = PlotCollection(pf)
        for ax in axes: 
            if opts.projection: p = pc.add_projection(opts.field, ax,
                                    weight_field=opts.weight)
            else: p = pc.add_slice(opts.field, ax)
            if opts.unit_boxes: p.modify["units"](factor=8)
            if opts.text is not None:
                p.modify["text"](
                    (0.02, 0.05), opts.text.replace(r"\n", "\n"),
                    text_args=dict(size="medium", color="w"))
        pc.set_width(opts.max_width,'1')
        # Check the output directory
        if not os.path.isdir(opts.output):
            os.mkdir(opts.output)
        # Figure out our zoom factor
        # Recall that factor^nframes = min_width / max_width
        # so factor = (log(min/max)/log(nframes))
        mylog.info("min_width: %0.3e max_width: %0.3e nframes: %0.3e",
                   min_width, opts.max_width, opts.nframes)
        factor=10**(math.log10(min_width/opts.max_width)/opts.nframes)
        mylog.info("Zoom factor: %0.3e", factor)
        w = 1.0
        for i in range(opts.nframes):
            mylog.info("Setting width to %0.3e", w)
            mylog.info("Saving frame %06i",i)
            pc.set_width(w,"1")
            if opts.zlim:
                pc.set_zlim(*opts.zlim)
            if opts.dex:
                pc.set_zlim('min', None, opts.dex)
            pc.set_cmap(opts.cmap)
            pc.save(os.path.join(opts.output,"%s_frame%06i" % (pf,i)))
            w = factor**i

    @add_cmd_options(["width", "unit", "bn", "proj", "center",
                      "zlim", "axis", "field", "weight", "skip",
                      "cmap", "output", "grids", "time"])
    @check_args
    def do_plot(self, subcmd, opts, arg):
        """
        Create a set of images

        ${cmd_usage}
        ${cmd_option_list}
        """
        pf = _fix_pf(arg)
        center = opts.center
        if opts.center == (-1,-1,-1):
            mylog.info("No center fed in; seeking.")
            v, center = pf.h.find_max("Density")
        center = na.array(center)
        pc=PlotCollection(pf, center=center)
        if opts.axis == 4:
            axes = range(3)
        else:
            axes = [opts.axis]
        for ax in axes:
            mylog.info("Adding plot for axis %i", ax)
            if opts.projection: pc.add_projection(opts.field, ax,
                                    weight_field=opts.weight, center=center)
            else: pc.add_slice(opts.field, ax, center=center)
            if opts.grids: pc.plots[-1].modify["grids"]()
            if opts.time: 
                time = pf.current_time*pf['Time']*pf['years']
                pc.plots[-1].modify["text"]((0.2,0.8), 't = %5.2e yr'%time)
        pc.set_width(opts.width, opts.unit)
        pc.set_cmap(opts.cmap)
        if opts.zlim: pc.set_zlim(*opts.zlim)
        if not os.path.isdir(opts.output): os.makedirs(opts.output)
        pc.save(os.path.join(opts.output,"%s" % (pf)))

    def do_rpdb(self, subcmd, opts, task):
        """
        Connect to a currently running (on localhost) rpd session.

        Commands run with --rpdb will trigger an rpdb session with any
        uncaught exceptions.

        ${cmd_usage} 
        ${cmd_option_list}
        """
        import rpdb
        rpdb.run_rpdb(int(task))

    @add_cmd_options(['outputfn','bn','skip'])
    @check_args
    def do_stats(self, subcmd, opts, arg):
        """
        Print stats and maximum density for one or more datasets

        ${cmd_option_list}
        """
        pf = _fix_pf(arg)
        pf.h.print_stats()
        v, c = pf.h.find_max("Density")
        print "Maximum density: %0.5e at %s" % (v, c)
        if opts.output is not None:
            t = pf.current_time * pf['years']
            open(opts.output, "a").write(
                "%s (%0.5e years): %0.5e at %s\n" % (pf, t, v, c))

    @add_cmd_options([])
    def do_analyze(self, subcmd, opts, arg):
        """
        Produce a set of analysis for a given output.  This includes
        HaloProfiler results with r200, as per the recipe file in the cookbook,
        profiles of a number of fields, projections of average Density and
        Temperature, and distribution functions for Density and Temperature.

        ${cmd_option_list}
        """
        # We will do the following things:
        #   Halo profiling (default parameters ONLY)
        #   Projections: Density, Temperature
        #   Full-box distribution functions
        import yt.extensions.HaloProfiler as HP
        hp = HP.HaloProfiler(arg)
        # Add a filter to remove halos that have no profile points with overdensity
        # above 200, and with virial masses less than 1e14 solar masses.
        # Also, return the virial mass and radius to be written out to a file.
        hp.add_halo_filter(HP.VirialFilter,must_be_virialized=True,
                           overdensity_field='ActualOverdensity',
                           virial_overdensity=200, virial_filters=[],
                           virial_quantities=['TotalMassMsun','RadiusMpc'])

        # Add profile fields.
        hp.add_profile('CellVolume',weight_field=None,accumulation=True)
        hp.add_profile('TotalMassMsun',weight_field=None,accumulation=True)
        hp.add_profile('Density',weight_field=None,accumulation=False)
        hp.add_profile('Temperature',weight_field='CellMassMsun',accumulation=False)
        hp.make_profiles(filename="FilteredQuantities.out")

        # Add projection fields.
        hp.add_projection('Density',weight_field=None)
        hp.add_projection('Temperature',weight_field='Density')
        hp.add_projection('Metallicity',weight_field='Density')

        # Make projections for all three axes using the filtered halo list and
        # save data to hdf5 files.
        hp.make_projections(save_cube=True,save_images=True,
                            halo_list='filtered',axes=[0,1,2])

        # Now we make full-box projections.
        pf = EnzoStaticOutput(arg)
        c = 0.5*(pf.domain_right_edge + pf.domain_left_edge)
        pc = PlotCollection(pf, center=c)
        for ax in range(3):
            pc.add_projection("Density", ax, "Density")
            pc.add_projection("Temperature", ax, "Density")
            pc.plots[-1].set_cmap("hot")

        # Time to add some phase plots
        dd = pf.h.all_data()
        ph = pc.add_phase_object(dd, ["Density", "Temperature", "CellMassMsun"],
                            weight=None)
        pc_dummy = PlotCollection(pf, center=c)
        pr = pc_dummy.add_profile_object(dd, ["Density", "Temperature"],
                            weight="CellMassMsun")
        ph.modify["line"](pr.data["Density"], pr.data["Temperature"])
        pc.save()
    
def run_main():
    for co in ["--parallel", "--paste"]:
        if co in sys.argv: del sys.argv[sys.argv.index(co)]
    YT = YTCommands()
    sys.exit(YT.main())

if __name__ == "__main__": run_main()
