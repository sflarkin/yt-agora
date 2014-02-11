"""
Very simple convenience function for importing all the modules, setting up
the namespace and getting the last argument on the command line.



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

# This handles the command line.

import argparse, os, sys

from yt.config import ytcfg
from yt.funcs import *

exe_name = os.path.basename(sys.executable)
# At import time, we determined whether or not we're being run in parallel.
def turn_on_parallelism():
    try:
        from mpi4py import MPI
    except ImportError as e:
        mylog.error("Warning: Attempting to turn on parallelism, " +
                    "but mpi4py import failed. Try pip install mpi4py.")
        raise e
    parallel_capable = (MPI.COMM_WORLD.size > 1)
    if parallel_capable:
        mylog.info("Global parallel computation enabled: %s / %s",
                   MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size)
        ytcfg["yt","__global_parallel_rank"] = str(MPI.COMM_WORLD.rank)
        ytcfg["yt","__global_parallel_size"] = str(MPI.COMM_WORLD.size)
        ytcfg["yt","__parallel"] = "True"
        if exe_name == "embed_enzo" or \
            ("_parallel" in dir(sys) and sys._parallel == True):
            ytcfg["yt","inline"] = "True"
        # I believe we do not need to turn this off manually
        #ytcfg["yt","StoreParameterFiles"] = "False"
        # Now let's make sure we have the right options set.
        if MPI.COMM_WORLD.rank > 0:
            if ytcfg.getboolean("yt","LogFile"):
                ytcfg["yt","LogFile"] = "False"
                yt.utilities.logger.disable_file_logging()
    return parallel_capable

# This fallback is for Paraview:

# We use two signals, SIGUSR1 and SIGUSR2.  In a non-threaded environment,
# we set up handlers to process these by printing the current stack and to
# raise a RuntimeError.  The latter can be used, inside pdb, to catch an error
# and then examine the current stack.
try:
    signal.signal(signal.SIGUSR1, signal_print_traceback)
    mylog.debug("SIGUSR1 registered for traceback printing")
    signal.signal(signal.SIGUSR2, signal_ipython)
    mylog.debug("SIGUSR2 registered for IPython Insertion")
except (ValueError, RuntimeError, AttributeError) as e:  # Not in main thread
    pass

class SetExceptionHandling(argparse.Action):
    def __call__(self, parser, namespace, values, option_string = None):
        # If we recognize one of the arguments on the command line as indicating a
        # different mechanism for handling tracebacks, we attach one of those handlers
        # and remove the argument from sys.argv.
        #
        if self.dest == "paste":
            sys.excepthook = paste_traceback
            mylog.debug("Enabling traceback pasting")
        elif self.dest == "paste-detailed":
            sys.excepthook = paste_traceback_detailed
            mylog.debug("Enabling detailed traceback pasting")
        elif self.dest == "detailed":
            import cgitb; cgitb.enable(format="text")
            mylog.debug("Enabling detailed traceback reporting")
        elif self.dest == "rpdb":
            sys.excepthook = rpdb.rpdb_excepthook
            mylog.debug("Enabling remote debugging")

class SetConfigOption(argparse.Action):
    def __call__(self, parser, namespace, values, option_string = None):
        param, val = values.split("=")
        mylog.debug("Overriding config: %s = %s", param, val)
        ytcfg["yt",param] = val
        if param == "loglevel": # special case
            mylog.setLevel(int(val))

class YTParser(argparse.ArgumentParser):
    def error(self, message):
        """error(message: string)

        Prints a help message that is more detailed than the argparse default
        and then exits.
        """
        self.print_help(sys.stderr)
        self.exit(2, '%s: error: %s\n' % (self.prog, message))

parser = YTParser(description = 'yt command line arguments')
parser.add_argument("--config", action=SetConfigOption,
    help = "Set configuration option, in the form param=value")
parser.add_argument("--paste", action=SetExceptionHandling,
    help = "Paste traceback to paste.yt-project.org", nargs = 0)
parser.add_argument("--paste-detailed", action=SetExceptionHandling,
    help = "Paste a detailed traceback with local variables to " +
           "paste.yt-project.org", nargs = 0)
parser.add_argument("--detailed", action=SetExceptionHandling,
    help = "Display detailed traceback.", nargs = 0)
parser.add_argument("--rpdb", action=SetExceptionHandling,
    help = "Enable remote pdb interaction (for parallel debugging).", nargs = 0)
parser.add_argument("--parallel", action="store_true", default=False,
    dest = "parallel",
    help = "Run in MPI-parallel mode (must be launched as an MPI task)")
if not hasattr(sys, 'argv') or sys.argv is None: sys.argv = []

unparsed_args = []

parallel_capable = False
if not ytcfg.getboolean("yt","__command_line"):
    opts, unparsed_args = parser.parse_known_args()
    # THIS IS NOT SUCH A GOOD IDEA:
    #sys.argv = [a for a in unparsed_args]
    if opts.parallel:
        parallel_capable = turn_on_parallelism()
    subparsers = parser.add_subparsers(title="subcommands",
                        dest='subcommands',
                        description="Valid subcommands",)
else:
    subparsers = parser.add_subparsers(title="subcommands",
                        dest='subcommands',
                        description="Valid subcommands",)
    def print_help(*args, **kwargs):
        parser.print_help()
    help_parser = subparsers.add_parser("help", help="Print help message")
    help_parser.set_defaults(func=print_help)


if parallel_capable == True:
    pass
elif exe_name in \
        ["mpi4py", "embed_enzo",
         "python"+sys.version[:3]+"-mpi"] \
        or '_parallel' in dir(sys) \
        or any(["ipengine" in arg for arg in sys.argv]) \
        or any(["cluster-id" in arg for arg in sys.argv]):
    parallel_capable = turn_on_parallelism()
else:
    parallel_capable = False
