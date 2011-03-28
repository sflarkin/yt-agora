import os, os.path
import sys
import time
import subprocess
import distribute_setup
distribute_setup.use_setuptools()

from numpy.distutils.misc_util import appendpath
from numpy.distutils import log

# Verify that we have Cython installed
try:
    import Cython
except ImportError as e:
    print "Received error on importing Cython:"
    print e
    print "Now attempting to install Cython"
    import pip
    rv = pip.main(["install",
              "http://yt.enzotools.org/dependencies/Cython-latest.tar.gz"])
    if rv == 1:
        print "Unable to install Cython.  Please report this bug to yt-users."
        sys.exit(1)

######
# This next bit comes from Matthew Brett, to get Cython working with NumPy
# distutils.  I added a bit to get C++ Cython working.
from os.path import join as pjoin, dirname
from distutils.dep_util import newer_group
from distutils.errors import DistutilsError


def generate_a_pyrex_source(self, base, ext_name, source, extension):
    ''' Monkey patch for numpy build_src.build_src method

    Uses Cython instead of Pyrex.

    Assumes Cython is present
    '''
    if self.inplace:
        target_dir = dirname(base)
    else:
        target_dir = appendpath(self.build_src, dirname(base))
    if extension.language == "c++":
        cplus = True
        file_ext = ".cpp"
    else:
        cplus = False
        file_ext = ".c"
    target_file = pjoin(target_dir, ext_name + file_ext)
    depends = [source] + extension.depends
    if self.force or newer_group(depends, target_file, 'newer'):
        import Cython.Compiler.Main
        log.info("cythonc:> %s" % (target_file))
        self.mkpath(target_dir)
        options = Cython.Compiler.Main.CompilationOptions(
            defaults=Cython.Compiler.Main.default_options,
            include_path=extension.include_dirs,
            language=extension.language, cplus = cplus,
            output_file=target_file)
        cython_result = Cython.Compiler.Main.compile(source,
                                                   options=options)
        if cython_result.num_errors != 0:
            raise DistutilsError("%d errors while compiling %r with Cython" \
                  % (cython_result.num_errors, source))
    return target_file


from numpy.distutils.command import build_src
build_src.build_src.generate_a_pyrex_source = generate_a_pyrex_source
# End snippet
######

import setuptools

DATA_FILES = []
VERSION = "2.1dev"

if os.path.exists('MANIFEST'): os.remove('MANIFEST')

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    
    config.make_config_py()
    config.make_svn_version_py()
    config.add_subpackage('yt','yt')
    config.add_scripts("scripts/*")

    return config

def setup_package():

    from numpy.distutils.core import setup

    setup(
        name = "yt",
        version = VERSION,
        description = "An analysis and visualization toolkit for Astrophysical "
                    + "simulations, focusing on Adaptive Mesh Refinement data "
                      "from Enzo, Orion, FLASH, and others.",
        classifiers = [ "Development Status :: 5 - Production/Stable",
                        "Environment :: Console",
                        "Intended Audience :: Science/Research",
                        "License :: OSI Approved :: GNU General Public License (GPL)",
                        "Operating System :: MacOS :: MacOS X",
                        "Operating System :: POSIX :: AIX",
                        "Operating System :: POSIX :: Linux",
                        "Programming Language :: C",
                        "Programming Language :: Python",
                        "Topic :: Scientific/Engineering :: Astronomy",
                        "Topic :: Scientific/Engineering :: Physics",
                        "Topic :: Scientific/Engineering :: Visualization", ],
        keywords='astronomy astrophysics visualization amr adaptivemeshrefinement',
        entry_points = { 'console_scripts' : [
                            'yt = yt.utilities.command_line:run_main',
                            'enzo_test = yt.utilities.answer_testing.runner:run_main',
                       ]},
        author="Matthew J. Turk",
        author_email="matthewturk@gmail.com",
        url = "http://yt.enzotools.org/",
        license="GPL-3",
        configuration=configuration,
        data_files=DATA_FILES,
        zip_safe=False,
        package_data = {'': ['*.so'], }
        )
    return

if __name__ == '__main__':
    setup_package()
