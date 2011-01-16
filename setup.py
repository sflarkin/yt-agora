import os, os.path
import sys
import time
import subprocess
import distribute_setup
distribute_setup.use_setuptools()

import setuptools

DATA_FILES = []
VERSION = "2.0dev"

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
        description = "An analysis and visualization toolkit for Adaptive Mesh " \
                    + "Refinement data, specifically for the Enzo and Orion codes.",
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
