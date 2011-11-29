#!/usr/bin/env python
import setuptools
import os, sys, os.path

import os.path

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('halo_finding',parent_package,top_path)
    config.add_subpackage("fof")
    config.add_subpackage("hop")
    config.add_subpackage("parallel_hop")
    if "ROCKSTAR_DIR" in os.environ:
        config.add_subpackage("rockstar")
    config.make_config_py() # installs __config__.py
    #config.make_svn_version_py()
    return config
