#!/usr/bin/env python
from __future__ import print_function
import setuptools
import os, sys, os.path

import os.path

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('sunrise_export',parent_package,top_path)
    config.make_config_py() # installs __config__.py
    #config.make_svn_version_py()
    config.add_extension("octree_to_depthFirstHilbert",
                         "yt/analysis_modules/sunrise_export/octree_to_depthFirstHilbert.pyx",
                         #define_macros = [("THREADSAFE", "__thread")],
                         define_macros = [("THREADSAFE", "")])
    return config

