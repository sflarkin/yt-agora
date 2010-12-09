#!/usr/bin/env python
import setuptools

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('frontends',parent_package,top_path)
    config.make_config_py() # installs __config__.py
    config.make_svn_version_py()
    config.add_subpackage("chombo")
    config.add_subpackage("enzo")
    config.add_subpackage("flash")
    config.add_subpackage("orion")
    config.add_subpackage("ramses")
    config.add_subpackage("tiger")
    config.add_subpackage("art")
    return config
