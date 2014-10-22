"""
EAGLE data-file handling function




"""

#-----------------------------------------------------------------------------
# Copyright (c) 2014, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

from yt.frontends.sph.owls.io import \
    IOHandlerOWLS

class IOHandlerEagleNetwork(IOHandlerOWLS):
    _dataset_type = "eagle_network"
