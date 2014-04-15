"""
Enzo frontend tests using moving7



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

from yt.testing import *
from yt.utilities.answer_testing.framework import \
    requires_pf, \
    small_patch_amr, \
    big_patch_amr, \
    data_dir_load
from yt.frontends.enzo.api import EnzoDataset

_fields = ("Temperature", "Density", "VelocityMagnitude", "DivV")

m7 = "DD0010/moving7_0010"
@requires_pf(m7)
def test_moving7():
    pf = data_dir_load(m7)
    yield assert_equal, str(pf), "moving7_0010"
    for test in small_patch_amr(m7, _fields):
        test_moving7.__name__ = test.description
        yield test

g30 = "IsolatedGalaxy/galaxy0030/galaxy0030"
@requires_pf(g30, big_data=True)
def test_galaxy0030():
    pf = data_dir_load(g30)
    yield assert_equal, str(pf), "galaxy0030"
    for test in big_patch_amr(g30, _fields):
        test_galaxy0030.__name__ = test.description
        yield test

hds0 = "rockstar_halos/halos_0.0.bin"
hds1 = "rockstar_halos/halos_0.1.bin"
@requires_pf(hds0)
@requires_pf(hds1)
def test_halo_mass_function():
	hds = data_dir_load(hds0)
	yield assert_equal, str(hds), "halos_0.0.bin"
	for test in hfm_sim_and_analytic(hds0):
		test_halo_mass_function.__name__ = test.description
		yield test