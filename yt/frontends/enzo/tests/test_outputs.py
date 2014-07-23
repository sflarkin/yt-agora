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
    requires_ds, \
    small_patch_amr, \
    big_patch_amr, \
    data_dir_load
from yt.frontends.enzo.api import EnzoDataset

_fields = ("temperature", "density", "velocity_magnitude",
           "velocity_divergence")

def check_color_conservation(ds):
    species_names = ds.field_info.species_names
    dd = ds.all_data()
    dens_yt = dd["density"].copy()
    # Enumerate our species here
    for s in sorted(species_names):
        if s == "El": continue
        dens_yt -= dd["%s_density" % s]
    dens_yt -= dd["metal_density"]
    delta_yt = np.abs(dens_yt / dd["density"])

    # Now we compare color conservation to Enzo's color conservation
    dd = ds.all_data()
    dens_enzo = dd["Density"].copy()
    for f in sorted(ds.field_list):
        if not f[1].endswith("_Density") or \
               f[1].startswith("Dark_Matter_")  or \
               f[1].startswith("Electron_") or \
               f[1].startswith("SFR_") or \
               f[1].startswith("Forming_Stellar_") or \
               f[1].startswith("Star_Particle_"):
            continue
        dens_enzo -= dd[f]
    delta_enzo = np.abs(dens_enzo / dd["Density"])
    return assert_almost_equal, delta_yt, delta_enzo

m7 = "DD0010/moving7_0010"
@requires_ds(m7)
def test_moving7():
    ds = data_dir_load(m7)
    yield assert_equal, str(ds), "moving7_0010"
    for test in small_patch_amr(m7, _fields):
        test_moving7.__name__ = test.description
        yield test

g30 = "IsolatedGalaxy/galaxy0030/galaxy0030"
@requires_ds(g30, big_data=True)
def test_galaxy0030():
    ds = data_dir_load(g30)
    yield check_color_conservation(ds)
    yield assert_equal, str(ds), "galaxy0030"
    for test in big_patch_amr(g30, _fields):
        test_galaxy0030.__name__ = test.description
        yield test

ecp = "enzo_cosmology_plus/DD0046/DD0046"
@requires_ds(ecp, big_data=True)
def test_ecp():
    ds = data_dir_load(ecp)
    # Now we test our species fields
    yield check_color_conservation(ds)
