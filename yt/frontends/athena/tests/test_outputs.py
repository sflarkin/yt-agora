"""
Athena frontend tests



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
from yt.frontends.athena.api import AthenaDataset

parameters_sloshing = {"time_unit":(1.0,"Myr"),
                       "length_unit":(1.0,"Mpc"),
                       "mass_unit":(1.0e14,"Msun")}

parameters_stripping = {"time_unit":3.086e14,
                        "length_unit":8.0236e22,
                        "mass_unit":9.999e-30*8.0236e22**3}

_fields_sloshing = ("temperature", "density", "magnetic_energy")

sloshing = "MHDSloshing/id0/virgo_cluster.0055.vtk"
@requires_ds(sloshing, big_data=True)
def test_sloshing():
    ds = data_dir_load(sloshing, kwargs={"parameters":parameters_sloshing})
    yield assert_equal, str(ds), "virgo_cluster.0055"
    for test in small_patch_amr(sloshing, _fields_sloshing):
        test_sloshing.__name__ = test.description
        yield test

_fields_blast = ("temperature", "density", "velocity_magnitude")

blast = "MHDBlast/Blast.0100.vtk"
@requires_ds(blast)
def test_blast():
    ds = data_dir_load(blast)
    yield assert_equal, str(ds), "Blast.0100"
    for test in small_patch_amr(blast, _fields_blast):
        test_blast.__name__ = test.description
        yield test

_fields_stripping = ("temperature", "density", "specific_scalar[0]")

stripping = "RamPressureStripping/id0/rps.0063.vtk"
@requires_ds(stripping, big_data=True)
def test_stripping():
    ds = data_dir_load(stripping, kwargs={"parameters":parameters_stripping})
    yield assert_equal, str(ds), "rps.0062"
    for test in small_patch_amr(stripping, _fields_stripping):
        test_stripping.__name__ = test.description
        yield test
