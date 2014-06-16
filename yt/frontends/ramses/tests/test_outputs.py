"""
RAMSES frontend tests 




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
    data_dir_load, \
    PixelizedProjectionValuesTest, \
    FieldValuesTest, \
    create_obj
from yt.frontends.artio.api import ARTIODataset

_fields = ("temperature", "density", "velocity_magnitude",
           ("deposit", "all_density"), ("deposit", "all_count")) 

output_00080 = "output_00080/info_00080.txt"
@requires_ds(output_00080)
def test_output_00080():
    ds = data_dir_load(output_00080)
    yield assert_equal, str(ds), "info_00080"
    dso = [ None, ("sphere", ("max", (0.1, 'unitary')))]
    for ds in dso:
        for field in _fields:
            for axis in [0, 1, 2]:
                for weight_field in [None, "density"]:
                    yield PixelizedProjectionValuesTest(
                        output_00080, axis, field, weight_field,
                        ds)
            yield FieldValuesTest(output_00080, field, ds)
        dobj = create_obj(ds, ds)
        s1 = dobj["ones"].sum()
        s2 = sum(mask.sum() for block, mask in dobj.blocks)
        yield assert_equal, s1, s2
