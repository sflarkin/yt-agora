"""
Tests for AMRSlice


"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import os
import numpy as np
import tempfile
from nose.tools import raises
from yt.testing import \
    fake_random_pf, assert_equal, assert_array_equal
from yt.utilities.definitions import \
    x_dict, y_dict
from yt.utilities.exceptions import \
    YTNoDataInObjectError

def setup():
    from yt.config import ytcfg
    ytcfg["yt", "__withintesting"] = "True"


def teardown_func(fns):
    for fn in fns:
        try:
            os.remove(fn)
        except OSError:
            pass


def test_slice():
    for nprocs in [8, 1]:
        # We want to test both 1 proc and 8 procs, to make sure that
        # parallelism isn't broken
        pf = fake_random_pf(64, nprocs=nprocs)
        dims = pf.domain_dimensions
        xn, yn, zn = pf.domain_dimensions
        xi, yi, zi = pf.domain_left_edge + 1.0 / (pf.domain_dimensions * 2)
        xf, yf, zf = pf.domain_right_edge - 1.0 / (pf.domain_dimensions * 2)
        coords = np.mgrid[xi:xf:xn * 1j, yi:yf:yn * 1j, zi:zf:zn * 1j]
        uc = [np.unique(c) for c in coords]
        slc_pos = 0.5
        # Some simple slice tests with single grids
        for ax, an in enumerate("xyz"):
            xax = x_dict[ax]
            yax = y_dict[ax]
            for wf in ["Density", None]:
                fns = []
                slc = pf.h.slice(ax, slc_pos, ["Ones", "Density"])
                yield assert_equal, slc["Ones"].sum(), slc["Ones"].size
                yield assert_equal, slc["Ones"].min(), 1.0
                yield assert_equal, slc["Ones"].max(), 1.0
                yield assert_equal, np.unique(slc["px"]), uc[xax]
                yield assert_equal, np.unique(slc["py"]), uc[yax]
                yield assert_equal, np.unique(slc["pdx"]), 0.5 / dims[xax]
                yield assert_equal, np.unique(slc["pdy"]), 0.5 / dims[yax]
                pw = slc.to_pw()
                tmpfd, tmpname = tempfile.mkstemp(suffix='.png')
                os.close(tmpfd)
                fns += pw.save(name=tmpname)
                frb = slc.to_frb((1.0, 'unitary'), 64)
                for slc_field in ['Ones', 'Density']:
                    yield assert_equal, frb[slc_field].info['data_source'], \
                        slc.__str__()
                    yield assert_equal, frb[slc_field].info['axis'], \
                        ax
                    yield assert_equal, frb[slc_field].info['field'], \
                        slc_field
                    yield assert_equal, frb[slc_field].info['units'], \
                        pf.field_info[slc_field].get_units()
                    yield assert_equal, frb[slc_field].info['xlim'], \
                        frb.bounds[:2]
                    yield assert_equal, frb[slc_field].info['ylim'], \
                        frb.bounds[2:]
                    yield assert_equal, frb[slc_field].info['length_to_cm'], \
                        pf['cm']
                    yield assert_equal, frb[slc_field].info['center'], \
                        slc.center
                    yield assert_equal, frb[slc_field].info['coord'], \
                        slc_pos
                teardown_func(fns)
            # wf == None
            yield assert_equal, wf, None


def test_slice_over_edges():
    pf = fake_random_pf(64, nprocs=8, fields=["Density"], negative=[False])

    slc = pf.h.slice(0, 0.0, "Density")
    yield assert_array_equal, slc.grid_left_edge[:, 0], np.zeros((4))
    slc = pf.h.slice(1, 0.5, "Density")
    yield assert_array_equal, slc.grid_left_edge[:, 1], np.ones((4)) * 0.5


@raises(YTNoDataInObjectError)
def test_slice_over_outer_boundary():
    pf = fake_random_pf(64, nprocs=8, fields=["Density"], negative=[False])
    slc = pf.h.slice(2, 1.0, "Density")
