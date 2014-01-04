from yt.testing import *
from yt.utilities.units import Unit
import os
import tempfile

def setup():
    from yt.config import ytcfg
    ytcfg["yt","__withintesting"] = "True"

def teardown_func(fns):
    for fn in fns:
        try:
            os.remove(fn)
        except OSError:
            pass

def test_cutting_plane():
    for nprocs in [8, 1]:
        # We want to test both 1 proc and 8 procs, to make sure that
        # parallelism isn't broken
        pf = fake_random_pf(64, nprocs = nprocs)
        dims = pf.domain_dimensions
        center = [0.5,0.5,0.5]
        normal = [1,1,1]
        fns = []
        cut = pf.h.cutting(normal, center)
        yield assert_equal, cut["ones"].sum(), cut["ones"].size
        yield assert_equal, cut["ones"].min(), 1.0
        yield assert_equal, cut["ones"].max(), 1.0
        pw = cut.to_pw()
        tmpfd, tmpname = tempfile.mkstemp(suffix='.png')
        os.close(tmpfd)
        fns += pw.save(name=tmpname)
        frb = cut.to_frb((1.0,'unitary'), 64)
        for cut_field in ['ones', 'density']:
            fi = pf._get_field_info("unknown", cut_field)
            yield assert_equal, frb[cut_field].info['data_source'], \
                cut.__str__()
            yield assert_equal, frb[cut_field].info['axis'], \
                4
            yield assert_equal, frb[cut_field].info['field'], \
                cut_field
            yield assert_equal, frb[cut_field].units, \
                Unit(fi.units)
            yield assert_equal, frb[cut_field].info['xlim'], \
                frb.bounds[:2]
            yield assert_equal, frb[cut_field].info['ylim'], \
                frb.bounds[2:]
            yield assert_equal, frb[cut_field].info['length_to_cm'], \
                pf.length_unit.in_cgs()
            yield assert_equal, frb[cut_field].info['center'], \
                cut.center
        teardown_func(fns)
