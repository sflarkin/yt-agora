from yt.testing import *
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

def test_projection():
    for nprocs in [8, 1]:
        # We want to test both 1 proc and 8 procs, to make sure that
        # parallelism isn't broken
        pf = fake_random_pf(64, nprocs = nprocs)
        dims = pf.domain_dimensions
        xn, yn, zn = pf.domain_dimensions
        xi, yi, zi = pf.domain_left_edge + 1.0/(pf.domain_dimensions * 2)
        xf, yf, zf = pf.domain_right_edge - 1.0/(pf.domain_dimensions * 2)
        dd = pf.h.all_data()
        rho_tot = dd.quantities["TotalQuantity"]("Density")[0]
        coords = np.mgrid[xi:xf:xn*1j, yi:yf:yn*1j, zi:zf:zn*1j]
        uc = [np.unique(c) for c in coords]
        # Some simple projection tests with single grids
        for ax, an in enumerate("xyz"):
            xax = x_dict[ax]
            yax = y_dict[ax]
            for wf in ["Density", None]:
                fns = []
                proj = pf.h.proj(ax, ["Ones", "Density"], weight_field = wf)
                yield assert_equal, proj["Ones"].sum(), proj["Ones"].size
                yield assert_equal, proj["Ones"].min(), 1.0
                yield assert_equal, proj["Ones"].max(), 1.0
                yield assert_equal, np.unique(proj["px"]), uc[xax]
                yield assert_equal, np.unique(proj["py"]), uc[yax]
                yield assert_equal, np.unique(proj["pdx"]), 1.0/(dims[xax]*2.0)
                yield assert_equal, np.unique(proj["pdy"]), 1.0/(dims[yax]*2.0)
                pw = proj.to_pw()
                tmpfd, tmpname = tempfile.mkstemp(suffix='.png')
                os.close(tmpfd)
                fns += pw.save(name=tmpname)
                frb = proj.to_frb((1.0,'unitary'), 64)
                for proj_field in ['Ones', 'Density']:
                    yield assert_equal, frb[proj_field].info['data_source'], \
                            proj.__str__()
                    yield assert_equal, frb[proj_field].info['axis'], \
                            ax
                    yield assert_equal, frb[proj_field].info['field'], \
                            proj_field
                    yield assert_equal, frb[proj_field].info['units'], \
                            pf.field_info[proj_field].get_units()
                    yield assert_equal, frb[proj_field].info['xlim'], \
                            frb.bounds[:2]
                    yield assert_equal, frb[proj_field].info['ylim'], \
                            frb.bounds[2:]
                    yield assert_equal, frb[proj_field].info['length_to_cm'], \
                            pf['cm']
                    yield assert_equal, frb[proj_field].info['center'], \
                            proj.center
                    yield assert_equal, frb[proj_field].info['weight_field'], \
                            wf
                teardown_func(fns)
            # wf == None
            yield assert_equal, wf, None
            v1 = proj["Density"].sum()
            v2 = (dd["Density"] * dd["d%s" % an]).sum()
            yield assert_rel_equal, v1, v2, 10


