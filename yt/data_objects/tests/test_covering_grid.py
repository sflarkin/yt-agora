from yt.testing import *
from yt.data_objects.profiles import \
    BinnedProfile1D, BinnedProfile2D, BinnedProfile3D

def setup():
    from yt.config import ytcfg
    ytcfg["yt","__withintesting"] = "True"

def test_covering_grid():
    # We decompose in different ways
    for level in [0, 1, 2]:
        for nprocs in [1, 2, 4, 8]:
            pf = fake_random_pf(16, nprocs = nprocs)
            dn = pf.refine_by**level 
            cg = pf.h.covering_grid(level, [0.0, 0.0, 0.0],
                    dn * pf.domain_dimensions)
            # Test coordinate generation
            yield assert_equal, np.unique(cg["dx"]).size, 1
            xmi = cg["x"].min()
            xma = cg["x"].max()
            dx = cg["dx"].flat[0:1]
            edges = pf.arr([[0,1],[0,1],[0,1]], 'code_length')
            yield assert_equal, xmi, edges[0,0] + dx/2.0
            yield assert_equal, xmi, cg["x"][0,0,0]
            yield assert_equal, xmi, cg["x"][0,1,1]
            yield assert_equal, xma, edges[0,1] - dx/2.0
            yield assert_equal, xma, cg["x"][-1,0,0]
            yield assert_equal, xma, cg["x"][-1,1,1]
            yield assert_equal, np.unique(cg["dy"]).size, 1
            ymi = cg["y"].min()
            yma = cg["y"].max()
            dy = cg["dy"][0]
            yield assert_equal, ymi, edges[1,0] + dy/2.0
            yield assert_equal, ymi, cg["y"][0,0,0]
            yield assert_equal, ymi, cg["y"][1,0,1]
            yield assert_equal, yma, edges[1,1] - dy/2.0
            yield assert_equal, yma, cg["y"][0,-1,0]
            yield assert_equal, yma, cg["y"][1,-1,1]
            yield assert_equal, np.unique(cg["dz"]).size, 1
            zmi = cg["z"].min()
            zma = cg["z"].max()
            dz = cg["dz"][0]
            yield assert_equal, zmi, edges[2,0] + dz/2.0
            yield assert_equal, zmi, cg["z"][0,0,0]
            yield assert_equal, zmi, cg["z"][1,1,0]
            yield assert_equal, zma, edges[2,1] - dz/2.0
            yield assert_equal, zma, cg["z"][0,0,-1]
            yield assert_equal, zma, cg["z"][1,1,-1]
            # Now we test other attributes
            yield assert_equal, cg["ones"].max(), 1.0
            yield assert_equal, cg["ones"].min(), 1.0
            yield assert_equal, cg["grid_level"], 0
            yield assert_equal, cg["cell_volume"].sum(), pf.domain_width.prod()
            for g in pf.h.grids:
                di = g.get_global_startindex()
                dd = g.ActiveDimensions
                for i in range(dn):
                    f = cg["density"][dn*di[0]+i:dn*(di[0]+dd[0])+i:dn,
                                      dn*di[1]+i:dn*(di[1]+dd[1])+i:dn,
                                      dn*di[2]+i:dn*(di[2]+dd[2])+i:dn]
                    yield assert_equal, f, g["density"]

def test_smoothed_covering_grid():
    # We decompose in different ways
    for level in [0, 1, 2]:
        for nprocs in [1, 2, 4, 8]:
            pf = fake_random_pf(16, nprocs = nprocs)
            dn = pf.refine_by**level 
            cg = pf.h.smoothed_covering_grid(level, [0.0, 0.0, 0.0],
                    dn * pf.domain_dimensions)
            yield assert_equal, cg["ones"].max(), 1.0
            yield assert_equal, cg["ones"].min(), 1.0
            yield assert_equal, cg["cell_volume"].sum(), pf.domain_width.prod()
            for g in pf.h.grids:
                if level != g.Level: continue
                di = g.get_global_startindex()
                dd = g.ActiveDimensions
                for i in range(dn):
                    f = cg["density"][dn*di[0]+i:dn*(di[0]+dd[0])+i:dn,
                                      dn*di[1]+i:dn*(di[1]+dd[1])+i:dn,
                                      dn*di[2]+i:dn*(di[2]+dd[2])+i:dn]
                    yield assert_equal, f, g["density"]
