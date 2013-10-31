from yt.testing import *

def setup():
    from yt.config import ytcfg
    ytcfg["yt","__withintesting"] = "True"

def test_cut_region():
    # We decompose in different ways
    return #TESTDISABLED
    for nprocs in [1, 2, 4, 8]:
        pf = fake_random_pf(64, nprocs = nprocs,
            fields = ("density", "Temperature", "velocity_x"))
        # We'll test two objects
        dd = pf.h.all_data()
        r = dd.cut_region( [ "grid['Temperature'] > 0.5",
                             "grid['density'] < 0.75",
                             "grid['velocity_x'] > 0.25" ])
        t = ( (dd["Temperature"] > 0.5 ) 
            & (dd["density"] < 0.75 )
            & (dd["velocity_x"] > 0.25 ) )
        yield assert_equal, np.all(r["Temperature"] > 0.5), True
        yield assert_equal, np.all(r["density"] < 0.75), True
        yield assert_equal, np.all(r["velocity_x"] > 0.25), True
        yield assert_equal, np.sort(dd["density"][t]), np.sort(r["density"])
        yield assert_equal, np.sort(dd["x"][t]), np.sort(r["x"])
        r2 = r.cut_region( [ "grid['Temperature'] < 0.75" ] )
        t2 = (r["Temperature"] < 0.75)
        yield assert_equal, np.sort(r2["Temperature"]), np.sort(r["Temperature"][t2])
        yield assert_equal, np.all(r2["Temperature"] < 0.75), True

def test_extract_region():
    # We decompose in different ways
    return #TESTDISABLED
    for nprocs in [1, 2, 4, 8]:
        pf = fake_random_pf(64, nprocs = nprocs,
            fields = ("density", "Temperature", "velocity_x"))
        # We'll test two objects
        dd = pf.h.all_data()
        t = ( (dd["Temperature"] > 0.5 ) 
            & (dd["density"] < 0.75 )
            & (dd["velocity_x"] > 0.25 ) )
        r = dd.extract_region(t)
        yield assert_equal, np.all(r["Temperature"] > 0.5), True
        yield assert_equal, np.all(r["density"] < 0.75), True
        yield assert_equal, np.all(r["velocity_x"] > 0.25), True
        yield assert_equal, np.sort(dd["density"][t]), np.sort(r["density"])
        yield assert_equal, np.sort(dd["x"][t]), np.sort(r["x"])
        t2 = (r["Temperature"] < 0.75)
        r2 = r.cut_region( [ "grid['Temperature'] < 0.75" ] )
        yield assert_equal, np.sort(r2["Temperature"]), np.sort(r["Temperature"][t2])
        yield assert_equal, np.all(r2["Temperature"] < 0.75), True
        t3 = (r["Temperature"] < 0.75)
        r3 = r.extract_region( t3 )
        yield assert_equal, np.sort(r3["Temperature"]), np.sort(r["Temperature"][t3])
        yield assert_equal, np.all(r3["Temperature"] < 0.75), True
