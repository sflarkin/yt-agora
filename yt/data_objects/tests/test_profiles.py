from yt.testing import *
from yt.data_objects.profiles import \
    BinnedProfile1D, BinnedProfile2D, BinnedProfile3D, \
    Profile1D, Profile2D, Profile3D

_fields = ("density", "temperature", "dinosaurs", "tribbles")

def test_binned_profiles():
    pf = fake_random_pf(64, nprocs = 8, fields = _fields)
    nv = pf.domain_dimensions.prod()
    dd = pf.h.all_data()
    (rmi, rma), (tmi, tma), (dmi, dma) = dd.quantities["Extrema"](
        ["density", "temperature", "dinosaurs"])
    rt, tt, dt = dd.quantities["TotalQuantity"](
        ["density", "temperature", "dinosaurs"])
    # First we look at the 
    for nb in [8, 16, 32, 64]:
        # We log all the fields or don't log 'em all.  No need to do them
        # individually.
        for lf in [True, False]: 
            # We have the min and the max, but to avoid cutting them off
            # since we aren't doing end-collect, we cut a bit off the edges
            for ec, e1, e2 in [(False, 0.9, 1.1), (True, 1.0, 1.0)]:
                p1d = BinnedProfile1D(dd, 
                    nb, "density", rmi*e1, rma*e2, lf,
                    end_collect=ec)
                p1d.add_fields(["ones", "temperature"], weight=None)
                yield assert_equal, p1d["ones"].sum(), nv
                yield assert_rel_equal, tt, p1d["temperature"].sum(), 7

                p2d = BinnedProfile2D(dd, 
                    nb, "density", rmi*e1, rma*e2, lf,
                    nb, "temperature", tmi*e1, tma*e2, lf,
                    end_collect=ec)
                p2d.add_fields(["ones", "temperature"], weight=None)
                yield assert_equal, p2d["ones"].sum(), nv
                yield assert_rel_equal, tt, p2d["temperature"].sum(), 7

                p3d = BinnedProfile3D(dd, 
                    nb, "density", rmi*e1, rma*e2, lf,
                    nb, "temperature", tmi*e1, tma*e2, lf,
                    nb, "dinosaurs", dmi*e1, dma*e2, lf,
                    end_collect=ec)
                p3d.add_fields(["ones", "temperature"], weight=None)
                yield assert_equal, p3d["ones"].sum(), nv
                yield assert_rel_equal, tt, p3d["temperature"].sum(), 7

        p1d = BinnedProfile1D(dd, nb, "x", 0.0, 1.0, log_space=False)
        p1d.add_fields("ones", weight=None)
        av = nv / nb
        yield assert_equal, p1d["ones"][:-1], np.ones(nb)*av
        # We re-bin ones with a weight now
        p1d.add_fields(["ones"], weight="temperature")
        yield assert_equal, p1d["ones"][:-1], np.ones(nb)

        p2d = BinnedProfile2D(dd, nb, "x", 0.0, 1.0, False,
                                  nb, "y", 0.0, 1.0, False)
        p2d.add_fields("ones", weight=None)
        av = nv / nb**2
        yield assert_equal, p2d["ones"][:-1,:-1], np.ones((nb, nb))*av
        # We re-bin ones with a weight now
        p2d.add_fields(["ones"], weight="temperature")
        yield assert_equal, p2d["ones"][:-1,:-1], np.ones((nb, nb))

        p3d = BinnedProfile3D(dd, nb, "x", 0.0, 1.0, False,
                                  nb, "y", 0.0, 1.0, False,
                                  nb, "z", 0.0, 1.0, False)
        p3d.add_fields("ones", weight=None)
        av = nv / nb**3
        yield assert_equal, p3d["ones"][:-1,:-1,:-1], np.ones((nb, nb, nb))*av
        # We re-bin ones with a weight now
        p3d.add_fields(["ones"], weight="temperature")
        yield assert_equal, p3d["ones"][:-1,:-1,:-1], np.ones((nb,nb,nb))

def test_profiles():
    pf = fake_random_pf(64, nprocs = 8, fields = _fields)
    nv = pf.domain_dimensions.prod()
    dd = pf.h.all_data()
    (rmi, rma), (tmi, tma), (dmi, dma) = dd.quantities["Extrema"](
        ["density", "temperature", "dinosaurs"])
    rt, tt, dt = dd.quantities["TotalQuantity"](
        ["density", "temperature", "dinosaurs"])
    # First we look at the 
    e1, e2 = 0.9, 1.1
    for nb in [8, 16, 32, 64]:
        # We log all the fields or don't log 'em all.  No need to do them
        # individually.
        for lf in [True, False]: 
            p1d = Profile1D(dd, 
                "density",     nb, rmi*e1, rma*e2, lf,
                weight_field = None)
            p1d.add_fields(["ones", "temperature"])
            yield assert_equal, p1d["ones"].sum(), nv
            yield assert_rel_equal, tt, p1d["temperature"].sum(), 7

            p2d = Profile2D(dd, 
                "density",     nb, rmi*e1, rma*e2, lf,
                "temperature", nb, tmi*e1, tma*e2, lf,
                weight_field = None)
            p2d.add_fields(["ones", "temperature"])
            yield assert_equal, p2d["ones"].sum(), nv
            yield assert_rel_equal, tt, p2d["temperature"].sum(), 7

            p3d = Profile3D(dd, 
                "density",     nb, rmi*e1, rma*e2, lf,
                "temperature", nb, tmi*e1, tma*e2, lf,
                "dinosaurs",   nb, dmi*e1, dma*e2, lf,
                weight_field = None)
            p3d.add_fields(["ones", "temperature"])
            yield assert_equal, p3d["ones"].sum(), nv
            yield assert_rel_equal, tt, p3d["temperature"].sum(), 7

        p1d = Profile1D(dd, "x", nb, 0.0, 1.0, False,
                        weight_field = None)
        p1d.add_fields("ones")
        av = nv / nb
        yield assert_equal, p1d["ones"], np.ones(nb)*av

        # We re-bin ones with a weight now
        p1d = Profile1D(dd, "x", nb, 0.0, 1.0, False,
                        weight_field = "temperature")
        p1d.add_fields(["ones"])
        yield assert_equal, p1d["ones"], np.ones(nb)

        p2d = Profile2D(dd, "x", nb, 0.0, 1.0, False,
                            "y", nb, 0.0, 1.0, False,
                            weight_field = None)
        p2d.add_fields("ones")
        av = nv / nb**2
        yield assert_equal, p2d["ones"], np.ones((nb, nb))*av

        # We re-bin ones with a weight now
        p2d = Profile2D(dd, "x", nb, 0.0, 1.0, False,
                            "y", nb, 0.0, 1.0, False,
                            weight_field = "temperature")
        p2d.add_fields(["ones"])
        yield assert_equal, p2d["ones"], np.ones((nb, nb))

        p3d = Profile3D(dd, "x", nb, 0.0, 1.0, False,
                            "y", nb, 0.0, 1.0, False,
                            "z", nb, 0.0, 1.0, False,
                            weight_field = None)
        p3d.add_fields("ones")
        av = nv / nb**3
        yield assert_equal, p3d["ones"], np.ones((nb, nb, nb))*av

        # We re-bin ones with a weight now
        p3d = Profile3D(dd, "x", nb, 0.0, 1.0, False,
                            "y", nb, 0.0, 1.0, False,
                            "z", nb, 0.0, 1.0, False,
                            weight_field = "temperature")
        p3d.add_fields(["ones"])
        yield assert_equal, p3d["ones"], np.ones((nb,nb,nb))

