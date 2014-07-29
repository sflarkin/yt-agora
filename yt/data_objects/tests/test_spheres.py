from yt.data_objects.profiles import create_profile
from yt.testing import *
import numpy as np

def setup():
    from yt.config import ytcfg
    ytcfg["yt","__withintesting"] = "True"

def test_domain_sphere():
    ds = fake_random_ds(16, fields = ("density"))
    sp = ds.sphere(ds.domain_center, ds.domain_width[0])

    # Now we test that we can get different radial velocities based on field
    # parameters.

    # Get the first sphere
    ds = fake_random_ds(16, fields = ("density",
      "velocity_x", "velocity_y", "velocity_z"))
    sp0 = ds.sphere(ds.domain_center, 0.25)

    # Compute the bulk velocity from the cells in this sphere
    bulk_vel = sp0.quantities.bulk_velocity()

    # Get the second sphere
    sp1 = ds.sphere(ds.domain_center, 0.25)

    # Set the bulk velocity field parameter
    sp1.set_field_parameter("bulk_velocity", bulk_vel)

    yield assert_equal, np.any(sp0["radial_velocity"] ==
                               sp1["radial_velocity"]), False

    # Radial profile without correction

    rp0 = create_profile(sp0, 'radius', 'radial_velocity',
                         units = {'radius': 'kpc'},
                         logs = {'radius': False})

    # Radial profile with correction for bulk velocity

    rp1 = create_profile(sp1, 'radius', 'radial_velocity',
                         units = {'radius': 'kpc'},
                         logs = {'radius': False})

    yield assert_equal, rp0.x_bins, rp1.x_bins
    yield assert_equal, np.any(rp0["radial_velocity"] ==
                               rp1["radial_velocity"]), False
