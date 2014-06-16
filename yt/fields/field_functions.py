"""
General field-related functions.



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2014, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np

def get_radius(data, field_prefix):
    center = data.get_field_parameter("center").in_units("cm")
    DW = (data.ds.domain_right_edge - data.ds.domain_left_edge).in_units("cm")
    # This is in cm**2 so it can be the destination for our r later.
    radius2 = data.ds.arr(np.zeros(data[field_prefix+"x"].shape,
                         dtype='float64'), 'cm**2')
    r = radius2.copy()
    if any(data.ds.periodicity):
        rdw = radius2.copy()
    for i, ax in enumerate('xyz'):
        # This will coerce the units, so we don't need to worry that we copied
        # it from a cm**2 array.
        np.subtract(data["%s%s" % (field_prefix, ax)].in_units("cm"),
                    center[i], r)
        if data.ds.periodicity[i] == True:
            np.abs(r, r)
            np.subtract(r, DW[i], rdw)
            np.abs(rdw, rdw)
            np.minimum(r, rdw, r)
        np.power(r, 2.0, r)
        np.add(radius2, r, radius2)
        if data.ds.dimensionality < i+1:
            break
    # Now it's cm.
    np.sqrt(radius2, radius2)
    # Alias it, just for clarity.
    radius = radius2
    return radius
