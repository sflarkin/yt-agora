"""
Miscellaneous FITS routines
"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
from yt.fields.api import add_field
from yt.fields.derived_field import ValidateSpatial
from yt.funcs import mylog
from .data_structures import ap

def _make_counts(emin, emax):
    def _counts(field, data):
        e = data["event_energy"].in_units("keV")
        mask = np.logical_and(e >= emin, e < emax)
        x = data["event_x"][mask]
        y = data["event_y"][mask]
        z = np.ones(x.shape)
        pos = np.array([x,y,z]).transpose()
        img = data.deposit(pos, method="count")
        if data.has_field_parameter("sigma"):
            sigma = data.get_field_parameter("sigma")
        else:
            sigma = None
        if sigma is not None and sigma > 0.0:
            kern = ap.conv.Gaussian2DKernel(stddev=sigma)
            img[:,:,0] = ap.conv.convolve(img[:,:,0], kern)
        return data.pf.arr(img, "counts/pixel")
    return _counts

def setup_counts_fields(ebounds):
    for (emin, emax) in ebounds:
        cfunc = _make_counts(emin, emax)
        fname = "counts_%s-%s" % (emin, emax)
        mylog.info("Creating counts field %s." % fname)
        add_field(("gas",fname), function=cfunc,
                  units="counts/pixel",
                  validators = [ValidateSpatial()],
                  display_name="Counts (%s-%s keV)" % (emin, emax))