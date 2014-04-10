"""
Geographic fields




"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
from .coordinate_handler import \
    CoordinateHandler, \
    _unknown_coord, \
    _get_coord_fields
import yt.visualization._MPL as _MPL
from yt.utilities.lib.misc_utilities import \
    pixelize_cylinder, pixelize_aitoff

class GeographicCoordinateHandler(CoordinateHandler):

    def __init__(self, pf, ordering = 'latlonalt'):
        if ordering != 'latlonalt': raise NotImplementedError
        super(GeographicCoordinateHandler, self).__init__(pf)

    def setup_fields(self, registry):
        # return the fields for r, z, theta
        registry.add_field(("index", "dx"), function=_unknown_coord)
        registry.add_field(("index", "dy"), function=_unknown_coord)
        registry.add_field(("index", "dz"), function=_unknown_coord)
        registry.add_field(("index", "x"), function=_unknown_coord)
        registry.add_field(("index", "y"), function=_unknown_coord)
        registry.add_field(("index", "z"), function=_unknown_coord)
        f1, f2 = _get_coord_fields(0, "")
        registry.add_field(("index", "dlatitude"), function = f1,
                           display_field = False,
                           units = "")
        registry.add_field(("index", "latitude"), function = f2,
                           display_field = False,
                           units = "")

        f1, f2 = _get_coord_fields(1, "")
        registry.add_field(("index", "dlongitude"), function = f1,
                           display_field = False,
                           units = "")
        registry.add_field(("index", "longitude"), function = f2,
                           display_field = False,
                           units = "")

        f1, f2 = _get_coord_fields(2)
        registry.add_field(("index", "daltitude"), function = f1,
                           display_field = False,
                           units = "code_length")
        registry.add_field(("index", "altitude"), function = f2,
                           display_field = False,
                           units = "code_length")

        def _SphericalVolume(field, data):
            # r**2 sin theta dr dtheta dphi
            # We can use the transformed coordinates here.
            vol = data["index", "r"]**2.0
            vol *= data["index", "dr"]
            vol *= np.sin(data["index", "theta"])
            vol *= data["index", "dtheta"]
            vol *= data["index", "dphi"]
            return vol
        registry.add_field(("index", "cell_volume"),
                 function=_SphericalVolume,
                 units = "code_length**3")

        # Altitude is the radius from the central zone minus the radius of the
        # surface.
        def _altitude_to_radius(field, data):
            surface_height = data.get_field_parameter("surface_height")
            if surface_height is None:
                surface_height = getattr(data.pf, "surface_height", 0.0)
            return data["altitude"] + surface_height
        registry.add_field(("index", "r"),
                 function=_altitude_to_radius,
                 units = "code_length")
        registry.alias(("index", "dr"), ("index", "daltitude"))

        def _longitude_to_theta(field, data):
            # longitude runs from -180 to 180.
            return (data["longitude"] + 180) * np.pi/180.0
        registry.add_field(("index", "theta"),
                 function = _longitude_to_theta,
                 units = "")
        def _dlongitude_to_dtheta(field, data):
            return data["dlongitude"] * np.pi/180.0
        registry.add_field(("index", "dtheta"),
                 function = _dlongitude_to_dtheta,
                 units = "")

        def _latitude_to_phi(field, data):
            # latitude runs from -180 to 180.
            return (data["latitude"] + 90) * np.pi/180.0
        registry.add_field(("index", "phi"),
                 function = _latitude_to_phi,
                 units = "")
        def _dlatitude_to_dphi(field, data):
            return data["dlatitude"] * np.pi/180.0
        registry.add_field(("index", "dphi"),
                 function = _dlatitude_to_dphi,
                 units = "")

    def pixelize(self, dimension, data_source, field, bounds, size,
                 antialias = True, periodic = True):
        if dimension in (0, 1):
            return self._cyl_pixelize(data_source, field, bounds, size,
                                          antialias, dimension)
        elif dimension == 2:
            return self._ortho_pixelize(data_source, field, bounds, size,
                                        antialias, dimension, periodic)
        else:
            raise NotImplementedError

    def _ortho_pixelize(self, data_source, field, bounds, size, antialias,
                        dim, periodic):
        # We should be using fcoords
        buff = pixelize_aitoff(data_source["theta"], data_source["dtheta"]/2.0,
                               data_source["phi"], data_source["dphi"]/2.0,
                               size, data_source[field], None, None).transpose()
        return buff

    def _cyl_pixelize(self, data_source, field, bounds, size, antialias,
                      dimension):
        if dimension == 0:
            buff = pixelize_cylinder(data_source['r'],
                                     data_source['dr'] / 2.0,
                                     data_source['theta'],
                                     data_source['dtheta'] / 2.0, # half-widths
                                     size, data_source[field], bounds)
            buff = pixelize_cylinder(data_source['r'],
                                     data_source['dr'] / 2.0,
                                     2.0*np.pi - data_source['theta'],
                                     data_source['dtheta'] / 2.0, # half-widths
                                     size, data_source[field], bounds,
                                     input_img = buff)
        elif dimension == 1:
            buff = pixelize_cylinder(data_source['r'],
                                     data_source['dr'] / 2.0,
                                     data_source['phi'],
                                     data_source['dphi'] / 2.0, # half-widths
                                     size, data_source[field], bounds)
        else:
            raise RuntimeError
        return buff


    def convert_from_cartesian(self, coord):
        raise NotImplementedError

    def convert_to_cartesian(self, coord):
        raise NotImplementedError

    def convert_to_cylindrical(self, coord):
        raise NotImplementedError

    def convert_from_cylindrical(self, coord):
        raise NotImplementedError

    def convert_to_spherical(self, coord):
        raise NotImplementedError

    def convert_from_spherical(self, coord):
        raise NotImplementedError

    # Despite being mutables, we uses these here to be clear about how these
    # are generated and to ensure that they are not re-generated unnecessarily
    axis_name = { 0  : 'latitude',  1  : 'longitude',  2  : 'altitude',
                 'latitude' : 'latitude',
                 'longitude' : 'longitude', 
                 'altitude' : 'altitude',
                 'Latitude' : 'latitude',
                 'Longitude' : 'longitude', 
                 'Altitude' : 'altitude',
                 'lat' : 'latitude',
                 'lon' : 'longitude', 
                 'alt' : 'altitude' }

    axis_id = { 'latitude' : 0, 'longitude' : 1, 'altitude' : 2,
                 0  : 0,  1  : 1,  2  : 2}

    x_axis = { 'latitude' : 1, 'longitude' : 0, 'altitude' : 0,
                0  : 1,  1  : 0,  2  : 0}

    y_axis = { 'latitude' : 2, 'longitude' : 2, 'altitude' : 1,
                0  : 2,  1  : 2,  2  : 1}

    @property
    def period(self):
        return self.pf.domain_width

