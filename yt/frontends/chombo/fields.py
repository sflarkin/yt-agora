"""
Chombo-specific fields

Author: J. S. Oishi <jsoishi@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2009-2011 J. S. Oishi, Matthew Turk.  All Rights Reserved.

  This file is part of yt.

  yt is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from yt.data_objects.field_info_container import \
    FieldInfoContainer, \
    FieldInfo, \
    ValidateParameter, \
    ValidateDataField, \
    ValidateProperty, \
    ValidateSpatial, \
    ValidateGridType
import yt.data_objects.universal_fields

ChomboFieldInfo = FieldInfoContainer.create_with_fallback(FieldInfo)
add_chombo_field = ChomboFieldInfo.add_field

add_field = add_chombo_field

add_field("density", function=NullFunc, take_log=True,
          validators = [ValidateDataField("density")],
          units=r"\rm{g}/\rm{cm}^3")

ChomboFieldInfo["density"]._projected_units =r"\rm{g}/\rm{cm}^2"

add_field("X-momentum", function=NullFunc, take_log=False,
          validators = [ValidateDataField("X-Momentum")],
          units=r"",display_name=r"B_x")
ChomboFieldInfo["X-momentum"]._projected_units=r""

add_field("Y-momentum", function=NullFunc, take_log=False,
          validators = [ValidateDataField("Y-Momentum")],
          units=r"",display_name=r"B_y")
ChomboFieldInfo["Y-momentum"]._projected_units=r""

add_field("Z-momentum", function=NullFunc, take_log=False,
          validators = [ValidateDataField("Z-Momentum")],
          units=r"",display_name=r"B_z")
ChomboFieldInfo["Z-momentum"]._projected_units=r""

add_field("X-magnfield", function=NullFunc, take_log=False,
          validators = [ValidateDataField("X-Magnfield")],
          units=r"",display_name=r"B_x")
ChomboFieldInfo["X-magnfield"]._projected_units=r""

add_field("Y-magnfield", function=NullFunc, take_log=False,
          validators = [ValidateDataField("Y-Magnfield")],
          units=r"",display_name=r"B_y")
ChomboFieldInfo["Y-magnfield"]._projected_units=r""

add_field("Z-magnfield", function=NullFunc, take_log=False,
          validators = [ValidateDataField("Z-Magnfield")],
          units=r"",display_name=r"B_z")
ChomboFieldInfo["Z-magnfield"]._projected_units=r""

def _MagneticEnergy(field,data):
    return (data["X-magnfield"]**2 +
            data["Y-magnfield"]**2 +
            data["Z-magnfield"]**2)/2.
add_field("MagneticEnergy", function=_MagneticEnergy, take_log=True,
          units=r"",display_name=r"B^2/8\pi")
ChomboFieldInfo["MagneticEnergy"]._projected_units=r""

def _xVelocity(field, data):
    """generate x-velocity from x-momentum and density

    """
    return data["X-momentum"]/data["density"]
add_field("x-velocity",function=_xVelocity, take_log=False,
          units=r'\rm{cm}/\rm{s}')

def _yVelocity(field,data):
    """generate y-velocity from y-momentum and density

    """
    #try:
    #    return data["xvel"]
    #except KeyError:
    return data["Y-momentum"]/data["density"]
add_field("y-velocity",function=_yVelocity, take_log=False,
          units=r'\rm{cm}/\rm{s}')

def _zVelocity(field,data):
    """generate z-velocity from z-momentum and density

    """
    return data["Z-momentum"]/data["density"]
add_field("z-velocity",function=_zVelocity, take_log=False,
          units=r'\rm{cm}/\rm{s}')
    
