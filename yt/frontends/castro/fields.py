"""
Castro-specific fields

Author: J. S. Oishi <jsoishi@gmail.com>
Affiliation: UC Berkeley
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2008-2010 J. S. Oishi, Matthew Turk.  All Rights Reserved.

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
from yt.utilities.physical_constants import \
    mh, kboltz
from yt.data_objects.field_info_container import \
    FieldInfoContainer, \
    FieldInfo, \
    NullFunc, \
    TranslationFunc, \
    ValidateParameter, \
    ValidateDataField, \
    ValidateProperty, \
    ValidateSpatial, \
    ValidateGridType
import yt.data_objects.universal_fields

CastroFieldInfo = FieldInfoContainer.create_with_fallback(FieldInfo)
add_field = CastroFieldInfo.add_field

KnownCastroFields = FieldInfoContainer()
add_castro_field = KnownCastroFields.add_field

# def _convertDensity(data):
#     return data.convert("Density")
add_castro_field("density", function=NullFunc, take_log=True,
          units=r"\rm{g}/\rm{cm}^3",
CastroFieldInfo["density"]._projected_units =r"\rm{g}/\rm{cm}^2"
#CastroFieldInfo["density"]._convert_function=_convertDensity

add_castro_field("eden", function=NullFunc, take_log=True,
          validators = [ValidateDataField("eden")],
          units=r"\rm{erg}/\rm{cm}^3")

add_castro_field("xmom", function=NullFunc, take_log=False,
          validators = [ValidateDataField("xmom")],
          units=r"\rm{g}/\rm{cm^2\ s}")

add_castro_field("ymom", function=NullFunc, take_log=False,
          validators = [ValidateDataField("ymom")],
          units=r"\rm{gm}/\rm{cm^2\ s}")

add_castro_field("zmom", function=NullFunc, take_log=False,
          validators = [ValidateDataField("zmom")],
          units=r"\rm{g}/\rm{cm^2\ s}")

translation_dict = {"x-velocity": "xvel",
                    "y-velocity": "yvel",
                    "z-velocity": "zvel",
                    "Density": "density",
                    "Total_Energy": "eden",
                    "Temperature": "temperature",
                    "x-momentum": "xmom",
                    "y-momentum": "ymom",
                    "z-momentum": "zmom"
                   }

for f, v in translation_dict.items():
    add_field(theirs, function=TranslationFunc(mine),
              take_log=KnownCastroFields[theirs].take_log)

# Now fallbacks, in case these fields are not output
def _xVelocity(field, data):
    """generate x-velocity from x-momentum and density

    """
    return data["xmom"]/data["density"]
add_field("x-velocity", function=_xVelocity, take_log=False,
          units=r'\rm{cm}/\rm{s}')

def _yVelocity(field, data):
    """generate y-velocity from y-momentum and density

    """
    return data["ymom"]/data["density"]
add_field("y-velocity", function=_yVelocity, take_log=False,
          units=r'\rm{cm}/\rm{s}')

def _zVelocity(field, data):
    """generate z-velocity from z-momentum and density

    """
    return data["zmom"]/data["density"]
add_field("z-velocity", function=_zVelocity, take_log=False,
          units=r'\rm{cm}/\rm{s}')

def _ThermalEnergy(field, data):
    """generate thermal (gas energy). Dual Energy Formalism was
        implemented by Stella, but this isn't how it's called, so I'll
        leave that commented out for now.
    """
    #if data.pf["DualEnergyFormalism"]:
    #    return data["Gas_Energy"]
    #else:
    return data["Total_Energy"] - 0.5 * data["density"] * (
        data["x-velocity"]**2.0
        + data["y-velocity"]**2.0
        + data["z-velocity"]**2.0 )
add_field("ThermalEnergy", function=_ThermalEnergy,
          units=r"\rm{ergs}/\rm{cm^3}")

def _Pressure(field, data):
    """M{(Gamma-1.0)*e, where e is thermal energy density
       NB: this will need to be modified for radiation
    """
    return (data.pf["Gamma"] - 1.0)*data["ThermalEnergy"]
add_field("Pressure", function=_Pressure, units=r"\rm{dyne}/\rm{cm}^{2}")

def _Temperature(field, data):
    return (data.pf["Gamma"]-1.0)*data.pf["mu"]*mh*data["ThermalEnergy"]/(kboltz*data["Density"])
add_field("Temperature", function=_Temperature, units=r"\rm{Kelvin}", take_log=False)

def _convertParticleMassMsun(data):
    return 1.0/1.989e33
def _ParticleMassMsun(field, data):
    return data["particle_mass"]
add_field("ParticleMassMsun",
          function=_ParticleMassMsun, validators=[ValidateSpatial(0)],
          particle_type=True, convert_function=_convertParticleMassMsun,
          particle_convert_function=_ParticleMassMsun)

# Fundamental fields that are usually/always output:
#   density
#   xmom
#   ymom
#   zmom
#   rho_E
#   rho_e
#   Temp
#
# "Derived" fields that are sometimes output:
#   x_velocity
#   y_velocity
#   z_velocity
#   magvel
#   grav_x
#   grav_y
#   grav_z
#   maggrav
#   magvort
#   pressure
#   entropy
#   divu
#   eint_e (e as derived from the "rho e" variable)
#   eint_E (e as derived from the "rho E" variable)
