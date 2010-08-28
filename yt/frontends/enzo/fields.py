"""
Fields specific to Enzo

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2008-2010 Matthew Turk.  All Rights Reserved.

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

import numpy as na

from yt.data_objects.field_info_container import \
    CodeFieldInfoContainer, \
    ValidateParameter, \
    ValidateDataField, \
    ValidateProperty, \
    ValidateSpatial, \
    ValidateGridType
import yt.data_objects.universal_fields
from yt.utilities.physical_constants import \
    mh
import yt.utilities.amr_utils as amr_utils

class EnzoFieldContainer(CodeFieldInfoContainer):
    """
    This is a container for Enzo-specific fields.
    """
    _shared_state = {}
    _field_list = {}
EnzoFieldInfo = EnzoFieldContainer()
add_enzo_field = EnzoFieldInfo.add_field

add_field = add_enzo_field

_speciesList = ["HI","HII","Electron",
               "HeI","HeII","HeIII",
               "H2I","H2II","HM",
               "DI","DII","HDI","Metal","PreShock"]
_speciesMass = {"HI":1.0,"HII":1.0,"Electron":1.0,
                "HeI":4.0,"HeII":4.0,"HeIII":4.0,
                "H2I":2.0,"H2II":2.0,"HM":1.0,
                "DI":2.0,"DII":2.0,"HDI":3.0}

def _SpeciesComovingDensity(field, data):
    sp = field.name.split("_")[0] + "_Density"
    ef = (1.0 + data.pf.current_redshift)**3.0
    return data[sp]/ef
def _SpeciesFraction(field, data):
    sp = field.name.split("_")[0] + "_Density"
    return data[sp]/data["Density"]
def _SpeciesNumberDensity(field, data):
    species = field.name.split("_")[0]
    sp = field.name.split("_")[0] + "_Density"
    return data[sp]/_speciesMass[species]
def _ConvertNumberDensity(data):
    return 1.0/mh

for species in _speciesList:
    add_field("%s_Fraction" % species,
             function=_SpeciesFraction,
             validators=ValidateDataField("%s_Density" % species))
    add_field("Comoving_%s_Density" % species,
             function=_SpeciesComovingDensity,
             validators=ValidateDataField("%s_Density" % species))
    if _speciesMass.has_key(species):
        add_field("%s_NumberDensity" % species,
                  function=_SpeciesNumberDensity,
                  convert_function=_ConvertNumberDensity,
                  validators=ValidateDataField("%s_Density" % species))

def _Metallicity(field, data):
    return data["Metal_Fraction"]
def _ConvertMetallicity(data):
    return 49.0196 # 1 / 0.0204
add_field("Metallicity", units=r"Z_{\rm{\odot}}",
          function=_Metallicity,
          convert_function=_ConvertMetallicity,
          validators=ValidateDataField("Metal_Density"),
          projection_conversion="1")

def _Metallicity3(field, data):
    return data["SN_Colour"]/data["Density"]
add_field("Metallicity3", units=r"Z_{\rm{\odot}}",
          function=_Metallicity3,
          convert_function=_ConvertMetallicity,
          validators=ValidateDataField("SN_Colour"),
          projection_conversion="1")

def _Cooling_Time(field, data):
    return data["Cooling_Time"]
add_field("Cooling_Time", units=r"\rm{s}",
          function=_Cooling_Time,
          validators=ValidateDataField("Cooling_Time"),
          projection_conversion="1")

def _ThermalEnergy(field, data):
    if data.pf["HydroMethod"] == 2:
        return data["Total_Energy"]
    else:
        if data.pf["DualEnergyFormalism"]:
            return data["GasEnergy"]
        else:
            return data["Total_Energy"] - 0.5*(
                   data["x-velocity"]**2.0
                 + data["y-velocity"]**2.0
                 + data["z-velocity"]**2.0 )
add_field("ThermalEnergy", function=_ThermalEnergy,
          units=r"\rm{ergs}/\rm{cm^3}")

def _KineticEnergy(field, data):
    return 0.5*data["Density"] * ( data["x-velocity"]**2.0
                                   + data["y-velocity"]**2.0
                                   + data["z-velocity"]**2.0 )
add_field("KineticEnergy",function=_KineticEnergy,
          units = r"\rm{ergs}/\rm{cm^3}")
# This next section is the energy field section
# Note that we have aliases that manually unconvert themselves.
# This is because numerous code branches use Gas_Energy or GasEnergy
# indiscriminately -- this is almost fixed with LCA1.5, but not everyone is
# moving to that branch.  So, because the actual function doesn't get called
# *unless* it's an alias, we simply de-convert -- since the input data is
# already converted to cgs.

def _convertEnergy(data):
    return data.convert("x-velocity")**2.0

def _GasEnergy(field, data):
    return data["Gas_Energy"] / _convertEnergy(data)
add_field("GasEnergy", function=_GasEnergy,
          units=r"\rm{ergs}/\rm{g}", convert_function=_convertEnergy)

def _Gas_Energy(field, data):
    return data["GasEnergy"] / _convertEnergy(data)
add_field("Gas_Energy", function=_Gas_Energy,
          units=r"\rm{ergs}/\rm{g}", convert_function=_convertEnergy)

def _TotalEnergy(field, data):
    return data["Total_Energy"] / _convertEnergy(data)
add_field("TotalEnergy", function=_TotalEnergy,
          display_name = "\mathrm{Total}\/\mathrm{Energy}",
          units=r"\rm{ergs}/\rm{g}", convert_function=_convertEnergy)

def _Total_Energy(field, data):
    return data["TotalEnergy"] / _convertEnergy(data)
add_field("Total_Energy", function=_Total_Energy,
          display_name = "\mathrm{Total}\/\mathrm{Energy}",
          units=r"\rm{ergs}/\rm{g}", convert_function=_convertEnergy)

def _NumberDensity(field, data):
    # We can assume that we at least have Density
    # We should actually be guaranteeing the presence of a .shape attribute,
    # but I am not currently implementing that
    fieldData = na.zeros(data["Density"].shape,
                         dtype = data["Density"].dtype)
    if data.pf["MultiSpecies"] == 0:
        if data.has_field_parameter("mu"):
            mu = data.get_field_parameter("mu")
        else:
            mu = 0.6
        fieldData += data["Density"] / mu
    if data.pf["MultiSpecies"] > 0:
        fieldData += data["HI_Density"] / 1.0
        fieldData += data["HII_Density"] / 1.0
        fieldData += data["HeI_Density"] / 4.0
        fieldData += data["HeII_Density"] / 4.0
        fieldData += data["HeIII_Density"] / 4.0
        fieldData += data["Electron_Density"] / 1.0
    if data.pf["MultiSpecies"] > 1:
        fieldData += data["HM_Density"] / 1.0
        fieldData += data["H2I_Density"] / 2.0
        fieldData += data["H2II_Density"] / 2.0
    if data.pf["MultiSpecies"] > 2:
        fieldData += data["DI_Density"] / 2.0
        fieldData += data["DII_Density"] / 2.0
        fieldData += data["HDI_Density"] / 3.0
    return fieldData
add_field("NumberDensity", units=r"\rm{cm}^{-3}",
          function=_NumberDensity,
          convert_function=_ConvertNumberDensity)

def _ComovingDensity(field,data):
    ef = (1.0 + data.pf.current_redshift)**3.0
    return data["Density"]/ef
add_field("ComovingDensity", function=_ComovingDensity, units=r"\rm{g}/\rm{cm}^3")

def Overdensity(field,data):
    return (data['Density'] + data['Dark_Matter_Density']) / \
        (rho_crit_now * (data.pf.hubble_constant**2) * ((1+data.pf.current_redshift)**3))
add_field("Overdensity",function=Overdensity,units=r"")

# Now we add all the fields that we want to control, but we give a null function
# This is every Enzo field we can think of.  This will be installation-dependent,

# removed: "Gas_Energy","Total_Energy",
# these are now aliases for each other

_default_fields = ["Density","Temperature",
                   "x-velocity","y-velocity","z-velocity",
                   "x-momentum","y-momentum","z-momentum"]
# else:
#     _default_fields = ["Density","Temperature","Gas_Energy","Total_Energy",
#                        "x-velocity","y-velocity","z-velocity"]
_default_fields += [ "%s_Density" % sp for sp in _speciesList ]

for field in _default_fields:
    add_field(field, function=lambda a, b: None, take_log=True,
              validators=[ValidateDataField(field)], units=r"\rm{g}/\rm{cm}^3")
EnzoFieldInfo["x-velocity"].projection_conversion='1'
EnzoFieldInfo["y-velocity"].projection_conversion='1'
EnzoFieldInfo["z-velocity"].projection_conversion='1'

# Now we override

def _convertDensity(data):
    return data.convert("Density")
for field in ["Density"] + [ "%s_Density" % sp for sp in _speciesList ]:
    EnzoFieldInfo[field]._units = r"\rm{g}/\rm{cm}^3"
    EnzoFieldInfo[field]._projected_units = r"\rm{g}/\rm{cm}^2"
    EnzoFieldInfo[field]._convert_function=_convertDensity

add_field("Dark_Matter_Density", function=lambda a,b: None,
          convert_function=_convertDensity,
          validators=[ValidateDataField("Dark_Matter_Density"),
                      ValidateSpatial(0)],
          display_name = "Dark\ Matter\ Density",
          not_in_all = True)

EnzoFieldInfo["Temperature"]._units = r"\rm{K}"
EnzoFieldInfo["Temperature"].units = r"K"

def _convertVelocity(data):
    return data.convert("x-velocity")
for ax in ['x','y','z']:
    f = EnzoFieldInfo["%s-velocity" % ax]
    f._units = r"\rm{cm}/\rm{s}"
    f._convert_function = _convertVelocity
    f.take_log = False

def _pdensity(field, data):
    blank = na.zeros(data.ActiveDimensions, dtype='float32')
    if data.NumberOfParticles == 0: return blank
    amr_utils.CICDeposit_3(data["particle_position_x"].astype(na.float64),
                           data["particle_position_y"].astype(na.float64),
                           data["particle_position_z"].astype(na.float64),
                           data["particle_mass"].astype(na.float32),
                           data["particle_mass"].size,
                           blank, na.array(data.LeftEdge).astype(na.float64),
                           na.array(data.ActiveDimensions).astype(na.int32), 
                           na.float64(data['dx']))
    return blank
add_field("particle_density", function=_pdensity,
          validators=[ValidateSpatial(0)], convert_function=_convertDensity)

def _spdensity(field, data):
    blank = na.zeros(data.ActiveDimensions, dtype='float32')
    if data.NumberOfParticles == 0: return blank
    filter = data['creation_time'] > 0.0
    if not filter.any(): return blank
    amr_utils.CICDeposit_3(data["particle_position_x"][filter].astype(na.float64),
                           data["particle_position_y"][filter].astype(na.float64),
                           data["particle_position_z"][filter].astype(na.float64),
                           data["particle_mass"][filter].astype(na.float32),
                           na.int64(na.where(filter)[0].size),
                           blank, na.array(data.LeftEdge).astype(na.float64),
                           na.array(data.ActiveDimensions).astype(na.int32), 
                           na.float64(data['dx']))
    return blank
add_field("star_density", function=_spdensity,
          validators=[ValidateSpatial(0)], convert_function=_convertDensity)

def _dmpdensity(field, data):
    blank = na.zeros(data.ActiveDimensions, dtype='float32')
    if data.NumberOfParticles == 0: return blank
    filter = data['creation_time'] <= 0.0
    if not filter.any(): return blank
    amr_utils.CICDeposit_3(data["particle_position_x"][filter].astype(na.float64),
                           data["particle_position_y"][filter].astype(na.float64),
                           data["particle_position_z"][filter].astype(na.float64),
                           data["particle_mass"][filter].astype(na.float32),
                           na.int64(na.where(filter)[0].size),
                           blank, na.array(data.LeftEdge).astype(na.float64),
                           na.array(data.ActiveDimensions).astype(na.int32), 
                           na.float64(data['dx']))
    return blank
add_field("dm_density", function=_dmpdensity,
          validators=[ValidateSpatial(0)], convert_function=_convertDensity)

def _star_field(field, data):
    """
    Create a grid field for star quantities, weighted by star mass.
    """
    particle_field = field.name[5:]
    top = na.zeros(data.ActiveDimensions, dtype='float32')
    if data.NumberOfParticles == 0: return top
    filter = data['creation_time'] > 0.0
    if not filter.any(): return top
    particle_field_data = data[particle_field][filter] * data['particle_mass'][filter]
    amr_utils.CICDeposit_3(data["particle_position_x"][filter].astype(na.float64),
                          data["particle_position_y"][filter].astype(na.float64),
                          data["particle_position_z"][filter].astype(na.float64),
                          particle_field_data.astype(na.float32),
                          na.int64(na.where(filter)[0].size),
                          top, na.array(data.LeftEdge).astype(na.float64),
                          na.array(data.ActiveDimensions).astype(na.int32), 
                          na.float64(data['dx']))
    del particle_field_data

    bottom = na.zeros(data.ActiveDimensions, dtype='float32')
    amr_utils.CICDeposit_3(data["particle_position_x"][filter].astype(na.float64),
                          data["particle_position_y"][filter].astype(na.float64),
                          data["particle_position_z"][filter].astype(na.float64),
                          data["particle_mass"][filter].astype(na.float32),
                          na.int64(na.where(filter)[0].size),
                          bottom, na.array(data.LeftEdge).astype(na.float64),
                          na.array(data.ActiveDimensions).astype(na.int32), 
                          na.float64(data['dx']))
    top[bottom == 0] = 0.0
    bnz = bottom.nonzero()
    top[bnz] /= bottom[bnz]
    return top

add_field('star_metallicity_fraction', function=_star_field,
          validators=[ValidateSpatial(0)])
add_field('star_creation_time', function=_star_field,
          validators=[ValidateSpatial(0)])
add_field('star_dynamical_time', function=_star_field,
          validators=[ValidateSpatial(0)])

def _StarMetallicity(field, data):
    return data['star_metallicity_fraction']
add_field('StarMetallicity', units=r"Z_{\rm{\odot}}",
          function=_StarMetallicity,
          convert_function=_ConvertMetallicity,
          projection_conversion="1")

def _StarCreationTime(field, data):
    return data['star_creation_time']
def _ConvertEnzoTimeYears(data):
    return data.pf.time_units['years']
add_field('StarCreationTimeYears', units=r"\mathrm{yr}",
          function=_StarCreationTime,
          convert_function=_ConvertEnzoTimeYears,
          projection_conversion="1")

def _StarDynamicalTime(field, data):
    return data['star_dynamical_time']
add_field('StarDynamicalTimeYears', units=r"\mathrm{yr}",
          function=_StarDynamicalTime,
          convert_function=_ConvertEnzoTimeYears,
          projection_conversion="1")

def _StarAge(field, data):
    star_age = na.zeros(data['StarCreationTimeYears'].shape)
    with_stars = data['StarCreationTimeYears'] > 0
    star_age[with_stars] = data.pf.time_units['years'] * \
        data.pf.current_time - \
        data['StarCreationTimeYears'][with_stars]
    return star_age
add_field('StarAgeYears', units=r"\mathrm{yr}",
          function=_StarAge,
          projection_conversion="1")

def _IsStarParticle(field, data):
    is_star = (data['creation_time'] > 0).astype('float64')
    return is_star
add_field('IsStarParticle', function=_IsStarParticle,
          particle_type = True)

#
# Now we do overrides for 2D fields
#

class Enzo2DFieldContainer(CodeFieldInfoContainer):
    _shared_state = {}
    _field_list = EnzoFieldContainer._field_list.copy()
# We make a copy of the dict from the other, so we
# can now update it...
Enzo2DFieldInfo = Enzo2DFieldContainer()
add_enzo_2d_field = Enzo2DFieldInfo.add_field

def _CellArea(field, data):
    if data['dx'].size == 1:
        try:
            return data['dx']*data['dy']*\
                na.ones(data.ActiveDimensions, dtype='float64')
        except AttributeError:
            return data['dx']*data['dy']
    return data["dx"]*data["dy"]
def _ConvertCellAreaMpc(data):
    return data.convert("mpc")**2.0
def _ConvertCellAreaCGS(data):
    return data.convert("cm")**2.0
add_enzo_2d_field("CellAreaCode", units=r"\rm{BoxArea}^2",
          function=_CellArea)
add_enzo_2d_field("CellAreaMpc", units=r"\rm{Mpc}^2",
          function=_CellArea,
          convert_function=_ConvertCellAreaMpc)
add_enzo_2d_field("CellArea", units=r"\rm{cm}^2",
          function=_CellArea,
          convert_function=_ConvertCellAreaCGS)

for a in ["Code", "Mpc", ""]:
    Enzo2DFieldInfo["CellVolume%s" % a] = \
        Enzo2DFieldInfo["CellArea%s" % a]

def _zvel(field, data):
    return na.zeros(data["x-velocity"].shape,
                    dtype='float64')
add_enzo_2d_field("z-velocity", function=_zvel)


#
# Now we do overrides for 1D fields
#

class Enzo1DFieldContainer(CodeFieldInfoContainer):
    _shared_state = {}
    _field_list = EnzoFieldContainer._field_list.copy()
# We make a copy of the dict from the other, so we
# can now update it...
Enzo1DFieldInfo = Enzo1DFieldContainer()
add_enzo_1d_field = Enzo1DFieldInfo.add_field

def _CellLength(field, data):
    return data["dx"]
def _ConvertCellLengthMpc(data):
    return data.convert("mpc")
def _ConvertCellLengthCGS(data):
    return data.convert("cm")
add_enzo_1d_field("CellLengthCode", units=r"\rm{BoxArea}^2",
          function=_CellLength)
add_enzo_1d_field("CellLengthMpc", units=r"\rm{Mpc}^2",
          function=_CellLength,
          convert_function=_ConvertCellLengthMpc)
add_enzo_1d_field("CellLength", units=r"\rm{cm}^2",
          function=_CellLength,
          convert_function=_ConvertCellLengthCGS)

for a in ["Code", "Mpc", ""]:
    Enzo1DFieldInfo["CellVolume%s" % a] = \
        Enzo1DFieldInfo["CellLength%s" % a]

def _yvel(field, data):
    return na.zeros(data["x-velocity"].shape,
                    dtype='float64')
add_enzo_1d_field("z-velocity", function=_zvel)
add_enzo_1d_field("y-velocity", function=_yvel)
