"""
FLASH-specific fields



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
from yt.utilities.exceptions import *
from yt.data_objects.field_info_container import \
    FieldInfoContainer, \
    NullFunc, \
    TranslationFunc, \
    FieldInfo, \
    ValidateParameter, \
    ValidateDataField, \
    ValidateProperty, \
    ValidateSpatial, \
    ValidateGridType
import yt.data_objects.universal_fields
from yt.utilities.physical_constants import \
    kboltz, mh, Na
KnownFLASHFields = FieldInfoContainer()
add_flash_field = KnownFLASHFields.add_field

FLASHFieldInfo = FieldInfoContainer.create_with_fallback(FieldInfo)
add_field = FLASHFieldInfo.add_field

# Common fields in FLASH: (Thanks to John ZuHone for this list)
#
# dens gas mass density (g/cc) --
# eint internal energy (ergs/g) --
# ener total energy (ergs/g), with 0.5*v^2 --
# gamc gamma defined as ratio of specific heats, no units
# game gamma defined as in , no units
# gpol gravitational potential from the last timestep (ergs/g)
# gpot gravitational potential from the current timestep (ergs/g)
# grac gravitational acceleration from the current timestep (cm s^-2)
# pden particle mass density (usually dark matter) (g/cc)
# pres pressure (erg/cc)
# temp temperature (K) --
# velx velocity x (cm/s) --
# vely velocity y (cm/s) --
# velz velocity z (cm/s) --

translation_dict = {"x-velocity": "velx",
                    "y-velocity": "vely",
                    "z-velocity": "velz",
                    "Density": "dens",
                    "Temperature": "temp",
                    "Pressure" : "pres", 
                    "Grav_Potential" : "gpot",
                    "particle_position_x" : "particle_posx",
                    "particle_position_y" : "particle_posy",
                    "particle_position_z" : "particle_posz",
                    "particle_velocity_x" : "particle_velx",
                    "particle_velocity_y" : "particle_vely",
                    "particle_velocity_z" : "particle_velz",
                    "particle_index" : "particle_tag",
                    "Electron_Fraction" : "elec",
                    "HI_Fraction" : "h   ",
                    "HD_Fraction" : "hd  ",
                    "HeI_Fraction": "hel ",
                    "HeII_Fraction": "hep ",
                    "HeIII_Fraction": "hepp",
                    "HM_Fraction": "hmin",
                    "HII_Fraction": "hp  ",
                    "H2I_Fraction": "htwo",
                    "H2II_Fraction": "htwp",
                    "DI_Fraction": "deut",
                    "DII_Fraction": "dplu",
                    "ParticleMass": "particle_mass",
                    "Flame_Fraction": "flam"}

def _get_density(fname):
    def _dens(field, data):
        return data[fname] * data['Density']
    return _dens

for fn1, fn2 in translation_dict.items():
    if fn1.endswith("_Fraction"):
        add_field(fn1.split("_")[0] + "_Density",
                  function=_get_density(fn1), take_log=True,
                  display_name="%s\/Density" % fn1.split("_")[0],
                  units = r"\rm{g}/\rm{cm}^{3}",
                  projected_units = r"\rm{g}/\rm{cm}^{2}",
                  )

def _get_convert(fname):
    def _conv(data):
        return data.convert(fname)
    return _conv

add_flash_field("dens", function=NullFunc, take_log=True,
                convert_function=_get_convert("dens"),
                units=r"\rm{g}/\rm{cm}^{3}",
                projected_units = r"\rm{g}/\rm{cm}^{2}"),
add_flash_field("velx", function=NullFunc, take_log=False,
                convert_function=_get_convert("velx"),
                units=r"\rm{cm}/\rm{s}")
add_flash_field("vely", function=NullFunc, take_log=False,
                convert_function=_get_convert("vely"),
                units=r"\rm{cm}/\rm{s}")
add_flash_field("velz", function=NullFunc, take_log=False,
                convert_function=_get_convert("velz"),
                units=r"\rm{cm}/\rm{s}")
add_flash_field("ener", function=NullFunc, take_log=True,
                convert_function=_get_convert("ener"),
                units=r"\rm{erg}/\rm{g}")
add_flash_field("eint", function=NullFunc, take_log=True,
                convert_function=_get_convert("eint"),
                units=r"\rm{erg}/\rm{g}")
add_flash_field("particle_posx", function=NullFunc, take_log=False,
                convert_function=_get_convert("particle_posx"),
                units=r"\rm{cm}", particle_type=True)
add_flash_field("particle_posy", function=NullFunc, take_log=False,
                convert_function=_get_convert("particle_posy"),
                units=r"\rm{cm}", particle_type=True)
add_flash_field("particle_posz", function=NullFunc, take_log=False,
                convert_function=_get_convert("particle_posz"),
                units=r"\rm{cm}", particle_type=True)
add_flash_field("particle_velx", function=NullFunc, take_log=False,
                convert_function=_get_convert("particle_velx"),
                units=r"\rm{cm}/\rm{s}", particle_type=True)
add_flash_field("particle_vely", function=NullFunc, take_log=False,
                convert_function=_get_convert("particle_vely"),
                units=r"\rm{cm}/\rm{s}", particle_type=True)
add_flash_field("particle_velz", function=NullFunc, take_log=False,
                convert_function=_get_convert("particle_velz"),
                units=r"\rm{cm}/\rm{s}", particle_type=True)
add_flash_field("particle_tag", function=NullFunc, take_log=False,
                convert_function=_get_convert("particle_tag"),
                particle_type=True)
add_flash_field("particle_mass", function=NullFunc, take_log=False,
                convert_function=_get_convert("particle_mass"),
                units=r"\rm{g}", particle_type=True)
add_flash_field("temp", function=NullFunc, take_log=True,
                convert_function=_get_convert("temp"),
                units=r"\rm{K}")
add_flash_field("tion", function=NullFunc, take_log=True,
                units=r"\rm{K}")
add_flash_field("tele", function=NullFunc, take_log=True,
                convert_function=_get_convert("tele"),
                units = r"\rm{K}")
add_flash_field("trad", function=NullFunc, take_log=True,
                units = r"\rm{K}")
add_flash_field("pres", function=NullFunc, take_log=True,
                convert_function=_get_convert("pres"),
                units=r"\rm{erg}/\rm{cm}^{3}")
add_flash_field("pion", function=NullFunc, take_log=True,
                display_name="Ion Pressure",
                units=r"\rm{erg}/\rm{cm}^3")
add_flash_field("pele", function=NullFunc, take_log=True,
                display_name="Electron Pressure, P_e",
                units=r"\rm{erg}/\rm{cm}^3")
add_flash_field("prad", function=NullFunc, take_log=True,
                display_name="Radiation Pressure",
                units = r"\rm{erg}/\rm{cm}^3")
add_flash_field("eion", function=NullFunc, take_log=True,
                display_name="Ion Internal Energy",
                units=r"\rm{erg}")
add_flash_field("eele", function=NullFunc, take_log=True,
                display_name="Electron Internal Energy",
                units=r"\rm{erg}")
add_flash_field("erad", function=NullFunc, take_log=True,
                display_name="Radiation Internal Energy",
                units=r"\rm{erg}")
add_flash_field("pden", function=NullFunc, take_log=True,
                convert_function=_get_convert("pden"),
                units=r"\rm{g}/\rm{cm}^{3}")
add_flash_field("depo", function=NullFunc, take_log=True,
                units = r"\rm{ergs}/\rm{g}")
add_flash_field("ye", function=NullFunc, take_log=True,)
add_flash_field("magx", function=NullFunc, take_log=False,
                convert_function=_get_convert("magx"),
                units = r"\mathrm{Gau\ss}")
add_flash_field("magy", function=NullFunc, take_log=False,
                convert_function=_get_convert("magy"),
                units = r"\mathrm{Gau\ss}")
add_flash_field("magz", function=NullFunc, take_log=False,
                convert_function=_get_convert("magz"),
                units = r"\mathrm{Gau\ss}")
add_flash_field("magp", function=NullFunc, take_log=True,
                convert_function=_get_convert("magp"),
                units = r"\rm{erg}/\rm{cm}^{3}")
add_flash_field("divb", function=NullFunc, take_log=False,
                convert_function=_get_convert("divb"),
                units = r"\mathrm{Gau\ss}\/\rm{cm}")
add_flash_field("game", function=NullFunc, take_log=False,
                convert_function=_get_convert("game"),
                units=r"\rm{ratio\/of\/specific\/heats}")
add_flash_field("gamc", function=NullFunc, take_log=False,
                convert_function=_get_convert("gamc"),
                units=r"\rm{ratio\/of\/specific\/heats}")
add_flash_field("gpot", function=NullFunc, take_log=False,
                convert_function=_get_convert("gpot"),
                units=r"\rm{ergs}/\rm{g}")
add_flash_field("gpol", function=NullFunc, take_log=False,
                convert_function=_get_convert("gpol"),
                units = r"\rm{ergs}/\rm{g}")
add_flash_field("flam", function=NullFunc, take_log=False,
                convert_function=_get_convert("flam"))
add_flash_field("absr", function=NullFunc, take_log=False,
                display_name="Absorption Coefficient")
add_flash_field("emis", function=NullFunc, take_log=False,
                display_name="Emissivity")
add_flash_field("cond", function=NullFunc, take_log=False,
                display_name="Conductivity")
add_flash_field("dfcf", function=NullFunc, take_log=False,
                display_name="Diffusion Equation Scalar")
add_flash_field("fllm", function=NullFunc, take_log=False,
                display_name="Flux Limit")
add_flash_field("pipe", function=NullFunc, take_log=False,
                display_name="P_i/P_e")
add_flash_field("tite", function=NullFunc, take_log=False,
                display_name="T_i/T_e")
add_flash_field("dbgs", function=NullFunc, take_log=False,
                display_name="Debug for Shocks")
add_flash_field("cham", function=NullFunc, take_log=False,
                display_name="Chamber Material Fraction")
add_flash_field("targ", function=NullFunc, take_log=False,
                display_name="Target Material Fraction")
add_flash_field("sumy", function=NullFunc, take_log=False)
add_flash_field("mgdc", function=NullFunc, take_log=False,
                display_name="Emission Minus Absorption Diffusion Terms")

for i in range(1, 1000):
    add_flash_field("r{0:03}".format(i), function=NullFunc, take_log=False,
        display_name="Energy Group {0}".format(i))


for f,v in translation_dict.items():
    if v not in KnownFLASHFields:
        pfield = v.startswith("particle")
        add_flash_field(v, function=NullFunc, take_log=False,
                  validators = [ValidateDataField(v)],
                  particle_type = pfield)
    if f.endswith("_Fraction") :
        dname = "%s\/Fraction" % f.split("_")[0]
    else :
        dname = f                    
    ff = KnownFLASHFields[v]
    pfield = f.startswith("particle")
    add_field(f, TranslationFunc(v),
              take_log=KnownFLASHFields[v].take_log,
              units = ff._units, display_name=dname,
              projected_units = ff._projected_units,
              particle_type = pfield)

def _convertParticleMassMsun(data):
    return 1.0/1.989e33
def _ParticleMassMsun(field, data):
    return data["ParticleMass"]
add_field("ParticleMassMsun",
          function=_ParticleMassMsun, validators=[ValidateSpatial(0)],
          particle_type=True, convert_function=_convertParticleMassMsun,
          particle_convert_function=_ParticleMassMsun)

def _ThermalEnergy(fields, data) :
    try:
        return data["eint"]
    except:
        pass
    try:
        return data["Pressure"] / (data.pf["Gamma"] - 1.0) / data["Density"]
    except:
        pass
    if data.has_field_parameter("mu") :
        mu = data.get_field_parameter("mu")
    else:
        mu = 0.6
    return kboltz*data["Density"]*data["Temperature"]/(mu*mh) / (data.pf["Gamma"] - 1.0)
    
add_field("ThermalEnergy", function=_ThermalEnergy,
          units=r"\rm{ergs}/\rm{g}")

def _TotalEnergy(fields, data) :
    try:
        etot = data["ener"]
    except:
        etot = data["ThermalEnergy"] + 0.5 * (
            data["x-velocity"]**2.0 +
            data["y-velocity"]**2.0 +
            data["z-velocity"]**2.0)
    try:
        etot += data['magp']/data["Density"]
    except:
        pass
    return etot

add_field("TotalEnergy", function=_TotalEnergy,
          units=r"\rm{ergs}/\rm{g}")

def _GasEnergy(fields, data) :
    return data["ThermalEnergy"]

add_field("GasEnergy", function=_GasEnergy, 
          units=r"\rm{ergs}/\rm{g}")

# See http://flash.uchicago.edu/pipermail/flash-users/2012-October/001180.html
# along with the attachment to that e-mail for details
def GetMagRescalingFactor(pf):
    if pf['unitsystem'].lower() == "cgs":
         factor = 1
    elif pf['unitsystem'].lower() == "si":
         factor = np.sqrt(4*np.pi/1e7)
    elif pf['unitsystem'].lower() == "none":
         factor = np.sqrt(4*np.pi)
    else:
        raise RuntimeError("Runtime parameter unitsystem with "
                           "value %s is unrecognized" % pf['unitsystem'])
    return factor

def _Bx(fields, data):
    factor = GetMagRescalingFactor(data.pf)
    return data['magx']*factor
add_field("Bx", function=_Bx, take_log=False,
          units=r"\rm{Gauss}", display_name=r"B_x")

def _By(fields, data):
    factor = GetMagRescalingFactor(data.pf)
    return data['magy']*factor
add_field("By", function=_By, take_log=False,
          units=r"\rm{Gauss}", display_name=r"B_y")

def _Bz(fields, data):
    factor = GetMagRescalingFactor(data.pf)
    return data['magz']*factor
add_field("Bz", function=_Bz, take_log=False,
          units=r"\rm{Gauss}", display_name=r"B_z")

def _DivB(fields, data):
    factor = GetMagRescalingFactor(data.pf)
    return data['divb']*factor
add_field("DivB", function=_DivB, take_log=False,
          units=r"\rm{Gauss}\/\rm{cm}^{-1}")



## Derived FLASH Fields
def _nele(field, data):
    return data['dens'] * data['ye'] * Na
add_field('nele', function=_nele, take_log=True, units=r"\rm{cm}^{-3}")
add_field('edens', function=_nele, take_log=True, units=r"\rm{cm}^{-3}")

def _nion(field, data):
    return data['dens'] * data['sumy'] * Na
add_field('nion', function=_nion, take_log=True, units=r"\rm{cm}^{-3}")

def _abar(field, data):
    try:
        return 1.0 / data['sumy']
    except:
        pass
    return data['dens']*Na*kboltz*data['temp']/data['pres']
add_field('abar', function=_abar, take_log=False)
	

def _NumberDensity(fields,data) :
    try:
        return data["nele"]+data["nion"]
    except:
        pass
    return data['pres']/(data['temp']*kboltz)
add_field("NumberDensity", function=_NumberDensity,
        units=r'\rm{cm}^{-3}')


