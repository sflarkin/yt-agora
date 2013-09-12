"""
Chombo-specific fields



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

from yt.data_objects.field_info_container import \
    FieldInfoContainer, \
    FieldInfo, \
    NullFunc, \
    ValidateParameter, \
    ValidateDataField, \
    ValidateProperty, \
    ValidateSpatial, \
    ValidateGridType
import yt.data_objects.universal_fields
import numpy as np

KnownChomboFields = FieldInfoContainer()
add_chombo_field = KnownChomboFields.add_field

ChomboFieldInfo = FieldInfoContainer.create_with_fallback(FieldInfo)
add_field = ChomboFieldInfo.add_field

add_chombo_field("density", function=NullFunc, take_log=True,
                 validators = [ValidateDataField("density")],
                 units=r"\rm{g}/\rm{cm}^3")

KnownChomboFields["density"]._projected_units =r"\rm{g}/\rm{cm}^2"

add_chombo_field("X-momentum", function=NullFunc, take_log=False,
                 validators = [ValidateDataField("X-Momentum")],
                 units=r"",display_name=r"M_x")
KnownChomboFields["X-momentum"]._projected_units=r""

add_chombo_field("Y-momentum", function=NullFunc, take_log=False,
                 validators = [ValidateDataField("Y-Momentum")],
                 units=r"",display_name=r"M_y")
KnownChomboFields["Y-momentum"]._projected_units=r""

add_chombo_field("Z-momentum", function=NullFunc, take_log=False,
                 validators = [ValidateDataField("Z-Momentum")],
                 units=r"",display_name=r"M_z")
KnownChomboFields["Z-momentum"]._projected_units=r""

add_chombo_field("X-magnfield", function=NullFunc, take_log=False,
                 validators = [ValidateDataField("X-Magnfield")],
                 units=r"",display_name=r"B_x")
KnownChomboFields["X-magnfield"]._projected_units=r""

add_chombo_field("Y-magnfield", function=NullFunc, take_log=False,
                 validators = [ValidateDataField("Y-Magnfield")],
                 units=r"",display_name=r"B_y")
KnownChomboFields["Y-magnfield"]._projected_units=r""

add_chombo_field("Z-magnfield", function=NullFunc, take_log=False,
                  validators = [ValidateDataField("Z-Magnfield")],
                  units=r"",display_name=r"B_z")
KnownChomboFields["Z-magnfield"]._projected_units=r""

add_chombo_field("energy-density", function=NullFunc, take_log=True,
                 validators = [ValidateDataField("energy-density")],
                 units=r"\rm{erg}/\rm{cm}^3")
KnownChomboFields["energy-density"]._projected_units =r""

add_chombo_field("radiation-energy-density", function=NullFunc, take_log=True,
                 validators = [ValidateDataField("radiation-energy-density")],
                 units=r"\rm{erg}/\rm{cm}^3")
KnownChomboFields["radiation-energy-density"]._projected_units =r""

def _Density(field,data):
    """A duplicate of the density field. This is needed because when you try 
    to instantiate a PlotCollection without passing in a center, the code
    will try to generate one for you using the "Density" field, which gives an error 
    if it isn't defined.

    """
    return data["density"]
add_field("Density",function=_Density, take_log=True,
          units=r'\rm{g}/\rm{cm^3}')

def _Bx(field,data):
    return data["X-magnfield"]
add_field("Bx", function=_Bx, take_log=False,
          units=r"\rm{Gauss}", display_name=r"B_x")

def _By(field,data):
    return data["Y-magnfield"]
add_field("By", function=_By, take_log=False,
          units=r"\rm{Gauss}", display_name=r"B_y")

def _Bz(field,data):
    return data["Z-magnfield"]
add_field("Bz", function=_Bz, take_log=False,
          units=r"\rm{Gauss}", display_name=r"B_z")

def _MagneticEnergy(field,data):
    return (data["X-magnfield"]**2 +
            data["Y-magnfield"]**2 +
            data["Z-magnfield"]**2)/2.
add_field("MagneticEnergy", function=_MagneticEnergy, take_log=True,
          units=r"", display_name=r"B^2 / 8 \pi")
ChomboFieldInfo["MagneticEnergy"]._projected_units=r""

def _xVelocity(field, data):
    """ Generate x-velocity from x-momentum and density. """
    return data["X-momentum"]/data["density"]
add_field("x-velocity",function=_xVelocity, take_log=False,
          units=r'\rm{cm}/\rm{s}')

def _yVelocity(field,data):
    """ Generate y-velocity from y-momentum and density. """
    #try:
    #    return data["xvel"]
    #except KeyError:
    return data["Y-momentum"]/data["density"]
add_field("y-velocity",function=_yVelocity, take_log=False,
          units=r'\rm{cm}/\rm{s}')

def _zVelocity(field,data):
    """ Generate z-velocity from z-momentum and density. """
    return data["Z-momentum"]/data["density"]
add_field("z-velocity",function=_zVelocity, take_log=False,
          units=r'\rm{cm}/\rm{s}')

def particle_func(p_field, dtype='float64'):
    def _Particles(field, data):
        io = data.hierarchy.io
        if not data.NumberOfParticles > 0:
            return np.array([], dtype=dtype)
        else:
            return io._read_particles(data, p_field).astype(dtype)
        
    return _Particles

_particle_field_list = ["mass",
                        "position_x",
                        "position_y",
                        "position_z",
                        "momentum_x",
                        "momentum_y",
                        "momentum_z",
                        "angmomen_x",
                        "angmomen_y",
                        "angmomen_z",
                        "mlast",
                        "r",
                        "mdeut",
                        "n",
                        "mdot",
                        "burnstate",
                        "luminosity",
                        "id"]

for pf in _particle_field_list:
    pfunc = particle_func("particle_%s" % (pf))
    add_field("particle_%s" % pf, function=pfunc,
              validators = [ValidateSpatial(0)],
              particle_type=True)

def _ParticleMass(field, data):
    particles = data["particle_mass"].astype('float64')
    return particles

def _ParticleMassMsun(field, data):
    particles = data["particle_mass"].astype('float64')
    return particles/1.989e33

add_field("ParticleMass",
          function=_ParticleMass, validators=[ValidateSpatial(0)],
          particle_type=True)
add_field("ParticleMassMsun",
          function=_ParticleMassMsun, validators=[ValidateSpatial(0)],
          particle_type=True)
