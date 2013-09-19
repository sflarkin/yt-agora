"""
Skeleton-specific fields



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
    kboltz

# The first field container is where any fields that exist on disk go, along
# with their conversion factors, display names, etc.

KnownSkeletonFields = FieldInfoContainer()
add_skeleton_field = KnownSkeletonFields.add_field

SkeletonFieldInfo = FieldInfoContainer.create_with_fallback(FieldInfo)
add_field = SkeletonFieldInfo.add_field

# Often, we want to translate between fields on disk and fields in yt.  This
# construct shows how to do that.  Note that we use TranslationFunc.

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

for f,v in translation_dict.items():
    if v not in KnownSkeletonFields:
        pfield = v.startswith("particle")
        add_skeleton_field(v, function=NullFunc, take_log=False,
                  validators = [ValidateDataField(v)],
                  particle_type = pfield)
    if f.endswith("_Fraction") :
        dname = "%s\/Fraction" % f.split("_")[0]
    else :
        dname = f                    
    ff = KnownSkeletonFields[v]
    pfield = f.startswith("particle")
    add_field(f, TranslationFunc(v),
              take_log=KnownSkeletonFields[v].take_log,
              units = ff._units, display_name=dname,
              particle_type = pfield)

# Here's an example of adding a new field:

add_skeleton_field("dens", function=NullFunc, take_log=True,
                convert_function=_get_convert("dens"),
                units=r"\rm{g}/\rm{cm}^3")
