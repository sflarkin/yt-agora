"""
Gadget-specific fields


Authors:
 * Christopher E Moody 


"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np

from yt.funcs import *
from yt.data_objects.field_info_container import \
    FieldInfoContainer, \
    FieldInfo, \
    ValidateParameter, \
    ValidateDataField, \
    ValidateProperty, \
    ValidateSpatial, \
    ValidateGridType
import yt.data_objects.universal_fields

GadgetFieldInfo = FieldInfoContainer.create_with_fallback(FieldInfo)
add_gadget_field = GadgetFieldInfo.add_field

add_field = add_gadget_field

translation_dict = {"particle_position_x" : "position_x",
                    "particle_position_y" : "position_y",
                    "particle_position_z" : "position_z",
                   }

def _generate_translation(mine, theirs):
    pfield = mine.startswith("particle")
    add_field(theirs, function=lambda a, b: b[mine], take_log=True,
              particle_type = pfield)

for f,v in translation_dict.items():
    if v not in GadgetFieldInfo:
        # Note here that it's the yt field that we check for particle nature
        pfield = f.startswith("particle")
        add_field(v, function=lambda a,b: None, take_log=False,
                  validators = [ValidateDataField(v)],
                  particle_type = pfield)
    print "Setting up translator from %s to %s" % (v, f)
    _generate_translation(v, f)


#for f,v in translation_dict.items():
#    add_field(f, function=lambda a,b: None, take_log=True,
#        validators = [ValidateDataField(v)],
#        units=r"\rm{cm}")
#    add_field(v, function=lambda a,b: None, take_log=True,
#        validators = [ValidateDataField(v)],
#        units=r"\rm{cm}")
          

          
add_field("position_x", function=lambda a,b: None, take_log=True,
          validators = [ValidateDataField("position_x")],
          particle_type = True,
          units=r"\rm{cm}")

add_field("position_y", function=lambda a,b: None, take_log=True,
          validators = [ValidateDataField("position_y")],
          particle_type = True,
          units=r"\rm{cm}")

add_field("position_z", function=lambda a,b: None, take_log=True,
          validators = [ValidateDataField("position_z")],
          particle_type = True,
          units=r"\rm{cm}")

add_field("VEL", function=lambda a,b: None, take_log=True,
          validators = [ValidateDataField("VEL")],
          units=r"")

add_field("id", function=lambda a,b: None, take_log=True,
          validators = [ValidateDataField("ID")],
          units=r"")

add_field("mass", function=lambda a,b: None, take_log=True,
          validators = [ValidateDataField("mass")],
          units=r"\rm{g}")
def _particle_mass(field, data):
    return data["mass"]/just_one(data["CellVolume"])
def _convert_particle_mass(data):
    return 1.0
add_field("particle_mass", function=_particle_mass, take_log=True,
          convert_function=_convert_particle_mass,
          validators = [ValidateSpatial(0)],
          units=r"\mathrm{g}\/\mathrm{cm}^{-3}")

add_field("U", function=lambda a,b: None, take_log=True,
          validators = [ValidateDataField("U")],
          units=r"")

add_field("NE", function=lambda a,b: None, take_log=True,
          validators = [ValidateDataField("NE")],
          units=r"")

add_field("POT", function=lambda a,b: None, take_log=True,
          validators = [ValidateDataField("POT")],
          units=r"")

add_field("ACCE", function=lambda a,b: None, take_log=True,
          validators = [ValidateDataField("ACCE")],
          units=r"")

add_field("ENDT", function=lambda a,b: None, take_log=True,
          validators = [ValidateDataField("ENDT")],
          units=r"")

add_field("TSTP", function=lambda a,b: None, take_log=True,
          validators = [ValidateDataField("TSTP")],
          units=r"")

