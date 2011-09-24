"""
GDF-specific fields

Author: J. S. Oishi <jsoishi@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
Homepage: http://yt-project.org/
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
    CodeFieldInfoContainer, \
    ValidateParameter, \
    ValidateDataField, \
    ValidateProperty, \
    ValidateSpatial, \
    ValidateGridType
import yt.data_objects.universal_fields

log_translation_dict = {"Density": "density",
                        "Pressure": "pressure"}

translation_dict = {"x-velocity": "velocity_x",
                    "y-velocity": "velocity_y",
                    "z-velocity": "velocity_z"}
                    
class GDFFieldContainer(CodeFieldInfoContainer):
    _shared_state = {}
    _field_list = {}
GDFFieldInfo = GDFFieldContainer()
add_gdf_field = GDFFieldInfo.add_field

add_field = add_gdf_field

add_field("density", function=lambda a,b: None, take_log=True,
          validators = [ValidateDataField("density")],
          units=r"\rm{g}/\rm{cm}^3")

GDFFieldInfo["density"]._projected_units =r"\rm{g}/\rm{cm}^2"

add_field("specific_energy", function=lambda a,b: None, take_log=True,
          validators = [ValidateDataField("specific_energy")],
          units=r"\rm{erg}/\rm{g}")

add_field("pressure", function=lambda a,b: None, take_log=True,
          validators = [ValidateDataField("pressure")],
          units=r"\rm{erg}/\rm{g}")

add_field("velocity_x", function=lambda a,b: None, take_log=False,
          validators = [ValidateDataField("velocity_x")],
          units=r"\rm{cm}/\rm{s}")

add_field("velocity_y", function=lambda a,b: None, take_log=False,
          validators = [ValidateDataField("velocity_y")],
          units=r"\rm{cm}/\rm{s}")

add_field("velocity_z", function=lambda a,b: None, take_log=False,
          validators = [ValidateDataField("velocity_z")],
          units=r"\rm{cm}/\rm{s}")

add_field("mag_field_x", function=lambda a,b: None, take_log=False,
          validators = [ValidateDataField("mag_field_x")],
          units=r"\rm{cm}/\rm{s}")

add_field("mag_field_y", function=lambda a,b: None, take_log=False,
          validators = [ValidateDataField("mag_field_y")],
          units=r"\rm{cm}/\rm{s}")

add_field("mag_field_z", function=lambda a,b: None, take_log=False,
          validators = [ValidateDataField("mag_field_z")],
          units=r"\rm{cm}/\rm{s}")

def _get_alias(alias):
    def _alias(field, data):
        return data[alias]
    return _alias

def _generate_translation(mine, theirs ,log_field=True):
    add_field(theirs, function=_get_alias(mine), take_log=log_field)

for f,v in log_translation_dict.items():
    _generate_translation(v, f, log_field=True)

for f,v in translation_dict.items():
    _generate_translation(v, f, log_field=False)

