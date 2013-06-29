"""
Fields specific to Streaming data

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
Homepage: http://yt-project.org/
License:
  Copyright (C) 2008-2011 Matthew Turk.  All Rights Reserved.

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
    NullFunc, \
    TranslationFunc, \
    FieldInfo, \
    ValidateParameter, \
    ValidateDataField, \
    ValidateProperty, \
    ValidateSpatial, \
    ValidateGridType
import yt.data_objects.universal_fields

KnownStreamFields = FieldInfoContainer()
add_stream_field = KnownStreamFields.add_field

StreamFieldInfo = FieldInfoContainer.create_with_fallback(FieldInfo)
add_field = StreamFieldInfo.add_field

add_stream_field("density", function = NullFunc)
add_field("Density", function = TranslationFunc("density"))

add_stream_field("particle_position_x", function = NullFunc, particle_type=True)
add_stream_field("particle_position_y", function = NullFunc, particle_type=True)
add_stream_field("particle_position_z", function = NullFunc, particle_type=True)
add_stream_field("particle_index", function = NullFunc, particle_type=True)
add_stream_field("particle_gas_density", function = NullFunc, particle_type=True)
add_stream_field("particle_gas_temperature", function = NullFunc, particle_type=True)
add_stream_field("particle_mass", function = NullFunc, particle_type=True)

add_field("particle_mass", function = TranslationFunc("particle_mass"),
          particle_type=True)

add_stream_field(("all", "particle_position_x"), function = NullFunc, particle_type=True)
add_stream_field(("all", "particle_position_y"), function = NullFunc, particle_type=True)
add_stream_field(("all", "particle_position_z"), function = NullFunc, particle_type=True)
add_stream_field(("all", "particle_index"), function = NullFunc, particle_type=True)
add_stream_field(("all", "particle_gas_density"), function = NullFunc, particle_type=True)
add_stream_field(("all", "particle_gas_temperature"), function = NullFunc, particle_type=True)
add_stream_field(("all", "particle_mass"), function = NullFunc, particle_type=True)

add_field(("all", "particle_mass"), function = TranslationFunc("particle_mass"),
          particle_type=True)
