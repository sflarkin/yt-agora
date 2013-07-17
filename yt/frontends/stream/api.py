"""
API for yt.frontends.stream

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: Columbia University
Homepage: http://yt-project.org/
License:
  Copyright (C) 2011 Matthew Turk.  All Rights Reserved.

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

from .data_structures import \
      StreamGrid, \
      StreamHierarchy, \
      StreamStaticOutput, \
      StreamHandler, \
      load_uniform_grid, \
      load_amr_grids, \
      refine_amr

from .fields import \
      KnownStreamFields, \
      StreamFieldInfo, \
      add_stream_field

from .io import \
      IOHandlerStream
