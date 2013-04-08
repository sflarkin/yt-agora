/**********************************************************************
Widget Store for Reason

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: Columbia University
Homepage: http://yt-project.org/
License:
  Copyright (C) 2012 Matthew Turk.  All Rights Reserved.

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
***********************************************************************/

Ext.define('Reason.store.widgets.SceneWidgets', {
    extend: 'Ext.data.Store',
    id: 'scenewidgets',
    fields: [
       {name: 'name', type: 'string'},
       {name: 'enabled', type: 'boolean', defaultValue: true},
       {name: 'type', type: 'string'},
       {name: 'widget'},
    ],
    data: [],
});
