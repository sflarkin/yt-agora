"""
Halo Catalog object

Author: Britton Smith <brittonsmith@gmail.com>
Affiliation: MSU
Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: Columbia University
Homepage: http://yt-project.org/
License:
  Copyright (C) 2013 Britton Smith, Matthew Turk.  All Rights Reserved.

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

from .operator_registry import \
    callback_registry, \
    quantity_registry, \
    hf_registry

class HaloCatalog(object):
    def __init__(self, data_pf, halos_pf):#, finding_method, data_source = None):
        self.data_pf = data_pf
        self.halos_pf = halos_pf
        # self.finding_method = hf_registry.find(finding_method)
        # if data_source is None:
        #     data_source = pf.h.all_data()
        # self.data_source = data_source
        self.values = []
        self.callbacks = []

    def add_callback(self, callback, *args, **kwargs):
        callback = callback_registry.find(callback, *args, **kwargs)
        self.callbacks.append(callback)

    def add_quantity(self, quantity, *args, **kwargs):
        quantity = quantity_registry.find(quantity, *args, **kwargs)
        self.callbacks.append(quantity)

    def add_filter(self, filter, *args, **kwargs):
        filter = callback_registry.find(filter, *args, **kwargs)
        self.callbacks.append(filter)

    def run(self):
        # Here's the basic rundown.
        # First we call the halo finding operation.  This is going to be handed
        # the data source, but we assume it already has all its arguments
        # necessary for the finding operation.
        # halo_list here will be a generator.
        # halo_list = self.finding_method(self)
        self.run_callbacks(halo_list)

    def run_callbacks(self, halo_list):
        for cb in self.callbacks:
            cb.initialize(self)
        for halo in halo_list:
            if all(cb(self, halo) for cb in self.callbacks):
                self.values.append(halo.quantities)
        for cb in self.callbacks:
            cb.finalize(self)

