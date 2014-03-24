"""
This is a container for storing local fields defined on each load of yt.



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np

from .field_plugin_registry import \
    register_field_plugin

from .field_info_container import \
    FieldInfoContainer

def ensure_user_field(func):
    def check_and_call(*args, **kwargs):
        if 'user_field' not in kwargs:
            kwargs['user_field'] = True
        ret = func(*args, **kwargs)
        return ret
    return check_and_call

# Empty FieldInfoContainer
local_fields = FieldInfoContainer(None, [], None)

add_field = derived_field = ensure_user_field(local_fields.add_field)

@register_field_plugin
def setup_local_fields(registry, ftype = "gas", slice_info = None):
    # This is easy.  We just update with the contents of the local_fields field
    # info container, and since they are not mutable in any real way, we are
    # fine.
    # Note that we actually don't care about the ftype here.
    for f in local_fields:
        if local_fields[f].user_field:
            registry._show_field_errors.append(f)
    registry.update(local_fields)
