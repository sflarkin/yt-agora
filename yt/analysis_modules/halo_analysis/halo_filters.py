"""
Halo filter object



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013-2014, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

from .halo_callbacks import HaloCallback
from .operator_registry import filter_registry

def add_filter(name, function):
    filter_registry[name] = HaloFilter(function)

class HaloFilter(HaloCallback):
    r"""
    A HaloFilter is a function that minimally takes a Halo object, performs 
    some analysis, and returns either True or False.  The return value determines 
    whether the Halo is added to the final halo catalog being generated by the 
    HaloCatalog object.
    """
    def __init__(self, function, *args, **kwargs):
        HaloCallback.__init__(self, function, args, kwargs)

    def __call__(self, halo):
        return self.function(halo, *self.args, **self.kwargs)

def quantity_value(halo, field, operator, value, units):
    r"""
    Filter based on a value in the halo quantities dictionary.

    Parameters
    ----------
    halo : Halo object
        The Halo object to be provided by the HaloCatalog.
    field : string
        The field used for the evaluation.
    operator : string 
        The comparison operator to be used ("<", "<=", "==", ">=", ">", etc.)
    value : numneric
        The value to be compared against.
    units : string
        Units of the value to be compared.
        
    """

    if field not in halo.quantities:
        raise RuntimeError("Halo object does not contain %s quantity." % field)

    h_value = halo.quantities[field].in_units(units).to_ndarray()
    return eval("%s %s %s" % (h_value, operator, value))

add_filter("quantity_value", quantity_value)
