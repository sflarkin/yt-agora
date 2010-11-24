"""
Generalized Enzo output objects, both static and time-series.

Presumably at some point EnzoRun will be absorbed into here.
Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2007-2009 Matthew Turk, J. S. Oishi.  All Rights Reserved.

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

import string, re, gc, time, os, os.path, weakref

from yt.funcs import *

from yt.config import ytcfg
from yt.utilities.parameter_file_storage import \
    ParameterFileStore, \
    NoParameterShelf, \
    output_type_registry

# We want to support the movie format in the future.
# When such a thing comes to pass, I'll move all the stuff that is contant up
# to here, and then have it instantiate EnzoStaticOutputs as appropriate.

_cached_pfs = weakref.WeakValueDictionary()
_pf_store = ParameterFileStore()

class StaticOutput(object):
    class __metaclass__(type):
        def __init__(cls, name, b, d):
            type.__init__(cls, name, b, d)
            output_type_registry[name]=cls
            mylog.debug("Registering: %s as %s", name, cls)

    def __new__(cls, filename=None, *args, **kwargs):
        if not isinstance(filename, types.StringTypes): 
            obj = object.__new__(cls)
            obj.__init__(filename, *args, **kwargs)
            return obj
        apath = os.path.abspath(filename)
        if not os.path.exists(apath): raise IOError(filename)
        if apath not in _cached_pfs:
            obj = object.__new__(cls)
            obj.__init__(filename, *args, **kwargs)
            _cached_pfs[apath] = obj
            if ytcfg.getboolean('yt','serialize'):
                try:
                    _pf_store.check_pf(obj)
                except NoParameterShelf:
                    pass
        return _cached_pfs[apath]

    def __init__(self, filename, data_style=None):
        """
        Base class for generating new output types.  Principally consists of
        a *filename* and a *data_style* which will be passed on to children.
        """
        self.data_style = data_style
        self.parameter_filename = str(filename)
        self.basename = os.path.basename(filename)
        self.directory = os.path.expanduser(os.path.dirname(filename))
        self.fullpath = os.path.abspath(self.directory)
        self._instantiated = time.time()
        if len(self.directory) == 0:
            self.directory = "."
        self.conversion_factors = {}
        self.parameters = {}
        self._parse_parameter_file()
        self._set_units()
        # These can be taken out if you so desire

    def __reduce__(self):
        args = (self._hash(),)
        return (_reconstruct_pf, args)

    def __repr__(self):
        return self.basename

    def _hash(self):
        s = "%s;%s;%s" % (self.basename,
            self.current_time, self.unique_identifier)
        try:
            import hashlib
            return hashlib.md5(s).hexdigest()
        except ImportError:
            return s.replace(";", "*")

    @classmethod
    def _is_valid(cls, *args, **kwargs):
        return False

    def __getitem__(self, key):
        """
        Returns _units, parameters, or _conversion_factors in that order
        """
        for d in [self.units, self.time_units, self.parameters, \
                  self.conversion_factors]:
            if key in d: return d[key]
        raise KeyError(key)

    def keys(self):
        """
        Returns a list of possible keys, from _units, parameters and
        _conversion_factors
        """
        return self.units.keys() \
             + self.time_units.keys() \
             + self.parameters.keys() \
             + self.conversion_factors.keys()

    def __iter__(self):
        for ll in [self.units, self.time_units,
                   self.parameters, self.conversion_factors]:
            for i in ll.keys(): yield i

    def get_smallest_appropriate_unit(self, v):
        max_nu = 1e30
        good_u = None
        for unit in ['mpc','kpc','pc','au','rsun','cm']:
            vv = v*self[unit]
            if vv < max_nu and vv > 1.0:
                good_u = unit
                max_nu = v*self[unit]
        return good_u

    def has_key(self, key):
        """
        Returns true or false
        """
        return key in self.units or \
               key in self.time_units or \
               key in self.parameters or \
               key in self.conversion_factors

    def _get_hierarchy(self):
        if self.__hierarchy == None:
            if self._hierarchy_class == None:
                raise RuntimeError("You should not instantiate StaticOutput.")
            self.__hierarchy = self._hierarchy_class(self, data_style=self.data_style)
        return self.__hierarchy

    def _set_hierarchy(self, newh):
        if self.__hierarchy != None:
            mylog.warning("Overriding hierarchy attribute!  This is probably unwise!")
        self.__hierarchy = newh

    __hierarchy = None
    hierarchy = property(_get_hierarchy, _set_hierarchy)
    h = property(_get_hierarchy, _set_hierarchy)

def _reconstruct_pf(*args, **kwargs):
    pfs = ParameterFileStore()
    pf = pfs.get_pf_hash(*args)
    return pf

