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

from yt.lagos import *
from yt.fido import ParameterFileStore, NoParameterShelf
from yt.funcs import *
import string, re, gc, time, os, os.path

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
            if ytcfg.getboolean('lagos','serialize'):
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
            self["InitialTime"], self["CurrentTimeIdentifier"])
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


class EnzoStaticOutput(StaticOutput):
    """
    Enzo-specific output, set at a fixed time.
    """
    _hierarchy_class = EnzoHierarchy
    _fieldinfo_class = EnzoFieldContainer
    def __init__(self, filename, data_style=None,
                 parameter_override = None,
                 conversion_override = None,
                 storage_filename = None):
        """
        This class is a stripped down class that simply reads and parses
        *filename* without looking at the hierarchy.  *data_style* gets passed
        to the hierarchy to pre-determine the style of data-output.  However,
        it is not strictly necessary.  Optionally you may specify a
        *parameter_override* dictionary that will override anything in the
        paarmeter file and a *conversion_override* dictionary that consists
        of {fieldname : conversion_to_cgs} that will override the #DataCGS.
        """
        if parameter_override is None: parameter_override = {}
        self.__parameter_override = parameter_override
        if conversion_override is None: conversion_override = {}
        self.__conversion_override = conversion_override
        self.storage_filename = storage_filename

        StaticOutput.__init__(self, filename, data_style)
        if "InitialTime" not in self.parameters:
            self.parameters["InitialTime"] = 0.0
        rp = os.path.join(self.directory, "rates.out")
        if os.path.exists(rp):
            try:
                self.rates = EnzoTable(rp, rates_out_key)
            except:
                pass
        cp = os.path.join(self.directory, "cool_rates.out")
        if os.path.exists(cp):
            try:
                self.cool = EnzoTable(cp, cool_out_key)
            except:
                pass

        # Now fixes for different types of Hierarchies
        # This includes changing the fieldinfo class!
        if self["TopGridRank"] == 1: self._setup_1d()
        elif self["TopGridRank"] == 2: self._setup_2d()

        self.field_info = self._fieldinfo_class()

    def _setup_1d(self):
        self._hierarchy_class = EnzoHierarchy1D
        self._fieldinfo_class = Enzo1DFieldContainer
        self.parameters["DomainLeftEdge"] = \
            na.concatenate([self["DomainLeftEdge"], [0.0, 0.0]])
        self.parameters["DomainRightEdge"] = \
            na.concatenate([self["DomainRightEdge"], [1.0, 1.0]])

    def _setup_2d(self):
        self._hierarchy_class = EnzoHierarchy2D
        self._fieldinfo_class = Enzo2DFieldContainer
        self.parameters["DomainLeftEdge"] = \
            na.concatenate([self["DomainLeftEdge"], [0.0]])
        self.parameters["DomainRightEdge"] = \
            na.concatenate([self["DomainRightEdge"], [1.0]])

    def get_parameter(self,parameter,type=None):
        """
        Gets a parameter not in the parameterDict.
        """
        if self.parameters.has_key(parameter):
            return self.parameters[parameter]

        # Let's read the file
        self.parameters["CurrentTimeIdentifier"] = \
            int(os.stat(self.parameter_filename)[ST_CTIME])
        lines = open(self.parameter_filename).readlines()
        for lineI, line in enumerate(lines):
            if line.find("#") >= 1: # Keep the commented lines
                line=line[:line.find("#")]
            line=line.strip().rstrip()
            if len(line) < 2:
                continue
            try:
                param, vals = map(strip,map(rstrip,line.split("=")))
            except ValueError:
                mylog.error("ValueError: '%s'", line)
            if parameter == param:
                if type is None:
                    t = vals.split()
                else:
                    t = map(type, vals.split())
                if len(t) == 1:
                    self.parameters[param] = t[0]
                else:
                    self.parameters[param] = t
                if param.endswith("Units") and not param.startswith("Temperature"):
                    dataType = param[:-5]
                    self.conversion_factors[dataType] = self.parameters[param]
                return self.parameters[parameter]

        return ""

    def _parse_parameter_file(self):
        """
        Parses the parameter file and establishes the various
        dictionaries.
        """
        # Let's read the file
        self.parameters["CurrentTimeIdentifier"] = \
            int(os.stat(self.parameter_filename)[ST_CTIME])
        lines = open(self.parameter_filename).readlines()
        for lineI, line in enumerate(lines):
            if line.find("#") >= 1: # Keep the commented lines
                line=line[:line.find("#")]
            line=line.strip().rstrip()
            if len(line) < 2:
                continue
            try:
                param, vals = map(strip,map(rstrip,line.split("=")))
            except ValueError:
                mylog.error("ValueError: '%s'", line)
            if parameterDict.has_key(param):
                t = map(parameterDict[param], vals.split())
                if len(t) == 1:
                    self.parameters[param] = t[0]
                else:
                    self.parameters[param] = t
                if param.endswith("Units") and not param.startswith("Temperature"):
                    dataType = param[:-5]
                    self.conversion_factors[dataType] = self.parameters[param]
            elif param.startswith("#DataCGS"):
                # Assume of the form: #DataCGSConversionFactor[7] = 2.38599e-26 g/cm^3
                if lines[lineI-1].find("Label") >= 0:
                    kk = lineI-1
                elif lines[lineI-2].find("Label") >= 0:
                    kk = lineI-2
                dataType = lines[kk].split("=")[-1].rstrip().strip()
                convFactor = float(line.split("=")[-1].split()[0])
                self.conversion_factors[dataType] = convFactor
            elif param.startswith("#CGSConversionFactor"):
                dataType = param[20:].rstrip()
                convFactor = float(line.split("=")[-1])
                self.conversion_factors[dataType] = convFactor
            elif param.startswith("DomainLeftEdge"):
                self.parameters["DomainLeftEdge"] = \
                    na.array([float(i) for i in vals.split()])
            elif param.startswith("DomainRightEdge"):
                self.parameters["DomainRightEdge"] = \
                    na.array([float(i) for i in vals.split()])
        for p, v in self.__parameter_override.items():
            self.parameters[p] = v
        for p, v in self.__conversion_override.items():
            self.conversion_factors[p] = v

    def _set_units(self):
        """
        Generates the conversion to various physical _units based on the parameter file
        """
        self.units = {}
        self.time_units = {}
        if len(self.parameters) == 0:
            self._parse_parameter_file()
        if "EOSType" not in self.parameters: self.parameters["EOSType"] = -1
        if self["ComovingCoordinates"]:
            self._setup_comoving_units()
        elif self.has_key("LengthUnit"):
            # 'Why share when we can reinvent incompatibly?'
            self.parameters["LengthUnits"] = self["LengthUnit"]
            self._setup_getunits_units()
        elif self.has_key("LengthUnits"):
            self._setup_getunits_units()
        else:
            self._setup_nounits_units()
        self.time_units['1'] = 1
        self.units['1'] = 1
        self.units['unitary'] = 1.0 / (self["DomainRightEdge"] - self["DomainLeftEdge"]).max()
        seconds = self["Time"]
        self.time_units['years'] = seconds / (365*3600*24.0)
        self.time_units['days']  = seconds / (3600*24.0)

    def _setup_comoving_units(self):
        z = self["CosmologyCurrentRedshift"]
        h = self["CosmologyHubbleConstantNow"]
        boxcm_cal = self["CosmologyComovingBoxSize"]
        boxcm_uncal = boxcm_cal / h
        box_proper = boxcm_uncal/(1+z)
        self.units['aye']  = (1.0 + self["CosmologyInitialRedshift"])/(z + 1.0)
        if not self.has_key("Time"):
            cu = self.cosmology_get_units()
            self.conversion_factors["Time"] = cu['utim']
        for unit in mpc_conversion:
            self.units[unit] = mpc_conversion[unit] * box_proper
            self.units[unit+'h'] = mpc_conversion[unit] * box_proper * h
            self.units[unit+'cm'] = mpc_conversion[unit] * boxcm_uncal
            self.units[unit+'hcm'] = mpc_conversion[unit] * boxcm_cal

    def _setup_getunits_units(self):
        # We are given LengthUnits, which is number of cm per box length
        # So we convert that to box-size in Mpc
        box_proper = 3.24077e-25 * self["LengthUnits"]
        self.units['aye']  = 1.0
        for unit in mpc_conversion.keys():
            self.units[unit] = mpc_conversion[unit] * box_proper
        if not self.has_key("TimeUnits"):
            self.conversion_factors["Time"] = self["LengthUnits"] / self["x-velocity"]

    def _setup_nounits_units(self):
        z = 0
        mylog.warning("Setting 1.0 in code units to be 1.0 cm")
        if not self.has_key("TimeUnits"):
            mylog.warning("No time units.  Setting 1.0 = 1 second.")
            self.conversion_factors["Time"] = 1.0
        for unit in mpc_conversion.keys():
            self.units[unit] = mpc_conversion[unit] / mpc_conversion["cm"]

    def cosmology_get_units(self):
        """
        Return an Enzo-fortran style dictionary of units to feed into custom
        routines.  This is typically only necessary if you are interacting
        with fortran code.
        """
        k = {}
        k["utim"] = 2.52e17/na.sqrt(self.parameters["CosmologyOmegaMatterNow"])\
                       / self.parameters["CosmologyHubbleConstantNow"] \
                       / (1+self.parameters["CosmologyInitialRedshift"])**1.5
        k["urho"] = 1.88e-29 * self.parameters["CosmologyOmegaMatterNow"] \
                        * self.parameters["CosmologyHubbleConstantNow"]**2 \
                        * (1.0 + self.parameters["CosmologyCurrentRedshift"])**3
        k["uxyz"] = 3.086e24 * \
               self.parameters["CosmologyComovingBoxSize"] / \
               self.parameters["CosmologyHubbleConstantNow"] / \
               (1.0 + self.parameters["CosmologyCurrentRedshift"])
        k["uaye"] = 1.0/(1.0 + self.parameters["CosmologyInitialRedshift"])
        k["uvel"] = 1.225e7*self.parameters["CosmologyComovingBoxSize"] \
                      *na.sqrt(self.parameters["CosmologyOmegaMatterNow"]) \
                      *na.sqrt(1+ self.parameters["CosmologyInitialRedshift"])
        k["utem"] = 1.88e6 * (self.parameters["CosmologyComovingBoxSize"]**2) \
                      * self.parameters["CosmologyOmegaMatterNow"] \
                      * (1.0 + self.parameters["CosmologyInitialRedshift"])
        k["aye"]  = (1.0 + self.parameters["CosmologyInitialRedshift"]) / \
               (1.0 + self.parameters["CosmologyCurrentRedshift"])
        return k

    @classmethod
    def _is_valid(cls, *args, **kwargs):
        return os.path.exists("%s.hierarchy" % args[0])

# We set our default output type to EnzoStaticOutput

output_type_registry[None] = EnzoStaticOutput

class EnzoStaticOutputInMemory(EnzoStaticOutput):
    _hierarchy_class = EnzoHierarchyInMemory
    _data_style = 'enzo_inline'

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj.__init__(*args, **kwargs)
        return obj

    def __init__(self, parameter_override=None, conversion_override=None):
        if parameter_override is None: parameter_override = {}
        self.__parameter_override = parameter_override
        if conversion_override is None: conversion_override = {}
        self.__conversion_override = conversion_override

        StaticOutput.__init__(self, "InMemoryParameterFile", self._data_style)

        self.field_info = self._fieldinfo_class()

    def _parse_parameter_file(self):
        enzo = self._obtain_enzo()
        self.basename = "cycle%08i" % (
            enzo.yt_parameter_file["NumberOfPythonCalls"])
        self.parameters['CurrentTimeIdentifier'] = time.time()
        self.parameters.update(enzo.yt_parameter_file)
        self.conversion_factors.update(enzo.conversion_factors)
        for i in self.parameters:
            if isinstance(self.parameters[i], types.TupleType):
                self.parameters[i] = na.array(self.parameters[i])
        for i in self.conversion_factors:
            if isinstance(self.conversion_factors[i], types.TupleType):
                self.conversion_factors[i] = na.array(self.conversion_factors[i])
        for p, v in self.__parameter_override.items():
            self.parameters[p] = v
        for p, v in self.__conversion_override.items():
            self.conversion_factors[p] = v

    def _obtain_enzo(self):
        import enzo; return enzo

    @classmethod
    def _is_valid(cls, *args, **kwargs):
        return False

class OrionStaticOutput(StaticOutput):
    """
    This class is a stripped down class that simply reads and parses
    *filename*, without looking at the Orion hierarchy.
    """
    _hierarchy_class = OrionHierarchy
    _fieldinfo_class = OrionFieldContainer

    def __init__(self, plotname, paramFilename=None, fparamFilename=None,
                 data_style='orion_native', paranoia=False,
                 storage_filename = None):
        """need to override for Orion file structure.

        the paramfile is usually called "inputs"
        and there may be a fortran inputs file usually called "probin"
        plotname here will be a directory name
        as per BoxLib, data_style will be one of
         * Native
         * IEEE (not implemented in yt)
         * ASCII (not implemented in yt)

        """
        self.storage_filename = storage_filename
        self.paranoid_read = paranoia
        self.parameter_filename = paramFilename
        self.fparameter_filename = fparamFilename
        self.__ipfn = paramFilename

        self.fparameters = {}

        StaticOutput.__init__(self, plotname.rstrip("/"),
                              data_style='orion_native')
        self.field_info = self._fieldinfo_class()
        self._parse_header_file()

        # These should maybe not be hardcoded?
        self.parameters["HydroMethod"] = 'orion' # always PPM DE
        self.parameters["Time"] = 1. # default unit is 1...
        self.parameters["DualEnergyFormalism"] = 0 # always off.
        self.parameters["EOSType"] = -1 # default
        if self.fparameters.has_key("mu"):
            self.parameters["mu"] = self.fparameters["mu"]

    def _localize(self, f, default):
        if f is None:
            return os.path.join(self.directory, default)
        return f

    @classmethod
    def _is_valid(cls, *args, **kwargs):
        # fill our args
        pname = args[0].rstrip("/")
        dn = os.path.dirname(pname)
        if len(args) > 1: kwargs['paramFilename'] = args[1]
        pfname = kwargs.get("paramFilename", os.path.join(dn, "inputs"))
        return os.path.exists(os.path.join(pfname))
        
    def _parse_parameter_file(self):
        """
        Parses the parameter file and establishes the various
        dictionaries.
        """
        self.fullplotdir = os.path.abspath(self.parameter_filename)
        self.parameter_filename = self._localize(
                self.__ipfn, 'inputs')
        self.fparameter_filename = self._localize(
                self.fparameter_filename, 'probin')
        if os.path.isfile(self.fparameter_filename):
            self._parse_fparameter_file()
        # Let's read the file
        self.parameters["CurrentTimeIdentifier"] = \
            int(os.stat(self.parameter_filename)[ST_CTIME])
        lines = open(self.parameter_filename).readlines()
        for lineI, line in enumerate(lines):
            if line.find("#") >= 1: # Keep the commented lines...
                line=line[:line.find("#")]
            line=line.strip().rstrip()
            if len(line) < 2 or line.find("#") == 0: # ...but skip comments
                continue
            try:
                param, vals = map(strip,map(rstrip,line.split("=")))
            except ValueError:
                mylog.error("ValueError: '%s'", line)
            if orion2enzoDict.has_key(param):
                paramName = orion2enzoDict[param]
                t = map(parameterDict[paramName], vals.split())
                if len(t) == 1:
                    self.parameters[paramName] = t[0]
                else:
                    if paramName == "RefineBy":
                        self.parameters[paramName] = t[0]
                    else:
                        self.parameters[paramName] = t
                
            elif param.startswith("geometry.prob_hi"):
                self.parameters["DomainRightEdge"] = \
                    na.array([float(i) for i in vals.split()])
            elif param.startswith("geometry.prob_lo"):
                self.parameters["DomainLeftEdge"] = \
                    na.array([float(i) for i in vals.split()])

    def _parse_fparameter_file(self):
        """
        Parses the fortran parameter file for Orion. Most of this will
        be useless, but this is where it keeps mu = mass per
        particle/m_hydrogen.
        """
        lines = open(self.fparameter_filename).readlines()
        for line in lines:
            if line.count("=") == 1:
                param, vals = map(strip,map(rstrip,line.split("=")))
                if vals.count("'") == 0:
                    t = map(float,[a.replace('D','e').replace('d','e') for a in vals.split()]) # all are floating point.
                else:
                    t = vals.split()
                if len(t) == 1:
                    self.fparameters[param] = t[0]
                else:
                    self.fparameters[param] = t

    def _parse_header_file(self):
        """
        Parses the BoxLib header file to get any parameters stored
        there. Hierarchy information is read out of this file in
        OrionHierarchy. 

        Currently, only Time is read here.
        """
        header_file = open(os.path.join(self.fullplotdir,'Header'))
        lines = header_file.readlines()
        header_file.close()
        n_fields = int(lines[1])
        self.parameters["InitialTime"] = float(lines[3+n_fields])

                
    def _set_units(self):
        """
        Generates the conversion to various physical _units based on the parameter file
        """
        self.units = {}
        self.time_units = {}
        if len(self.parameters) == 0:
            self._parse_parameter_file()
        self._setup_nounits_units()
        self.conversion_factors = defaultdict(lambda: 1.0)
        self.time_units['1'] = 1
        self.units['1'] = 1.0
        self.units['unitary'] = 1.0 / (self["DomainRightEdge"] - self["DomainLeftEdge"]).max()
        seconds = 1 #self["Time"]
        self.time_units['years'] = seconds / (365*3600*24.0)
        self.time_units['days']  = seconds / (3600*24.0)
        for key in yt2orionFieldsDict:
            self.conversion_factors[key] = 1.0

    def _setup_nounits_units(self):
        z = 0
        mylog.warning("Setting 1.0 in code units to be 1.0 cm")
        if not self.has_key("TimeUnits"):
            mylog.warning("No time units.  Setting 1.0 = 1 second.")
            self.conversion_factors["Time"] = 1.0
        for unit in mpc_conversion.keys():
            self.units[unit] = mpc_conversion[unit] / mpc_conversion["cm"]

def _reconstruct_pf(*args, **kwargs):
    pfs = ParameterFileStore()
    pf = pfs.get_pf_hash(*args)
    return pf

class GadgetStaticOutput(StaticOutput):
    _hierarchy_class = GadgetHierarchy
    _fieldinfo_class = GadgetFieldContainer
    def __init__(self, h5filename,storage_filename=None) :
        StaticOutput.__init__(self, h5filename, 'gadget_hdf5')
        self.storage_filename = storage_filename #Don't know what this is
        self.field_info = self._fieldinfo_class()
        x = self._get_param('maxlevel')**2
        self.max_grid_size = (x,x,x)
        self.parameters["InitialTime"] = 0.0
        # These should be explicitly obtained from the file, but for now that
        # will wait until a reorganization of the source tree and better
        # generalization.
        self.parameters["TopGridRank"] = 3
        self.parameters["RefineBy"] = 2
        self.parameters["DomainLeftEdge"] = self.leftedge
        self.parameters["DomainRightEdge"] = self.rightedge
        
        
    def _parse_parameter_file(self):
        # read the units in from the hdf5 file 
        #fill in self.units dict
        #fill in self.time_units dict (keys: 'days','years', '1')
        
        #import all of the parameter file params 
        #this is NOT originally from the gadget snapshot but instead
        #from the paramfile starting the sim
        skips = ('TITLE','CLASS','VERSION') #these are just hdf5 crap
        fh = h5py.File(self.parameter_filename)
        for kw in fh['root'].attrs.keys():
            if any([skip in kw for skip in skips]):
                continue
            val = fh['root'].attrs[kw]
            if type(val)==type(''):
                try:    val = cPickle.loads(val)
                except: pass
            #also, includes unit info
            setattr(self,kw,val)
            
    def _get_param(self,kw,location='/root'):
        fh = h5py.File(self.parameter_filename)
        val = fh[location].attrs[kw]
        try:    val = cPickle.loads(val)
        except: pass
        return val
            
    def _set_units(self):
        #check out the unit params from _parse_parameter_file and use them
        #code below is all filler
        self.units = {}
        self.time_units = {}
        self.conversion_factors = defaultdict(lambda: 1.0)
        self.time_units['1'] = 1
        self.units['1'] = 1.0
        self.units['unitary'] = 1.0
        self.units['cm'] = 1.0
        seconds = 1 #self["Time"]
        self.time_units['years'] = seconds / (365*3600*24.0)
        self.time_units['days']  = seconds / (3600*24.0)
        for key in yt2orionFieldsDict:
            self.conversion_factors[key] = 1.0
        
        
    @classmethod
    def _is_valid(cls, *args, **kwargs):
        # check for a /root to exist in the h5 file
        try:
            h5f=h5py.File(self.h5filename)
            valid = 'root' in h5f.items()[0]
            h5f.close()
            return valid
        except:
            pass
        return False

class ChomboStaticOutput(StaticOutput):
    _hierarchy_class = ChomboHierarchy
    _fieldinfo_class = ChomboFieldContainer
    
    def __init__(self, filename, data_style='chombo_hdf5',
                 storage_filename = None):
        StaticOutput.__init__(self,filename,data_style)
        self.storage_filename = storage_filename

        self.field_info = self._fieldinfo_class()
        # hardcoded for now
        self.parameters["InitialTime"] = 0.0
        # These should be explicitly obtained from the file, but for now that
        # will wait until a reorganization of the source tree and better
        # generalization.
        self.parameters["TopGridRank"] = 3
        self.parameters["RefineBy"] = 2
        
    def _set_units(self):
        """
        Generates the conversion to various physical _units based on the parameter file
        """
        self.units = {}
        self.time_units = {}
        if len(self.parameters) == 0:
            self._parse_parameter_file()
        self._setup_nounits_units()
        self.conversion_factors = defaultdict(lambda: 1.0)
        self.time_units['1'] = 1
        self.units['1'] = 1.0
        self.units['unitary'] = 1.0 / (self["DomainRightEdge"] - self["DomainLeftEdge"]).max()
        seconds = 1 #self["Time"]
        self.time_units['years'] = seconds / (365*3600*24.0)
        self.time_units['days']  = seconds / (3600*24.0)
        for key in yt2orionFieldsDict:
            self.conversion_factors[key] = 1.0

    def _setup_nounits_units(self):
        z = 0
        mylog.warning("Setting 1.0 in code units to be 1.0 cm")
        if not self.has_key("TimeUnits"):
            mylog.warning("No time units.  Setting 1.0 = 1 second.")
            self.conversion_factors["Time"] = 1.0
        for unit in mpc_conversion.keys():
            self.units[unit] = mpc_conversion[unit] / mpc_conversion["cm"]


    def _parse_parameter_file(self):
        self.parameters["CurrentTimeIdentifier"] = \
            int(os.stat(self.parameter_filename)[ST_CTIME])
        self.parameters["DomainLeftEdge"] = na.array([0.,0.,0.])
        self.parameters["DomainRightEdge"] = self.__calc_right_edge()
        

    def __calc_right_edge(self):
        fileh = h5py.File(self.parameter_filename,'r')
        dx0 = fileh['/level_0'].attrs['dx']
        RE = dx0*((na.array(fileh['/level_0'].attrs['prob_domain']))[3:] + 1)
        fileh.close()
        return RE
                   
    @classmethod
    def _is_valid(self, *args, **kwargs):
        try:
            fileh = h5py.File(args[0],'r')
            if (fileh.listnames())[0] == 'Chombo_global':
                return True
        except:
            pass
        return False

class TigerStaticOutput(StaticOutput):
    _hierarchy_class = TigerHierarchy
    _fieldinfo_class = TigerFieldContainer

    def __init__(self, rhobname, root_size, max_grid_size=128,
                 data_style='tiger', storage_filename = None):
        StaticOutput.__init__(self, rhobname, data_style)
        self.storage_filename = storage_filename
        self.basename = rhobname[:-4]
        if not os.path.exists(self.basename + "rhob"):
            print "%s doesn't exist, don't know how to handle this!" % (
                        self.basename + "rhob")
            raise IOError
        if not iterable(root_size): root_size = (root_size,) * 3
        self.root_size = root_size
        if not iterable(max_grid_size): max_grid_size = (max_grid_size,) * 3
        self.max_grid_size = max_grid_size

        self.field_info = self._fieldinfo_class()

        # We assume that we have basename + "rhob" and basename + "temp"
        # to get at our various parameters.

        # First we get our our header:
        
        header = [
            ('i', 'dummy0'),
            ('f', 'ZR'),
            ('f', 'OMEGA0'),
            ('f', 'FLAM0'),
            ('f', 'OMEGAB'),
            ('f', 'H0'),
            ('f', 'BOXL0'),
            ('i', 'dummy1'),
            ]

        h_fmt, h_key = zip(*header)
        header_string = "".join(h_fmt)

        fs = open(self.basename + "rhob")
        header_raw = read_struct(fs, header_string)
        self.parameters.update(dict(zip(h_key, header_raw)))

        if "InitialTime" not in self.parameters:
            self.parameters["InitialTime"] = 0.0
        self.parameters["CurrentTimeIdentifier"] = \
            int(os.stat(self.parameter_filename)[ST_CTIME])
        self.parameters['TopGridDimensions'] = root_size
        self.parameters['TopGridRank'] = 3
        self.units["Density"] = 1.0
        self.parameters['RefineBy'] = 2

    def _set_units(self):
        self.parameters["DomainLeftEdge"] = na.zeros(3, dtype='float64')
        self.parameters["DomainRightEdge"] = na.ones(3, dtype='float64')
        self.units = {}
        self.time_units = {}
        self.time_units['1'] = 1
        self.units['1'] = 1.0
        self.units['cm'] = 1.0 # This is just plain false
        self.units['unitary'] = 1.0 / (self["DomainRightEdge"] - self["DomainLeftEdge"]).max()

    def _parse_parameter_file(self):
        pass

    @classmethod
    def _is_valid(self, *args, **kwargs):
        return os.path.exists(args[0] + "rhob")

class FLASHStaticOutput(StaticOutput):
    _hierarchy_class = FLASHHierarchy
    _fieldinfo_class = FLASHFieldContainer
    _handle = None
    
    def __init__(self, filename, data_style='flash_hdf5',
                 storage_filename = None):
        StaticOutput.__init__(self, filename, data_style)
        self.storage_filename = storage_filename

        self.field_info = self._fieldinfo_class()
        # hardcoded for now
        self.parameters["InitialTime"] = 0.0
        # These should be explicitly obtained from the file, but for now that
        # will wait until a reorganization of the source tree and better
        # generalization.
        self.parameters["TopGridRank"] = 3
        self.parameters["RefineBy"] = 2
        self.parameters["HydroMethod"] = 'flash' # always PPM DE
        self.parameters["Time"] = 1. # default unit is 1...
        
    def _set_units(self):
        """
        Generates the conversion to various physical _units based on the parameter file
        """
        self.units = {}
        self.time_units = {}
        if len(self.parameters) == 0:
            self._parse_parameter_file()
        self._setup_nounits_units()
        self.conversion_factors = defaultdict(lambda: 1.0)
        self.time_units['1'] = 1
        self.units['1'] = 1.0
        self.units['unitary'] = 1.0 / (self["DomainRightEdge"] - self["DomainLeftEdge"]).max()
        seconds = 1 #self["Time"]
        self.time_units['years'] = seconds / (365*3600*24.0)
        self.time_units['days']  = seconds / (3600*24.0)
        for key in yt2orionFieldsDict:
            self.conversion_factors[key] = 1.0

    def _setup_nounits_units(self):
        z = 0
        mylog.warning("Setting 1.0 in code units to be 1.0 cm")
        if not self.has_key("TimeUnits"):
            mylog.warning("No time units.  Setting 1.0 = 1 second.")
            self.conversion_factors["Time"] = 1.0
        for unit in mpc_conversion.keys():
            self.units[unit] = mpc_conversion[unit] / mpc_conversion["cm"]

    def _find_parameter(self, ptype, pname, scalar = False, handle = None):
        # We're going to implement handle caching eventually
        if handle is None: handle = self._handle
        if handle is None:
            handle = h5py.File(self.parameter_filename, "r")
        nn = "/%s %s" % (ptype,
                {False: "runtime parameters", True: "scalars"}[scalar])
        for tpname, pval in handle[nn][:]:
            if tpname.strip() == pname:
                return pval
        raise KeyError(pname)

    def _parse_parameter_file(self):
        self.parameters["CurrentTimeIdentifier"] = \
            int(os.stat(self.parameter_filename)[ST_CTIME])
        self._handle = h5py.File(self.parameter_filename, "r")
        self.parameters["DomainLeftEdge"] = na.array(
            [self._find_parameter("real", "%smin" % ax) for ax in 'xyz'])
        self.parameters["DomainRightEdge"] = na.array(
            [self._find_parameter("real", "%smax" % ax) for ax in 'xyz'])
        self._handle.close()

    @classmethod
    def _is_valid(self, *args, **kwargs):
        try:
            fileh = h5py.File(args[0],'r')
            if "bounding box" in fileh["/"].keys():
                return True
        except:
            pass
        return False

class RAMSESStaticOutput(StaticOutput):
    _hierarchy_class = RAMSESHierarchy
    _fieldinfo_class = RAMSESFieldContainer
    _handle = None
    
    def __init__(self, filename, data_style='ramses',
                 storage_filename = None):
        StaticOutput.__init__(self, filename, data_style)
        self.storage_filename = storage_filename

        self.field_info = self._fieldinfo_class()
        # hardcoded for now
        self.parameters["InitialTime"] = 0.0
        # These should be explicitly obtained from the file, but for now that
        # will wait until a reorganization of the source tree and better
        # generalization.
        self.parameters["TopGridRank"] = 3
        self.parameters["RefineBy"] = 2
        self.parameters["HydroMethod"] = 'ramses'
        self.parameters["Time"] = 1. # default unit is 1...

    def __repr__(self):
        return self.basename.rsplit(".", 1)[0]
        
    def _set_units(self):
        """
        Generates the conversion to various physical _units based on the parameter file
        """
        self.units = {}
        self.time_units = {}
        if len(self.parameters) == 0:
            self._parse_parameter_file()
        self._setup_nounits_units()
        self.conversion_factors = defaultdict(lambda: 1.0)
        self.time_units['1'] = 1
        self.units['1'] = 1.0
        self.units['unitary'] = 1.0 / (self["DomainRightEdge"] - self["DomainLeftEdge"]).max()
        seconds = 1 #self["Time"]
        self.time_units['years'] = seconds / (365*3600*24.0)
        self.time_units['days']  = seconds / (3600*24.0)
        for key in yt2orionFieldsDict:
            self.conversion_factors[key] = 1.0

    def _setup_nounits_units(self):
        z = 0
        mylog.warning("Setting 1.0 in code units to be 1.0 cm")
        if not self.has_key("TimeUnits"):
            mylog.warning("No time units.  Setting 1.0 = 1 second.")
            self.conversion_factors["Time"] = 1.0
        for unit in mpc_conversion.keys():
            self.units[unit] = mpc_conversion[unit] / mpc_conversion["cm"]

    def _parse_parameter_file(self):
        self.parameters["CurrentTimeIdentifier"] = \
            int(os.stat(self.parameter_filename)[ST_CTIME])
        import yt.ramses_reader as rr
        self.ramses_tree = rr.RAMSES_tree_proxy(self.parameter_filename)
        rheader = self.ramses_tree.get_file_info()
        self.parameters.update(rheader)
        self.parameters["DomainRightEdge"] = na.ones(3, dtype='float64') \
                                           * rheader['boxlen']
        self.parameters["DomainLeftEdge"] = na.zeros(3, dtype='float64')
        self.parameters["TopGridDimensions"] = rheader["nx"]

    @classmethod
    def _is_valid(self, *args, **kwargs):
        if not os.path.basename(args[0]).startswith("info_"): return False
        fn = args[0].replace("info_", "amr_").replace(".txt", ".out00001")
        print fn
        return os.path.exists(fn)

