"""
Generalized Enzo output objects, both static and time-series.

Presumably at some point EnzoRun will be absorbed into here.


"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import string, re, gc, time, os, os.path, weakref
import functools

from yt.funcs import *
from yt.extern.six import add_metaclass

from yt.config import ytcfg
from yt.utilities.cosmology import \
     Cosmology
from yt.utilities.parallel_tools.parallel_analysis_interface import \
    parallel_root_only
from yt.utilities.parameter_file_storage import \
    ParameterFileStore, \
    NoParameterShelf, \
    output_type_registry
from yt.units.unit_object import Unit
from yt.units.unit_registry import UnitRegistry
from yt.fields.field_info_container import \
    FieldInfoContainer, NullFunc
from yt.data_objects.particle_filters import \
    filter_registry
from yt.data_objects.particle_unions import \
    ParticleUnion
from yt.data_objects.data_containers import \
    data_object_registry
from yt.utilities.minimal_representation import \
    MinimalDataset
from yt.units.yt_array import \
    YTArray, \
    YTQuantity

from yt.geometry.cartesian_coordinates import \
    CartesianCoordinateHandler
from yt.geometry.polar_coordinates import \
    PolarCoordinateHandler
from yt.geometry.cylindrical_coordinates import \
    CylindricalCoordinateHandler
from yt.geometry.spherical_coordinates import \
    SphericalCoordinateHandler
from yt.geometry.geographic_coordinates import \
    GeographicCoordinateHandler
from yt.geometry.ppv_coordinates import \
    PPVCoordinateHandler

# We want to support the movie format in the future.
# When such a thing comes to pass, I'll move all the stuff that is contant up
# to here, and then have it instantiate EnzoDatasets as appropriate.

_cached_pfs = weakref.WeakValueDictionary()
_pf_store = ParameterFileStore()

def _unsupported_object(pf, obj_name):
    def _raise_unsupp(*args, **kwargs):
        raise YTObjectNotImplemented(pf, obj_name)
    return _raise_unsupp

class RegisteredDataset(type):
    def __init__(cls, name, b, d):
        type.__init__(cls, name, b, d)
        output_type_registry[name] = cls
        mylog.debug("Registering: %s as %s", name, cls)

class IndexProxy(object):
    # This is a simple proxy for Index objects.  It enables backwards
    # compatibility so that operations like .h.sphere, .h.print_stats and
    # .h.grid_left_edge will correctly pass through to the various dataset or
    # index objects.
    def __init__(self, ds):
        self.ds = weakref.proxy(ds)
        ds.index

    def __getattr__(self, name):
        # Check the ds first
        if hasattr(self.ds, name):
            return getattr(self.ds, name)
        # Now for a subset of the available items, check the ds.index.
        elif name in self.ds.index._index_properties:
            return getattr(self.ds.index, name)
        raise AttributeError

def requires_index(attr_name):
    @property
    def ireq(self):
        self.index
        # By now it should have been set
        attr = self.__dict__[attr_name]
        return attr

    @ireq.setter
    def ireq(self, value):
        self.__dict__[attr_name] = value

    return ireq

@add_metaclass(RegisteredDataset)
class Dataset(object):

    default_fluid_type = "gas"
    fluid_types = ("gas", "deposit", "index")
    particle_types = ("io",) # By default we have an 'all'
    particle_types_raw = ("io",)
    geometry = "cartesian"
    coordinates = None
    max_level = 99
    storage_filename = None
    particle_unions = None
    known_filters = None
    _index_class = None
    field_units = None
    derived_field_list = requires_index("derived_field_list")
    _instantiated = False

    def __new__(cls, filename=None, *args, **kwargs):
        from yt.frontends.stream.data_structures import StreamHandler
        if not isinstance(filename, types.StringTypes):
            obj = object.__new__(cls)
            # The Stream frontend uses a StreamHandler object to pass metadata
            # to __init__.
            is_stream = (hasattr(filename, 'get_fields') and
                         hasattr(filename, 'get_particle_type'))
            if not is_stream:
                obj.__init__(filename, *args, **kwargs)
            return obj
        apath = os.path.abspath(filename)
        #if not os.path.exists(apath): raise IOError(filename)
        if apath not in _cached_pfs:
            obj = object.__new__(cls)
            if obj._skip_cache is False:
                _cached_pfs[apath] = obj
        else:
            obj = _cached_pfs[apath]
        return obj

    def __init__(self, filename, dataset_type=None, file_style=None):
        """
        Base class for generating new output types.  Principally consists of
        a *filename* and a *dataset_type* which will be passed on to children.
        """
        # We return early and do NOT initialize a second time if this file has
        # already been initialized.
        if self._instantiated: return
        self.dataset_type = dataset_type
        self.file_style = file_style
        self.conversion_factors = {}
        self.parameters = {}
        self.known_filters = self.known_filters or {}
        self.particle_unions = self.particle_unions or {}
        self.field_units = self.field_units or {}

        # path stuff
        self.parameter_filename = str(filename)
        self.basename = os.path.basename(filename)
        self.directory = os.path.expanduser(os.path.dirname(filename))
        self.fullpath = os.path.abspath(self.directory)
        self.backup_filename = self.parameter_filename + '_backup.gdf'
        self.read_from_backup = False
        if os.path.exists(self.backup_filename):
            self.read_from_backup = True
        if len(self.directory) == 0:
            self.directory = "."

        # to get the timing right, do this before the heavy lifting
        self._instantiated = time.time()

        self.min_level = 0
        self.no_cgs_equiv_length = False

        self._create_unit_registry()
        self._parse_parameter_file()
        self.set_units()
        self._setup_coordinate_handler()

        # Because we need an instantiated class to check the pf's existence in
        # the cache, we move that check to here from __new__.  This avoids
        # double-instantiation.
        try:
            _pf_store.check_pf(self)
        except NoParameterShelf:
            pass
        self.print_key_parameters()

        self._set_derived_attrs()
        self._setup_classes()

    def _set_derived_attrs(self):
        if self.domain_left_edge is None or self.domain_right_edge is None:
            self.domain_center = np.zeros(3)
            self.domain_width = np.zeros(3)
        else:
            self.domain_center = 0.5 * (self.domain_right_edge + self.domain_left_edge)
            self.domain_width = self.domain_right_edge - self.domain_left_edge
        if not isinstance(self.current_time, YTQuantity):
            self.current_time = self.quan(self.current_time, "code_time")
        for attr in ("center", "width", "left_edge", "right_edge"):
            n = "domain_%s" % attr
            v = getattr(self, n)
            v = self.arr(v, "code_length")
            setattr(self, n, v)

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
            return hashlib.md5(s.encode('utf-8')).hexdigest()
        except ImportError:
            return s.replace(";", "*")

    @property
    def _mrep(self):
        return MinimalDataset(self)

    @property
    def _skip_cache(self):
        return False

    def hub_upload(self):
        self._mrep.upload()

    @classmethod
    def _is_valid(cls, *args, **kwargs):
        return False

    def __getitem__(self, key):
        """ Returns units, parameters, or conversion_factors in that order. """
        return self.parameters[key]

    def __iter__(self):
      for i in self.parameters: yield i

    def get_smallest_appropriate_unit(self, v):
        max_nu = 1e30
        good_u = None
        for unit in ['Mpc', 'kpc', 'pc', 'au', 'rsun', 'km', 'cm']:
            vv = v * self.length_unit.in_units(unit)
            if vv < max_nu and vv > 1.0:
                good_u = unit
                max_nu = v * self.length_unit.in_units(unit)
        if good_u is None : good_u = 'cm'
        return good_u

    def has_key(self, key):
        """
        Checks units, parameters, and conversion factors. Returns a boolean.

        """
        return key in self.parameters

    _instantiated_index = None
    @property
    def index(self):
        if self._instantiated_index is None:
            if self._index_class is None:
                raise RuntimeError("You should not instantiate Dataset.")
            self._instantiated_index = self._index_class(
                self, dataset_type=self.dataset_type)
            # Now we do things that we need an instantiated index for
            # ...first off, we create our field_info now.
            oldsettings = np.geterr()
            np.seterr(all='ignore')
            self.create_field_info()
            np.seterr(**oldsettings)
        return self._instantiated_index

    _index_proxy = None
    @property
    def h(self):
        if self._index_proxy is None:
            self._index_proxy = IndexProxy(self)
        return self._index_proxy
    hierarchy = h

    @parallel_root_only
    def print_key_parameters(self):
        for a in ["current_time", "domain_dimensions", "domain_left_edge",
                  "domain_right_edge", "cosmological_simulation"]:
            if not hasattr(self, a):
                mylog.error("Missing %s in parameter file definition!", a)
                continue
            v = getattr(self, a)
            mylog.info("Parameters: %-25s = %s", a, v)
        if hasattr(self, "cosmological_simulation") and \
           getattr(self, "cosmological_simulation"):
            for a in ["current_redshift", "omega_lambda", "omega_matter",
                      "hubble_constant"]:
                if not hasattr(self, a):
                    mylog.error("Missing %s in parameter file definition!", a)
                    continue
                v = getattr(self, a)
                mylog.info("Parameters: %-25s = %s", a, v)

    @parallel_root_only
    def print_stats(self):
        self.index.print_stats()

    @property
    def field_list(self):
        return self.index.field_list

    def create_field_info(self):
        self.field_dependencies = {}
        self.derived_field_list = []
        self.filtered_particle_types = []
        self.field_info = self._field_info_class(self, self.field_list)
        self.coordinates.setup_fields(self.field_info)
        self.field_info.setup_fluid_fields()
        for ptype in self.particle_types:
            self.field_info.setup_particle_fields(ptype)
        if "all" not in self.particle_types:
            mylog.debug("Creating Particle Union 'all'")
            pu = ParticleUnion("all", list(self.particle_types_raw))
            self.add_particle_union(pu)
        deps, unloaded = self.field_info.check_derived_fields()
        self.field_dependencies.update(deps)
        mylog.info("Loading field plugins.")
        self.field_info.load_all_plugins()

    def setup_deprecated_fields(self):
        from yt.fields.field_aliases import _field_name_aliases
        added = []
        for old_name, new_name in _field_name_aliases:
            try:
                fi = self._get_field_info(new_name)
            except YTFieldNotFound:
                continue
            self.field_info.alias(("gas", old_name), fi.name)
            added.append(("gas", old_name))
        self.field_info.find_dependencies(added)

    def _setup_coordinate_handler(self):
        if self.geometry == "cartesian":
            self.coordinates = CartesianCoordinateHandler(self)
        elif self.geometry == "cylindrical":
            self.coordinates = CylindricalCoordinateHandler(self)
        elif self.geometry == "polar":
            self.coordinates = PolarCoordinateHandler(self)
        elif self.geometry == "spherical":
            self.coordinates = SphericalCoordinateHandler(self)
        elif self.geometry == "geographic":
            self.coordinates = GeographicCoordinateHandler(self)
        elif self.geometry == "ppv":
            self.coordinates = PPVCoordinateHandler(self)
        else:
            raise YTGeometryNotSupported(self.geometry)

    def add_particle_union(self, union):
        # No string lookups here, we need an actual union.
        f = self.particle_fields_by_type
        fields = set_intersection([f[s] for s in union
                                   if s in self.particle_types_raw
                                   and len(f[s]) > 0])
        for field in fields:
            units = set([])
            for s in union:
                # First we check our existing fields for units
                funits = self._get_field_info(s, field).units
                # Then we override with field_units settings.
                funits = self.field_units.get((s, field), funits)
                units.add(funits)
            if len(units) == 1:
                self.field_units[union.name, field] = list(units)[0]
        self.particle_types += (union.name,)
        self.particle_unions[union.name] = union
        fields = [ (union.name, field) for field in fields]
        self.field_list.extend(fields)
        # Give ourselves a chance to add them here, first, then...
        # ...if we can't find them, we set them up as defaults.
        new_fields = self._setup_particle_types([union.name])
        rv = self.field_info.find_dependencies(new_fields)

    def add_particle_filter(self, filter):
        # This requires an index
        self.index
        # This is a dummy, which we set up to enable passthrough of "all"
        # concatenation fields.
        n = getattr(filter, "name", filter)
        self.known_filters[n] = None
        if isinstance(filter, types.StringTypes):
            used = False
            for f in filter_registry[filter]:
                used = self._setup_filtered_type(f)
                if used:
                    filter = f
                    break
        else:
            used = self._setup_filtered_type(filter)
        if not used:
            self.known_filters.pop(n, None)
            return False
        self.known_filters[filter.name] = filter
        return True

    def _setup_filtered_type(self, filter):
        if not filter.available(self.derived_field_list):
            return False
        fi = self.field_info
        fd = self.field_dependencies
        available = False
        for fn in self.derived_field_list:
            if fn[0] == filter.filtered_type:
                # Now we can add this
                available = True
                self.derived_field_list.append(
                    (filter.name, fn[1]))
                fi[filter.name, fn[1]] = filter.wrap_func(fn, fi[fn])
                # Now we append the dependencies
                fd[filter.name, fn[1]] = fd[fn]
        if available:
            self.particle_types += (filter.name,)
            self.filtered_particle_types.append(filter.name)
            self._setup_particle_types([filter.name])
        return available

    def _setup_particle_types(self, ptypes = None):
        df = []
        if ptypes is None: ptypes = self.pf.particle_types_raw
        for ptype in set(ptypes):
            df += self._setup_particle_type(ptype)
        return df

    _last_freq = (None, None)
    _last_finfo = None
    def _get_field_info(self, ftype, fname = None):
        self.index
        if fname is None:
            ftype, fname = "unknown", ftype
        guessing_type = False
        if ftype == "unknown":
            guessing_type = True
            ftype = self._last_freq[0] or ftype
        field = (ftype, fname)
        if field == self._last_freq:
            return self._last_finfo
        if field in self.field_info:
            self._last_freq = field
            self._last_finfo = self.field_info[(ftype, fname)]
            return self._last_finfo
        if fname == self._last_freq[1]:
            return self._last_finfo
        if fname in self.field_info:
            # Sometimes, if guessing_type == True, this will be switched for
            # the type of field it is.  So we look at the field type and
            # determine if we need to change the type.
            fi = self._last_finfo = self.field_info[fname]
            if fi.particle_type and self._last_freq[0] \
                not in self.particle_types:
                    field = "all", field[1]
            elif not fi.particle_type and self._last_freq[0] \
                not in self.fluid_types:
                    field = self.default_fluid_type, field[1]
            self._last_freq = field
            return self._last_finfo
        # We also should check "all" for particles, which can show up if you're
        # mixing deposition/gas fields with particle fields.
        if guessing_type:
            to_guess = ["all", self.default_fluid_type] \
                     + list(self.fluid_types) \
                     + list(self.particle_types)
            for ftype in to_guess:
                if (ftype, fname) in self.field_info:
                    self._last_freq = (ftype, fname)
                    self._last_finfo = self.field_info[(ftype, fname)]
                    return self._last_finfo
        raise YTFieldNotFound((ftype, fname), self)

    def _setup_classes(self):
        # Called by subclass
        self.object_types = []
        self.objects = []
        self.plots = []
        for name, cls in sorted(data_object_registry.items()):
            if name in self._index_class._unsupported_objects:
                setattr(self, name,
                    _unsupported_object(self, name))
                continue
            cname = cls.__name__
            if cname.endswith("Base"): cname = cname[:-4]
            self._add_object_class(name, cname, cls, {'pf':self})
        if self.refine_by != 2 and hasattr(self, 'proj') and \
            hasattr(self, 'overlap_proj'):
            mylog.warning("Refine by something other than two: reverting to"
                        + " overlap_proj")
            self.proj = self.overlap_proj
        if self.dimensionality < 3 and hasattr(self, 'proj') and \
            hasattr(self, 'overlap_proj'):
            mylog.warning("Dimensionality less than 3: reverting to"
                        + " overlap_proj")
            self.proj = self.overlap_proj
        self.object_types.sort()

    def _add_object_class(self, name, class_name, base, dd):
        self.object_types.append(name)
        dd.update({'__doc__': base.__doc__})
        obj = type(class_name, (base,), dd)
        setattr(self, name, obj)

    def find_max(self, field):
        """
        Returns (value, location) of the maximum of a given field.
        """
        mylog.debug("Searching for maximum value of %s", field)
        source = self.all_data()
        max_val, maxi, mx, my, mz = \
            source.quantities.max_location(field)
        mylog.info("Max Value is %0.5e at %0.16f %0.16f %0.16f",
              max_val, mx, my, mz)
        return max_val, np.array([mx, my, mz], dtype="float64")

    def find_min(self, field):
        """
        Returns (value, location) for the minimum of a given field.
        """
        mylog.debug("Searching for minimum value of %s", field)
        source = self.all_data()
        min_val, maxi, mx, my, mz = \
            source.quantities.min_location(field)
        mylog.info("Min Value is %0.5e at %0.16f %0.16f %0.16f",
              min_val, mx, my, mz)
        return min_val, np.array([mx, my, mz], dtype="float64")

    # Now all the object related stuff
    def all_data(self, find_max=False):
        if find_max: c = self.find_max("density")[1]
        else: c = (self.domain_right_edge + self.domain_left_edge)/2.0
        return self.region(c,
            self.domain_left_edge, self.domain_right_edge)

    def _setup_particle_type(self, ptype):
        orig = set(self.field_info.items())
        self.field_info.setup_particle_fields(ptype)
        return [n for n, v in set(self.field_info.items()).difference(orig)]

    @property
    def particle_fields_by_type(self):
        fields = defaultdict(list)
        for field in self.field_list:
            if field[0] in self.particle_types_raw:
                fields[field[0]].append(field[1])
        return fields

    @property
    def ires_factor(self):
        o2 = np.log2(self.refine_by)
        if o2 != int(o2):
            raise RuntimeError
        return int(o2)

    def relative_refinement(self, l0, l1):
        return self.refine_by**(l1-l0)

    def _create_unit_registry(self):
        self.unit_registry = UnitRegistry()
        import yt.units.dimensions as dimensions
        self.unit_registry.add("code_length", 1.0, dimensions.length)
        self.unit_registry.add("code_mass", 1.0, dimensions.mass)
        self.unit_registry.add("code_time", 1.0, dimensions.time)
        self.unit_registry.add("code_magnetic", 1.0, dimensions.magnetic_field)
        self.unit_registry.add("code_temperature", 1.0, dimensions.temperature)
        self.unit_registry.add("code_velocity", 1.0, dimensions.velocity)
        self.unit_registry.add("code_metallicity", 1.0,
                               dimensions.dimensionless)

    def set_units(self):
        """
        Creates the unit registry for this dataset.

        """
        from yt.units.dimensions import length
        if hasattr(self, "cosmological_simulation") \
           and getattr(self, "cosmological_simulation"):
            # this dataset is cosmological, so add cosmological units.
            self.unit_registry.modify("h", self.hubble_constant)
            # Comoving lengths
            for my_unit in ["m", "pc", "AU", "au"]:
                new_unit = "%scm" % my_unit
                self.unit_registry.add(new_unit, self.unit_registry.lut[my_unit][0] /
                                       (1 + self.current_redshift),
                                       length, "\\rm{%s}/(1+z)" % my_unit)

        self.set_code_units()

        if hasattr(self, "cosmological_simulation") \
           and getattr(self, "cosmological_simulation"):
            # this dataset is cosmological, add a cosmology object
            setattr(self, "cosmology",
                    Cosmology(hubble_constant=self.hubble_constant,
                              omega_matter=self.omega_matter,
                              omega_lambda=self.omega_lambda,
                              unit_registry=self.unit_registry))
            setattr(self, "critical_density",
                    self.cosmology.critical_density(self.current_redshift))

    def get_unit_from_registry(self, unit_str):
        """
        Creates a unit object matching the string expression, using this
        dataset's unit registry.

        Parameters
        ----------
        unit_str : str
            string that we can parse for a sympy Expr.

        """
        new_unit = Unit(unit_str, registry=self.unit_registry)
        return new_unit

    def set_code_units(self):
        self._set_code_unit_attributes()
        self.unit_registry.modify("code_length", self.length_unit)
        self.unit_registry.modify("code_mass", self.mass_unit)
        self.unit_registry.modify("code_time", self.time_unit)
        vel_unit = getattr(self, "velocity_unit",
                    self.length_unit / self.time_unit)
        self.unit_registry.modify("code_velocity", vel_unit)
        # domain_width does not yet exist
        if None not in (self.domain_left_edge, self.domain_right_edge):
            DW = self.arr(self.domain_right_edge - self.domain_left_edge, "code_length")
            self.unit_registry.add("unitary", float(DW.max() * DW.units.cgs_value),
                                   DW.units.dimensions)

    _arr = None
    @property
    def arr(self):
        if self._arr is not None:
            return self._arr
        self._arr = functools.partial(YTArray, registry = self.unit_registry)
        return self._arr

    _quan = None
    @property
    def quan(self):
        if self._quan is not None:
            return self._quan
        self._quan = functools.partial(YTQuantity,
                registry = self.unit_registry)
        return self._quan

    def add_field(self, name, function=None, **kwargs):
        """
        Dataset-specific call to add_field
        """
        self.index
        self.field_info.add_field(name, function=function, **kwargs)
        deps, _ = self.field_info.check_derived_fields([name])
        self.field_dependencies.update(deps)

def _reconstruct_pf(*args, **kwargs):
    pfs = ParameterFileStore()
    pf = pfs.get_pf_hash(*args)
    return pf

class ParticleFile(object):
    def __init__(self, pf, io, filename, file_id):
        self.pf = pf
        self.io = weakref.proxy(io)
        self.filename = filename
        self.file_id = file_id
        self.total_particles = self.io._count_particles(self)

    def select(self, selector):
        pass

    def count(self, selector):
        pass

    def _calculate_offsets(self, fields):
        pass
