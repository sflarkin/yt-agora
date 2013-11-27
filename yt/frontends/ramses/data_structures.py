"""
RAMSES-specific data structures



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import os
import numpy as np
import stat
import weakref
import cStringIO

from yt.funcs import *
from yt.geometry.oct_geometry_handler import \
    OctreeGeometryHandler
from yt.geometry.geometry_handler import \
    GeometryHandler, YTDataChunk
from yt.data_objects.static_output import \
    StaticOutput
from yt.data_objects.octree_subset import \
    OctreeSubset

from .definitions import ramses_header
from yt.utilities.definitions import \
    mpc_conversion, sec_conversion
from yt.utilities.lib import \
    get_box_grids_level
from yt.utilities.io_handler import \
    io_registry
from yt.data_objects.field_info_container import \
    FieldInfoContainer, NullFunc
import yt.utilities.fortran_utils as fpu
from yt.geometry.oct_container import \
    RAMSESOctreeContainer
from .fields import \
    RAMSESFieldInfo, \
    KnownRAMSESFields, \
    create_cooling_fields, \
    _setup_particle_fields
from yt.fields.particle_fields import \
    standard_particle_fields

class RAMSESDomainFile(object):
    _last_mask = None
    _last_selector_id = None

    def __init__(self, pf, domain_id):
        self.pf = pf
        self.domain_id = domain_id
        self.nvar = 0 # Set this later!
        num = os.path.basename(pf.parameter_filename).split("."
                )[0].split("_")[1]
        basename = "%s/%%s_%s.out%05i" % (
            os.path.abspath(
              os.path.dirname(pf.parameter_filename)),
            num, domain_id)
        for t in ['grav', 'hydro', 'part', 'amr']:
            setattr(self, "%s_fn" % t, basename % t)
        self._read_amr_header()
        self._read_hydro_header()
        self._read_particle_header()
        self._read_amr()

    _hydro_offset = None
    _level_count = None

    def __repr__(self):
        return "RAMSESDomainFile: %i" % self.domain_id

    @property
    def level_count(self):
        if self._level_count is not None: return self._level_count
        self.hydro_offset
        return self._level_count

    @property
    def hydro_offset(self):
        if self._hydro_offset is not None: return self._hydro_offset
        # We now have to open the file and calculate it
        f = open(self.hydro_fn, "rb")
        fpu.skip(f, 6)
        # It goes: level, CPU, 8-variable
        min_level = self.pf.min_level
        n_levels = self.amr_header['nlevelmax'] - min_level
        hydro_offset = np.zeros(n_levels, dtype='int64')
        hydro_offset -= 1
        level_count = np.zeros(n_levels, dtype='int64')
        skipped = []
        for level in range(self.amr_header['nlevelmax']):
            for cpu in range(self.amr_header['nboundary'] +
                             self.amr_header['ncpu']):
                header = ( ('file_ilevel', 1, 'I'),
                           ('file_ncache', 1, 'I') )
                try:
                    hvals = fpu.read_attrs(f, header, "=")
                except AssertionError:
                    print "You are running with the wrong number of fields."
                    print "If you specified these in the load command, check the array length."
                    print "In this file there are %s hydro fields." % skipped
                    #print "The last set of field sizes was: %s" % skipped
                    raise
                if hvals['file_ncache'] == 0: continue
                assert(hvals['file_ilevel'] == level+1)
                if cpu + 1 == self.domain_id and level >= min_level:
                    hydro_offset[level - min_level] = f.tell()
                    level_count[level - min_level] = hvals['file_ncache']
                skipped = fpu.skip(f, 8 * self.nvar)
        self._hydro_offset = hydro_offset
        self._level_count = level_count
        return self._hydro_offset

    def _read_hydro_header(self):
        if self.nvar > 0: return self.nvar
        # Read the number of hydro  variables
        f = open(self.hydro_fn, "rb")
        fpu.skip(f, 1)
        self.nvar = fpu.read_vector(f, "i")[0]

    def _read_particle_header(self):
        if not os.path.exists(self.part_fn):
            self.local_particle_count = 0
            self.particle_field_offsets = {}
            return
        f = open(self.part_fn, "rb")
        f.seek(0, os.SEEK_END)
        flen = f.tell()
        f.seek(0)
        hvals = {}
        attrs = ( ('ncpu', 1, 'I'),
                  ('ndim', 1, 'I'),
                  ('npart', 1, 'I') )
        hvals.update(fpu.read_attrs(f, attrs))
        fpu.read_vector(f, 'I')

        attrs = ( ('nstar_tot', 1, 'I'),
                  ('mstar_tot', 1, 'd'),
                  ('mstar_lost', 1, 'd'),
                  ('nsink', 1, 'I') )
        hvals.update(fpu.read_attrs(f, attrs))
        self.particle_header = hvals
        self.local_particle_count = hvals['npart']
        particle_fields = [
                ("particle_position_x", "d"),
                ("particle_position_y", "d"),
                ("particle_position_z", "d"),
                ("particle_velocity_x", "d"),
                ("particle_velocity_y", "d"),
                ("particle_velocity_z", "d"),
                ("particle_mass", "d"),
                ("particle_identifier", "I"),
                ("particle_refinement_level", "I")]
        if hvals["nstar_tot"] > 0:
            particle_fields += [("particle_age", "d"),
                                ("particle_metallicity", "d")]
        field_offsets = {}
        _pfields = {}
        for field, vtype in particle_fields:
            if f.tell() >= flen: break
            field_offsets["io", field] = f.tell()
            _pfields["io", field] = vtype
            fpu.skip(f, 1)
        self.particle_field_offsets = field_offsets
        self.particle_field_types = _pfields
        self.particle_types = self.particle_types_raw = ("io",)

    def _read_amr_header(self):
        hvals = {}
        f = open(self.amr_fn, "rb")
        for header in ramses_header(hvals):
            hvals.update(fpu.read_attrs(f, header))
        # That's the header, now we skip a few.
        hvals['numbl'] = np.array(hvals['numbl']).reshape(
            (hvals['nlevelmax'], hvals['ncpu']))
        fpu.skip(f)
        if hvals['nboundary'] > 0:
            fpu.skip(f, 2)
            self.ngridbound = fpu.read_vector(f, 'i').astype("int64")
        else:
            self.ngridbound = np.zeros(hvals['nlevelmax'], dtype='int64')
        free_mem = fpu.read_attrs(f, (('free_mem', 5, 'i'), ) )
        ordering = fpu.read_vector(f, 'c')
        fpu.skip(f, 4)
        # Now we're at the tree itself
        # Now we iterate over each level and each CPU.
        self.amr_header = hvals
        self.amr_offset = f.tell()
        self.local_oct_count = hvals['numbl'][self.pf.min_level:, self.domain_id - 1].sum()
        self.total_oct_count = hvals['numbl'][self.pf.min_level:,:].sum(axis=0)

    def _read_amr(self):
        """Open the oct file, read in octs level-by-level.
           For each oct, only the position, index, level and domain 
           are needed - its position in the octree is found automatically.
           The most important is finding all the information to feed
           oct_handler.add
        """
        self.oct_handler = RAMSESOctreeContainer(self.pf.domain_dimensions/2,
                self.pf.domain_left_edge, self.pf.domain_right_edge)
        root_nodes = self.amr_header['numbl'][self.pf.min_level,:].sum()
        self.oct_handler.allocate_domains(self.total_oct_count, root_nodes)
        fb = open(self.amr_fn, "rb")
        fb.seek(self.amr_offset)
        f = cStringIO.StringIO()
        f.write(fb.read())
        f.seek(0)
        mylog.debug("Reading domain AMR % 4i (%0.3e, %0.3e)",
            self.domain_id, self.total_oct_count.sum(), self.ngridbound.sum())
        def _ng(c, l):
            if c < self.amr_header['ncpu']:
                ng = self.amr_header['numbl'][l, c]
            else:
                ng = self.ngridbound[c - self.amr_header['ncpu'] +
                                self.amr_header['nboundary']*l]
            return ng
        min_level = self.pf.min_level
        max_level = min_level
        nx, ny, nz = (((i-1.0)/2.0) for i in self.amr_header['nx'])
        for level in range(self.amr_header['nlevelmax']):
            # Easier if do this 1-indexed
            for cpu in range(self.amr_header['nboundary'] + self.amr_header['ncpu']):
                #ng is the number of octs on this level on this domain
                ng = _ng(cpu, level)
                if ng == 0: continue
                ind = fpu.read_vector(f, "I").astype("int64")
                fpu.skip(f, 2)
                pos = np.empty((ng, 3), dtype='float64')
                pos[:,0] = fpu.read_vector(f, "d") - nx
                pos[:,1] = fpu.read_vector(f, "d") - ny
                pos[:,2] = fpu.read_vector(f, "d") - nz
                #pos *= self.pf.domain_width
                #pos += self.parameter_file.domain_left_edge
                fpu.skip(f, 31)
                #parents = fpu.read_vector(f, "I")
                #fpu.skip(f, 6)
                #children = np.empty((ng, 8), dtype='int64')
                #for i in range(8):
                #    children[:,i] = fpu.read_vector(f, "I")
                #cpu_map = np.empty((ng, 8), dtype="int64")
                #for i in range(8):
                #    cpu_map[:,i] = fpu.read_vector(f, "I")
                #rmap = np.empty((ng, 8), dtype="int64")
                #for i in range(8):
                #    rmap[:,i] = fpu.read_vector(f, "I")
                # We don't want duplicate grids.
                # Note that we're adding *grids*, not individual cells.
                if level >= min_level:
                    assert(pos.shape[0] == ng)
                    n = self.oct_handler.add(cpu + 1, level - min_level, pos,
                                count_boundary = 1)
                    self._error_check(cpu, level, pos, n, ng, (nx, ny, nz))
                    if n > 0: max_level = max(level - min_level, max_level)
        self.max_level = max_level
        self.oct_handler.finalize()

    def _error_check(self, cpu, level, pos, n, ng, nn):
        # NOTE: We have the second conditional here because internally, it will
        # not add any octs in that case.
        if n == ng or cpu + 1 > self.oct_handler.num_domains:
            return
        # This is where we now check for issues with creating the new octs, and
        # we attempt to determine what precisely is going wrong.
        # These are all print statements.
        print "We have detected an error with the construction of the Octree."
        print "  The number of Octs to be added :  %s" % ng
        print "  The number of Octs added       :  %s" % n
        print "  Level                          :  %s" % level
        print "  CPU Number (0-indexed)         :  %s" % cpu
        for i, ax in enumerate('xyz'):
            print "  extent [%s]                     :  %s %s" % \
            (ax, pos[:,i].min(), pos[:,i].max())
        print "  domain left                    :  %s" % \
            (self.pf.domain_left_edge,)
        print "  domain right                   :  %s" % \
            (self.pf.domain_right_edge,)
        print "  offset applied                 :  %s %s %s" % \
            (nn[0], nn[1], nn[2])
        print "AMR Header:"
        for key in sorted(self.amr_header):
            print "   %-30s: %s" % (key, self.amr_header[key])
        raise RuntimeError

    def included(self, selector):
        if getattr(selector, "domain_id", None) is not None:
            return selector.domain_id == self.domain_id
        domain_ids = self.oct_handler.domain_identify(selector)
        return self.domain_id in domain_ids

class RAMSESDomainSubset(OctreeSubset):

    _domain_offset = 1

    def fill(self, content, fields, selector):
        # Here we get a copy of the file, which we skip through and read the
        # bits we want.
        oct_handler = self.oct_handler
        all_fields = self.domain.pf.h.fluid_field_list
        fields = [f for ft, f in fields]
        tr = {}
        cell_count = selector.count_oct_cells(self.oct_handler, self.domain_id)
        levels, cell_inds, file_inds = self.oct_handler.file_index_octs(
            selector, self.domain_id, cell_count)
        for field in fields:
            tr[field] = np.zeros(cell_count, 'float64')
        for level, offset in enumerate(self.domain.hydro_offset):
            if offset == -1: continue
            content.seek(offset)
            nc = self.domain.level_count[level]
            temp = {}
            for field in all_fields:
                temp[field] = np.empty((nc, 8), dtype="float64")
            for i in range(8):
                for field in all_fields:
                    if field not in fields:
                        fpu.skip(content)
                    else:
                        temp[field][:,i] = fpu.read_vector(content, 'd') # cell 1
            oct_handler.fill_level(level, levels, cell_inds, file_inds, tr, temp)
        return tr

class RAMSESGeometryHandler(OctreeGeometryHandler):

    def __init__(self, pf, data_style='ramses'):
        self._pf = pf # TODO: Figure out the class composition better!
        self.fluid_field_list = pf._fields_in_file
        self.data_style = data_style
        self.parameter_file = weakref.proxy(pf)
        # for now, the hierarchy file is the parameter file!
        self.hierarchy_filename = self.parameter_file.parameter_filename
        self.directory = os.path.dirname(self.hierarchy_filename)
        self.max_level = None

        self.float_type = np.float64
        super(RAMSESGeometryHandler, self).__init__(pf, data_style)

    def _initialize_oct_handler(self):
        self.domains = [RAMSESDomainFile(self.parameter_file, i + 1)
                        for i in range(self.parameter_file['ncpu'])]
        total_octs = sum(dom.local_oct_count #+ dom.ngridbound.sum()
                         for dom in self.domains)
        self.max_level = max(dom.max_level for dom in self.domains)
        self.num_grids = total_octs

    def _detect_fields(self):
        # Do we want to attempt to figure out what the fields are in the file?
        pfl = set([])
        if len(self.fluid_field_list) <= 0:
            self._setup_auto_fields()
        for domain in self.domains:
            pfl.update(set(domain.particle_field_offsets.keys()))
        self.particle_field_list = list(pfl)
        self.field_list = [("gas", f) for f in self.fluid_field_list] \
                        + self.particle_field_list

    def _setup_auto_fields(self):
        '''
        If no fluid fields are set, the code tries to set up a fluids array by hand
        '''
        # TODO: SUPPORT RT - THIS REQUIRES IMPLEMENTING A NEW FILE READER!
        # Find nvar
        # TODO: copy/pasted from DomainFile; needs refactoring!
        num = os.path.basename(self._pf.parameter_filename).split("."
                )[0].split("_")[1]
        testdomain = 1 # Just pick the first domain file to read
        basename = "%s/%%s_%s.out%05i" % (
            os.path.abspath(
              os.path.dirname(self._pf.parameter_filename)),
            num, testdomain)
        hydro_fn = basename % "hydro"
        # Do we have a hydro file?
        if hydro_fn:
            # Read the number of hydro  variables
            f = open(hydro_fn, "rb")
            fpu.skip(f, 1)
            nvar = fpu.read_vector(f, "i")[0]
        # OK, we got NVAR, now set up the arrays depending on what NVAR is
        # Allow some wiggle room for users to add too many variables
        if nvar < 5:
            mylog.debug("nvar=%s is too small! YT doesn't currently support 1D/2D runs in RAMSES %s")
            raise ValueError
        # Basic hydro runs
        if nvar == 5:
            fields = ["Density", 
                      "x-velocity", "y-velocity", "z-velocity", 
                      "Pressure"]
        if nvar > 5 and nvar < 11:
            fields = ["Density", 
                      "x-velocity", "y-velocity", "z-velocity", 
                      "Pressure", "Metallicity"]
        # MHD runs - NOTE: THE MHD MODULE WILL SILENTLY ADD 3 TO THE NVAR IN THE MAKEFILE
        if nvar == 11:
            fields = ["Density", 
                      "x-velocity", "y-velocity", "z-velocity", 
                      "x-Bfield-left", "y-Bfield-left", "z-Bfield-left", 
                      "x-Bfield-right", "y-Bfield-right", "z-Bfield-right", 
                      "Pressure"]
        if nvar > 11:
            fields = ["Density", 
                      "x-velocity", "y-velocity", "z-velocity", 
                      "x-Bfield-left", "y-Bfield-left", "z-Bfield-left", 
                      "x-Bfield-right", "y-Bfield-right", "z-Bfield-right", 
                      "Pressure","Metallicity"]
        while len(fields) < nvar:
            fields.append("var"+str(len(fields)))
        mylog.debug("No fields specified by user; automatically setting fields array to %s", str(fields))
        self.fluid_field_list = fields

    def _setup_derived_fields(self):
        self._parse_cooling()
        super(RAMSESGeometryHandler, self)._setup_derived_fields()
    
    def _setup_classes(self):
        dd = self._get_data_reader_dict()
        super(RAMSESGeometryHandler, self)._setup_classes(dd)
        self.object_types.sort()

    def _identify_base_chunk(self, dobj):
        if getattr(dobj, "_chunk_info", None) is None:
            domains = [dom for dom in self.domains if
                       dom.included(dobj.selector)]
            base_region = getattr(dobj, "base_region", dobj)
            if len(domains) > 1:
                mylog.debug("Identified %s intersecting domains", len(domains))
            subsets = [RAMSESDomainSubset(base_region, domain, self.parameter_file)
                       for domain in domains]
            dobj._chunk_info = subsets
        dobj._current_chunk = list(self._chunk_all(dobj))[0]

    def _chunk_all(self, dobj):
        oobjs = getattr(dobj._current_chunk, "objs", dobj._chunk_info)
        yield YTDataChunk(dobj, "all", oobjs, None)

    def _chunk_spatial(self, dobj, ngz, sort = None, preload_fields = None):
        sobjs = getattr(dobj._current_chunk, "objs", dobj._chunk_info)
        for i,og in enumerate(sobjs):
            if ngz > 0:
                g = og.retrieve_ghost_zones(ngz, [], smoothed=True)
            else:
                g = og
            yield YTDataChunk(dobj, "spatial", [g], None)

    def _chunk_io(self, dobj, cache = True):
        oobjs = getattr(dobj._current_chunk, "objs", dobj._chunk_info)
        for subset in oobjs:
            yield YTDataChunk(dobj, "io", [subset], None, cache = cache)

    def _parse_cooling(self):
        pf = self.parameter_file
        num = os.path.basename(pf.parameter_filename).split("."
                )[0].split("_")[1]
        basename = "%s/cooling_%05i.out" % (
            os.path.dirname(pf.parameter_filename), int(num))
        create_cooling_fields(basename, pf.field_info)

class RAMSESStaticOutput(StaticOutput):
    _hierarchy_class = RAMSESGeometryHandler
    _fieldinfo_fallback = RAMSESFieldInfo
    _fieldinfo_known = KnownRAMSESFields
    _particle_mass_name = "ParticleMass"
    _particle_coordinates_name = "Coordinates"
    
    def __init__(self, filename, data_style='ramses',
                 fields = [],
                 storage_filename = None):
        '''
        fields: An array of hydro variable fields in order of position in the hydro_XXXXX.outYYYYY file
                If set to None, will try a default set of fields
        '''
        self._fields_in_file = fields
        StaticOutput.__init__(self, filename, data_style)
        self.storage_filename = storage_filename

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
        self.units['unitary'] = 1.0 / (self.domain_right_edge - self.domain_left_edge).max()
        rho_u = self.parameters['unit_d']
        self.conversion_factors["Density"] = rho_u
        vel_u = self.parameters['unit_l'] / self.parameters['unit_t']
        self.conversion_factors["Pressure"] = rho_u*vel_u**2
        self.conversion_factors["x-velocity"] = vel_u
        self.conversion_factors["y-velocity"] = vel_u
        self.conversion_factors["z-velocity"] = vel_u
        # Necessary to get the length units in, which are needed for Mass
        # We also have to multiply by the boxlength here to scale into our
        # domain.
        self.conversion_factors['mass'] = rho_u * self.parameters['unit_l']**3

    def _setup_nounits_units(self):
        # Note that unit_l *already* converts to proper!
        # Also note that unit_l must be multiplied by the boxlen parameter to
        # ensure we are correctly set up for the current domain.
        unit_l = self.parameters['unit_l'] * self.parameters['boxlen']
        for unit in mpc_conversion.keys():
            self.units[unit] = unit_l * mpc_conversion[unit] / mpc_conversion["cm"]
            self.units['%sh' % unit] = self.units[unit] * self.hubble_constant
            self.units['%scm' % unit] = (self.units[unit] *
                                          (1 + self.current_redshift))
            self.units['%shcm' % unit] = (self.units['%sh' % unit] *
                                          (1 + self.current_redshift))
        for unit in sec_conversion.keys():
            self.time_units[unit] = self.parameters['unit_t'] / sec_conversion[unit]

    def _parse_parameter_file(self):
        # hardcoded for now
        # These should be explicitly obtained from the file, but for now that
        # will wait until a reorganization of the source tree and better
        # generalization.
        self.dimensionality = 3
        self.refine_by = 2
        self.parameters["HydroMethod"] = 'ramses'
        self.parameters["Time"] = 1. # default unit is 1...

        self.unique_identifier = \
            int(os.stat(self.parameter_filename)[stat.ST_CTIME])
        # We now execute the same logic Oliver's code does
        rheader = {}
        f = open(self.parameter_filename)
        def read_rhs(cast):
            line = f.readline()
            p, v = line.split("=")
            rheader[p.strip()] = cast(v)
        for i in range(6): read_rhs(int)
        f.readline()
        for i in range(11): read_rhs(float)
        f.readline()
        read_rhs(str)
        # This next line deserves some comment.  We specify a min_level that
        # corresponds to the minimum level in the RAMSES simulation.  RAMSES is
        # one-indexed, but it also does refer to the *oct* dimensions -- so
        # this means that a levelmin of 1 would have *1* oct in it.  So a
        # levelmin of 2 would have 8 octs at the root mesh level.
        self.min_level = rheader['levelmin'] - 1
        # Now we read the hilbert indices
        self.hilbert_indices = {}
        if rheader['ordering type'] == "hilbert":
            f.readline() # header
            for n in range(rheader['ncpu']):
                dom, mi, ma = f.readline().split()
                self.hilbert_indices[int(dom)] = (float(mi), float(ma))
        self.parameters.update(rheader)
        self.current_time = self.parameters['time'] * self.parameters['unit_t']
        self.domain_left_edge = np.zeros(3, dtype='float64')
        self.domain_dimensions = np.ones(3, dtype='int32') * \
                        2**(self.min_level+1)
        self.domain_right_edge = np.ones(3, dtype='float64')
        # This is likely not true, but I am not sure how to otherwise
        # distinguish them.
        mylog.warning("RAMSES frontend assumes all simulations are cosmological!")
        self.cosmological_simulation = 1
        self.periodicity = (True, True, True)
        self.current_redshift = (1.0 / rheader["aexp"]) - 1.0
        self.omega_lambda = rheader["omega_l"]
        self.omega_matter = rheader["omega_m"]
        self.hubble_constant = rheader["H0"] / 100.0 # This is H100
        self.max_level = rheader['levelmax'] - self.min_level

    def _setup_particle_type(self, ptype):
        orig = set(self.field_info.items())
        _setup_particle_fields(self.field_info, ptype)
        standard_particle_fields(self.field_info, ptype)
        return [n for n, v in set(self.field_info.items()).difference(orig)]

    @classmethod
    def _is_valid(self, *args, **kwargs):
        if not os.path.basename(args[0]).startswith("info_"): return False
        fn = args[0].replace("info_", "amr_").replace(".txt", ".out00001")
        return os.path.exists(fn)
