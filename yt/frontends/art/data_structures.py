"""
ART-specific data structures

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: UCSD
Author: Christopher Moody <cemoody@ucsc.edu>
Affiliation: UCSC
Homepage: http://yt-project.org/
License:
  Copyright (C) 2010-2011 Matthew Turk.  All Rights Reserved.

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
import numpy as np
import os.path
import glob
import stat
import weakref
import cStringIO
import difflib
import glob

from yt.funcs import *
from yt.geometry.oct_geometry_handler import \
    OctreeGeometryHandler
from yt.geometry.geometry_handler import \
    GeometryHandler, YTDataChunk
from yt.data_objects.static_output import \
    StaticOutput
from yt.data_objects.octree_subset import \
    OctreeSubset
from yt.geometry.oct_container import \
    ARTOctreeContainer
from yt.data_objects.field_info_container import \
    FieldInfoContainer, NullFunc
from .fields import \
    ARTFieldInfo, add_art_field, KnownARTFields
from yt.utilities.definitions import \
    mpc_conversion
from yt.utilities.io_handler import \
    io_registry
from yt.utilities.lib import \
    get_box_grids_level
import yt.utilities.lib as amr_utils

from yt.frontends.art.definitions import *
from yt.utilities.fortran_utils import *
from .io import _read_art_level_info
from .io import _read_child_mask_level
from .io import _read_child_level
from .io import _read_root_level
from .io import _count_art_octs
from .io import b2t

import yt.frontends.ramses._ramses_reader as _ramses_reader

from .fields import ARTFieldInfo, KnownARTFields
from yt.utilities.definitions import \
    mpc_conversion, sec_conversion
from yt.utilities.lib import \
    get_box_grids_level
from yt.utilities.io_handler import \
    io_registry
from yt.data_objects.field_info_container import \
    FieldInfoContainer, NullFunc
from yt.utilities.physical_constants import \
    mass_hydrogen_cgs, sec_per_Gyr


class ARTGeometryHandler(OctreeGeometryHandler):
    def __init__(self, pf, data_style="art"):
        self.fluid_field_list = fluid_fields
        self.data_style = data_style
        self.parameter_file = weakref.proxy(pf)
        self.hierarchy_filename = self.parameter_file.parameter_filename
        self.directory = os.path.dirname(self.hierarchy_filename)
        self.max_level = pf.max_level
        self.float_type = np.float64
        super(ARTGeometryHandler, self).__init__(pf, data_style)

    def get_smallest_dx(self):
        """
        Returns (in code units) the smallest cell size in the simulation.
        """
        # Overloaded
        pf = self.parameter_file
        return (1.0/pf.domain_dimensions.astype('f8') /
                (2**self.max_level)).min()

    def _initialize_oct_handler(self):
        """
        Just count the number of octs per domain and
        allocate the requisite memory in the oct tree
        """
        nv = len(self.fluid_field_list)
        self.oct_handler = ARTOctreeContainer(
            self.parameter_file.domain_dimensions/2,  # dd is # of root cells
            self.parameter_file.domain_left_edge,
            self.parameter_file.domain_right_edge)
        self.domains = [ARTDomainFile(self.parameter_file, 0, nv, l,
                                      self.oct_handler)]
        self.octs_per_domain = [dom.level_count.sum() for dom in self.domains]
        self.total_octs = sum(self.octs_per_domain)
        mylog.debug("Allocating %s octs", self.total_octs)
        self.oct_handler.allocate_domains(self.octs_per_domain)
        for domain in self.domains:
            if domain.domain_level == 0:
                domain._read_amr_root(self.oct_handler)
            else:
                domain._read_amr_level(self.oct_handler)

    def _detect_fields(self):
        self.particle_field_list = particle_fields
        self.field_list = set(fluid_fields + particle_fields +
                              particle_star_fields)
        self.field_list = list(self.field_list)
        # now generate all of the possible particle fields
        if "wspecies" in self.parameter_file.parameters.keys():
            wspecies = self.parameter_file.parameters['wspecies']
            nspecies = len(wspecies)
            self.parameter_file.particle_types = ["all", "darkmatter", "stars"]
            for specie in range(nspecies):
                self.parameter_file.particle_types.append("specie%i" % specie)
        else:
            self.parameter_file.particle_types = []

    def _setup_classes(self):
        dd = self._get_data_reader_dict()
        super(ARTGeometryHandler, self)._setup_classes(dd)
        self.object_types.sort()

    def _identify_base_chunk(self, dobj):
        """
        Take the passed in data source dobj, and use its embedded selector
        to calculate the domain mask, build the reduced domain
        subsets and oct counts. Attach this information to dobj.
        """
        if getattr(dobj, "_chunk_info", None) is None:
            # Get all octs within this oct handler
            domains = [dom for dom in self.domains if
                       dom.included(dobj.selector)]
            base_region = getattr(dobj, "base_region", dobj)
            if len(domains) > 1:
                mylog.debug("Identified %s intersecting domains", len(domains))
            subsets = [ARTDomainSubset(base_region, domain, self.parameter_file)
                       for domain in domains]
            dobj._chunk_info = subsets
        dobj._current_chunk = list(self._chunk_all(dobj))[0]

    def _chunk_all(self, dobj):
        oobjs = getattr(dobj._current_chunk, "objs", dobj._chunk_info)
        # We pass the chunk both the current chunk and list of chunks,
        # as well as the referring data source
        yield YTDataChunk(dobj, "all", oobjs, None)

    def _chunk_spatial(self, dobj, ngz, sort = None):
        sobjs = getattr(dobj._current_chunk, "objs", dobj._chunk_info)
        for i,og in enumerate(sobjs):
            if ngz > 0:
                g = og.retrieve_ghost_zones(ngz, [], smoothed=True)
            else:
                g = og
            yield YTDataChunk(dobj, "spatial", [g], None)

    def _chunk_io(self, dobj):
        """
        Since subsets are calculated per domain,
        i.e. per file, yield each domain at a time to
        organize by IO. We will eventually chunk out NMSU ART
        to be level-by-level.
        """
        oobjs = getattr(dobj._current_chunk, "objs", dobj._chunk_info)
        for subset in oobjs:
            yield YTDataChunk(dobj, "io", [subset], None)


class ARTStaticOutput(StaticOutput):
    _hierarchy_class = ARTGeometryHandler
    _fieldinfo_fallback = ARTFieldInfo
    _fieldinfo_known = KnownARTFields

    def __init__(self, filename, data_style='art',
                 fields=None, storage_filename=None,
                 skip_particles=False, skip_stars=False,
                 limit_level=None, spread_age=True,
                 force_max_level=None, file_particle_header=None,
                 file_particle_data=None, file_particle_stars=None):
        if fields is None:
            fields = fluid_fields
        filename = os.path.abspath(filename)
        self._fields_in_file = fields
        self._file_amr = filename
        self._file_particle_header = file_particle_header
        self._file_particle_data = file_particle_data
        self._file_particle_stars = file_particle_stars
        self._find_files(filename)
        self.parameter_filename = filename
        self.skip_particles = skip_particles
        self.skip_stars = skip_stars
        self.limit_level = limit_level
        self.max_level = limit_level
        self.force_max_level = force_max_level
        self.spread_age = spread_age
        self.domain_left_edge = np.zeros(3, dtype='float')
        self.domain_right_edge = np.zeros(3, dtype='float')+1.0
        StaticOutput.__init__(self, filename, data_style)
        self.storage_filename = storage_filename

    def _find_files(self, file_amr):
        """
        Given the AMR base filename, attempt to find the
        particle header, star files, etc.
        """
        base_prefix, base_suffix = filename_pattern['amr']
        possibles = glob.glob(os.path.dirname(file_amr)+"/*")
        for filetype, (prefix, suffix) in filename_pattern.iteritems():
            # if this attribute is already set skip it
            if getattr(self, "_file_"+filetype, None) is not None:
                continue
            stripped = file_amr.replace(base_prefix, prefix)
            stripped = stripped.replace(base_suffix, suffix)
            match, = difflib.get_close_matches(stripped, possibles, 1, 0.6)
            if match is not None:
                mylog.info('discovered %s:%s', filetype, match)
                setattr(self, "_file_"+filetype, match)
            else:
                setattr(self, "_file_"+filetype, None)

    def __repr__(self):
        return self._file_amr.split('/')[-1]

    def _set_units(self):
        """
        Generates the conversion to various physical units based
                on the parameters from the header
        """
        self.units = {}
        self.time_units = {}
        self.time_units['1'] = 1
        self.units['1'] = 1.0
        self.units['unitary'] = 1.0

        # spatial units
        z = self.current_redshift
        h = self.hubble_constant
        boxcm_cal = self.parameters["boxh"]
        boxcm_uncal = boxcm_cal / h
        box_proper = boxcm_uncal/(1+z)
        aexpn = self["aexpn"]
        for unit in mpc_conversion:
            self.units[unit] = mpc_conversion[unit] * box_proper
            self.units[unit+'h'] = mpc_conversion[unit] * box_proper * h
            self.units[unit+'cm'] = mpc_conversion[unit] * boxcm_uncal
            self.units[unit+'hcm'] = mpc_conversion[unit] * boxcm_cal

        # all other units
        wmu = self.parameters["wmu"]
        Om0 = self.parameters['Om0']
        ng = self.parameters['ng']
        wmu = self.parameters["wmu"]
        boxh = self.parameters['boxh']
        aexpn = self.parameters["aexpn"]
        hubble = self.parameters['hubble']

        cf = defaultdict(lambda: 1.0)
        r0 = boxh/ng
        P0 = 4.697e-16 * Om0**2.0 * r0**2.0 * hubble**2.0
        T_0 = 3.03e5 * r0**2.0 * wmu * Om0  # [K]
        S_0 = 52.077 * wmu**(5.0/3.0)
        S_0 *= hubble**(-4.0/3.0)*Om0**(1.0/3.0)*r0**2.0
        # v0 =  r0 * 50.0*1.0e5 * np.sqrt(self.omega_matter)  #cm/s
        v0 = 50.0*r0*np.sqrt(Om0)
        t0 = r0/v0
        rho1 = 1.8791e-29 * hubble**2.0 * self.omega_matter
        rho0 = 2.776e11 * hubble**2.0 * Om0
        tr = 2./3. * (3.03e5*r0**2.0*wmu*self.omega_matter)*(1.0/(aexpn**2))
        aM0 = rho0 * (boxh/hubble)**3.0 / ng**3.0
        cf['r0'] = r0
        cf['P0'] = P0
        cf['T_0'] = T_0
        cf['S_0'] = S_0
        cf['v0'] = v0
        cf['t0'] = t0
        cf['rho0'] = rho0
        cf['rho1'] = rho1
        cf['tr'] = tr
        cf['aM0'] = aM0

        # factors to multiply the native code units to CGS
        cf['Pressure'] = P0  # already cgs
        cf['Velocity'] = v0/aexpn*1.0e5  # proper cm/s
        cf["Mass"] = aM0 * 1.98892e33
        cf["Density"] = rho1*(aexpn**-3.0)
        cf["GasEnergy"] = rho0*v0**2*(aexpn**-5.0)
        cf["Potential"] = 1.0
        cf["Entropy"] = S_0
        cf["Temperature"] = tr
        cf["Time"] = 1.0
        cf["particle_mass"] = cf['Mass']
        cf["particle_mass_initial"] = cf['Mass']
        self.cosmological_simulation = True
        self.conversion_factors = cf

        for ax in 'xyz':
            self.conversion_factors["%s-velocity" % ax] = cf["Velocity"]
            self.conversion_factors["particle_velocity_%s" % ax] = cf["Velocity"]
        for pt in particle_fields:
            if pt not in self.conversion_factors.keys():
                self.conversion_factors[pt] = 1.0
        for unit in sec_conversion.keys():
            self.time_units[unit] = 1.0 / sec_conversion[unit]

    def _parse_parameter_file(self):
        """
        Get the various simulation parameters & constants.
        """
        self.dimensionality = 3
        self.refine_by = 2
        self.periodicity = (True, True, True)
        self.cosmological_simulation = True
        self.parameters = {}
        self.unique_identifier = \
            int(os.stat(self.parameter_filename)[stat.ST_CTIME])
        self.parameters.update(constants)
        self.parameters['Time'] = 1.0
        # read the amr header
        with open(self._file_amr, 'rb') as f:
            amr_header_vals = read_attrs(f, amr_header_struct, '>')
            for to_skip in ['tl', 'dtl', 'tlold', 'dtlold', 'iSO']:
                skipped = skip(f, endian='>')
            (self.ncell) = read_vector(f, 'i', '>')[0]
            # Try to figure out the root grid dimensions
            est = int(np.rint(self.ncell**(1.0/3.0)))
            # Note here: this is the number of *cells* on the root grid.
            # This is not the same as the number of Octs.
            # domain dimensions is the number of root *cells*
            self.domain_dimensions = np.ones(3, dtype='int64')*est
            self.root_grid_mask_offset = f.tell()
            self.root_nocts = self.domain_dimensions.prod()/8
            self.root_ncells = self.root_nocts*8
            mylog.debug("Estimating %i cells on a root grid side," +
                        "%i root octs", est, self.root_nocts)
            self.root_iOctCh = read_vector(f, 'i', '>')[:self.root_ncells]
            self.root_iOctCh = self.root_iOctCh.reshape(self.domain_dimensions,
                                                        order='F')
            self.root_grid_offset = f.tell()
            self.root_nhvar = skip(f, endian='>')
            self.root_nvar = skip(f, endian='>')
            # make sure that the number of root variables is a multiple of
            # rootcells
            assert self.root_nhvar % self.root_ncells == 0
            assert self.root_nvar % self.root_ncells == 0
            self.nhydro_variables = ((self.root_nhvar+self.root_nvar) /
                                     self.root_ncells)
            self.iOctFree, self.nOct = read_vector(f, 'i', '>')
            self.child_grid_offset = f.tell()
            self.parameters.update(amr_header_vals)
            self.parameters['ncell0'] = self.parameters['ng']**3
            # estimate the root level
            float_center, fl, iocts, nocts, root_level = _read_art_level_info(
                f,
                [0, self.child_grid_offset], 1,
                coarse_grid=self.domain_dimensions[0])
            del float_center, fl, iocts, nocts
            self.root_level = root_level
            mylog.info("Using root level of %02i", self.root_level)
        # read the particle header
        if not self.skip_particles and self._file_particle_header:
            with open(self._file_particle_header, "rb") as fh:
                particle_header_vals = read_attrs(
                    fh, particle_header_struct, '>')
                fh.seek(seek_extras)
                n = particle_header_vals['Nspecies']
                wspecies = np.fromfile(fh, dtype='>f', count=10)
                lspecies = np.fromfile(fh, dtype='>i', count=10)
            self.parameters['wspecies'] = wspecies[:n]
            self.parameters['lspecies'] = lspecies[:n]
            ls_nonzero = np.diff(lspecies)[:n-1]
            self.star_type = len(ls_nonzero)
            mylog.info("Discovered %i species of particles", len(ls_nonzero))
            mylog.info("Particle populations: "+'%1.1e '*len(ls_nonzero),
                       *ls_nonzero)
            for k, v in particle_header_vals.items():
                if k in self.parameters.keys():
                    if not self.parameters[k] == v:
                        mylog.info(
                            "Inconsistent parameter %s %1.1e  %1.1e", k, v,
                            self.parameters[k])
                else:
                    self.parameters[k] = v
            self.parameters_particles = particle_header_vals

        # setup standard simulation params yt expects to see
        self.current_redshift = self.parameters["aexpn"]**-1.0 - 1.0
        self.omega_lambda = amr_header_vals['Oml0']
        self.omega_matter = amr_header_vals['Om0']
        self.hubble_constant = amr_header_vals['hubble']
        self.min_level = amr_header_vals['min_level']
        self.max_level = amr_header_vals['max_level']
        if self.limit_level is not None:
            self.max_level = min(
                self.limit_level, amr_header_vals['max_level'])
        if self.force_max_level is not None:
            self.max_level = self.force_max_level
        self.hubble_time = 1.0/(self.hubble_constant*100/3.08568025e19)
        self.current_time = b2t(self.parameters['t']) * sec_per_Gyr
        mylog.info("Max level is %02i", self.max_level)

    @classmethod
    def _is_valid(self, *args, **kwargs):
        """
        Defined for the NMSU file naming scheme.
        This could differ for other formats.
        """
        f = ("%s" % args[0])
        prefix, suffix = filename_pattern['amr']
        with open(f, 'rb') as fh:
            try:
                amr_header_vals = read_attrs(fh, amr_header_struct, '>')
                return True
            except AssertionError:
                return False
        return False

class ARTDomainSubset(OctreeSubset):
    def __init__(self, base_region, domain, pf):
        super(ARTDomainSubset, self).__init__(base_region, domain, pf)
        self.domain_level = domain.domain_level

    def fill_root(self, content, ftfields):
        """
        This is called from IOHandler. It takes content
        which is a binary stream, reads the requested field
        over this while domain. It then uses oct_handler fill
        to reorgnize values from IO read index order to
        the order they are in in the octhandler.
        """
        oct_handler = self.oct_handler
        all_fields = self.domain.pf.h.fluid_field_list
        fields = [f for ft, f in ftfields]
        field_idxs = [all_fields.index(f) for f in fields]
        source = {}
        data = _read_root_level(content, self.domain.level_child_offsets,
                                self.domain.level_count)

        for field, i in zip(fields, field_idxs):
            temp = np.reshape(data[i, :], self.domain.pf.domain_dimensions,
                              order='F').astype('float64').T
            source[field] = temp
        dest = oct_handler.fill_level_from_grid(
            self.selector, self.domain_id, source)
        return dest

    def fill_level(self, content, ftfields):
        oct_handler = self.oct_handler
        fields = [f for ft, f in ftfields]
        level_offset = 0
        dest = {}
        for field in fields:
            dest[field] = np.zeros(self.cell_count, 'float64')-1.
        level = self.domain_level
        no = self.domain.level_count[level]
        noct_range = [0, no]
        source = _read_child_level(
            content, self.domain.level_child_offsets,
            self.domain.level_offsets,
            self.domain.level_count, level, fields,
            self.domain.pf.domain_dimensions,
            self.domain.pf.parameters['ncell0'],
            noct_range=noct_range)
        nocts_filling = noct_range[1]-noct_range[0]
        level_offset += oct_handler.fill_level(self.domain.domain_id,
                                               level, dest, source,
                                               self.mask, level_offset,
                                               noct_range[0],
                                               nocts_filling)
        return dest

class ARTDomainFile(object):
    """
    Read in the AMR, left/right edges, fill out the octhandler
    """
    # We already read in the header in static output,
    # and since these headers are defined in only a single file it's
    # best to leave them in the static output
    _last_mask = None
    _last_seletor_id = None

    def __init__(self, pf, nvar, level, oct_handler):
        self.nvar = nvar
        self.pf = pf
        self.domain_id = domain_id
        self._level_count = None
        self._level_oct_offsets = None
        self._level_child_offsets = None
        self.oct_handler = oct_handler

    @property
    def level_count(self):
        # this is number of *octs*
        if self._level_count is not None:
            return self._level_count
        self.level_offsets
        return self._level_count

    @property
    def level_child_offsets(self):
        if self._level_count is not None:
            return self._level_child_offsets
        self.level_offsets
        return self._level_child_offsets

    @property
    def level_offsets(self):
        # this is used by the IO operations to find the file offset,
        # and then start reading to fill values
        # note that this is called hydro_offset in ramses
        if self._level_oct_offsets is not None:
            return self._level_oct_offsets
        # We now have to open the file and calculate it
        f = open(self.pf._file_amr, "rb")
        nhydrovars, inoll, _level_oct_offsets, _level_child_offsets = \
            _count_art_octs(f,  self.pf.child_grid_offset, self.pf.min_level,
                            self.pf.max_level)
        # remember that the root grid is by itself; manually add it back in
        inoll[0] = self.pf.domain_dimensions.prod()/8
        _level_child_offsets[0] = self.pf.root_grid_offset
        self.nhydrovars = nhydrovars
        self.inoll = inoll  # number of octs
        self._level_oct_offsets = _level_oct_offsets
        self._level_child_offsets = _level_child_offsets
        self._level_count = inoll
        return self._level_oct_offsets

    def _read_amr_level(self, oct_handler):
        """Open the oct file, read in octs level-by-level.
           For each oct, only the position, index, level and domain
           are needed - its position in the octree is found automatically.
           The most important is finding all the information to feed
           oct_handler.add
        """
        self.level_offsets
        f = open(self.pf._file_amr, "rb")
        for level in range(self.pf.max_level + 1):
            unitary_center, fl, iocts, nocts, root_level = \
                _read_art_level_info( f,
                    self._level_oct_offsets, level,
                    coarse_grid=self.pf.domain_dimensions[0],
                    root_level=self.pf.root_level)
            nocts_check = oct_handler.add(self.domain_id, level,
                                          unitary_center)
            assert(nocts_check == nocts)
            mylog.debug("Added %07i octs on level %02i, cumulative is %07i",
                        nocts, level, oct_handler.nocts)

    def _read_amr_root(self, oct_handler):
        self.level_offsets
        f = open(self.pf._file_amr, "rb")
        # add the root *cell* not *oct* mesh
        root_octs_side = self.pf.domain_dimensions[0]/2
        NX = np.ones(3)*root_octs_side
        octs_side = NX*2 # Level == 0
        LE = np.array([0.0, 0.0, 0.0], dtype='float64')
        RE = np.array([1.0, 1.0, 1.0], dtype='float64')
        root_dx = (RE - LE) / NX
        LL = LE + root_dx/2.0
        RL = RE - root_dx/2.0
        # compute floating point centers of root octs
        root_fc = np.mgrid[LL[0]:RL[0]:NX[0]*1j,
                           LL[1]:RL[1]:NX[1]*1j,
                           LL[2]:RL[2]:NX[2]*1j]
        root_fc = np.vstack([p.ravel() for p in root_fc]).T
        nocts_check = oct_handler.add(self.domain_id, 0, root_fc)
        assert(oct_handler.nocts == root_fc.shape[0])
        mylog.debug("Added %07i octs on level %02i, cumulative is %07i",
                    root_octs_side**3, 0, oct_handler.nocts)

    def included(self, selector):
        return True
        if getattr(selector, "domain_id", None) is not None:
            return selector.domain_id == self.domain_id
        domain_ids = self.pf.h.oct_handler.domain_identify(selector)
        return self.domain_id in domain_ids
