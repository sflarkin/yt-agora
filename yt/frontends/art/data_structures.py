"""
ART-specific data structures



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import numpy as np
import os.path
import stat
import weakref
import cStringIO
import difflib
import glob

from yt.funcs import *
from yt.geometry.oct_geometry_handler import \
    OctreeIndex
from yt.geometry.geometry_handler import \
    Index, YTDataChunk
from yt.data_objects.static_output import \
    Dataset
from yt.data_objects.octree_subset import \
    OctreeSubset
from yt.geometry.oct_container import \
    ARTOctreeContainer
from .fields import \
    ARTFieldInfo
from yt.utilities.definitions import \
    mpc_conversion
from yt.utilities.io_handler import \
    io_registry
from yt.utilities.lib.misc_utilities import \
    get_box_grids_level

from yt.frontends.art.definitions import *
import yt.utilities.fortran_utils as fpu
from .io import _read_art_level_info
from .io import _read_child_level
from .io import _read_root_level
from .io import b2t

from yt.utilities.definitions import \
    mpc_conversion, sec_conversion
from yt.utilities.io_handler import \
    io_registry
from yt.fields.field_info_container import \
    FieldInfoContainer, NullFunc
from yt.utilities.physical_constants import \
    mass_hydrogen_cgs, sec_per_Gyr


class ARTIndex(OctreeIndex):
    def __init__(self, ds, dataset_type="art"):
        self.fluid_field_list = fluid_fields
        self.dataset_type = dataset_type
        self.dataset = weakref.proxy(ds)
        self.index_filename = self.dataset.parameter_filename
        self.directory = os.path.dirname(self.index_filename)
        self.max_level = ds.max_level
        self.float_type = np.float64
        super(ARTIndex, self).__init__(ds, dataset_type)

    def get_smallest_dx(self):
        """
        Returns (in code units) the smallest cell size in the simulation.
        """
        # Overloaded
        ds = self.dataset
        return (1.0/pf.domain_dimensions.astype('f8') /
                (2**self.max_level)).min()

    def _initialize_oct_handler(self):
        """
        Just count the number of octs per domain and
        allocate the requisite memory in the oct tree
        """
        nv = len(self.fluid_field_list)
        self.oct_handler = ARTOctreeContainer(
            self.dataset.domain_dimensions/2,  # dd is # of root cells
            self.dataset.domain_left_edge,
            self.dataset.domain_right_edge,
            1)
        # The 1 here refers to domain_id == 1 always for ARTIO.
        self.domains = [ARTDomainFile(self.dataset, nv, 
                                      self.oct_handler, 1)]
        self.octs_per_domain = [dom.level_count.sum() for dom in self.domains]
        self.total_octs = sum(self.octs_per_domain)
        mylog.debug("Allocating %s octs", self.total_octs)
        self.oct_handler.allocate_domains(self.octs_per_domain)
        domain = self.domains[0]
        domain._read_amr_root(self.oct_handler)
        domain._read_amr_level(self.oct_handler)
        self.oct_handler.finalize()

    def _detect_output_fields(self):
        self.particle_field_list = [f for f in particle_fields]
        self.field_list = [("gas", f) for f in fluid_fields]
        # now generate all of the possible particle fields
        if "wspecies" in self.dataset.parameters.keys():
            wspecies = self.dataset.parameters['wspecies']
            nspecies = len(wspecies)
            self.dataset.particle_types = ["darkmatter", "stars"]
            for specie in range(nspecies):
                self.dataset.particle_types.append("specie%i" % specie)
            self.dataset.particle_types_raw = tuple(
                self.dataset.particle_types)
        else:
            self.dataset.particle_types = []
        for ptype in self.dataset.particle_types:
            for pfield in self.particle_field_list:
                pfn = (ptype, pfield)
                self.field_list.append(pfn)

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
            subsets = [ARTDomainSubset(base_region, domain, self.dataset)
                       for domain in domains]
            dobj._chunk_info = subsets
        dobj._current_chunk = list(self._chunk_all(dobj))[0]

    def _chunk_all(self, dobj):
        oobjs = getattr(dobj._current_chunk, "objs", dobj._chunk_info)
        # We pass the chunk both the current chunk and list of chunks,
        # as well as the referring data source
        yield YTDataChunk(dobj, "all", oobjs, None)

    def _chunk_spatial(self, dobj, ngz, sort = None, preload_fields = None):
        sobjs = getattr(dobj._current_chunk, "objs", dobj._chunk_info)
        for i,og in enumerate(sobjs):
            if ngz > 0:
                g = og.retrieve_ghost_zones(ngz, [], smoothed=True)
            else:
                g = og
            yield YTDataChunk(dobj, "spatial", [g], None)

    def _chunk_io(self, dobj, cache = True, local_only = False):
        """
        Since subsets are calculated per domain,
        i.e. per file, yield each domain at a time to
        organize by IO. We will eventually chunk out NMSU ART
        to be level-by-level.
        """
        oobjs = getattr(dobj._current_chunk, "objs", dobj._chunk_info)
        for subset in oobjs:
            yield YTDataChunk(dobj, "io", [subset], None,
                              cache = cache)


class ARTDataset(Dataset):
    _index_class = ARTIndex
    _field_info_class = ARTFieldInfo

    def __init__(self, filename, dataset_type='art',
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
        Dataset.__init__(self, filename, dataset_type)
        self.storage_filename = storage_filename

    def _find_files(self, file_amr):
        """
        Given the AMR base filename, attempt to find the
        particle header, star files, etc.
        """
        base_prefix, base_suffix = filename_pattern['amr']
        aexpstr = 'a'+file_amr.rsplit('a',1)[1].replace(base_suffix,'')
        possibles = glob.glob(os.path.dirname(file_amr)+"/*")
        for filetype, (prefix, suffix) in filename_pattern.iteritems():
            # if this attribute is already set skip it
            if getattr(self, "_file_"+filetype, None) is not None:
                continue
            match = None
            for possible in possibles:
                if possible.endswith(aexpstr+suffix):
                    if os.path.basename(possible).startswith(prefix):
                        match = possible
            if match is not None:
                mylog.info('discovered %s:%s', filetype, match)
                setattr(self, "_file_"+filetype, match)
            else:
                setattr(self, "_file_"+filetype, None)

    def __repr__(self):
        return self._file_amr.split('/')[-1]

    def _set_code_unit_attributes(self):
        """
        Generates the conversion to various physical units based
                on the parameters from the header
        """

        # spatial units
        z = self.current_redshift
        h = self.hubble_constant
        boxcm_cal = self.parameters["boxh"]
        boxcm_uncal = boxcm_cal / h
        box_proper = boxcm_uncal/(1+z)
        aexpn = self.parameters["aexpn"]

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

        # I have left much of the old code above.for historical reference, but
        # it is largely or completely vestigial.  The subsequent unit setting
        # is from ARTIO, which I have been assured has identical units to
        # NMSU-ART for hydrodynamic quantities..
        #
        # That being said, these are a bit uncertain to me.

        self.mass_unit = self.quan(cf["Mass"], "g*%s" % ng**3)
        self.length_unit = self.quan(box_proper, "Mpc")
        self.velocity_unit = self.quan(cf["Velocity"], "cm/s")
        self.time_unit = self.length_unit / self.velocity_unit

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
            amr_header_vals = fpu.read_attrs(f, amr_header_struct, '>')
            for to_skip in ['tl', 'dtl', 'tlold', 'dtlold', 'iSO']:
                skipped = fpu.skip(f, endian='>')
            (self.ncell) = fpu.read_vector(f, 'i', '>')[0]
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
            self.root_iOctCh = fpu.read_vector(f, 'i', '>')[:self.root_ncells]
            self.root_iOctCh = self.root_iOctCh.reshape(self.domain_dimensions,
                                                        order='F')
            self.root_grid_offset = f.tell()
            self.root_nhvar = fpu.skip(f, endian='>')
            self.root_nvar = fpu.skip(f, endian='>')
            # make sure that the number of root variables is a multiple of
            # rootcells
            assert self.root_nhvar % self.root_ncells == 0
            assert self.root_nvar % self.root_ncells == 0
            self.nhydro_variables = ((self.root_nhvar+self.root_nvar) /
                                     self.root_ncells)
            self.iOctFree, self.nOct = fpu.read_vector(f, 'i', '>')
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
                particle_header_vals = fpu.read_attrs(
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
        self.gamma = self.parameters["gamma"]
        mylog.info("Max level is %02i", self.max_level)

    @classmethod
    def _is_valid(self, *args, **kwargs):
        """
        Defined for the NMSU file naming scheme.
        This could differ for other formats.
        """
        f = ("%s" % args[0])
        prefix, suffix = filename_pattern['amr']
        if not os.path.isfile(f): return False
        with open(f, 'rb') as fh:
            try:
                amr_header_vals = fpu.read_attrs(fh, amr_header_struct, '>')
                return True
            except:
                return False
        return False

class ARTDomainSubset(OctreeSubset):

    def fill(self, content, ftfields, selector):
        """
        This is called from IOHandler. It takes content
        which is a binary stream, reads the requested field
        over this while domain. It then uses oct_handler fill
        to reorgnize values from IO read index order to
        the order they are in in the octhandler.
        """
        oct_handler = self.oct_handler
        all_fields = self.domain.ds.index.fluid_field_list
        fields = [f for ft, f in ftfields]
        field_idxs = [all_fields.index(f) for f in fields]
        source, tr = {}, {}
        cell_count = selector.count_oct_cells(self.oct_handler, self.domain_id)
        levels, cell_inds, file_inds = self.oct_handler.file_index_octs(
            selector, self.domain_id, cell_count)
        for field in fields:
            tr[field] = np.zeros(cell_count, 'float64')
        data = _read_root_level(content, self.domain.level_child_offsets,
                                self.domain.level_count)
        ns = (self.domain.ds.domain_dimensions.prod() / 8, 8)
        for field, fi in zip(fields, field_idxs):
            source[field] = np.empty(ns, dtype="float64", order="C")
            dt = data[fi,:].reshape(self.domain.ds.domain_dimensions,
                                    order="F")
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        ii = ((k*2)+j)*2+i
                        # Note: C order because our index converts C to F.
                        source[field][:,ii] = \
                            dt[i::2,j::2,k::2].ravel(order="C")
        oct_handler.fill_level(0, levels, cell_inds, file_inds, tr, source)
        del source
        # Now we continue with the additional levels.
        for level in range(1, self.ds.max_level + 1):
            no = self.domain.level_count[level]
            noct_range = [0, no]
            source = _read_child_level(
                content, self.domain.level_child_offsets,
                self.domain.level_offsets,
                self.domain.level_count, level, fields,
                self.domain.ds.domain_dimensions,
                self.domain.ds.parameters['ncell0'],
                noct_range=noct_range)
            oct_handler.fill_level(level, levels, cell_inds, file_inds, tr,
                source)
        return tr

class ARTDomainFile(object):
    """
    Read in the AMR, left/right edges, fill out the octhandler
    """
    # We already read in the header in static output,
    # and since these headers are defined in only a single file it's
    # best to leave them in the static output
    _last_mask = None
    _last_seletor_id = None

    def __init__(self, ds, nvar, oct_handler, domain_id):
        self.nvar = nvar
        self.ds = ds
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
        f = open(self.ds._file_amr, "rb")
        nhydrovars, inoll, _level_oct_offsets, _level_child_offsets = \
            self._count_art_octs(f,  self.ds.child_grid_offset,
                self.ds.min_level, self.ds.max_level)
        # remember that the root grid is by itself; manually add it back in
        inoll[0] = self.ds.domain_dimensions.prod()/8
        _level_child_offsets[0] = self.ds.root_grid_offset
        self.nhydrovars = nhydrovars
        self.inoll = inoll  # number of octs
        self._level_oct_offsets = _level_oct_offsets
        self._level_child_offsets = _level_child_offsets
        self._level_count = inoll
        return self._level_oct_offsets

    def _count_art_octs(self, f, offset, MinLev, MaxLevelNow):
        level_oct_offsets = [0, ]
        level_child_offsets = [0, ]
        f.seek(offset)
        nchild, ntot = 8, 0
        Level = np.zeros(MaxLevelNow+1 - MinLev, dtype='int64')
        iNOLL = np.zeros(MaxLevelNow+1 - MinLev, dtype='int64')
        iHOLL = np.zeros(MaxLevelNow+1 - MinLev, dtype='int64')
        for Lev in xrange(MinLev + 1, MaxLevelNow+1):
            level_oct_offsets.append(f.tell())

            # Get the info for this level, skip the rest
            # print "Reading oct tree data for level", Lev
            # print 'offset:',f.tell()
            Level[Lev], iNOLL[Lev], iHOLL[Lev] = fpu.read_vector(f, 'i', '>')
            # print 'Level %i : '%Lev, iNOLL
            # print 'offset after level record:',f.tell()
            iOct = iHOLL[Lev] - 1
            nLevel = iNOLL[Lev]
            nLevCells = nLevel * nchild
            ntot = ntot + nLevel

            # Skip all the oct hierarchy data
            ns = fpu.peek_record_size(f, endian='>')
            size = struct.calcsize('>i') + ns + struct.calcsize('>i')
            f.seek(f.tell()+size * nLevel)

            level_child_offsets.append(f.tell())
            # Skip the child vars data
            ns = fpu.peek_record_size(f, endian='>')
            size = struct.calcsize('>i') + ns + struct.calcsize('>i')
            f.seek(f.tell()+size * nLevel*nchild)

            # find nhydrovars
            nhydrovars = 8+2
        f.seek(offset)
        return nhydrovars, iNOLL, level_oct_offsets, level_child_offsets


    def _read_amr_level(self, oct_handler):
        """Open the oct file, read in octs level-by-level.
           For each oct, only the position, index, level and domain
           are needed - its position in the octree is found automatically.
           The most important is finding all the information to feed
           oct_handler.add
        """
        self.level_offsets
        f = open(self.ds._file_amr, "rb")
        for level in range(1, self.ds.max_level + 1):
            unitary_center, fl, iocts, nocts, root_level = \
                _read_art_level_info( f,
                    self._level_oct_offsets, level,
                    coarse_grid=self.ds.domain_dimensions[0],
                    root_level=self.ds.root_level)
            nocts_check = oct_handler.add(self.domain_id, level,
                                          unitary_center)
            assert(nocts_check == nocts)
            mylog.debug("Added %07i octs on level %02i, cumulative is %07i",
                        nocts, level, oct_handler.nocts)

    def _read_amr_root(self, oct_handler):
        self.level_offsets
        f = open(self.ds._file_amr, "rb")
        # add the root *cell* not *oct* mesh
        root_octs_side = self.ds.domain_dimensions[0]/2
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
        domain_ids = self.ds.index.oct_handler.domain_identify(selector)
        return self.domain_id in domain_ids
