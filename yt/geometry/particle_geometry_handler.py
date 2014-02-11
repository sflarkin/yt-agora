"""
Particle-only geometry handler




"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import h5py
import numpy as na
import string, re, gc, time, cPickle
import weakref

from itertools import chain, izip

from yt.funcs import *
from yt.utilities.logger import ytLogger as mylog
from yt.arraytypes import blankRecordArray
from yt.config import ytcfg
from yt.geometry.geometry_handler import Index, YTDataChunk
from yt.geometry.particle_oct_container import \
    ParticleOctreeContainer, ParticleRegions
from yt.utilities.definitions import MAXLEVEL
from yt.utilities.io_handler import io_registry
from yt.utilities.parallel_tools.parallel_analysis_interface import \
    ParallelAnalysisInterface, parallel_splitter

from yt.data_objects.data_containers import data_object_registry
from yt.data_objects.octree_subset import ParticleOctreeSubset

class ParticleIndex(Index):
    _global_mesh = False

    def __init__(self, pf, dataset_type):
        self.dataset_type = dataset_type
        self.parameter_file = weakref.proxy(pf)
        # for now, the hierarchy file is the parameter file!
        self.hierarchy_filename = self.parameter_file.parameter_filename
        self.directory = os.path.dirname(self.hierarchy_filename)
        self.float_type = np.float64
        super(ParticleIndex, self).__init__(pf, dataset_type)

    def _setup_geometry(self):
        mylog.debug("Initializing Particle Geometry Handler.")
        self._initialize_particle_handler()


    def get_smallest_dx(self):
        """
        Returns (in code units) the smallest cell size in the simulation.
        """
        dx = 1.0/(2**self.oct_handler.max_level)
        dx *= (self.parameter_file.domain_right_edge -
               self.parameter_file.domain_left_edge)
        return dx.min()

    def convert(self, unit):
        return self.parameter_file.conversion_factors[unit]

    def _initialize_particle_handler(self):
        self._setup_data_io()
        template = self.parameter_file.filename_template
        ndoms = self.parameter_file.file_count
        cls = self.parameter_file._file_class
        self.data_files = [cls(self.parameter_file, self.io, template % {'num':i}, i)
                           for i in range(ndoms)]
        self.total_particles = sum(
                sum(d.total_particles.values()) for d in self.data_files)
        pf = self.parameter_file
        self.oct_handler = ParticleOctreeContainer(
            [1, 1, 1], pf.domain_left_edge, pf.domain_right_edge,
            over_refine = pf.over_refine_factor)
        self.oct_handler.n_ref = pf.n_ref
        mylog.info("Allocating for %0.3e particles", self.total_particles)
        # No more than 256^3 in the region finder.
        N = min(len(self.data_files), 256) 
        self.regions = ParticleRegions(
                pf.domain_left_edge, pf.domain_right_edge,
                [N, N, N], len(self.data_files))
        self._initialize_indices()
        self.oct_handler.finalize()
        self.max_level = self.oct_handler.max_level
        tot = sum(self.oct_handler.recursively_count().values())
        mylog.info("Identified %0.3e octs", tot)

    def _initialize_indices(self):
        # This will be replaced with a parallel-aware iteration step.
        # Roughly outlined, what we will do is:
        #   * Generate Morton indices on each set of files that belong to
        #     an individual processor
        #   * Create a global, accumulated histogram
        #   * Cut based on estimated load balancing
        #   * Pass particles to specific processors, along with NREF buffer
        #   * Broadcast back a serialized octree to join
        #
        # For now we will do this in serial.
        morton = np.empty(self.total_particles, dtype="uint64")
        ind = 0
        for data_file in self.data_files:
            npart = sum(data_file.total_particles.values())
            morton[ind:ind + npart] = \
                self.io._initialize_index(data_file, self.regions)
            ind += npart
        morton.sort()
        # Now we add them all at once.
        self.oct_handler.add(morton)

    def _detect_output_fields(self):
        # TODO: Add additional fields
        pfl = []
        for dom in self.data_files:
            fl = self.io._identify_fields(dom)
            dom._calculate_offsets(fl)
            for f in fl:
                if f not in pfl: pfl.append(f)
        self.field_list = pfl
        pf = self.parameter_file
        pf.particle_types = tuple(set(pt for pt, pf in pfl))
        # This is an attribute that means these particle types *actually*
        # exist.  As in, they are real, in the dataset.
        pf.particle_types_raw = pf.particle_types

    def _setup_classes(self):
        dd = self._get_data_reader_dict()
        super(ParticleIndex, self)._setup_classes(dd)
        self.object_types.sort()

    def _identify_base_chunk(self, dobj):
        if getattr(dobj, "_chunk_info", None) is None:
            data_files = getattr(dobj, "data_files", None)
            if data_files is None:
                data_files = [self.data_files[i] for i in
                              self.regions.identify_data_files(dobj.selector)]
            base_region = getattr(dobj, "base_region", dobj)
            oref = self.parameter_file.over_refine_factor
            subset = [ParticleOctreeSubset(base_region, data_files, 
                        self.parameter_file, over_refine_factor = oref)]
            dobj._chunk_info = subset
        dobj._current_chunk = list(self._chunk_all(dobj))[0]

    def _chunk_all(self, dobj):
        oobjs = getattr(dobj._current_chunk, "objs", dobj._chunk_info)
        yield YTDataChunk(dobj, "all", oobjs, None)

    def _chunk_spatial(self, dobj, ngz, sort = None, preload_fields = None):
        sobjs = getattr(dobj._current_chunk, "objs", dobj._chunk_info)
        # We actually do not really use the data files except as input to the
        # ParticleOctreeSubset.
        # This is where we will perform cutting of the Octree and
        # load-balancing.  That may require a specialized selector object to
        # cut based on some space-filling curve index.
        for i,og in enumerate(sobjs):
            if ngz > 0:
                g = og.retrieve_ghost_zones(ngz, [], smoothed=True)
            else:
                g = og
            yield YTDataChunk(dobj, "spatial", [g])

    def _chunk_io(self, dobj, cache = True):
        oobjs = getattr(dobj._current_chunk, "objs", dobj._chunk_info)
        for subset in oobjs:
            yield YTDataChunk(dobj, "io", [subset], None, cache = cache)

class ParticleDataChunk(YTDataChunk):
    def __init__(self, oct_handler, regions, *args, **kwargs):
        self.oct_handler = oct_handler
        self.regions = regions
        super(ParticleDataChunk, self).__init__(*args, **kwargs)
