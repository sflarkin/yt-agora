"""
Particle operations for Lagrangian Volume



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
import os, sys
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc
from yt.utilities.parallel_tools.parallel_analysis_interface import \
    parallel_objects

from yt.config import ytcfg

cdef import from "particle.h":
    struct particle:
        np.int64_t id
        float pos[6]
        float mass, energy
        float softening
        float metallicity
        np.int32_t type

ctypedef struct particleflat:
    np.int64_t id
    float pos_x
    float pos_y
    float pos_z
    float vel_x
    float vel_y
    float vel_z
    float mass, energy
    float softening
    float metallicity
    np.int32_t type    

cdef import from "halo.h":
    struct halo:
        np.int64_t id
        float pos[6], corevel[3], bulkvel[3]
        float m, r, child_r, mgrav, vmax, rvmax, rs, vrms, J[3], energy, spin
        np.int64_t num_p, num_child_particles, p_start, desc, flags, n_core
        float min_pos_err, min_vel_err, min_bulkvel_err
        
        np.int32_t type
        float sm, gas, bh, peak_density, av_density
        #float m_hires, m_lowres

cdef import from "io_generic.h":
    ctypedef void (*LPG) (char *filename, particle **p, np.int64_t *num_p)
    ctypedef void (*AHG) (halo *h, particle *hp)
    void set_load_particles_generic(LPG func, AHG afunc)

cdef import from "rockstar.h":
    void rockstar(float *bounds, np.int64_t manual_subs)

cdef import from "config.h":
    void setup_config()

cdef import from "server.h" nogil:
    int server()
    np.int64_t READER_TYPE
    np.int64_t WRITER_TYPE

cdef import from "client.h" nogil:
    void client(np.int64_t in_type)

cdef import from "meta_io.h":
    void read_particles(char *filename)
    void output_halos(np.int64_t id_offset, np.int64_t snap, 
			   np.int64_t chunk, float *bounds)

cdef import from "config_vars.h":
    # Rockstar cleverly puts all of the config variables inside a templated
    # definition of their vaiables.
    char *FILE_FORMAT
    np.float64_t PARTICLE_MASS

    char *MASS_DEFINITION
    np.int64_t MIN_HALO_OUTPUT_SIZE
    np.float64_t FORCE_RES
    np.float64_t INITIAL_METRIC_SCALING
    np.float64_t NON_DM_METRIC_SCALING
    np.int64_t SUPPRESS_GALAXIES

    np.float64_t SCALE_NOW
    np.float64_t h0
    np.float64_t Ol
    np.float64_t Om

    np.int64_t GADGET_ID_BYTES
    np.float64_t GADGET_MASS_CONVERSION
    np.float64_t GADGET_LENGTH_CONVERSION
    np.int64_t GADGET_SKIP_NON_HALO_PARTICLES
    np.int64_t RESCALE_PARTICLE_MASS

    np.int64_t PARALLEL_IO
    char *PARALLEL_IO_SERVER_ADDRESS
    char *PARALLEL_IO_SERVER_PORT
    np.int64_t PARALLEL_IO_WRITER_PORT
    char *PARALLEL_IO_SERVER_INTERFACE
    char *RUN_ON_SUCCESS

    char *INBASE
    char *FILENAME
    np.int64_t STARTING_SNAP
    np.int64_t RESTART_SNAP
    np.int64_t NUM_SNAPS
    np.int64_t NUM_BLOCKS
    np.int64_t NUM_READERS
    np.int64_t PRELOAD_PARTICLES
    char *SNAPSHOT_NAMES
    char *LIGHTCONE_ALT_SNAPS
    char *BLOCK_NAMES

    char *OUTBASE
    np.float64_t OVERLAP_LENGTH
    np.int64_t NUM_WRITERS
    np.int64_t FORK_READERS_FROM_WRITERS
    np.int64_t FORK_PROCESSORS_PER_MACHINE

    char *OUTPUT_FORMAT
    np.int64_t DELETE_BINARY_OUTPUT_AFTER_FINISHED
    np.int64_t FULL_PARTICLE_CHUNKS
    char *BGC2_SNAPNAMES

    np.int64_t BOUND_PROPS
    np.int64_t BOUND_OUT_TO_HALO_EDGE
    np.int64_t DO_MERGER_TREE_ONLY
    np.int64_t IGNORE_PARTICLE_IDS
    np.float64_t TRIM_OVERLAP
    np.float64_t ROUND_AFTER_TRIM
    np.int64_t LIGHTCONE
    np.int64_t PERIODIC

    np.float64_t LIGHTCONE_ORIGIN[3]
    np.float64_t LIGHTCONE_ALT_ORIGIN[3]

    np.float64_t LIMIT_CENTER[3]
    np.float64_t LIMIT_RADIUS

    np.int64_t SWAP_ENDIANNESS
    np.int64_t GADGET_VARIANT

    np.float64_t FOF_FRACTION
    np.float64_t FOF_LINKING_LENGTH
    np.float64_t INCLUDE_HOST_POTENTIAL_RATIO
    np.float64_t DOUBLE_COUNT_SUBHALO_MASS_RATIO
    np.int64_t TEMPORAL_HALO_FINDING
    np.int64_t MIN_HALO_PARTICLES
    np.float64_t UNBOUND_THRESHOLD
    np.int64_t ALT_NFW_METRIC

    np.int64_t TOTAL_PARTICLES
    np.float64_t BOX_SIZE
    np.int64_t OUTPUT_HMAD
    np.int64_t OUTPUT_PARTICLES
    np.int64_t OUTPUT_LEVELS
    np.float64_t DUMP_PARTICLES[3]

    np.float64_t AVG_PARTICLE_SPACING
    np.int64_t SINGLE_SNAP

# Forward declare
cdef class RockstarInterface

cdef void rh_analyze_halo(halo *h, particle *hp):
    # I don't know why, but sometimes we get halos with 0 particles.
    if h.num_p == 0: return
    cdef particleflat[:] pslice
    pslice = <particleflat[:h.num_p]> (<particleflat *>hp)

    '''
    parray = np.asarray(pslice) # This line produces the following error

    #project/projectdirs/agora/.AGORA_PIPE_INSTALL/yt-agora_install/lib/python2.7/site-packages/numpy/core/numeric.py:320: RuntimeWarning: Item size computed from the PEP 3118 buffer format string does not match the actual item size.
    #return array(a, dtype, copy=False, order=order)
    #Exception TypeError: 'expected a readable buffer object' in 'yt.analysis_modules.halo_finding.rockstar.rockstar_interface.rh_analyze_halo' ignored

    # This is where we call our functions   
    for cb in rh.callbacks:
        cb(rh.ds, parray)
    ''' 

cdef void rh_read_particles(char *filename, particle **p, np.int64_t *num_p):
    global SCALE_NOW
    cdef np.float64_t left_edge[6]
    cdef np.ndarray[np.int64_t, ndim=1] arri #index 
    cdef np.ndarray[np.float64_t, ndim=1] arr # pos/vel 
    cdef np.ndarray[np.float64_t, ndim=1] marr # mass
    #cdef np.ndarray[np.float64_t, ndim=1] earr # energy
    #cdef np.ndarray[np.float64_t, ndim=1] sarr # softening
    #cdef np.ndarray[np.float64_t, ndim=1] mtarr # metalicity
    #cdef np.ndarray[np.int32_t, ndim=1] tarr # type  
    cdef unsigned long long pi,fi,i
    cdef np.int64_t local_parts = 0
    ds = rh.ds = rh.tsl.next()
    block = int(str(filename).rsplit(".")[-1])
    n = rh.block_ratio

    SCALE_NOW = 1.0/(ds.current_redshift+1.0)
    # Now we want to grab data from only a subset of the grids for each reader.
    all_fields = set(ds.derived_field_list + ds.field_list)

    # First we need to find out how many this reader is going to read in
    # if the number of readers > 1.
    dd = ds.all_data()

    # Add particle type filter if not defined
    particle_types = np.unique([field[0] for field in ds.field_list if 'particle' in field[1]])
    if rh.particle_type not in particle_types and rh.particle_type != 'all':
        ds.add_particle_filter(rh.particle_type)

    if NUM_BLOCKS > 1:
        local_parts = 0
        for chunk in parallel_objects(
                dd.chunks([(rh.particle_type, "particle_ones")], "io")):
            local_parts += chunk[rh.particle_type, "particle_ones"].sum()
    else:
        local_parts = TOTAL_PARTICLES

    p[0] = <particle *> malloc(sizeof(particle) * local_parts)

    left_edge[0] = ds.domain_left_edge[0]
    left_edge[1] = ds.domain_left_edge[1]
    left_edge[2] = ds.domain_left_edge[2]
    left_edge[3] = left_edge[4] = left_edge[5] = 0.0
    pi = 0
    fields = [ (rh.particle_type, f) for f in
                ["particle_position_%s" % ax for ax in 'xyz'] + 
                ["particle_velocity_%s" % ax for ax in 'xyz'] +
                ["particle_index"] + ["particle_mass"]]
    for chunk in parallel_objects(dd.chunks(fields, "io")):
        arri = np.asarray(chunk[rh.particle_type, "particle_index"], dtype="int64")
        marr = chunk[rh.particle_type, "particle_mass"].in_units("Msun/h").astype("float64")
        #earr = chunk[rh.particle_type, "particle_mass"].in_units("Msun/h").astype("float64")
        #sarr = chunk[rh.particle_type, "particle_mass"].in_units("Msun/h").astype("float64")
        #mtarr= chunk[rh.particle_type, "particle_mass"].in_units("Msun/h").astype("float64") 
        #tarr = chunk[rh.particle_type, "particle_mass"].in_units("Msun/h").astype("int32")
        npart = arri.size
        for i in range(npart):
            p[0][i+pi].id = <np.int64_t> arri[i]
            p[0][i+pi].mass = <np.float64_t> marr[i]
            #p[0][i+pi].energy = <np.float64_t> earr[i]
            #p[0][i+pi].softening = <np.float64_t> sarr[i]
            #p[0][i+pi].metallicity = <np.float64_t> mtarr[i]
            p[0][i+pi].type = <np.int32_t> 0  # For now only dm particles are supported 
        fi = 0
        for field in ["particle_position_x", "particle_position_y",
                      "particle_position_z",
                      "particle_velocity_x", "particle_velocity_y",
                      "particle_velocity_z"]:
            if "position" in field:
                unit = "Mpccm/h"
            else:
                unit = "km/s"
            arr = chunk[rh.particle_type, field].in_units(unit).astype("float64")
            for i in range(npart):
                p[0][i+pi].pos[fi] = (arr[i]-left_edge[fi])
            fi += 1
        pi += npart
    num_p[0] = local_parts
    del ds

cdef class RockstarInterface:

    cdef public object data_source
    cdef public object ts
    cdef public object tsl
    cdef public object ds
    cdef int rank
    cdef int size
    cdef public int block_ratio
    cdef public object particle_type
    cdef public int total_particles
    cdef public object callbacks

    def __cinit__(self, ts):
        self.ts = ts
        self.tsl = ts.__iter__() #timseries generator used by read

    def setup_rockstar(self, char *server_address, char *server_port,
                       int num_snaps, int total_particles,
                       particle_type,
                       np.float64_t particle_mass,
                       int parallel = False, int num_readers = 1,
                       int num_writers = 1,
                       int writing_port = -1, int block_ratio = 1,
                       outbase = "None",
                       force_res = None,
                       initial_metric_scaling = 1, 
                       non_dm_metric_scaling = 10,
                       int suppress_galaxies = 1,
                       callbacks = None, int restart_num = 0,
                       int periodic = 1, int min_halo_size = 25,):
        global PARALLEL_IO, PARALLEL_IO_SERVER_ADDRESS, PARALLEL_IO_SERVER_PORT
        global FILENAME, FILE_FORMAT, NUM_SNAPS, STARTING_SNAP, h0, Ol, Om
        global BOX_SIZE, PERIODIC, PARTICLE_MASS, NUM_BLOCKS, NUM_READERS
        global FORK_READERS_FROM_WRITERS, PARALLEL_IO_WRITER_PORT, NUM_WRITERS
        global rh, SCALE_NOW, OUTBASE, MIN_HALO_OUTPUT_SIZE, 
        global OVERLAP_LENGTH, TOTAL_PARTICLES, FORCE_RES, RESTART_SNAP
        global INITIAL_METRIC_SCALING, NON_DM_METRIC_SCALING, SUPPRESS_GALAXIES

        if force_res is not None:
            FORCE_RES=np.float64(force_res)
            #print "set force res to ",FORCE_RES
        INITIAL_METRIC_SCALING=np.float64(initial_metric_scaling)
        NON_DM_METRIC_SCALING=np.float64(non_dm_metric_scaling)
        SUPPRESS_GALAXIES=np.int64(suppress_galaxies)    
        OVERLAP_LENGTH = 0.0
        if parallel:
            PARALLEL_IO = 1
            PARALLEL_IO_SERVER_ADDRESS = server_address
            PARALLEL_IO_SERVER_PORT = server_port
            if writing_port > 0:
                PARALLEL_IO_WRITER_PORT = writing_port
        else:
            PARALLEL_IO = 0
            PARALLEL_IO_SERVER_ADDRESS = server_address
            PARALLEL_IO_SERVER_PORT = server_port
        FILENAME = "inline.<block>"
        FILE_FORMAT = "GENERIC"
        OUTPUT_FORMAT = "ASCII"
        NUM_SNAPS = num_snaps
        RESTART_SNAP = restart_num
        NUM_READERS = num_readers
        NUM_WRITERS = num_writers
        NUM_BLOCKS = num_readers
        MIN_HALO_OUTPUT_SIZE=min_halo_size
        TOTAL_PARTICLES = total_particles
        self.block_ratio = block_ratio
        self.particle_type = particle_type
        
        tds = self.ts[0]
        h0 = tds.hubble_constant
        Ol = tds.omega_lambda
        Om = tds.omega_matter
        SCALE_NOW = 1.0/(tds.current_redshift+1.0)
        if callbacks is None: callbacks = []
        self.callbacks = callbacks
        if not outbase =='None'.decode('UTF-8'):
            #output directory. since we can't change the output filenames
            #workaround is to make a new directory
            OUTBASE = outbase 

        PARTICLE_MASS = particle_mass
        PERIODIC = periodic
        BOX_SIZE = tds.domain_width[0].in_units("Mpccm/h")
        setup_config()
        rh = self
        cdef LPG func = rh_read_particles
        cdef AHG afunc = rh_analyze_halo
        set_load_particles_generic(func, afunc)

    def call_rockstar(self):
        read_particles("generic")
        rockstar(NULL, 0)
        output_halos(0, 0, 0, NULL)

    def start_server(self):
        with nogil:
            server()

    def start_reader(self):
        cdef np.int64_t in_type = np.int64(READER_TYPE)
        client(in_type)

    def start_writer(self):
        cdef np.int64_t in_type = np.int64(WRITER_TYPE)
        client(in_type)

