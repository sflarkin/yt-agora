"""
Particle Deposition onto Octs




"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free, qsort
cimport cython
from libc.math cimport sqrt

from fp_utils cimport *
from oct_container cimport Oct, OctAllocationContainer, OctreeContainer
from .particle_deposit cimport sph_kernel, gind

cdef extern from "alloca.h":
    void *alloca(int)

cdef struct NeighborList
cdef struct NeighborList:
    np.int64_t pn       # Particle number
    np.float64_t r2     # radius**2

cdef inline np.float64_t r2dist(np.float64_t ppos[3],
                                np.float64_t cpos[3],
                                np.float64_t DW[3],
                                bint periodicity[3]):
    cdef int i
    cdef np.float64_t r2, DR
    r2 = 0.0
    for i in range(3):
        DR = (ppos[i] - cpos[i])
        if periodicity[i] == 0:
            pass
        elif (DR > DW[i]/2.0):
            DR -= DW[i]
        elif (DR < -DW[i]/2.0):
            DR += DW[i]
        r2 += DR * DR
    return r2

cdef class ParticleSmoothOperation:
    # We assume each will allocate and define their own temporary storage
    cdef public object nvals
    cdef np.float64_t DW[3]
    cdef int nfields
    cdef int maxn
    cdef int curn
    cdef bint periodicity[3]
    cdef np.int64_t *doffs
    cdef np.int64_t *pinds
    cdef np.int64_t *pcounts
    cdef np.float64_t *ppos
    # Note that we are preallocating here, so this is *not* threadsafe.
    cdef NeighborList *neighbors
    cdef void neighbor_process(self, int dim[3], np.float64_t left_edge[3],
                               np.float64_t dds[3], np.float64_t *ppos,
                               np.float64_t **fields, np.int64_t nneighbors,
                               np.int64_t *nind, np.int64_t *doffs,
                               np.int64_t *pinds, np.int64_t *pcounts,
                               np.int64_t offset, np.float64_t **index_fields)
    cdef void neighbor_eval(self, np.int64_t pn, np.float64_t ppos[3],
                            np.float64_t cpos[3])
    cdef void neighbor_reset(self)
    cdef void neighbor_find(self,
                            np.int64_t nneighbors,
                            np.int64_t *nind,
                            np.int64_t *doffs,
                            np.int64_t *pcounts,
                            np.int64_t *pinds,
                            np.float64_t *ppos,
                            np.float64_t cpos[3])
    cdef void process(self, np.int64_t offset, int i, int j, int k,
                      int dim[3], np.float64_t cpos[3], np.float64_t **fields,
                      np.float64_t **index_fields)
