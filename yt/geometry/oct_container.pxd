"""
Oct definitions file

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: Columbia University
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2012 Matthew Turk.  All Rights Reserved.

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

cimport numpy as np
from fp_utils cimport *

cdef struct ParticleArrays

cdef struct Oct
cdef struct Oct:
    np.int64_t ind          # index
    np.int64_t local_ind
    np.int64_t domain       # (opt) addl int index
    np.int64_t pos[3]       # position in ints
    np.int8_t level
    ParticleArrays *sd
    Oct *children[2][2][2]
    Oct *parent

cdef struct OctAllocationContainer
cdef struct OctAllocationContainer:
    np.int64_t n
    np.int64_t n_assigned
    np.int64_t offset
    OctAllocationContainer *next
    Oct *my_octs

cdef class OctreeContainer:
    cdef OctAllocationContainer *cont
    cdef Oct ****root_mesh
    cdef int nn[3]
    cdef np.float64_t DLE[3], DRE[3]
    cdef public int nocts
    cdef public int max_domain
    cdef Oct* get(self, ppos)
    cdef void neighbors(self, Oct *, Oct **)
    cdef void oct_bounds(self, Oct *, np.float64_t *, np.float64_t *)

cdef class ARTIOOctreeContainer(OctreeContainer):
    cdef OctAllocationContainer **domains
    cdef Oct *get_root_oct(self, np.float64_t ppos[3])
    cdef Oct *next_free_oct( self, int curdom )
    cdef int valid_domain_oct(self, int curdom, Oct *parent)
    cdef Oct *add_oct(self, int curdom, Oct *parent, int curlevel, double pp[3])

cdef class RAMSESOctreeContainer(OctreeContainer):
    cdef OctAllocationContainer **domains
    cdef Oct *next_root(self, int domain_id, int ind[3])
    cdef Oct *next_child(self, int domain_id, int ind[3], Oct *parent)

cdef struct ParticleArrays:
    Oct *oct
    ParticleArrays *next
    np.float64_t **pos
    np.int64_t np
