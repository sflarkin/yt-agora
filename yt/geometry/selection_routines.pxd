"""
Geometry selection routine imports.




"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

cimport numpy as np
from oct_visitors cimport Oct, OctVisitorData, \
    oct_visitor_function

cdef class SelectorObject:
    cdef public np.int32_t min_level
    cdef public np.int32_t max_level
    cdef int overlap_cells
    cdef np.float64_t domain_width[3]
    cdef bint periodicity[3]

    cdef void recursively_visit_octs(self, Oct *root,
                        np.float64_t pos[3], np.float64_t dds[3],
                        int level,
                        oct_visitor_function *func,
                        OctVisitorData *data,
                        int visit_covered = ?)
    cdef void visit_oct_cells(self, OctVisitorData *data, Oct *root, Oct *ch,
                              np.float64_t spos[3], np.float64_t sdds[3],
                              oct_visitor_function *func, int i, int j, int k)
    cdef int select_grid(self, np.float64_t left_edge[3],
                               np.float64_t right_edge[3],
                               np.int32_t level, Oct *o = ?) nogil
    cdef int select_cell(self, np.float64_t pos[3], np.float64_t dds[3]) nogil

    cdef int select_point(self, np.float64_t pos[3]) nogil
    cdef int select_sphere(self, np.float64_t pos[3], np.float64_t radius) nogil
    cdef int select_bbox(self, np.float64_t left_edge[3],
                               np.float64_t right_edge[3]) nogil

    # compute periodic distance (if periodicity set) assuming 0->domain_width[i] coordinates
    cdef np.float64_t difference(self, np.float64_t x1, np.float64_t x2, int d) nogil

cdef class AlwaysSelector(SelectorObject):
    pass

cdef class OctreeSubsetSelector(SelectorObject):
    cdef SelectorObject base_selector
    cdef public np.int64_t domain_id
