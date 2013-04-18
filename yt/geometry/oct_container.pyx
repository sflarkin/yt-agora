"""
Oct container

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: Columbia University
Author: Christopher Moody <chris.e.moody@gmail.com>
Affiliation: UC Santa Cruz
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2011 Matthew Turk.  All Rights Reserved.

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

from libc.stdlib cimport malloc, free, qsort
from libc.math cimport floor
cimport numpy as np
import numpy as np
from oct_container cimport Oct, OctAllocationContainer, OctreeContainer
from selection_routines cimport SelectorObject
cimport cython

cdef extern from "stdlib.h":
    # NOTE that size_t might not be int
    void *alloca(int)

cdef OctAllocationContainer *allocate_octs(
        int n_octs, OctAllocationContainer *prev):
    cdef OctAllocationContainer *n_cont
    cdef Oct *oct
    cdef int n, i, j, k
    n_cont = <OctAllocationContainer *> malloc(
        sizeof(OctAllocationContainer))
    if prev == NULL:
        n_cont.offset = 0
    else:
        n_cont.offset = prev.offset + prev.n
    n_cont.my_octs = <Oct *> malloc(sizeof(Oct) * n_octs)
    if n_cont.my_octs == NULL:
        raise MemoryError
    n_cont.n = n_octs
    n_cont.n_assigned = 0
    for n in range(n_octs):
        oct = &n_cont.my_octs[n]
        oct.parent = NULL
        oct.ind = oct.domain = -1
        oct.local_ind = n + n_cont.offset
        oct.level = -1
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    oct.children[i][j][k] = NULL
    if prev != NULL:
        prev.next = n_cont
    n_cont.next = NULL
    return n_cont

cdef void free_octs(
        OctAllocationContainer *first):
    cdef OctAllocationContainer *cur
    while first != NULL:
        cur = first
        free(first.my_octs)
        first = cur.next
        free(cur)

# Here is the strategy for RAMSES containers:
#   * Read each domain individually, creating *all* octs found in that domain
#     file, even if they reside on other CPUs.
#   * Only allocate octs that reside on >= domain
#   * For all octs, insert into tree, which may require traversing existing
#     octs
#   * Note that this does not allow OctAllocationContainer to exactly be a
#     chunk, but it is close.  For IO chunking, we can theoretically examine
#     those octs that live inside a given allocator.

cdef class OctreeContainer:

    def __init__(self, oct_domain_dimensions, domain_left_edge, domain_right_edge):
        # This will just initialize the root mesh octs
        cdef int i, j, k, p
        for i in range(3):
            self.nn[i] = oct_domain_dimensions[i]
        self.max_domain = -1
        p = 0
        self.nocts = 0 # Increment when initialized
        self.root_mesh = <Oct****> malloc(sizeof(void*) * self.nn[0])
        for i in range(self.nn[0]):
            self.root_mesh[i] = <Oct ***> malloc(sizeof(void*) * self.nn[1])
            for j in range(self.nn[1]):
                self.root_mesh[i][j] = <Oct **> malloc(sizeof(void*) * self.nn[2])
                for k in range(self.nn[2]):
                    self.root_mesh[i][j][k] = NULL
        # We don't initialize the octs yet
        for i in range(3):
            self.DLE[i] = domain_left_edge[i] #0
            self.DRE[i] = domain_right_edge[i] #num_grid

    def __dealloc__(self):
        free_octs(self.cont)
        for i in range(self.nn[0]):
            for j in range(self.nn[1]):
                if self.root_mesh[i][j] == NULL: continue
                free(self.root_mesh[i][j])
            if self.root_mesh[i] == NULL: continue
            free(self.root_mesh[i])
        if self.root_mesh == NULL: return
        free(self.root_mesh)

    def __iter__(self):
        #Get the next oct, will traverse domains
        #Note that oct containers can be sorted 
        #so that consecutive octs are on the same domain
        cdef OctAllocationContainer *cur = self.cont
        cdef Oct *this
        cdef int i
        while cur != NULL:
            for i in range(cur.n_assigned):
                this = &cur.my_octs[i]
                yield (this.ind, this.local_ind, this.domain)
            cur = cur.next

    cdef void oct_bounds(self, Oct *o, np.float64_t *corner, np.float64_t *size):
        cdef int i
        for i in range(3):
            size[i] = (self.DRE[i] - self.DLE[i]) / (self.nn[i] << o.level)
            corner[i] = o.pos[i] * size[i] + self.DLE[i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef Oct *get(self, ppos):
        #Given a floating point position, retrieve the most
        #refined oct at that time
        cdef np.int64_t ind[3]
        cdef np.float64_t dds[3], cp[3], pp[3]
        cdef Oct *cur
        cdef int i
        for i in range(3):
            pp[i] = ppos[i] - self.DLE[i]
            dds[i] = (self.DRE[i] - self.DLE[i])/self.nn[i]
            ind[i] = <np.int64_t> ((pp[i] - self.DLE[i])/dds[i])
            cp[i] = (ind[i] + 0.5) * dds[i]
        cur = self.root_mesh[ind[0]][ind[1]][ind[2]]
        while cur.children[0][0][0] != NULL:
            for i in range(3):
                dds[i] = dds[i] / 2.0
                if cp[i] > pp[i]:
                    ind[i] = 0
                    cp[i] -= dds[i] / 2.0
                else:
                    ind[i] = 1
                    cp[i] += dds[i]/2.0
            cur = cur.children[ind[0]][ind[1]][ind[2]]
        return cur

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def count_cells(self, SelectorObject selector,
              np.ndarray[np.uint8_t, ndim=2, cast=True] mask):
        cdef int i, j, k
        cdef np.int64_t oi
        # pos here is CELL center, not OCT center.
        cdef np.float64_t pos[3]
        cdef int n = mask.shape[0]
        cdef np.ndarray[np.int64_t, ndim=1] count
        count = np.zeros(self.max_domain, 'int64')
        # 
        cur = self.cont
        for oi in range(n):
            if oi - cur.offset >= cur.n_assigned:
                cur = cur.next
            o = &cur.my_octs[oi - cur.offset]
            for i in range(8):
                count[o.domain - 1] += mask[o.local_ind,i]
        return count

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void neighbors(self, Oct* o, Oct* neighbors[27]):
        #Get 3x3x3 neighbors, although the 1,1,1 oct is the
        #central one. 
        #Return an array of Octs
        cdef np.int64_t curopos[3]
        cdef np.int64_t curnpos[3]
        cdef np.int64_t npos[3]
        cdef int i, j, k, ni, nj, nk, ind[3], nn, dl, skip
        cdef np.float64_t dds[3], cp[3], pp[3]
        cdef Oct* candidate
        for i in range(27): neighbors[i] = NULL
        nn = 0
        for ni in range(3):
            for nj in range(3):
                for nk in range(3):
                    if ni == nj == nk == 1:
                        neighbors[nn] = o
                        nn += 1
                        continue
                    npos[0] = o.pos[0] + (ni - 1)
                    npos[1] = o.pos[1] + (nj - 1)
                    npos[2] = o.pos[2] + (nk - 1)
                    for i in range(3):
                        # Periodicity
                        if npos[i] == -1:
                            npos[i] = (self.nn[i]  << o.level) - 1
                        elif npos[i] == (self.nn[i] << o.level):
                            npos[i] = 0
                        curopos[i] = o.pos[i]
                        curnpos[i] = npos[i] 
                    # Now we have our neighbor position and a safe place to
                    # keep it.  curnpos will be the root index of the neighbor
                    # at a given level, and npos will be constant.  curopos is
                    # the candidate root at a level.
                    candidate = o
                    while candidate != NULL:
                        if ((curopos[0] == curnpos[0]) and 
                            (curopos[1] == curnpos[1]) and
                            (curopos[2] == curnpos[2])):
                            break
                        # This one doesn't meet it, so we pop up a level.
                        # First we update our positions, then we update our
                        # candidate.
                        for i in range(3):
                            # We strip a digit off the right
                            curopos[i] = (curopos[i] >> 1)
                            curnpos[i] = (curnpos[i] >> 1)
                        # Now we update to the candidate's parent, which should
                        # have a matching position to curopos[]
                        candidate = candidate.parent
                    if candidate == NULL:
                        # Worst case scenario
                        for i in range(3):
                            ind[i] = (npos[i] >> (o.level))
                        candidate = self.root_mesh[ind[0]][ind[1]][ind[2]]
                    # Now we have the common root, which may be NULL
                    while candidate.level < o.level:
                        dl = o.level - (candidate.level + 1)
                        for i in range(3):
                            ind[i] = (npos[i] >> dl) & 1
                        if candidate.children[0][0][0] == NULL: break
                        candidate = candidate.children[ind[0]][ind[1]][ind[2]]
                    neighbors[nn] = candidate
                    nn += 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def get_neighbor_boundaries(self, ppos):
        cdef Oct *main = self.get(ppos)
        cdef Oct* neighbors[27]
        self.neighbors(main, neighbors)
        cdef np.ndarray[np.float64_t, ndim=2] bounds
        cdef np.float64_t corner[3], size[3]
        bounds = np.zeros((27,6), dtype="float64")
        cdef int i, ii
        tnp = 0
        for i in range(27):
            self.oct_bounds(neighbors[i], corner, size)
            for ii in range(3):
                bounds[i, ii] = corner[ii]
                bounds[i, 3+ii] = size[ii]
        return bounds

cdef class ARTIOOctreeContainer(OctreeContainer):

    def allocate_domains(self, domain_counts):
        cdef int count, i
        cdef OctAllocationContainer *cur = self.cont
        assert(cur == NULL)
        self.max_domain = len(domain_counts) # 1-indexed
        self.domains = <OctAllocationContainer **> malloc(
            sizeof(OctAllocationContainer *) * len(domain_counts))
        for i, count in enumerate(domain_counts):
            cur = allocate_octs(count, cur)
            if self.cont == NULL: self.cont = cur
            self.domains[i] = cur
        
    def __dealloc__(self):
        # This gets called BEFORE the superclass deallocation.  But, both get
        # called.
        if self.domains != NULL: free(self.domains)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def count(self, np.ndarray[np.uint8_t, ndim=1, cast=True] mask,
                     split = False):
        cdef int n = mask.shape[0]
        cdef int i, dom
        cdef OctAllocationContainer *cur
        cdef np.ndarray[np.int64_t, ndim=1] count
        count = np.zeros(self.max_domain, 'int64')
        # This is the idiom for iterating over many containers.
        cur = self.cont
        for i in range(n):
            if i - cur.offset >= cur.n: cur = cur.next
            if mask[i] == 1:
                count[cur.my_octs[i - cur.offset].domain - 1] += 1
        return count

    def check(self, int curdom):
        cdef int dind, pi
        cdef Oct oct
        cdef OctAllocationContainer *cont = self.domains[curdom - 1]
        cdef int nbad = 0
        for pi in range(cont.n_assigned):
            oct = cont.my_octs[pi]
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        if oct.children[i][j][k] != NULL and \
                           oct.children[i][j][k].level != oct.level + 1:
                            if curdom == 61:
                                print pi, oct.children[i][j][k].level,
                                print oct.level
                            nbad += 1
        print "DOMAIN % 3i HAS % 9i BAD OCTS (%s / %s / %s)" % (curdom, nbad, 
            cont.n - cont.n_assigned, cont.n_assigned, cont.n)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef Oct *next_free_oct( self, int curdom ) :
        cdef OctAllocationContainer *cont
        cdef Oct *next_oct

        if curdom < 1 or curdom > self.max_domain or self.domains == NULL  :
            print "Error, invalid domain or unallocated domains"
            raise RuntimeError
        
        cont = self.domains[curdom - 1]
        if cont.n_assigned >= cont.n :
            print "Error, ran out of octs in domain curdom"
            raise RuntimeError

        self.nocts += 1
        next_oct = &cont.my_octs[cont.n_assigned]
        cont.n_assigned += 1
        return next_oct

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int valid_domain_oct(self, int curdom, Oct *parent) :
        cdef OctAllocationContainer *cont

        if curdom < 1 or curdom > self.max_domain or self.domains == NULL  :
            raise RuntimeError
        cont = self.domains[curdom - 1]

        if parent == NULL or parent < &cont.my_octs[0] or \
                parent > &cont.my_octs[cont.n_assigned] :
            return 0
        else :
            return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef Oct *get_root_oct(self, np.float64_t ppos[3]):
        cdef np.int64_t ind[3]
        cdef np.float64_t dds
        cdef int i
        for i in range(3):
            dds = (self.DRE[i] - self.DLE[i])/self.nn[i]
            ind[i] = <np.int64_t> floor((ppos[i]-self.DLE[i])/dds)
        return self.root_mesh[ind[0]][ind[1]][ind[2]]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef Oct *add_oct(self, int curdom, Oct *parent, 
                    int curlevel, np.float64_t pp[3]):

        cdef int level, i, ind[3]
        cdef Oct *cur, *next_oct
        cdef np.int64_t pos[3]
        cdef np.float64_t dds

        if curlevel < 0 :
            raise RuntimeError
        for i in range(3):
            if pp[i] < self.DLE[i] or pp[i] > self.DRE[i] :
                raise RuntimeError
            dds = (self.DRE[i] - self.DLE[i])/(<np.int64_t>self.nn[i])
            pos[i] = <np.int64_t> floor((pp[i]-self.DLE[i])*<np.float64_t>(1<<curlevel)/dds)

        if curlevel == 0 :
            cur = NULL
        elif parent == NULL :
            cur = self.get_root_oct(pp)
            assert( cur != NULL )

            # Now we find the location we want
            for level in range(1,curlevel):
                # At every level, find the cell this oct lives inside
                for i in range(3) :
                    if pos[i] < (2*cur.pos[i]+1)<<(curlevel-level) :
                        ind[i] = 0
                    else :
                        ind[i] = 1
                cur = cur.children[ind[0]][ind[1]][ind[2]]
                if cur == NULL:
                    # in ART we don't allocate down to curlevel 
                    # if parent doesn't exist
                    print "Error, no oct exists at that level"
                    raise RuntimeError
        else :
            if not self.valid_domain_oct(curdom,parent) or \
                    parent.level != curlevel - 1:
                raise RuntimeError
            cur = parent
 
        next_oct = self.next_free_oct( curdom )
        if cur == NULL :
            self.root_mesh[pos[0]][pos[1]][pos[2]] = next_oct
        else :
            for i in range(3) :
                if pos[i] < 2*cur.pos[i]+1 :
                    ind[i] = 0
                else :
                    ind[i] = 1
            if cur.level != curlevel - 1 or  \
                    cur.children[ind[0]][ind[1]][ind[2]] != NULL :
                print "Error in add_oct: child already filled!"
                raise RuntimeError

            cur.children[ind[0]][ind[1]][ind[2]] = next_oct
        for i in range(3) :
            next_oct.pos[i] = pos[i]
        next_oct.domain = curdom
        next_oct.parent = cur
        next_oct.ind = 1
        next_oct.level = curlevel
        return next_oct

    # ii:mask/art ; ci=ramses loop backward (k<-fast, j ,i<-slow) 
    # ii=0 000 art 000 ci 000 
    # ii=1 100 art 100 ci 001 
    # ii=2 010 art 010 ci 010 
    # ii=3 110 art 110 ci 011
    # ii=4 001 art 001 ci 100
    # ii=5 101 art 011 ci 101
    # ii=6 011 art 011 ci 110
    # ii=7 111 art 111 ci 111
    # keep coords ints so multiply by pow(2,1) when increasing level.
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def icoords(self, int domain_id,
                np.ndarray[np.uint8_t, ndim=2, cast=True] mask,
                np.int64_t cell_count,
                np.ndarray[np.int64_t, ndim=1] level_counts):
        # Wham, bam, it's a scam
        cdef np.int64_t i, j, k, oi, ci, n, ii, level
        cdef OctAllocationContainer *cur = self.domains[domain_id - 1]
        cdef Oct *o
        n = mask.shape[0]
        cdef np.ndarray[np.int64_t, ndim=2] coords
        coords = np.empty((cell_count, 3), dtype="int64")
        ci=0
        for oi in range(cur.n):
            o = &cur.my_octs[oi]
            for k in range(2):
                for j in range(2):
                    for i in range(2):
                        ii = ((k*2)+j)*2+i
                        if mask[o.local_ind, ii] == 0: continue
                        coords[ci, 0] = (o.pos[0] << 1) + i
                        coords[ci, 1] = (o.pos[1] << 1) + j
                        coords[ci, 2] = (o.pos[2] << 1) + k
                        ci += 1
        return coords

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def ires(self, int domain_id,
                np.ndarray[np.uint8_t, ndim=2, cast=True] mask,
                np.int64_t cell_count,
                np.ndarray[np.int64_t, ndim=1] level_counts):
        # Wham, bam, it's a scam
        cdef np.int64_t i, j, k, oi, ci, n
        cdef OctAllocationContainer *cur = self.domains[domain_id - 1]
        cdef Oct *o
        n = mask.shape[0]
        cdef np.ndarray[np.int64_t, ndim=1] levels
        levels = np.empty(cell_count, dtype="int64")
        ci = 0
        for oi in range(cur.n):
            o = &cur.my_octs[oi]
            for i in range(8):
                if mask[oi + cur.offset, i] == 0: continue
                levels[ci] = o.level
                ci +=1
        return levels

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def count_levels(self, int max_level, int domain_id,
                     np.ndarray[np.uint8_t, ndim=2, cast=True] mask):
        cdef np.ndarray[np.int64_t, ndim=1] level_count
        cdef OctAllocationContainer *cur = self.domains[domain_id - 1]
        cdef Oct *o
        cdef int oi, i
        level_count = np.zeros(max_level+1, 'int64')
        for oi in range(cur.n):
            o = &cur.my_octs[oi]
            for i in range(8):
                if mask[o.local_ind, i] == 0: continue
                level_count[o.level] += 1
        return level_count

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def fcoords(self, int domain_id,
                np.ndarray[np.uint8_t, ndim=2, cast=True] mask,
                np.int64_t cell_count,
                np.ndarray[np.int64_t, ndim=1] level_counts):
        # Wham, bam, it's a scam
        cdef np.int64_t i, j, k, oi, ci, n, ii
        cdef OctAllocationContainer *cur = self.domains[domain_id - 1]
        cdef Oct *o
        cdef np.float64_t pos[3]
        cdef np.float64_t base_dx[3], dx[3]
        n = mask.shape[0]
        cdef np.ndarray[np.float64_t, ndim=2] coords
        coords = np.empty((cell_count, 3), dtype="float64")
        ci =0 
        for i in range(3):
            # This is the base_dx, but not the base distance from the center
            # position.  Note that the positions will also all be offset by
            # dx/2.0.  This is also for *oct grids*, not cells.
            base_dx[i] = (self.DRE[i] - self.DLE[i])/self.nn[i]
        for oi in range(cur.n):
            o = &cur.my_octs[oi]
            for i in range(3):
                # This gives the *grid* width for this level
                dx[i] = base_dx[i] / (1 << o.level)
                # o.pos is the *grid* index, so pos[i] is the center of the
                # first cell in the grid
                pos[i] = self.DLE[i] + o.pos[i]*dx[i] + dx[i]/4.0
                dx[i] = dx[i] / 2.0 # This is now the *offset* 
            for k in range(2):
                for j in range(2):
                    for i in range(2):
                        ii = ((k*2)+j)*2+i
                        if mask[o.local_ind, ii] == 0: continue
                        coords[ci, 0] = pos[0] + dx[0] * i
                        coords[ci, 1] = pos[1] + dx[1] * j
                        coords[ci, 2] = pos[2] + dx[2] * k
                        ci +=1 
        return coords

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def fill_mask(self, int domain, dest_fields, source_fields,
                   np.ndarray[np.uint8_t, ndim=2, cast=True] mask, int offset):
        cdef np.ndarray[np.float32_t, ndim=1] source
        cdef np.ndarray[np.float64_t, ndim=1] dest
        cdef OctAllocationContainer *dom = self.domains[domain - 1]
        cdef Oct *o
        cdef int n
        cdef int i, j, k, ii
        cdef int local_pos, local_filled
        cdef np.float64_t val
        for key in dest_fields:
            local_filled = 0
            dest = dest_fields[key]
            source = source_fields[key]
            # snl: an alternative to filling level 0 yt-octs is to produce a 
            # mapping between the mask and the source read order
            for n in range(dom.n):
                o = &dom.my_octs[n]
                for k in range(2):
                    for j in range(2):
                        for i in range(2):
                            ii = ((k*2)+j)*2+i
                            if mask[o.local_ind, ii] == 0: continue
                            dest[local_filled + offset] = source[o.local_ind*8+ii]
                            # print 'oct_container.pyx:sourcemasked',o.level,local_filled, o.local_ind*8+ii, source[o.local_ind*8+ii]
                            local_filled += 1
        return local_filled

cdef class RAMSESOctreeContainer(OctreeContainer):

    cdef Oct* next_root(self, int domain_id, int ind[3]):
        cdef Oct *next = self.root_mesh[ind[0]][ind[1]][ind[2]]
        if next != NULL: return next
        cdef OctAllocationContainer *cont = self.domains[domain_id - 1]
        if cont.n_assigned >= cont.n: raise RuntimeError
        next = &cont.my_octs[cont.n_assigned]
        cont.n_assigned += 1
        self.root_mesh[ind[0]][ind[1]][ind[2]] = next
        next.parent = NULL
        next.level = 0
        for i in range(3):
            next.pos[i] = ind[i]
        self.nocts += 1
        return next

    cdef Oct* next_child(self, int domain_id, int ind[3], Oct *parent):
        cdef Oct *next = parent.children[ind[0]][ind[1]][ind[2]]
        if next != NULL: return next
        cdef OctAllocationContainer *cont = self.domains[domain_id - 1]
        if cont.n_assigned >= cont.n: raise RuntimeError
        next = &cont.my_octs[cont.n_assigned]
        cont.n_assigned += 1
        parent.children[ind[0]][ind[1]][ind[2]] = next
        next.parent = parent
        next.level = parent.level + 1
        for i in range(3):
            next.pos[i] = ind[i] + (parent.pos[i] << 1)
        self.nocts += 1
        return next

    def allocate_domains(self, domain_counts):
        cdef int count, i
        cdef OctAllocationContainer *cur = self.cont
        assert(cur == NULL)
        self.max_domain = len(domain_counts) # 1-indexed
        self.domains = <OctAllocationContainer **> malloc(
            sizeof(OctAllocationContainer *) * len(domain_counts))
        for i, count in enumerate(domain_counts):
            cur = allocate_octs(count, cur)
            if self.cont == NULL: self.cont = cur
            self.domains[i] = cur
        
    def __dealloc__(self):
        # This gets called BEFORE the superclass deallocation.  But, both get
        # called.
        if self.domains != NULL: free(self.domains)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def count(self, np.ndarray[np.uint8_t, ndim=1, cast=True] mask,
                     split = False):
        cdef int n = mask.shape[0]
        cdef int i, dom
        cdef OctAllocationContainer *cur
        cdef np.ndarray[np.int64_t, ndim=1] count
        count = np.zeros(self.max_domain, 'int64')
        # This is the idiom for iterating over many containers.
        cur = self.cont
        for i in range(n):
            if i - cur.offset >= cur.n_assigned: cur = cur.next
            if mask[i] == 1:
                count[cur.my_octs[i - cur.offset].domain - 1] += 1
        return count

    def domain_and(self, np.ndarray[np.uint8_t, ndim=2, cast=True] mask,
                   int domain_id):
        cdef np.int64_t i, oi, n, 
        cdef OctAllocationContainer *cur = self.domains[domain_id - 1]
        cdef Oct *o
        cdef np.ndarray[np.uint8_t, ndim=2] m2 = \
                np.zeros((mask.shape[0], 8), 'uint8')
        n = mask.shape[0]
        for oi in range(cur.n_assigned):
            o = &cur.my_octs[oi]
            for i in range(8):
                m2[o.local_ind, i] = mask[o.local_ind, i]
        return m2 # NOTE: This is uint8_t

    def domain_mask(self,
                    # mask is the base selector's *global* mask
                    np.ndarray[np.uint8_t, ndim=2, cast=True] mask,
                    int domain_id):
        # What distinguishes this one from domain_and is that we have a mask,
        # which covers the whole domain, but our output will only be of a much
        # smaller subset of octs that belong to a given domain *and* the mask.
        # Note also that typically when something calls domain_and, they will 
        # use a logical_any along the oct axis.  Here we don't do that.
        # Note also that we change the shape of the returned array.
        cdef np.int64_t i, j, k, oi, n, nm
        cdef OctAllocationContainer *cur = self.domains[domain_id - 1]
        cdef Oct *o
        n = mask.shape[0]
        nm = 0
        for oi in range(cur.n_assigned):
            o = &cur.my_octs[oi]
            use = 0
            for i in range(8):
                if mask[o.local_ind, i] == 1: use = 1
            nm += use
        cdef np.ndarray[np.uint8_t, ndim=4] m2 = \
                np.zeros((2, 2, 2, nm), 'uint8')
        nm = 0
        for oi in range(cur.n_assigned):
            o = &cur.my_octs[oi]
            use = 0
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        ii = ((k*2)+j)*2+i
                        if mask[o.local_ind, ii] == 0: continue
                        use = m2[i, j, k, nm] = 1
            nm += use
        return m2.astype("bool")

    def check(self, int curdom):
        cdef int dind, pi
        cdef Oct oct
        cdef OctAllocationContainer *cont = self.domains[curdom - 1]
        cdef int nbad = 0
        cdef int nmissed = 0
        cdef int unassigned = 0
        for pi in range(cont.n_assigned):
            oct = cont.my_octs[pi]
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        if oct.children[i][j][k] != NULL and \
                           oct.children[i][j][k].level != oct.level + 1:
                            nbad += 1
                        if oct.domain != curdom:
                            print curdom, oct.domain
                            nmissed += 1
                        if oct.domain == -1:
                            unassigned += 1
        print "DOMAIN % 3i HAS % 9i BAD OCTS (%s / %s / %s)" % (curdom, nbad, 
            cont.n - cont.n_assigned, cont.n_assigned, cont.n)
        print "DOMAIN % 3i HAS % 9i MISSED OCTS" % (curdom, nmissed)
        print "DOMAIN % 3i HAS % 9i UNASSIGNED OCTS" % (curdom, unassigned)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def add(self, int curdom, int curlevel, int ng,
            np.ndarray[np.float64_t, ndim=2] pos,
            int local_domain, int skip_boundary = 1):
        cdef int level, no, p, i, j, k, ind[3]
        cdef int local = (local_domain == curdom)
        cdef Oct *cur, *next = NULL
        cdef np.float64_t pp[3], cp[3], dds[3]
        no = pos.shape[0] #number of octs
        if curdom > self.max_domain: return 0
        cdef OctAllocationContainer *cont = self.domains[curdom - 1]
        cdef int initial = cont.n_assigned
        cdef int in_boundary = 0
        # How do we bootstrap ourselves?
        for p in range(no):
            #for every oct we're trying to add find the 
            #floating point unitary position on this level
            in_boundary = 0
            for i in range(3):
                pp[i] = pos[p, i]
                dds[i] = (self.DRE[i] - self.DLE[i])/self.nn[i]
                ind[i] = <np.int64_t> ((pp[i] - self.DLE[i])/dds[i])
                cp[i] = (ind[i] + 0.5) * dds[i] + self.DLE[i]
                if ind[i] < 0 or ind[i] >= self.nn[i]:
                    in_boundary = 1
            if skip_boundary == in_boundary == 1: continue
            cur = self.next_root(curdom, ind)
            # Now we find the location we want
            # Note that RAMSES I think 1-findiceses levels, but we don't.
            for level in range(curlevel):
                # At every level, find the cell this oct
                # lives inside
                for i in range(3):
                    #as we get deeper, oct size halves
                    dds[i] = dds[i] / 2.0
                    if cp[i] > pp[i]: 
                        ind[i] = 0
                        cp[i] -= dds[i]/2.0
                    else:
                        ind[i] = 1
                        cp[i] += dds[i]/2.0
                # Check if it has not been allocated
                cur = self.next_child(curdom, ind, cur)
            # Now we should be at the right level
            cur.domain = curdom
            if local == 1:
                cur.ind = p
            cur.level = curlevel
        return cont.n_assigned - initial

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def icoords(self, int domain_id,
                np.ndarray[np.uint8_t, ndim=2, cast=True] mask,
                np.int64_t cell_count,
                np.ndarray[np.int64_t, ndim=1] level_counts):
        # Wham, bam, it's a scam
        cdef np.int64_t i, j, k, oi, ci, n, ii, level
        cdef OctAllocationContainer *cur = self.domains[domain_id - 1]
        cdef Oct *o
        n = mask.shape[0]
        cdef np.ndarray[np.int64_t, ndim=2] coords
        coords = np.empty((cell_count, 3), dtype="int64")
        for oi in range(cur.n_assigned):
            o = &cur.my_octs[oi]
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        ii = ((k*2)+j)*2+i
                        if mask[o.local_ind, ii] == 0: continue
                        ci = level_counts[o.level]
                        coords[ci, 0] = (o.pos[0] << 1) + i
                        coords[ci, 1] = (o.pos[1] << 1) + j
                        coords[ci, 2] = (o.pos[2] << 1) + k
                        level_counts[o.level] += 1
        return coords

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def ires(self, int domain_id,
                np.ndarray[np.uint8_t, ndim=2, cast=True] mask,
                np.int64_t cell_count,
                np.ndarray[np.int64_t, ndim=1] level_counts):
        # Wham, bam, it's a scam
        cdef np.int64_t i, j, k, oi, ci, n
        cdef OctAllocationContainer *cur = self.domains[domain_id - 1]
        cdef Oct *o
        n = mask.shape[0]
        cdef np.ndarray[np.int64_t, ndim=1] levels
        levels = np.empty(cell_count, dtype="int64")
        ci = 0
        for oi in range(cur.n):
            o = &cur.my_octs[oi]
            for i in range(8):
                if mask[oi + cur.offset, i] == 0: continue
                ci = level_counts[o.level]
                levels[ci] = o.level
                level_counts[o.level] += 1
        return levels

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def count_levels(self, int max_level, int domain_id,
                     np.ndarray[np.uint8_t, ndim=2, cast=True] mask):
        cdef np.ndarray[np.int64_t, ndim=1] level_count
        cdef OctAllocationContainer *cur = self.domains[domain_id - 1]
        cdef Oct *o
        cdef int oi, i
        level_count = np.zeros(max_level+1, 'int64')
        for oi in range(cur.n_assigned):
            o = &cur.my_octs[oi]
            for i in range(8):
                if mask[o.local_ind, i] == 0: continue
                level_count[o.level] += 1
        return level_count

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def fcoords(self, int domain_id,
                np.ndarray[np.uint8_t, ndim=2, cast=True] mask,
                np.int64_t cell_count,
                np.ndarray[np.int64_t, ndim=1] level_counts):
        # Wham, bam, it's a scam
        cdef np.int64_t i, j, k, oi, ci, n, ii
        cdef OctAllocationContainer *cur = self.domains[domain_id - 1]
        cdef Oct *o
        cdef np.float64_t pos[3]
        cdef np.float64_t base_dx[3], dx[3]
        n = mask.shape[0]
        cdef np.ndarray[np.float64_t, ndim=2] coords
        coords = np.empty((cell_count, 3), dtype="float64")
        for i in range(3):
            # This is the base_dx, but not the base distance from the center
            # position.  Note that the positions will also all be offset by
            # dx/2.0.  This is also for *oct grids*, not cells.
            base_dx[i] = (self.DRE[i] - self.DLE[i])/self.nn[i]
        for oi in range(cur.n):
            o = &cur.my_octs[oi]
            for i in range(3):
                # This gives the *grid* width for this level
                dx[i] = base_dx[i] / (1 << o.level)
                # o.pos is the *grid* index, so pos[i] is the center of the
                # first cell in the grid
                pos[i] = self.DLE[i] + o.pos[i]*dx[i] + dx[i]/4.0
                dx[i] = dx[i] / 2.0 # This is now the *offset* 
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        ii = ((k*2)+j)*2+i
                        if mask[o.local_ind, ii] == 0: continue
                        ci = level_counts[o.level]
                        coords[ci, 0] = pos[0] + dx[0] * i
                        coords[ci, 1] = pos[1] + dx[1] * j
                        coords[ci, 2] = pos[2] + dx[2] * k
                        level_counts[o.level] += 1
        return coords

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def fill_level(self, int domain, int level, dest_fields, source_fields,
                   np.ndarray[np.uint8_t, ndim=2, cast=True] mask, int offset):
        cdef np.ndarray[np.float64_t, ndim=2] source
        cdef np.ndarray[np.float64_t, ndim=1] dest
        cdef OctAllocationContainer *dom = self.domains[domain - 1]
        cdef Oct *o
        cdef int n
        cdef int i, j, k, ii
        cdef int local_pos, local_filled
        cdef np.float64_t val
        for key in dest_fields:
            local_filled = 0
            dest = dest_fields[key]
            source = source_fields[key]
            for n in range(dom.n):
                o = &dom.my_octs[n]
                if o.level != level: continue
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            ii = ((k*2)+j)*2+i
                            if mask[o.local_ind, ii] == 0: continue
                            dest[local_filled + offset] = source[o.ind, ii]
                            local_filled += 1
        return local_filled



cdef class ARTOctreeContainer(RAMSESOctreeContainer):
    #this class is specifically for the NMSU ART
    @cython.boundscheck(True)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def fill_level(self, int domain, int level, dest_fields, source_fields,
                   np.ndarray[np.uint8_t, ndim=2, cast=True] mask, int offset,
                   np.int64_t subchunk_offset, np.int64_t subchunk_max):
        #Only minorly different from the RAMSES version
        #The source array is in chunks, just stop when we hit the end
        cdef np.ndarray[np.float64_t, ndim=2] source
        cdef np.ndarray[np.float64_t, ndim=1] dest
        cdef OctAllocationContainer *dom = self.domains[domain - 1]
        cdef Oct *o
        cdef int n
        cdef int i, j, k, ii
        cdef int local_pos, local_filled
        cdef np.float64_t val
        cdef np.int64_t index
        for key in dest_fields:
            local_filled = 0
            dest = dest_fields[key]
            source = source_fields[key]
            for n in range(dom.n):
                o = &dom.my_octs[n]
                index = o.ind-subchunk_offset
                if o.level != level: continue
                if index < 0: continue
                if index >= subchunk_max: 
                    #if we hit the end of the array,
                    #immeditely discontinue
                    return local_filled
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            ii = ((k*2)+j)*2+i
                            if mask[o.local_ind, ii] == 0: continue
                            dest[local_filled + offset] = \
                                source[index,ii]
                            local_filled += 1
        return local_filled


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def fill_level_from_grid(self, int domain, int level, dest_fields, 
                             source_fields, 
                             np.ndarray[np.uint8_t, ndim=2, cast=True] mask,
                             int offset):
        #Fill  level, but instead of assuming that the source
        #order is that of the oct order, we look up the oct position
        #and fill its children from the the source field
        #As a result, source is 3D grid with 8 times as many
        #elements as the number of octs on this level in this domain
        #and with the shape of an equal-sided cube
        cdef np.ndarray[np.float64_t, ndim=3] source
        cdef np.ndarray[np.float64_t, ndim=1] dest
        cdef OctAllocationContainer *dom = self.domains[domain - 1]
        cdef Oct *o
        cdef int n
        cdef int i, j, k, ii
        cdef int local_pos, local_filled
        cdef np.float64_t val
        cdef np.int64_t ox,oy,oz
        for key in dest_fields:
            local_filled = 0
            dest = dest_fields[key]
            source = source_fields[key]
            for n in range(dom.n):
                o = &dom.my_octs[n]
                if o.level != level: continue
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            ii = ((k*2)+j)*2+i
                            if mask[o.local_ind, ii] == 0: continue
                            ox = (o.pos[0] << 1) + i
                            oy = (o.pos[1] << 1) + j
                            oz = (o.pos[2] << 1) + k
                            dest[local_filled + offset] = source[ox,oy,oz]
                            local_filled += 1
        return local_filled


cdef int compare_octs(void *vo1, void *vo2) nogil:
    #This only compares if the octs live on the
    #domain, not if they are actually equal
    #Used to sort octs into consecutive domains
    cdef Oct *o1 = (<Oct**> vo1)[0]
    cdef Oct *o2 = (<Oct**> vo2)[0]
    if o1.domain < o2.domain: return -1
    elif o1.domain == o2.domain:
        if o1.level < o2.level: return -1
        if o1.level > o2.level: return 1
        else: return 0
    elif o1.domain > o2.domain: return 1

cdef class ParticleOctreeContainer(OctreeContainer):
    #Each ParticleArrays contains an Oct
    #a reference to the next ParticleArrays
    #its index and the number of particles 
    cdef ParticleArrays *first_sd
    cdef ParticleArrays *last_sd
    cdef Oct** oct_list
    #The starting oct index of each domain
    cdef np.int64_t *dom_offsets 
    cdef public int max_level
    #How many particles do we keep befor refining
    cdef public int n_ref

    def allocate_root(self):
        cdef int i, j, k
        cdef Oct *cur
        for i in range(self.nn[0]):
            for j in range(self.nn[1]):
                for k in range(self.nn[2]):
                    cur = self.allocate_oct()
                    cur.level = 0
                    cur.pos[0] = i
                    cur.pos[1] = j
                    cur.pos[2] = k
                    cur.parent = NULL
                    self.root_mesh[i][j][k] = cur

    def __dealloc__(self):
        #Call the freemem ops on every ocy
        #of the root mesh recursively
        cdef i, j, k
        for i in range(self.nn[0]):
            for j in range(self.nn[1]):
                for k in range(self.nn[2]):
                    self.visit_free(self.root_mesh[i][j][k])
        free(self.oct_list)
        free(self.dom_offsets)

    cdef void visit_free(self, Oct *o):
        #Free the memory for this oct recursively
        cdef int i, j, k
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    if o.children[i][j][k] == NULL: continue
                    self.visit_free(o.children[i][j][k])
        if o.sd.np >= 0:
            if o.sd.pos != NULL:
                for i in range(3):
                    free(o.sd.pos[i])
                free(o.sd.pos)
        free(o)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def icoords(self, int domain_id,
                np.ndarray[np.uint8_t, ndim=2, cast=True] mask,
                np.int64_t cell_count,
                np.ndarray[np.int64_t, ndim=1] level_counts):
        #Return the integer positions of the cells
        #Limited to this domain and within the mask
        #Positions are binary; aside from the root mesh
        #to each digit we just add a << 1 and a 0 or 1 
        #for each child recursively
        cdef np.ndarray[np.int64_t, ndim=2] coords
        coords = np.empty((cell_count, 3), dtype="int64")
        cdef int oi, i, ci, ii
        ci = 0
        for oi in range(self.nocts):
            o = self.oct_list[oi]
            if o.domain != domain_id: continue
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        ii = ((k*2)+j)*2+i
                        if mask[oi, ii] == 1:
                            coords[ci, 0] = (o.pos[0] << 1) + i
                            coords[ci, 1] = (o.pos[1] << 1) + j
                            coords[ci, 2] = (o.pos[2] << 1) + k
                            ci += 1
        return coords

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def ires(self, int domain_id,
                np.ndarray[np.uint8_t, ndim=2, cast=True] mask,
                np.int64_t cell_count,
                np.ndarray[np.int64_t, ndim=1] level_counts):
        #Return the 'resolution' of each cell; ie the level
        cdef np.ndarray[np.int64_t, ndim=1] res
        res = np.empty(cell_count, dtype="int64")
        cdef int oi, i, ci
        ci = 0
        for oi in range(self.nocts):
            o = self.oct_list[oi]
            if o.domain != domain_id: continue
            for i in range(8):
                if mask[oi, i] == 1:
                    res[ci] = o.level
                    ci += 1
        return res

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def fcoords(self, int domain_id,
                np.ndarray[np.uint8_t, ndim=2, cast=True] mask,
                np.int64_t cell_count,
                np.ndarray[np.int64_t, ndim=1] level_counts):
        #Return the floating point unitary position of every cell
        cdef np.ndarray[np.float64_t, ndim=2] coords
        coords = np.empty((cell_count, 3), dtype="float64")
        cdef int oi, i, ci
        cdef np.float64_t base_dx[3], dx[3], pos[3]
        for i in range(3):
            # This is the base_dx, but not the base distance from the center
            # position.  Note that the positions will also all be offset by
            # dx/2.0.  This is also for *oct grids*, not cells.
            base_dx[i] = (self.DRE[i] - self.DLE[i])/self.nn[i]
        ci = 0
        cdef int proc
        for oi in range(self.nocts):
            proc = 0
            for i in range(8):
                if mask[oi, i] == 1:
                    proc = 1
                    break
            if proc == 0: continue
            o = self.oct_list[oi]
            if o.domain != domain_id: continue
            for i in range(3):
                # This gives the *grid* width for this level
                dx[i] = base_dx[i] / (1 << o.level)
                # o.pos is the *grid* index, so pos[i] is the center of the
                # first cell in the grid
                pos[i] = self.DLE[i] + o.pos[i]*dx[i] + dx[i]/4.0
                dx[i] = dx[i] / 2.0 # This is now the *offset* 
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        ii = ((k*2)+j)*2+i
                        if mask[oi, ii] == 0: continue
                        coords[ci, 0] = pos[0] + dx[0] * i
                        coords[ci, 1] = pos[1] + dx[1] * j
                        coords[ci, 2] = pos[2] + dx[2] * k
                        ci += 1
        return coords

    def allocate_domains(self, domain_counts):
        pass

    def finalize(self):
        #This will sort the octs in the oct list
        #so that domains appear consecutively
        #And then find the oct index/offset for
        #every domain
        cdef int max_level = 0
        self.oct_list = <Oct**> malloc(sizeof(Oct*)*self.nocts)
        cdef np.int64_t i = 0
        cdef ParticleArrays *c = self.first_sd
        while c != NULL:
            self.oct_list[i] = c.oct
            max_level = imax(max_level, c.oct.level)
            if c.np >= 0:
                for j in range(3):
                    free(c.pos[j])
                free(c.pos)
                c.pos = NULL
            c = c.next
            i += 1
        self.max_level = max_level
        qsort(self.oct_list, self.nocts, sizeof(Oct*), &compare_octs)
        cdef int cur_dom = -1
        # We always need at least 2, and if max_domain is 0, we need 3.
        self.dom_offsets = <np.int64_t *>malloc(sizeof(np.int64_t) *
                                                (self.max_domain + 3))
        self.dom_offsets[0] = 0
        for i in range(self.nocts):
            self.oct_list[i].local_ind = i
            if self.oct_list[i].domain > cur_dom:
                cur_dom = self.oct_list[i].domain
                self.dom_offsets[cur_dom + 1] = i
        self.dom_offsets[cur_dom + 2] = self.nocts

    cdef Oct* allocate_oct(self):
        #Allocate the memory, set to NULL or -1
        #We reserve space for n_ref particles, but keep
        #track of how many are used with np initially 0
        self.nocts += 1
        cdef Oct *my_oct = <Oct*> malloc(sizeof(Oct))
        cdef ParticleArrays *sd = <ParticleArrays*> \
            malloc(sizeof(ParticleArrays))
        cdef int i, j, k
        my_oct.ind = my_oct.domain = -1
        my_oct.local_ind = self.nocts - 1
        my_oct.pos[0] = my_oct.pos[1] = my_oct.pos[2] = -1
        my_oct.level = -1
        my_oct.sd = sd
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    my_oct.children[i][j][k] = NULL
        my_oct.parent = NULL
        if self.first_sd == NULL:
            self.first_sd = sd
        if self.last_sd != NULL:
            self.last_sd.next = sd
        self.last_sd = sd
        sd.oct = my_oct
        sd.next = NULL
        sd.pos = <np.float64_t **> malloc(sizeof(np.float64_t*) * 3)
        for i in range(3):
            sd.pos[i] = <np.float64_t *> malloc(sizeof(np.float64_t) * self.n_ref)
        for i in range(self.n_ref):
            sd.pos[0][i] = sd.pos[1][i] = sd.pos[2][i] = 0.0
        sd.np = 0
        return my_oct

    def linearly_count(self):
        #Without visiting oct and cells
        #jump from particle arrays to the next one
        #counting the total # of particles en route
        cdef np.int64_t total = 0
        cdef ParticleArrays *c = self.first_sd
        while c != NULL:
            total += 1
            c = c.next
        return total

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def count_levels(self, int max_level, int domain_id,
                     np.ndarray[np.uint8_t, ndim=2, cast=True] mask):
        cdef np.ndarray[np.int64_t, ndim=1] level_count
        cdef Oct *o
        cdef int oi, i
        level_count = np.zeros(max_level+1, 'int64')
        cdef np.int64_t ndo, doff
        ndo = self.dom_offsets[domain_id + 2] \
            - self.dom_offsets[domain_id + 1]
        doff = self.dom_offsets[domain_id + 1]
        for oi in range(ndo):
            o = self.oct_list[oi + doff]
            for i in range(8):
                if mask[o.local_ind, i] == 0: continue
                level_count[o.level] += 1
        return level_count

    def add(self, np.ndarray[np.float64_t, ndim=2] pos, np.int64_t domain_id):
        #Add this particle to the root oct
        #Then if that oct has children, add it to them recursively
        #If the child needs to be refined because of max particles, do so
        cdef int no = pos.shape[0]
        cdef int p, i, level
        cdef np.float64_t dds[3], cp[3], pp[3]
        cdef int ind[3]
        self.max_domain = max(self.max_domain, domain_id)
        cdef int mid, mad
        if self.root_mesh[0][0][0] == NULL: self.allocate_root()
        for p in range(no):
            level = 0
            for i in range(3):
                #PP Calculate the unitary position, 
                #DDS Domain dimensions
                #IND Corresponding integer index on the root octs
                #CP Center  point of that oct
                pp[i] = pos[p, i]
                dds[i] = (self.DRE[i] + self.DLE[i])/self.nn[i]
                ind[i] = <np.int64_t> ((pp[i] - self.DLE[i])/dds[i])
                cp[i] = (ind[i] + 0.5) * dds[i] + self.DLE[i]
            cur = self.root_mesh[ind[0]][ind[1]][ind[2]]
            if cur == NULL:
                raise RuntimeError
            if self._check_refine(cur, cp, domain_id) == 1:
                self.refine_oct(cur, cp)
            while cur.sd.np < 0:
                if level > 100:
                    raise RuntimeError
                for i in range(3):
                    dds[i] = dds[i] / 2.0
                    if cp[i] > pp[i]:
                        ind[i] = 0
                        cp[i] -= dds[i] / 2.0
                    else:
                        ind[i] = 1
                        cp[i] += dds[i]/2.0
                cur = cur.children[ind[0]][ind[1]][ind[2]]
                level += 1
                if self._check_refine(cur, cp, domain_id) == 1:
                    self.refine_oct(cur, cp)
            # Now we copy in our particle 
            cur.level = level
            for i in range(3):
                cur.sd.pos[i][cur.sd.np] = pp[i]
            cur.domain = domain_id
            cur.sd.np += 1

    cdef int _check_refine(self, Oct *cur, np.float64_t cp[3], int domain_id):
        #Answers: should we refine this oct?
        #False if refined, 
        #False if not refined, but doesn't need refinement
        #True if particles need refinement, 
        #True if not in domain
        if cur.children[0][0][0] != NULL:
            return 0
        elif cur.sd.np >= self.n_ref:
            return 1
        elif cur.domain >= 0 and cur.domain != domain_id:
            return 1
        return 0

    cdef void refine_oct(self, Oct *o, np.float64_t pos[3]):
        #Allocate and initialize child octs
        #Attach particles to child octs
        #Remove particles from this oct entirely
        cdef int i, j, k, m, ind[3]
        cdef Oct *noct
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    noct = self.allocate_oct()
                    noct.level = o.level + 1
                    noct.pos[0] = (o.pos[0] << 1) + i
                    noct.pos[1] = (o.pos[1] << 1) + j
                    noct.pos[2] = (o.pos[2] << 1) + k
                    noct.parent = o
                    o.children[i][j][k] = noct
        for m in range(o.sd.np):
            for i in range(3):
                if o.sd.pos[i][m] < pos[i]:
                    ind[i] = 0
                else:
                    ind[i] = 1
            noct = o.children[ind[0]][ind[1]][ind[2]]
            k = noct.sd.np
            for i in range(3):
                noct.sd.pos[i][k] = o.sd.pos[i][m]
            noct.domain = o.domain
            noct.sd.np += 1
        o.sd.np = -1
        o.domain = -1
        for i in range(3):
            free(o.sd.pos[i])
        free(o.sd.pos)

    def recursively_count(self):
        #Visit every cell, accumulate the # of cells per level
        cdef int i, j, k
        cdef np.int64_t counts[128]
        for i in range(128): counts[i] = 0
        for i in range(self.nn[0]):
            for j in range(self.nn[1]):
                for k in range(self.nn[2]):
                    if self.root_mesh[i][j][k] != NULL:
                        self.visit(self.root_mesh[i][j][k], counts)
        level_counts = {}
        for i in range(128):
            if counts[i] == 0: break
            level_counts[i] = counts[i]
        return level_counts
        
    cdef visit(self, Oct *o, np.int64_t *counts, level = 0):
        cdef int i, j, k
        counts[level] += 1
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    if o.children[i][j][k] != NULL:
                        self.visit(o.children[i][j][k], counts, level + 1)
        return

    def domain_identify(self, np.ndarray[np.uint8_t, ndim=2, cast=True] mask):
        #Return an array of length # of domains
        #Every element is True if there is at least one
        #fully refined *cell* in that domain that isn't masked out
        cdef int i, oi, m
        cdef Oct *o
        cdef np.ndarray[np.uint8_t, ndim=1, cast=True] dmask
        dmask = np.zeros(self.max_domain+1, dtype='uint8')
        for oi in range(self.nocts):
            m = 0
            o = self.oct_list[oi]
            if o.sd.np <= 0 or o.domain == -1: continue
            for i in range(8):
                if mask[oi, i] == 1:
                    m = 1
                    break
            if m == 0: continue
            dmask[o.domain] = 1
        return dmask.astype("bool")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def count_neighbor_particles(self, ppos):
        #How many particles are in my neighborhood
        cdef Oct *main = self.get(ppos)
        cdef Oct* neighbors[27]
        self.neighbors(main, neighbors)
        cdef int i, ni, dl, tnp
        tnp = 0
        for i in range(27):
            if neighbors[i].sd != NULL:
                tnp += neighbors[i].sd.np
        return tnp

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def count_cells(self, SelectorObject selector,
              np.ndarray[np.uint8_t, ndim=2, cast=True] mask):
        #Count how many cells per level there are
        cdef int i, j, k, oi
        # pos here is CELL center, not OCT center.
        cdef np.float64_t pos[3]
        cdef int n = mask.shape[0]
        cdef int eterm[3]
        cdef np.ndarray[np.int64_t, ndim=1] count
        count = np.zeros(self.max_domain + 1, 'int64')
        for oi in range(n):
            o = self.oct_list[oi]
            if o.domain == -1: continue
            for i in range(8):
                count[o.domain] += mask[oi,i]
        return count

    def domain_and(self, np.ndarray[np.uint8_t, ndim=2, cast=True] mask,
                   int domain_id):
        cdef np.int64_t i, oi, n, 
        cdef Oct *o
        cdef np.ndarray[np.uint8_t, ndim=2] m2 = \
                np.zeros((mask.shape[0], 8), 'uint8')
        n = mask.shape[0]
        for oi in range(n):
            o = self.oct_list[oi]
            if o.domain != domain_id: continue
            for i in range(8):
                m2[o.local_ind, i] = mask[o.local_ind, i]
        return m2

