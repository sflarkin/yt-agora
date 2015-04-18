"""
This module contains cython tools to write octree data to a depth-first
Hilbert ordered oct-tree structure. The data is written to the output array, 
which is depth first Hilbert ordered, following the Sunrise convention. 
"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

cimport cython
cimport numpy as np
import numpy as np
from yt.funcs import get_pbar
import time


@cython.boundscheck(False)
def fill_octree_arrays(ad, fields, np.uint64_t nocts,
                       np.ndarray[np.float64_t, ndim=2] LeftEdge,
                       np.ndarray[np.float64_t, ndim=1] dx,
                       np.ndarray[np.int32_t, ndim=1] Level,
                       np.ndarray[np.float64_t, ndim=5] Fields):

     cdef np.uint64_t i
     block_iter = ad.blocks.__iter__()  
    
     pbar = get_pbar("Filling oct-tree arrays", nocts)
     for i in range(nocts):
         oct, mask = block_iter.next() 
         LeftEdge[i] = oct.LeftEdge 
         dx[i] = oct.dds[0]
         Level[i] = oct.Level
         Fields[i] = np.array([oct[f] for f in fields])
         pbar.update(i)
     pbar.finish()

     return LeftEdge, dx, Level, Fields


@cython.boundscheck(False)
@cython.wraparound(False)
def RecurseOctreeDepthFirstHilbert_c(np.ndarray[np.int32_t, ndim=1] child_index,# Integer position index of the children oct 
                                                                                # within a parent oct or within the root 
                                                                                # grid of octs (super level) 
                            np.int64_t oct_id, # ID of parent oct (-1 if root grid of octs) 
                            position_c pos, # The output hydro data position and refinement position
                            hilbert_state_c hilbert,  # The hilbert state
                            np.ndarray[np.float64_t, ndim=2] output, # Holds the hydro data
                            np.ndarray[np.uint32_t, ndim=1] refined, # Holds the refinement status  of Octs, 0s and 1s
                            np.ndarray[np.int32_t, ndim=1] levels, # For a given Oct, what is the level
                            np.ndarray[np.float64_t, ndim=2] LeftEdge, # Array with LeftEdges for all octs
                            np.ndarray[np.float64_t, ndim=1] dx,  # Array with dxs for all octs
                            np.ndarray[np.int32_t, ndim=1] Level, # Array with Leves for all octs
                            np.ndarray[np.float64_t, ndim=5] Fields, # Array with Field data for all octs
                            np.int32_t oct_level,  # level of the parent oct, if level < 0 we are at the 
                                                  # super level (root grid of octs)
                            np.float64_t start_time, 
                            max_level=None,                           
                            tracker=None):


    cdef int is_leaf
    cdef unsigned int field_index		    
    cdef np.int64_t child_oct_id, next_oct_id, 
    cdef np.float64_t cdx, pdx, child_oct_dx
    cdef object vertex
    cdef hilbert_state_c hilbert_child
    cdef np.ndarray[np.int32_t, ndim=1] vtx
    cdef np.ndarray[np.float64_t, ndim=2] cLE, 
    cdef np.ndarray[np.float64_t, ndim=1] child_oct_le, parent_oct_le
    cdef np.ndarray[np.float64_t, ndim=4] oct_fields
    cdef np.ndarray[np.int32_t, ndim=1] next_child_index, child_oct_ile
    cdef np.ndarray[np.int64_t, ndim=3] child_index_mask

    if pos.refined_pos%100000==0 and pos.refined_pos!=0:
        print '\nOcts visited = ', pos.refined_pos
        print 'Leaves written = ', pos.output_pos 
        print 'Time spent = ', time.time()-start_time			    
        if tracker: tracker.update(pos.refined_pos)
	
    if oct_id == -1:
        # we are at the root grid of octs
        children = np.argwhere(Level == 0)[:,0]
        cLE =  LeftEdge[children]
        cdx = np.unique(dx[children])
        assert cdx == np.max(dx) 
        pdx = 2.0*cdx
        thischild =  np.all(cLE ==  np.array([child_index[0]*pdx,
                                              child_index[1]*pdx,
                                              child_index[2]*pdx]), axis=1)
        assert len(children[thischild] == 1)				      
        child_oct_id = children[thischild][0] 
    else: 
        if oct_level==max_level:   
            child_oct_id = -1
        else:
            children, child_index_mask = get_oct_children_c(oct_id, LeftEdge, dx, Level)   
            child_oct_id = child_index_mask[child_index[0], child_index[1], child_index[2]]

    #record the refinement state
    levels[pos.refined_pos]  = oct_level
    is_leaf = (child_oct_id == -1) and (oct_level > 0) #Don't subdivide if we are on a superlevel
    refined[pos.refined_pos] = not is_leaf #True is oct, False is leaf
    pos.refined_pos += 1 

    if is_leaf: 
        #then we have hit a leaf cell; write it out
        oct_fields = Fields[oct_id]
        for field_index in range(oct_fields.shape[0]):
            output[pos.output_pos,field_index] = \
                    oct_fields[field_index,child_index[0],child_index[1],child_index[2]]
        pos.output_pos+= 1
    else:
        assert child_oct_id > -1
        for (vertex, hilbert_child) in hilbert:
            #vertex is a combination of three 0s and 1s to 
            #denote each of the 8 octs
            vtx = np.array(vertex).astype('int32')
            if oct_level < 0:
                next_oct_id = oct_id #Don't descend if we're a superlevel
                next_child_index = child_index + vtx*2**(-(oct_level+1))
            else:
                next_oct_id = child_oct_id  #Descend

                # Get the floating point left edge of the oct we descended into and the current oct/grid
                child_oct_le = LeftEdge[child_oct_id]
                child_oct_dx = dx[child_oct_id]
                if oct_id == -1:
                    parent_oct_le = child_index*pdx
                else:    
                    parent_oct_le = LeftEdge[oct_id] + child_index*dx[oct_id]
    
                # Then translate onto the subgrid integer index 
                child_oct_ile = np.floor((parent_oct_le - child_oct_le)/child_oct_dx).astype('int32')
                next_child_index = child_oct_ile + vtx

            RecurseOctreeDepthFirstHilbert_c(next_child_index, next_oct_id, pos,
                                             hilbert_child, output, refined, levels,
                                             LeftEdge, dx, Level, Fields,
					     oct_level + 1, 
                                             start_time,
                                             max_level=max_level,
                                             tracker=tracker) 


@cython.boundscheck(False)
@cython.wraparound(False)
def get_oct_children_c(np.int64_t parent_oct_id, 
                      np.ndarray[np.float64_t, ndim=2] LE,
                      np.ndarray[np.float64_t, ndim=1] dx,
                      np.ndarray[np.int32_t, ndim=1] Level):
    """
    Find the children of the given parent oct
    """
    cdef unsigned int i, j, k
    cdef np.float64_t pdx
    cdef np.int32_t pLevel
    cdef np.int64_t child
    cdef np.ndarray[np.int64_t, ndim=1] children
    cdef np.ndarray[np.int64_t, ndim=3] child_index_mask 
    cdef np.ndarray[np.float64_t, ndim=2] RE
    cdef np.ndarray[np.float64_t, ndim=1] pLE, pRE

    RE = LE + 2*dx.reshape((dx.size,1))
    pLE = LE[parent_oct_id]
    pdx = dx[parent_oct_id]
    pRE = pLE + 2.0*pdx
    pLevel = Level[parent_oct_id]

    children = np.argwhere(np.all(LE >= pLE, axis=1) &
                           np.all(RE <= pRE, axis=1) & 
                           (Level == pLevel + 1))[:,0] 

    child_index_mask = np.zeros((2, 2, 2), 'int64') - 1   

    for child in children:
        i,j,k = np.floor((LE[child] - pLE)/pdx).astype('uint32')
        child_index_mask[i,j,k] = child
 
    return children, child_index_mask


cdef class position_c:
    cdef public int output_pos, refined_pos
    def __init__(self):
        self.output_pos = 0
        self.refined_pos = 0

cdef class hilbert_state_c:

    cdef public hilbert_state_c child
    cdef public object vertex, dim, sgn, ndim, nsgn
    cdef public int octant
    cdef int temp, axis, i, j, k
 
    def __init__(self, dim=None, sgn=None, octant=None):
        if dim is None: dim = [0,1,2]
        if sgn is None: sgn = [1,1,1]
        if octant is None: octant = 5
        self.dim = dim
        self.sgn = sgn
        self.octant = octant
    def flip(self,i):
        self.sgn[i]*=-1
    def swap(self,i,j):
        temp = self.dim[i]
        self.dim[i]=self.dim[j]
        self.dim[j]=temp
        axis = self.sgn[i]
        self.sgn[i] = self.sgn[j]
        self.sgn[j] = axis
    def reorder(self,i,j,k):
        ndim = [self.dim[i],self.dim[j],self.dim[k]] 
        nsgn = [self.sgn[i],self.sgn[j],self.sgn[k]]
        self.dim = ndim
        self.sgn = nsgn
    def copy(self):
        return hilbert_state_c([self.dim[0],self.dim[1],self.dim[2]],
                               [self.sgn[0],self.sgn[1],self.sgn[2]],
                                self.octant)
    def descend(self,o):
        child = self.copy()
        child.octant = o
        if o==0:
            child.swap(0,2)
        elif o==1:
            child.swap(1,2)
        elif o==2:
            pass
        elif o==3:
            child.flip(0)
            child.flip(2)
            child.reorder(2,0,1)
        elif o==4:
            child.flip(0)
            child.flip(1)
            child.reorder(2,0,1)
        elif o==5:
            pass
        elif o==6:
            child.flip(1)
            child.flip(2)
            child.swap(1,2)
        elif o==7:
            child.flip(0)
            child.flip(2)
            child.swap(0,2)
        return child

    def __iter__(self):
        vertex = [0,0,0]
        j=0
        for i in range(3):
            vertex[self.dim[i]] = 0 if self.sgn[i]>0 else 1
        yield vertex, self.descend(j)
        vertex[self.dim[0]] += self.sgn[0]
        j+=1
        yield vertex, self.descend(j)
        vertex[self.dim[1]] += self.sgn[1] 
        j+=1
        yield vertex, self.descend(j)
        vertex[self.dim[0]] -= self.sgn[0] 
        j+=1
        yield vertex, self.descend(j)
        vertex[self.dim[2]] += self.sgn[2] 
        j+=1
        yield vertex, self.descend(j)
        vertex[self.dim[0]] += self.sgn[0] 
        j+=1
        yield vertex, self.descend(j)
        vertex[self.dim[1]] -= self.sgn[1] 
        j+=1
        yield vertex, self.descend(j)
        vertex[self.dim[0]] -= self.sgn[0] 
        j+=1
        yield vertex, self.descend(j)
