"""
Wrapping code for Oliver Hahn's RamsesRead++

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: UCSD
Author: Oliver Hahn <ohahn@stanford.edu>
Affiliation: KIPAC / Stanford
Homepage: http://yt.enzotools.org/
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

# Cython wrapping code for Oliver Hahn's RAMSES reader
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdlib cimport malloc, free, abs, calloc, labs

import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
    double ceil(double x)
    double log2(double x)

cdef inline np.int64_t i64max(np.int64_t i0, np.int64_t i1):
    if i0 > i1: return i0
    return i1

cdef inline np.int64_t i64min(np.int64_t i0, np.int64_t i1):
    if i0 < i1: return i0
    return i1

cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        cppclass iterator:
            T operator*()
            iterator operator++()
            bint operator==(iterator)
            bint operator!=(iterator)
        vector()
        void push_back(T&)
        T& operator[](int)
        T& at(int)
        iterator begin()
        iterator end()

cdef extern from "<map>" namespace "std":
    cdef cppclass map[A,B]:
        pass

cdef extern from "string" namespace "std":
    cdef cppclass string:
        string(char *cstr)
        char *c_str()
        string operator*()

cdef extern from "RAMSES_typedefs.h":
    pass

cdef extern from "RAMSES_info.hh" namespace "RAMSES":
    enum codeversion:
        version1
        version2
        version3

    cdef cppclass snapshot:
        string m_filename
        codeversion m_version
        cppclass info_data:
            unsigned ncpu
            unsigned ndim
            unsigned levelmin
            unsigned levelmax
            unsigned ngridmax
            unsigned nstep_coarse
            double boxlen
            double time
            double aexp
            double H0
            double omega_m
            double omega_l
            double omega_k
            double omega_b
            double unit_l
            double unit_d
            double unit_t

        info_data m_header
        vector[double] ind_min
        vector[double] ind_max

        snapshot(string info_filename, codeversion ver)

        unsigned get_snapshot_num()
        unsigned getdomain_bykey( double key )

    #void hilbert3d(vector[double] &x, vector[double] &y, vector[double] &z,
    #               vector[double] &order, unsigned bit_length)

cdef extern from "RAMSES_amr_data.hh" namespace "RAMSES::AMR":
    cdef cppclass vec[real_t]:
        real_t x, y, z
        vec( real_t x_, real_t y_, real_t z_)
        vec ( vec& v)
        vec ( )

    # This class definition is out of date.  I have unrolled the template
    # below.
    cdef cppclass cell_locally_essential[id_t, real_t]:
        id_t m_neighbor[6]
        id_t m_father
        id_t m_son[8]
        real_t m_xg[3]
        id_t m_cpu

        char m_pos

        cell_locally_essential()

        bint is_refined(int ison)

    cdef cppclass RAMSES_cell:
        unsigned m_neighbour[6]
        unsigned m_father
        unsigned m_son[8]
        float m_xg[3]
        unsigned m_cpu

        char m_pos

        RAMSES_cell()

        bint is_refined(int ison)

    cdef cppclass cell_simple[id_t, real_t]:
        id_t m_son[1]
        real_t m_xg[3]
        id_t m_cpu

        char m_pos
        cell_simple()
        bint is_refined(int ison)

    # AMR level implementation

    # This class definition is out of date.  I have unrolled the template
    # below.
    cdef cppclass level[Cell_]:
        unsigned m_ilevel
        vector[Cell_] m_level_cells
        double m_xc[8]
        double m_yc[8]
        double m_zc[8]

        # I skipped the typedefs here

        double m_dx
        unsigned m_nx
        level (unsigned ilevel)

        void register_cell( Cell_ cell )
        vector[Cell_].iterator begin()
        vector[Cell_].iterator end()

        Cell_& operator[]( unsigned i)
        unsigned size()

    cdef cppclass RAMSES_level:
        unsigned m_ilevel
        vector[RAMSES_cell] m_level_cells
        double m_xc[8]
        double m_yc[8]
        double m_zc[8]

        # I skipped the typedefs here

        double m_dx
        unsigned m_nx
        RAMSES_level (unsigned ilevel)

        void register_cell( RAMSES_cell cell )
        vector[RAMSES_cell].iterator begin()
        vector[RAMSES_cell].iterator end()

        RAMSES_cell& operator[]( unsigned i)
        unsigned size()

    # This class definition is out of date.  I have unrolled the template
    # below.
    cdef cppclass tree[Cell_, Level_]:
        cppclass header:
            vector[int] nx
            vector[int] nout
            vector[int] nsteps
            int ncpu
            int ndim
            int nlevelmax
            int ngridmax
            int nboundary
            int ngrid_current
            double boxlen
            vector[double] tout
            vector[double] aout
            vector[double] dtold
            vector[double] dtnew
            vector[double] cosm
            vector[double] timing
            double t
            double mass_sph

        vector[Level_] m_AMR_levels
        vector[unsigned] mheadl, m_numbl, m_taill

        int m_cpu
        int m_minlevel
        int m_maxlevel
        string m_fname
        unsigned m_ncoarse
        header m_header

        # This is from later on in the .hh file ... I don't think we need them
        # typedef proto_iterator<const tree*> const_iterator;
        # typedef proto_iterator<tree *> iterator;

        tree (snapshot &snap, int cpu, int maxlevel, int minlevel = 1)

    cppclass tree_iterator "RAMSES::AMR::RAMSES_tree::iterator":
        tree_iterator operator*()
        RAMSES_cell operator*()
        tree_iterator begin()
        tree_iterator end()
        tree_iterator to_parent()
        tree_iterator get_parent()
        void next()
        bint operator!=(tree_iterator other)
        unsigned get_cell_father()
        bint is_finest(int ison)
        int get_absolute_position()
        int get_domain()

    cdef cppclass RAMSES_tree:
        # This is, strictly speaking, not a header.  But, I believe it is
        # going to work alright.
        cppclass header:
            vector[int] nx
            vector[int] nout
            vector[int] nsteps
            int ncpu
            int ndim
            int nlevelmax
            int ngridmax
            int nboundary
            int ngrid_current
            double boxlen
            vector[double] tout
            vector[double] aout
            vector[double] dtold
            vector[double] dtnew
            vector[double] cosm
            vector[double] timing
            double t
            double mass_sph

        vector[RAMSES_level] m_AMR_levels
        vector[unsigned] mheadl, m_numbl, m_taill

        int m_cpu
        int m_minlevel
        int m_maxlevel
        string m_fname
        unsigned m_ncoarse
        header m_header

        unsigned size()

        # This is from later on in the .hh file ... I don't think we need them
        # typedef proto_iterator<const tree*> const_iterator;
        # typedef proto_iterator<tree *> iterator;

        RAMSES_tree(snapshot &snap, int cpu, int maxlevel, int minlevel)
        void read()

        tree_iterator begin(int ilevel)
        tree_iterator end(int ilevel)

        tree_iterator begin()
        tree_iterator end()

        vec[double] cell_pos_double "cell_pos<double>" (tree_iterator it, unsigned ind) 
        vec[double] grid_pos_double "grid_pos<double>" (tree_iterator it)
        vec[float] cell_pos_float "cell_pos<float>" (tree_iterator it, unsigned ind) 
        vec[float] grid_pos_float "grid_pos<float>" (tree_iterator it)

cdef extern from "RAMSES_amr_data.hh" namespace "RAMSES::HYDRO":
    enum hydro_var:
        density
        velocity_x
        velocity_y
        velocity_z
        pressure
        metallicit

    char ramses_hydro_variables[][64]

    # There are a number of classes here that we could wrap and utilize.
    # However, I will only wrap the methods I know we need.

    # I have no idea if this will work.
    cdef cppclass TreeTypeIterator[TreeType_]:
        pass

    # This class definition is out of date.  I have unrolled the template
    # below.
    cdef cppclass data[TreeType_, Real_]:
        cppclass header:
            unsigned ncpu
            unsigned nvar
            unsigned ndim
            unsigned nlevelmax
            unsigned nboundary
            double gamma
        string m_fname
        header m_header

        # I don't want to implement proto_data, so we put this here
        Real_& cell_value(TreeTypeIterator[TreeType_] &it, int ind)

        unsigned m_nvars
        vector[string] m_varnames
        map[string, unsigned] m_var_name_map

        data(TreeType_ &AMRtree)
        #_OutputIterator get_var_names[_OutputIterator](_OutputIterator names)
        void read(string varname)

    cdef cppclass RAMSES_hydro_data:
        cppclass header:
            unsigned ncpu
            unsigned nvar
            unsigned ndim
            unsigned nlevelmax
            unsigned nboundary
            double gamma
        string m_fname
        header m_header

        # I don't want to implement proto_data, so we put this here
        double cell_value (tree_iterator &it, int ind)

        unsigned m_nvars
        vector[string] m_varnames
        map[string, unsigned] m_var_name_map

        RAMSES_hydro_data(RAMSES_tree &AMRtree)
        #_OutputIterator get_var_names[_OutputIterator](_OutputIterator names)
        void read(string varname)

        vector[vector[double]] m_var_array

cdef class RAMSES_tree_proxy:
    cdef string *snapshot_name
    cdef snapshot *rsnap

    cdef RAMSES_tree** trees
    cdef RAMSES_hydro_data*** hydro_datas

    cdef int **loaded

    cdef public object field_ind
    cdef public object field_names

    # We will store this here so that we have a record, independent of the
    # header, of how many things we have allocated
    cdef int ndomains, nfields
    
    def __cinit__(self, char *fn):
        cdef int idomain, ifield, ii
        cdef RAMSES_tree *local_tree
        cdef RAMSES_hydro_data *local_hydro_data
        self.snapshot_name = new string(fn)
        self.rsnap = new snapshot(deref(self.snapshot_name), version3)
        # We now have to get our field names to fill our array
        self.trees = <RAMSES_tree**>\
            malloc(sizeof(RAMSES_tree*) * self.rsnap.m_header.ncpu)
        self.hydro_datas = <RAMSES_hydro_data ***>\
                       malloc(sizeof(RAMSES_hydro_data**) * self.rsnap.m_header.ncpu)
        self.ndomains = self.rsnap.m_header.ncpu
        #for ii in range(self.ndomains): self.trees[ii] = NULL
        # Note we don't do ncpu + 1
        for idomain in range(self.rsnap.m_header.ncpu):
            # we don't delete local_tree
            local_tree = new RAMSES_tree(deref(self.rsnap), idomain + 1,
                                         self.rsnap.m_header.levelmax, 0)
            local_tree.read()
            local_hydro_data = new RAMSES_hydro_data(deref(local_tree))
            self.hydro_datas[idomain] = <RAMSES_hydro_data **>\
                malloc(sizeof(RAMSES_hydro_data*) * local_hydro_data.m_nvars)
            for ii in range(local_hydro_data.m_nvars):
                self.hydro_datas[idomain][ii] = \
                    new RAMSES_hydro_data(deref(local_tree))
            self.trees[idomain] = local_tree
            # We do not delete the final snapshot, which we'll use later
            if idomain + 1 < self.rsnap.m_header.ncpu:
                del local_hydro_data
        # Only once, we read all the field names
        self.nfields = local_hydro_data.m_nvars
        cdef string *field_name
        self.field_names = []
        self.field_ind = {}
        self.loaded = <int **> malloc(sizeof(int) * local_hydro_data.m_nvars)
        for idomain in range(self.ndomains):
            self.loaded[idomain] = <int *> malloc(
                sizeof(int) * local_hydro_data.m_nvars)
            for ifield in range(local_hydro_data.m_nvars):
                self.loaded[idomain][ifield] = 0
        for ifield in range(local_hydro_data.m_nvars):
            field_name = &(local_hydro_data.m_varnames[ifield])
            # Does this leak?
            self.field_names.append(field_name.c_str())
            self.field_ind[self.field_names[-1]] = ifield
        # This all needs to be cleaned up in the deallocator
        del local_hydro_data

    def __dealloc__(self):
        import traceback; traceback.print_stack()
        cdef int idomain, ifield
        # To ensure that 'delete' is used, not 'free',
        # we allocate temporary variables.
        cdef RAMSES_tree *temp_tree
        cdef RAMSES_hydro_data *temp_hdata
        for idomain in range(self.ndomains):
            for ifield in range(self.nfields):
                temp_hdata = self.hydro_datas[idomain][ifield]
                del temp_hdata
            temp_tree = self.trees[idomain]
            del temp_tree
            free(self.loaded[idomain])
            free(self.hydro_datas[idomain])
        free(self.trees)
        free(self.hydro_datas)
        free(self.loaded)
        if self.snapshot_name != NULL: del self.snapshot_name
        if self.rsnap != NULL: del self.rsnap
        
    def count_zones(self):
        # We need to do simulation domains here

        cdef unsigned idomain, ilevel
        cdef RAMSES_tree *local_tree
        cdef RAMSES_hydro_data *local_hydro_data
        cdef RAMSES_level *local_level
        cdef tree_iterator grid_it, grid_end

        # All the loop-local pointers must be declared up here

        cell_count = []
        cdef int local_count = 0
        for ilevel in range(self.rsnap.m_header.levelmax + 1):
            cell_count.append(0)
        for idomain in range(1, self.rsnap.m_header.ncpu + 1):
            local_tree = new RAMSES_tree(deref(self.rsnap), idomain,
                                         self.rsnap.m_header.levelmax, 0)
            local_tree.read()
            local_hydro_data = new RAMSES_hydro_data(deref(local_tree))
            for ilevel in range(local_tree.m_maxlevel + 1):
                local_count = 0
                local_level = &local_tree.m_AMR_levels[ilevel]
                grid_it = local_tree.begin(ilevel)
                grid_end = local_tree.end(ilevel)
                while grid_it != grid_end:
                    local_count += (grid_it.get_domain() == idomain)
                    grid_it.next()
                cell_count[ilevel] += local_count
            del local_tree, local_hydro_data

        return cell_count

    def ensure_loaded(self, char *varname, int domain_index):
        # this domain_index must be zero-indexed
        cdef int varindex = self.field_ind[varname]
        cdef string *field_name = new string(varname)
        if self.loaded[domain_index][varindex] == 1:
            return
        print "READING FROM DISK", varname, domain_index, varindex
        self.hydro_datas[domain_index][varindex].read(deref(field_name))
        self.loaded[domain_index][varindex] = 1
        del field_name

    def clear_tree(self, char *varname, int domain_index):
        # this domain_index must be zero-indexed
        # We delete and re-create
        cdef int varindex = self.field_ind[varname]
        cdef string *field_name = new string(varname)
        if self.loaded[domain_index][varindex] == 0: return
        cdef RAMSES_hydro_data *temp_hdata = self.hydro_datas[domain_index][varindex]
        del temp_hdata
        self.hydro_datas[domain_index - 1][varindex] = \
            new RAMSES_hydro_data(deref(self.trees[domain_index]))
        self.loaded[domain_index][varindex] = 0
        del field_name

    def get_file_info(self):
        header_info = {}
        header_info["ncpu"] = self.rsnap.m_header.ncpu
        header_info["ndim"] = self.rsnap.m_header.ndim
        header_info["levelmin"] = self.rsnap.m_header.levelmin
        header_info["levelmax"] = self.rsnap.m_header.levelmax
        header_info["ngridmax"] = self.rsnap.m_header.ngridmax
        header_info["nstep_coarse"] = self.rsnap.m_header.nstep_coarse
        header_info["boxlen"] = self.rsnap.m_header.boxlen
        header_info["time"] = self.rsnap.m_header.time
        header_info["aexp"] = self.rsnap.m_header.aexp
        header_info["H0"] = self.rsnap.m_header.H0
        header_info["omega_m"] = self.rsnap.m_header.omega_m
        header_info["omega_l"] = self.rsnap.m_header.omega_l
        header_info["omega_k"] = self.rsnap.m_header.omega_k
        header_info["omega_b"] = self.rsnap.m_header.omega_b
        header_info["unit_l"] = self.rsnap.m_header.unit_l
        header_info["unit_d"] = self.rsnap.m_header.unit_d
        header_info["unit_t"] = self.rsnap.m_header.unit_t

        # Now we grab some from the trees
        cdef np.ndarray[np.int32_t, ndim=1] top_grid_dims = np.zeros(3, "int32")
        cdef int i
        for i in range(3):
            # Note that nx is really boundary conditions.  We always have 2.
            top_grid_dims[i] = self.trees[0].m_header.nx[i]
            top_grid_dims[i] = 2
        header_info["nx"] = top_grid_dims

        return header_info

    def fill_hierarchy_arrays(self, 
                              np.ndarray[np.int32_t, ndim=1] top_grid_dims,
                              np.ndarray[np.float64_t, ndim=2] left_edges,
                              np.ndarray[np.float64_t, ndim=2] right_edges,
                              np.ndarray[np.int32_t, ndim=2] grid_levels,
                              np.ndarray[np.int64_t, ndim=2] grid_file_locations,
                              np.ndarray[np.uint64_t, ndim=1] grid_hilbert_indices,
                              np.ndarray[np.int32_t, ndim=2] child_mask,
                              np.ndarray[np.float64_t, ndim=1] domain_left,
                              np.ndarray[np.float64_t, ndim=1] domain_right):
        # We need to do simulation domains here

        cdef unsigned idomain, ilevel

        # All the loop-local pointers must be declared up here
        cdef RAMSES_tree *local_tree
        cdef RAMSES_hydro_data *local_hydro_data
        cdef unsigned father

        cdef tree_iterator grid_it, grid_end, father_it
        cdef vec[double] gvec
        cdef int grid_ind = 0, grid_aind = 0
        cdef unsigned parent_ind
        cdef bint ci
        cdef double pos[3]
        cdef double grid_half_width 
        cdef unsigned long rv

        cdef np.int32_t rr
        cdef int i
        cell_count = []
        level_cell_counts = {}
        for idomain in range(1, self.rsnap.m_header.ncpu + 1):
            local_tree = new RAMSES_tree(deref(self.rsnap), idomain,
                                         self.rsnap.m_header.levelmax, 0)
            local_tree.read()
            local_hydro_data = new RAMSES_hydro_data(deref(local_tree))
            for ilevel in range(local_tree.m_maxlevel + 1):
                # this gets overwritten for every domain, which is okay
                level_cell_counts[ilevel] = grid_ind 
                #grid_half_width = self.rsnap.m_header.boxlen / \
                grid_half_width = 1.0 / \
                    (2**(ilevel) * top_grid_dims[0])
                grid_it = local_tree.begin(ilevel)
                grid_end = local_tree.end(ilevel)
                while grid_it != grid_end:
                    if grid_it.get_domain() != idomain:
                        grid_ind += 1
                        grid_it.next()
                        continue
                    gvec = local_tree.grid_pos_double(grid_it)
                    left_edges[grid_aind, 0] = pos[0] = gvec.x - grid_half_width
                    left_edges[grid_aind, 1] = pos[1] = gvec.y - grid_half_width
                    left_edges[grid_aind, 2] = pos[2] = gvec.z - grid_half_width
                    for i in range(3):
                        pos[i] = (pos[i] - domain_left[i]) / (domain_right[i] - domain_left[i])
                    right_edges[grid_aind, 0] = gvec.x + grid_half_width
                    right_edges[grid_aind, 1] = gvec.y + grid_half_width
                    right_edges[grid_aind, 2] = gvec.z + grid_half_width
                    grid_levels[grid_aind, 0] = ilevel
                    # Now the harder part
                    father_it = grid_it.get_parent()
                    grid_file_locations[grid_aind, 0] = <np.int64_t> idomain
                    grid_file_locations[grid_aind, 1] = grid_ind - level_cell_counts[ilevel]
                    parent_ind = father_it.get_absolute_position()
                    if ilevel > 0:
                        # We calculate the REAL parent index
                        grid_file_locations[grid_aind, 2] = \
                            level_cell_counts[ilevel - 1] + parent_ind
                    else:
                        grid_file_locations[grid_aind, 2] = -1
                    for ci in range(8):
                        rr = <np.int32_t> grid_it.is_finest(ci)
                        child_mask[grid_aind, ci] = rr
                    grid_ind += 1
                    grid_aind += 1
                    grid_it.next()
            del local_tree, local_hydro_data

    def read_oct_grid(self, char *field, int level, int domain, int grid_id):

        self.ensure_loaded(field, domain - 1)
        cdef int varindex = self.field_ind[field]
        cdef int i

        cdef np.ndarray[np.float64_t, ndim=3] tr = np.empty((2,2,2), dtype='float64',
                                                   order='F')
        cdef tree_iterator grid_it, grid_end
        cdef double* data = <double*> tr.data

        cdef RAMSES_tree *local_tree = self.trees[domain - 1]
        cdef RAMSES_hydro_data *local_hydro_data = self.hydro_datas[domain - 1][varindex]

        #inline ValueType_& cell_value( const typename TreeType_::iterator& it,
        #                               int ind )
        #{
        #   unsigned ipos   = it.get_absolute_position();
        #   unsigned ilevel = it.get_level();//-m_minlevel;
        #   return (m_var_array[ilevel])[m_twotondim*ipos+ind];
        #}
        
        for i in range(8): 
            data[i] = local_hydro_data.m_var_array[level][8*grid_id+i]
        return tr

    def read_grid(self, char *field, 
                  np.ndarray[np.int64_t, ndim=1] start_index,
                  np.ndarray[np.int32_t, ndim=1] grid_dims,
                  np.ndarray[np.float64_t, ndim=3] data,
                  np.ndarray[np.int32_t, ndim=3] filled,
                  int level, int ref_factor,
                  component_grid_info):
        cdef int varindex = self.field_ind[field]
        cdef RAMSES_tree *local_tree = NULL
        cdef RAMSES_hydro_data *local_hydro_data = NULL

        cdef int gi, i, j, k, domain, offset
        cdef int ir, jr, kr
        cdef int offi, offj, offk, odind
        cdef np.int64_t di, dj, dk
        cdef np.ndarray[np.int64_t, ndim=1] ogrid_info
        cdef np.ndarray[np.int64_t, ndim=1] og_start_index
        cdef np.float64_t temp_data
        cdef np.int64_t end_index[3]
        cdef int to_fill = 0
        # Note that indexing into a cell is:
        #   (k*2 + j)*2 + i
        for i in range(3):
            end_index[i] = start_index[i] + grid_dims[i]
        for gi in range(len(component_grid_info)):
            ogrid_info = component_grid_info[gi]
            domain = ogrid_info[0]
            #print "Loading", domain, ogrid_info
            self.ensure_loaded(field, domain - 1)
            local_tree = self.trees[domain - 1]
            local_hydro_data = self.hydro_datas[domain - 1][varindex]
            offset = ogrid_info[1]
            og_start_index = ogrid_info[3:]
            for i in range(2*ref_factor):
                di = i + og_start_index[0] * ref_factor
                if di < start_index[0] or di >= end_index[0]: continue
                ir = <int> (i / ref_factor)
                for j in range(2 * ref_factor):
                    dj = j + og_start_index[1] * ref_factor
                    if dj < start_index[1] or dj >= end_index[1]: continue
                    jr = <int> (j / ref_factor)
                    for k in range(2 * ref_factor):
                        dk = k + og_start_index[2] * ref_factor
                        if dk < start_index[2] or dk >= end_index[2]: continue
                        kr = <int> (k / ref_factor)
                        offi = di - start_index[0]
                        offj = dj - start_index[1]
                        offk = dk - start_index[2]
                        #print offi, filled.shape[0],
                        #print offj, filled.shape[1],
                        #print offk, filled.shape[2]
                        if filled[offi, offj, offk] == 1: continue

                        odind = (kr*2 + jr)*2 + ir
                        temp_data = local_hydro_data.m_var_array[
                                level][8*offset + odind]
                        data[offi, offj, offk] = temp_data
                        filled[offi, offj, offk] = 1
                        to_fill += 1
        return to_fill

cdef class ProtoSubgrid:
    cdef np.int64_t *signature[3]
    cdef np.int64_t left_edge[3]
    cdef np.int64_t right_edge[3]
    cdef np.int64_t dimensions[3]
    cdef public np.float64_t efficiency
    cdef public object sigs
    cdef public object grid_file_locations
    cdef public object dd
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self,
                   np.ndarray[np.int64_t, ndim=1] left_index,
                   np.ndarray[np.int64_t, ndim=1] dimensions, 
                   np.ndarray[np.int64_t, ndim=2] left_edges,
                   np.ndarray[np.int64_t, ndim=2] right_edges,
                   np.ndarray[np.int64_t, ndim=2] grid_dimensions,
                   np.ndarray[np.int64_t, ndim=2] grid_file_locations):
        # This also includes the shrinking step.
        cdef int i, ci, ng = left_edges.shape[0]
        cdef np.ndarray temp_arr
        cdef int l0, r0, l1, r1, l2, r2, i0, i1, i2
        cdef np.int64_t temp_l[3], temp_r[3], ncells
        cdef np.float64_t efficiency
        self.sigs = []
        for i in range(3):
            temp_l[i] = left_index[i] + dimensions[i]
            temp_r[i] = left_index[i]
            self.signature[i] = NULL
        for gi in range(ng):
            if left_edges[gi,0] > left_index[0]+dimensions[0] or \
               right_edges[gi,0] < left_index[0] or \
               left_edges[gi,1] > left_index[1]+dimensions[1] or \
               right_edges[gi,1] < left_index[1] or \
               left_edges[gi,2] > left_index[2]+dimensions[2] or \
               right_edges[gi,2] < left_index[2]:
               #print "Skipping grid", gi, "which lies outside out box"
               continue
            for i in range(3):
                temp_l[i] = i64min(left_edges[gi,i], temp_l[i])
                temp_r[i] = i64max(right_edges[gi,i], temp_r[i])
        for i in range(3):
            self.left_edge[i] = i64max(temp_l[i], left_index[i])
            self.right_edge[i] = i64min(temp_r[i], left_index[i] + dimensions[i])
            self.dimensions[i] = self.right_edge[i] - self.left_edge[i]
            if self.dimensions[i] <= 0:
                self.efficiency = -1.0
                return
            self.sigs.append(np.zeros(self.dimensions[i], 'int64'))
        #print self.sigs[0].size, self.sigs[1].size, self.sigs[2].size
        
        # My guess is that this whole loop could be done more efficiently.
        # However, this is clear and straightforward, so it is a good first
        # pass.
        cdef np.ndarray[np.int64_t, ndim=1] sig0, sig1, sig2
        sig0 = self.sigs[0]
        sig1 = self.sigs[1]
        sig2 = self.sigs[2]
        efficiency = 0.0
        cdef int used
        self.grid_file_locations = []
        for gi in range(ng):
            used = 0
            nnn = 0
            for l0 in range(grid_dimensions[gi, 0]):
                i0 = left_edges[gi, 0] + l0
                if i0 < self.left_edge[0]: continue
                if i0 >= self.right_edge[0]: break
                for l1 in range(grid_dimensions[gi, 1]):
                    i1 = left_edges[gi, 1] + l1
                    if i1 < self.left_edge[1]: continue
                    if i1 >= self.right_edge[1]: break
                    for l2 in range(grid_dimensions[gi, 2]):
                        i2 = left_edges[gi, 2] + l2
                        if i2 < self.left_edge[2]: continue
                        if i2 >= self.right_edge[2]: break
                        i = i0 - self.left_edge[0]
                        sig0[i] += 1
                        i = i1 - self.left_edge[1]
                        sig1[i] += 1
                        i = i2 - self.left_edge[2]
                        sig2[i] += 1
                        efficiency += 1
                        used = 1
            if used == 1:
                grid_file_locations[gi,3] = left_edges[gi, 0]
                grid_file_locations[gi,4] = left_edges[gi, 1]
                grid_file_locations[gi,5] = left_edges[gi, 2]
                self.grid_file_locations.append(grid_file_locations[gi,:])
         
        self.dd = np.ones(3, dtype='int64')
        for i in range(3):
            efficiency /= self.dimensions[i]
            self.dd[i] = self.dimensions[i]
        #print "Efficiency is %0.3e" % (efficiency)
        self.efficiency = efficiency

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def find_split(self):
        # First look for zeros
        cdef int i, center, ax
        cdef np.ndarray[ndim=1, dtype=np.int64_t] axes
        cdef np.int64_t strength, zcstrength, zcp
        axes = np.argsort(self.dd)[::-1]
        cdef np.ndarray[np.int64_t] sig
        for axi in range(3):
            ax = axes[axi]
            center = self.dimensions[ax] / 2
            sig = self.sigs[ax]
            for i in range(self.dimensions[ax]):
                if sig[i] == 0 and i > 0 and i < self.dimensions[ax] - 1:
                    #print "zero: %s (%s)" % (i, self.dimensions[ax])
                    return 0, ax, i
        zcstrength = 0
        zcp = 0
        zca = -1
        cdef int temp
        cdef np.int64_t *sig2d
        for axi in range(3):
            ax = axes[axi]
            sig = self.sigs[ax]
            sig2d = <np.int64_t *> malloc(sizeof(np.int64_t) * self.dimensions[ax])
            sig2d[0] = sig2d[self.dimensions[ax]-1] = 0
            for i in range(1, self.dimensions[ax] - 1):
                sig2d[i] = sig[i-1] - 2*sig[i] + sig[i+1]
            for i in range(1, self.dimensions[ax] - 1):
                if sig2d[i] * sig2d[i+1] <= 0:
                    strength = labs(sig2d[i] - sig2d[i+1])
                    if (strength > zcstrength) or \
                       (strength == zcstrength and (abs(center - i) <
                                                    abs(center - zcp))):
                        zcstrength = strength
                        zcp = i
                        zca = ax
            free(sig2d)
        #print "zcp: %s (%s)" % (zcp, self.dimensions[ax])
        return 1, ax, zcp

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_properties(self):
        cdef np.ndarray[np.int64_t, ndim=2] tr = np.empty((3,3), dtype='int64')
        cdef int i
        for i in range(3):
            tr[0,i] = self.left_edge[i]
            tr[1,i] = self.right_edge[i]
            tr[2,i] = self.dimensions[i]
        return tr

@cython.cdivision
cdef np.int64_t graycode(np.int64_t x):
    return x^(x>>1)

@cython.cdivision
cdef np.int64_t igraycode(np.int64_t x):
    cdef np.int64_t i, j
    if x == 0:
        return x
    m = <np.int64_t> ceil(log2(x)) + 1
    i, j = x, 1
    while j < m:
        i = i ^ (x>>j)
        j += 1
    return i

@cython.cdivision
cdef np.int64_t direction(np.int64_t x, np.int64_t n):
    #assert x < 2**n
    if x == 0:
        return 0
    elif x%2 == 0:
        return tsb(x-1, n)%n
    else:
        return tsb(x, n)%n

@cython.cdivision
cdef np.int64_t tsb(np.int64_t x, np.int64_t width):
    #assert x < 2**width
    cdef np.int64_t i = 0
    while x&1 and i <= width:
        x = x >> 1
        i += 1
    return i

@cython.cdivision
cdef np.int64_t bitrange(np.int64_t x, np.int64_t width,
                         np.int64_t start, np.int64_t end):
    return x >> (width-end) & ((2**(end-start))-1)

@cython.cdivision
cdef np.int64_t rrot(np.int64_t x, np.int64_t i, np.int64_t width):
    i = i%width
    x = (x>>i) | (x<<width-i)
    return x&(2**width-1)

@cython.cdivision
cdef np.int64_t lrot(np.int64_t x, np.int64_t i, np.int64_t width):
    i = i%width
    x = (x<<i) | (x>>width-i)
    return x&(2**width-1)

@cython.cdivision
cdef np.int64_t transform(np.int64_t entry, np.int64_t direction,
                          np.int64_t width, np.int64_t x):
    return rrot((x^entry), direction + 1, width)

@cython.cdivision
cdef np.int64_t entry(np.int64_t x):
    if x == 0: return 0
    return graycode(2*((x-1)/2))

@cython.cdivision
def get_hilbert_indices(int order, np.ndarray[np.int64_t, ndim=2] left_index):
    # This is inspired by the scurve package by user cortesi on GH.
    cdef int o, i
    cdef np.int64_t x, w, h, e, d, l, b
    cdef np.int64_t p[3]
    cdef np.ndarray[np.int64_t, ndim=1] hilbert_indices
    hilbert_indices = np.zeros(left_index.shape[0], 'int64')
    for o in range(left_index.shape[0]):
        p[0] = left_index[o, 0]
        p[1] = left_index[o, 1]
        p[2] = left_index[o, 2]
        h = e = d = 0
        for i in range(order):
            l = 0
            for x in range(3):
                b = bitrange(p[3-x-1], order, i, i+1)
                l |= (b<<x)
            l = transform(e, d, 3, l)
            w = igraycode(l)
            e = e ^ lrot(entry(w), d+1, 3)
            d = (d + direction(w, 3) + 1)%3
            h = (h<<3)|w
        hilbert_indices[o] = h
    return hilbert_indices

