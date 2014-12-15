"""
Simple utilities that don't fit anywhere else



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
from yt.units.yt_array import YTArray
cimport numpy as np
cimport cython
cimport libc.math as math
from fp_utils cimport fmin, fmax, i64min, i64max

cdef extern from "stdlib.h":
    # NOTE that size_t might not be int
    void *alloca(int)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def new_bin_profile1d(np.ndarray[np.intp_t, ndim=1] bins_x,
                  np.ndarray[np.float64_t, ndim=1] wsource,
                  np.ndarray[np.float64_t, ndim=2] bsource,
                  np.ndarray[np.float64_t, ndim=1] wresult,
                  np.ndarray[np.float64_t, ndim=2] bresult,
                  np.ndarray[np.float64_t, ndim=2] mresult,
                  np.ndarray[np.float64_t, ndim=2] qresult,
                  np.ndarray[np.uint8_t, ndim=1, cast=True] used):
    cdef int n, fi, bin
    cdef np.float64_t wval, bval, oldwr
    cdef int nb = bins_x.shape[0]
    cdef int nf = bsource.shape[1]
    for n in range(nb):
        bin = bins_x[n]
        wval = wsource[n]
        oldwr = wresult[bin]
        wresult[bin] += wval
        for fi in range(nf):
            bval = bsource[n,fi]
            # qresult has to have the previous wresult
            qresult[bin,fi] += (oldwr * wval * (bval - mresult[bin,fi])**2) / \
                (oldwr + wval)
            bresult[bin,fi] += wval*bval
            # mresult needs the new wresult
            mresult[bin,fi] += wval * (bval - mresult[bin,fi]) / wresult[bin]
        used[bin] = 1
    return

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def new_bin_profile2d(np.ndarray[np.intp_t, ndim=1] bins_x,
                  np.ndarray[np.intp_t, ndim=1] bins_y,
                  np.ndarray[np.float64_t, ndim=1] wsource,
                  np.ndarray[np.float64_t, ndim=2] bsource,
                  np.ndarray[np.float64_t, ndim=2] wresult,
                  np.ndarray[np.float64_t, ndim=3] bresult,
                  np.ndarray[np.float64_t, ndim=3] mresult,
                  np.ndarray[np.float64_t, ndim=3] qresult,
                  np.ndarray[np.uint8_t, ndim=2, cast=True] used):
    cdef int n, fi, bin_x, bin_y
    cdef np.float64_t wval, bval, oldwr
    cdef int nb = bins_x.shape[0]
    cdef int nf = bsource.shape[1]
    for n in range(nb):
        bin_x = bins_x[n]
        bin_y = bins_y[n]
        wval = wsource[n]
        oldwr = wresult[bin_x, bin_y]
        wresult[bin_x,bin_y] += wval
        for fi in range(nf):
            bval = bsource[n,fi]
            # qresult has to have the previous wresult
            qresult[bin_x,bin_y,fi] += (oldwr * wval * (bval - mresult[bin_x,bin_y,fi])**2) / \
                (oldwr + wval)
            bresult[bin_x,bin_y,fi] += wval*bval
            # mresult needs the new wresult
            mresult[bin_x,bin_y,fi] += wval * (bval - mresult[bin_x,bin_y,fi]) / wresult[bin_x,bin_y]
        used[bin_x,bin_y] = 1
    return

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def new_bin_profile3d(np.ndarray[np.intp_t, ndim=1] bins_x,
                  np.ndarray[np.intp_t, ndim=1] bins_y,
                  np.ndarray[np.intp_t, ndim=1] bins_z,
                  np.ndarray[np.float64_t, ndim=1] wsource,
                  np.ndarray[np.float64_t, ndim=2] bsource,
                  np.ndarray[np.float64_t, ndim=3] wresult,
                  np.ndarray[np.float64_t, ndim=4] bresult,
                  np.ndarray[np.float64_t, ndim=4] mresult,
                  np.ndarray[np.float64_t, ndim=4] qresult,
                  np.ndarray[np.uint8_t, ndim=3, cast=True] used):
    cdef int n, fi, bin_x, bin_y, bin_z
    cdef np.float64_t wval, bval, oldwr
    cdef int nb = bins_x.shape[0]
    cdef int nf = bsource.shape[1]
    for n in range(nb):
        bin_x = bins_x[n]
        bin_y = bins_y[n]
        bin_z = bins_z[n]
        wval = wsource[n]
        oldwr = wresult[bin_x, bin_y, bin_z]
        wresult[bin_x,bin_y,bin_z] += wval
        for fi in range(nf):
            bval = bsource[n,fi]
            # qresult has to have the previous wresult
            qresult[bin_x,bin_y,bin_z,fi] += \
                (oldwr * wval * (bval - mresult[bin_x,bin_y,bin_z,fi])**2) / \
                (oldwr + wval)
            bresult[bin_x,bin_y,bin_z,fi] += wval*bval
            # mresult needs the new wresult
            mresult[bin_x,bin_y,bin_z,fi] += wval * \
                (bval - mresult[bin_x,bin_y,bin_z,fi]) / \
                 wresult[bin_x,bin_y,bin_z]
        used[bin_x,bin_y,bin_z] = 1
    return

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def bin_profile1d(np.ndarray[np.int64_t, ndim=1] bins_x,
                  np.ndarray[np.float64_t, ndim=1] wsource,
                  np.ndarray[np.float64_t, ndim=1] bsource,
                  np.ndarray[np.float64_t, ndim=1] wresult,
                  np.ndarray[np.float64_t, ndim=1] bresult,
                  np.ndarray[np.float64_t, ndim=1] mresult,
                  np.ndarray[np.float64_t, ndim=1] qresult,
                  np.ndarray[np.float64_t, ndim=1] used):
    cdef int n, bin
    cdef np.float64_t wval, bval
    for n in range(bins_x.shape[0]):
        bin = bins_x[n]
        bval = bsource[n]
        wval = wsource[n]
        qresult[bin] += (wresult[bin] * wval * (bval - mresult[bin])**2) / \
            (wresult[bin] + wval)
        wresult[bin] += wval
        bresult[bin] += wval*bval
        mresult[bin] += wval * (bval - mresult[bin]) / wresult[bin]
        used[bin] = 1
    return

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def bin_profile2d(np.ndarray[np.int64_t, ndim=1] bins_x,
                  np.ndarray[np.int64_t, ndim=1] bins_y,
                  np.ndarray[np.float64_t, ndim=1] wsource,
                  np.ndarray[np.float64_t, ndim=1] bsource,
                  np.ndarray[np.float64_t, ndim=2] wresult,
                  np.ndarray[np.float64_t, ndim=2] bresult,
                  np.ndarray[np.float64_t, ndim=2] mresult,
                  np.ndarray[np.float64_t, ndim=2] qresult,
                  np.ndarray[np.float64_t, ndim=2] used):
    cdef int n, bini, binj
    cdef np.int64_t bin
    cdef np.float64_t wval, bval
    for n in range(bins_x.shape[0]):
        bini = bins_x[n]
        binj = bins_y[n]
        bval = bsource[n]
        wval = wsource[n]
        qresult[bini, binj] += (wresult[bini, binj] * wval * (bval - mresult[bini, binj])**2) / \
            (wresult[bini, binj] + wval)
        wresult[bini, binj] += wval
        bresult[bini, binj] += wval*bval
        mresult[bini, binj] += wval * (bval - mresult[bini, binj]) / wresult[bini, binj]
        used[bini, binj] = 1
    return

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def bin_profile3d(np.ndarray[np.int64_t, ndim=1] bins_x,
                  np.ndarray[np.int64_t, ndim=1] bins_y,
                  np.ndarray[np.int64_t, ndim=1] bins_z,
                  np.ndarray[np.float64_t, ndim=1] wsource,
                  np.ndarray[np.float64_t, ndim=1] bsource,
                  np.ndarray[np.float64_t, ndim=3] wresult,
                  np.ndarray[np.float64_t, ndim=3] bresult,
                  np.ndarray[np.float64_t, ndim=3] mresult,
                  np.ndarray[np.float64_t, ndim=3] qresult,
                  np.ndarray[np.float64_t, ndim=3] used):
    cdef int n, bini, binj, bink
    cdef np.int64_t bin
    cdef np.float64_t wval, bval
    for n in range(bins_x.shape[0]):
        bini = bins_x[n]
        binj = bins_y[n]
        bink = bins_z[n]
        bval = bsource[n]
        wval = wsource[n]
        qresult[bini, binj, bink] += (wresult[bini, binj, bink] * wval * (bval - mresult[bini, binj, bink])**2) / \
            (wresult[bini, binj, bink] + wval)
        wresult[bini, binj, bink] += wval
        bresult[bini, binj, bink] += wval*bval
        mresult[bini, binj, bink] += wval * (bval - mresult[bini, binj, bink]) / wresult[bini, binj, bink]
        used[bini, binj, bink] = 1
    return

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lines(np.ndarray[np.float64_t, ndim=3] image,
          np.ndarray[np.int64_t, ndim=1] xs,
          np.ndarray[np.int64_t, ndim=1] ys,
          np.ndarray[np.float64_t, ndim=2] colors,
          int points_per_color=1,
          int thick=1,
	  int flip=0):

    cdef int nx = image.shape[0]
    cdef int ny = image.shape[1]
    cdef int nl = xs.shape[0]
    cdef np.float64_t alpha[4], outa
    cdef int i, j
    cdef int dx, dy, sx, sy, e2, err
    cdef np.int64_t x0, x1, y0, y1
    cdef int has_alpha = (image.shape[2] == 4)
    for j in range(0, nl, 2):
        # From wikipedia http://en.wikipedia.org/wiki/Bresenham's_line_algorithm
        x0 = xs[j]; y0 = ys[j]; x1 = xs[j+1]; y1 = ys[j+1]
        dx = abs(x1-x0)
        dy = abs(y1-y0)
        err = dx - dy
        if has_alpha:
            for i in range(4):
                alpha[i] = colors[j/points_per_color,i]
        else:
            for i in range(3):
                alpha[i] = colors[j/points_per_color,3]*\
                        colors[j/points_per_color,i]
        if x0 < x1:
            sx = 1
        else:
            sx = -1
        if y0 < y1:
            sy = 1
        else:
            sy = -1
        while(1):
            if (x0 < thick and sx == -1): break
            elif (x0 >= nx-thick+1 and sx == 1): break
            elif (y0 < thick and sy == -1): break
            elif (y0 >= ny-thick+1 and sy == 1): break
            if x0 >= thick and x0 < nx-thick and y0 >= thick and y0 < ny-thick:
                for xi in range(x0-thick/2, x0+(1+thick)/2):
                    for yi in range(y0-thick/2, y0+(1+thick)/2):
                        if flip: 
                            yi0 = ny - yi
                        else:
                            yi0 = yi

                        if has_alpha:
                            image[xi, yi0, 3] = outa = alpha[3] + image[xi, yi0, 3]*(1-alpha[3])
                            if outa != 0.0:
                                outa = 1.0/outa
                            for i in range(3):
                                image[xi, yi0, i] = \
                                        ((1.-alpha[3])*image[xi, yi0, i]*image[xi, yi0, 3]
                                         + alpha[3]*alpha[i])*outa
                        else:
                            for i in range(3):
                                image[xi, yi0, i] = \
                                        (1.-alpha[i])*image[xi,yi0,i] + alpha[i]

            if (x0 == x1 and y0 == y1):
                break
            e2 = 2*err
            if e2 > -dy:
                err = err - dy
                x0 += sx
            if e2 < dx :
                err = err + dx
                y0 += sy
    return

def rotate_vectors(np.ndarray[np.float64_t, ndim=3] vecs,
        np.ndarray[np.float64_t, ndim=2] R):
    cdef int nx = vecs.shape[0]
    cdef int ny = vecs.shape[1]
    rotated = np.empty((nx,ny,3),dtype='float64')
    for i in range(nx):
        for j in range(ny):
            for k in range(3):
                rotated[i,j,k] =\
                    R[k,0]*vecs[i,j,0]+R[k,1]*vecs[i,j,1]+R[k,2]*vecs[i,j,2]
    return rotated

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def get_color_bounds(np.ndarray[np.float64_t, ndim=1] px,
                     np.ndarray[np.float64_t, ndim=1] py,
                     np.ndarray[np.float64_t, ndim=1] pdx,
                     np.ndarray[np.float64_t, ndim=1] pdy,
                     np.ndarray[np.float64_t, ndim=1] value,
                     np.float64_t leftx, np.float64_t rightx,
                     np.float64_t lefty, np.float64_t righty,
                     np.float64_t mindx = -1, np.float64_t maxdx = -1):
    cdef int i
    cdef np.float64_t mi = 1e100, ma = -1e100, v
    cdef int np = px.shape[0]
    with nogil:
        for i in range(np):
            v = value[i]
            if v < mi or v > ma:
                if px[i] + pdx[i] < leftx: continue
                if px[i] - pdx[i] > rightx: continue
                if py[i] + pdy[i] < lefty: continue
                if py[i] - pdy[i] > righty: continue
                if pdx[i] < mindx or pdy[i] < mindx: continue
                if maxdx > 0 and (pdx[i] > maxdx or pdy[i] > maxdx): continue
                if v < mi: mi = v
                if v > ma: ma = v
    return (mi, ma)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def kdtree_get_choices(np.ndarray[np.float64_t, ndim=3] data,
                       np.ndarray[np.float64_t, ndim=1] l_corner,
                       np.ndarray[np.float64_t, ndim=1] r_corner):
    cdef int i, j, k, dim, n_unique, best_dim, n_best, n_grids, addit, my_split
    n_grids = data.shape[0]
    cdef np.float64_t **uniquedims, *uniques, split
    uniquedims = <np.float64_t **> alloca(3 * sizeof(np.float64_t*))
    for i in range(3):
        uniquedims[i] = <np.float64_t *> \
                alloca(2*n_grids * sizeof(np.float64_t))
    my_max = 0
    best_dim = -1
    for dim in range(3):
        n_unique = 0
        uniques = uniquedims[dim]
        for i in range(n_grids):
            # Check for disqualification
            for j in range(2):
                #print "Checking against", i,j,dim,data[i,j,dim]
                if not (l_corner[dim] < data[i, j, dim] and
                        data[i, j, dim] < r_corner[dim]):
                    #print "Skipping ", data[i,j,dim]
                    continue
                skipit = 0
                # Add our left ...
                for k in range(n_unique):
                    if uniques[k] == data[i, j, dim]:
                        skipit = 1
                        #print "Identified", uniques[k], data[i,j,dim], n_unique
                        break
                if skipit == 0:
                    uniques[n_unique] = data[i, j, dim]
                    n_unique += 1
        if n_unique > my_max:
            best_dim = dim
            my_max = n_unique
            my_split = (n_unique-1)/2
    # I recognize how lame this is.
    cdef np.ndarray[np.float64_t, ndim=1] tarr = np.empty(my_max, dtype='float64')
    for i in range(my_max):
        #print "Setting tarr: ", i, uniquedims[best_dim][i]
        tarr[i] = uniquedims[best_dim][i]
    tarr.sort()
    split = tarr[my_split]
    cdef np.ndarray[np.uint8_t, ndim=1] less_ids = np.empty(n_grids, dtype='uint8')
    cdef np.ndarray[np.uint8_t, ndim=1] greater_ids = np.empty(n_grids, dtype='uint8')
    for i in range(n_grids):
        if data[i, 0, best_dim] < split:
            less_ids[i] = 1
        else:
            less_ids[i] = 0
        if data[i, 1, best_dim] > split:
            greater_ids[i] = 1
        else:
            greater_ids[i] = 0
    # Return out unique values
    return best_dim, split, less_ids.view("bool"), greater_ids.view("bool")

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def get_box_grids_level(np.ndarray[np.float64_t, ndim=1] left_edge,
                        np.ndarray[np.float64_t, ndim=1] right_edge,
                        int level,
                        np.ndarray[np.float64_t, ndim=2] left_edges,
                        np.ndarray[np.float64_t, ndim=2] right_edges,
                        np.ndarray[np.int32_t, ndim=2] levels,
                        np.ndarray[np.int32_t, ndim=1] mask,
                        int min_index = 0):
    cdef int i, n
    cdef int nx = left_edges.shape[0]
    cdef int inside
    cdef np.float64_t eps = np.finfo(np.float64).eps
    for i in range(nx):
        if i < min_index or levels[i,0] != level:
            mask[i] = 0
            continue
        inside = 1
        for n in range(3):
            if (right_edges[i,n] - left_edge[n]) <= eps or \
               (right_edge[n] - left_edges[i,n]) <= eps:
                inside = 0
                break
        if inside == 1: mask[i] = 1
        else: mask[i] = 0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def get_box_grids_below_level(
                        np.ndarray[np.float64_t, ndim=1] left_edge,
                        np.ndarray[np.float64_t, ndim=1] right_edge,
                        int level,
                        np.ndarray[np.float64_t, ndim=2] left_edges,
                        np.ndarray[np.float64_t, ndim=2] right_edges,
                        np.ndarray[np.int32_t, ndim=2] levels,
                        np.ndarray[np.int32_t, ndim=1] mask,
                        int min_level = 0):
    cdef int i, n
    cdef int nx = left_edges.shape[0]
    cdef int inside
    cdef np.float64_t eps = np.finfo(np.float64).eps
    for i in range(nx):
        mask[i] = 0
        if levels[i,0] <= level and levels[i,0] >= min_level:
            inside = 1
            for n in range(3):
                if (right_edges[i,n] - left_edge[n]) <= eps or \
                   (right_edge[n] - left_edges[i,n]) <= eps:
                    inside = 0
                    break
            if inside == 1: mask[i] = 1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def find_values_at_point(np.ndarray[np.float64_t, ndim=1] point,
                         np.ndarray[np.float64_t, ndim=2] left_edges,
                         np.ndarray[np.float64_t, ndim=2] right_edges,
                         np.ndarray[np.int32_t, ndim=2] dimensions,
                         field_names, grid_objects):
    # This iterates in order, first to last, and then returns with the first
    # one in which the point is located; this means if you order from highest
    # level to lowest, you will find the correct grid without consulting child
    # masking.  Note also that we will do a few relatively slow operations on
    # strings and whatnot, but they should not be terribly slow.
    cdef int ind[3], gi, fi
    cdef int nf = len(field_names)
    cdef np.float64_t dds
    cdef np.ndarray[np.float64_t, ndim=3] field
    cdef np.ndarray[np.float64_t, ndim=1] rv = np.zeros(nf, dtype='float64')
    for gi in range(left_edges.shape[0]):
        if not ((left_edges[gi,0] < point[0] < right_edges[gi,0])
            and (left_edges[gi,1] < point[1] < right_edges[gi,1])
            and (left_edges[gi,2] < point[2] < right_edges[gi,2])):
            continue
        # We found our grid!
        for fi in range(3):
            dds = ((right_edges[gi,fi] - left_edges[gi,fi])/
                   (<np.float64_t> dimensions[gi,fi]))
            ind[fi] = <int> ((point[fi] - left_edges[gi,fi])/dds)
        grid = grid_objects[gi]
        for fi in range(nf):
            field = grid[field_names[fi]]
            rv[fi] = field[ind[0], ind[1], ind[2]]
        return rv
    raise KeyError

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def pixelize_cylinder(np.ndarray[np.float64_t, ndim=1] radius,
                      np.ndarray[np.float64_t, ndim=1] dradius,
                      np.ndarray[np.float64_t, ndim=1] theta,
                      np.ndarray[np.float64_t, ndim=1] dtheta,
                      buff_size,
                      np.ndarray[np.float64_t, ndim=1] field,
                      extents, input_img = None):

    cdef np.ndarray[np.float64_t, ndim=2] img
    cdef np.float64_t x, y, dx, dy, r0, theta0
    cdef np.float64_t rmax, x0, y0, x1, y1
    cdef np.float64_t r_i, theta_i, dr_i, dtheta_i, dthetamin
    cdef np.float64_t costheta, sintheta
    cdef int i, pi, pj
    
    imax = radius.argmax()
    rmax = radius[imax] + dradius[imax]
          
    if input_img is None:
        img = np.zeros((buff_size[0], buff_size[1]))
        img[:] = np.nan
    else:
        img = input_img
    x0, x1, y0, y1 = extents
    dx = (x1 - x0) / img.shape[0]
    dy = (y1 - y0) / img.shape[1]
    cdef np.float64_t rbounds[2]
    cdef np.float64_t corners[8]
    # Find our min and max r
    corners[0] = x0*x0+y0*y0
    corners[1] = x1*x1+y0*y0
    corners[2] = x0*x0+y1*y1
    corners[3] = x1*x1+y1*y1
    corners[4] = x0*x0
    corners[5] = x1*x1
    corners[6] = y0*y0
    corners[7] = y1*y1
    rbounds[0] = rbounds[1] = corners[0]
    for i in range(8):
        rbounds[0] = fmin(rbounds[0], corners[i])
        rbounds[1] = fmax(rbounds[1], corners[i])
    rbounds[0] = rbounds[0]**0.5
    rbounds[1] = rbounds[1]**0.5
    # If we include the origin in either direction, we need to have radius of
    # zero as our lower bound.
    if x0 < 0 and x1 > 0:
        rbounds[0] = 0.0
    if y0 < 0 and y1 > 0:
        rbounds[0] = 0.0
    dthetamin = dx / rmax
    for i in range(radius.shape[0]):

        r0 = radius[i]
        theta0 = theta[i]
        dr_i = dradius[i]
        dtheta_i = dtheta[i]
        # Skip out early if we're offsides, for zoomed in plots
        if r0 + dr_i < rbounds[0] or r0 - dr_i > rbounds[1]:
            continue
        theta_i = theta0 - dtheta_i
        # Buffer of 0.5 here
        dthetamin = 0.5*dx/(r0 + dr_i)
        while theta_i < theta0 + dtheta_i:
            r_i = r0 - dr_i
            costheta = math.cos(theta_i)
            sintheta = math.sin(theta_i)
            while r_i < r0 + dr_i:
                if rmax <= r_i:
                    r_i += 0.5*dx 
                    continue
                y = r_i * costheta
                x = r_i * sintheta
                pi = <int>((x - x0)/dx)
                pj = <int>((y - y0)/dy)
                if pi >= 0 and pi < img.shape[0] and \
                   pj >= 0 and pj < img.shape[1]:
                    if img[pi, pj] != img[pi, pj]:
                        img[pi, pj] = 0.0
                    img[pi, pj] = field[i]
                r_i += 0.5*dx 
            theta_i += dthetamin

    return img

cdef void aitoff_thetaphi_to_xy(np.float64_t theta, np.float64_t phi,
                                np.float64_t *x, np.float64_t *y):
    cdef np.float64_t z = math.sqrt(1 + math.cos(phi) * math.cos(theta / 2.0))
    x[0] = math.cos(phi) * math.sin(theta / 2.0) / z
    y[0] = math.sin(phi) / z

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def pixelize_aitoff(np.ndarray[np.float64_t, ndim=1] theta,
                    np.ndarray[np.float64_t, ndim=1] dtheta,
                    np.ndarray[np.float64_t, ndim=1] phi,
                    np.ndarray[np.float64_t, ndim=1] dphi,
                    buff_size,
                    np.ndarray[np.float64_t, ndim=1] field,
                    extents, input_img = None):
    # http://paulbourke.net/geometry/transformationprojection/
    # longitude is -pi to pi
    # latitude is -pi/2 to pi/2
    # z^2 = 1 + cos(latitude) cos(longitude/2)
    # x = cos(latitude) sin(longitude/2) / z
    # y = sin(latitude) / z
    cdef np.ndarray[np.float64_t, ndim=2] img
    cdef int i, j, nf, fi
    cdef np.float64_t x, y, z, zb
    cdef np.float64_t dx, dy, inside
    cdef np.float64_t theta1, dtheta1, phi1, dphi1
    cdef np.float64_t theta0, phi0, theta_p, dtheta_p, phi_p, dphi_p
    cdef np.float64_t PI = np.pi
    cdef np.float64_t s2 = math.sqrt(2.0)
    cdef np.float64_t xmax, ymax, xmin, ymin
    nf = field.shape[0]
    
    if input_img is None:
        img = np.zeros((buff_size[0], buff_size[1]))
        img[:] = np.nan
    else:
        img = input_img
    # Okay, here's our strategy.  We compute the bounds in x and y, which will
    # be a rectangle, and then for each x, y position we check to see if it's
    # within our theta.  This will cost *more* computations of the
    # (x,y)->(theta,phi) calculation, but because we no longer have to search
    # through the theta, phi arrays, it should be faster.
    dx = 2.0 / (img.shape[0] - 1)
    dy = 2.0 / (img.shape[1] - 1)
    for fi in range(nf):
        theta_p = theta[fi] - PI
        dtheta_p = dtheta[fi]
        phi_p = phi[fi] - PI/2.0
        dphi_p = dphi[fi]
        # Four transformations
        aitoff_thetaphi_to_xy(theta_p - dtheta_p, phi_p - dphi_p, &x, &y)
        xmin = x
        xmax = x
        ymin = y
        ymax = y
        aitoff_thetaphi_to_xy(theta_p - dtheta_p, phi_p + dphi_p, &x, &y)
        xmin = fmin(xmin, x)
        xmax = fmax(xmax, x)
        ymin = fmin(ymin, y)
        ymax = fmax(ymax, y)
        aitoff_thetaphi_to_xy(theta_p + dtheta_p, phi_p - dphi_p, &x, &y)
        xmin = fmin(xmin, x)
        xmax = fmax(xmax, x)
        ymin = fmin(ymin, y)
        ymax = fmax(ymax, y)
        aitoff_thetaphi_to_xy(theta_p + dtheta_p, phi_p + dphi_p, &x, &y)
        xmin = fmin(xmin, x)
        xmax = fmax(xmax, x)
        ymin = fmin(ymin, y)
        ymax = fmax(ymax, y)
        # Now we have the (projected rectangular) bounds.
        xmin = (xmin + 1) # Get this into normalized image coords
        xmax = (xmax + 1) # Get this into normalized image coords
        ymin = (ymin + 1) # Get this into normalized image coords
        ymax = (ymax + 1) # Get this into normalized image coords
        x0 = <int> (xmin / dx)
        x1 = <int> (xmax / dx) + 1
        y0 = <int> (ymin / dy)
        y1 = <int> (ymax / dy) + 1
        for i in range(x0, x1):
            x = (-1.0 + i*dx)*s2*2.0
            for j in range(y0, y1):
                y = (-1.0 + j * dy)*s2
                zb = (x*x/8.0 + y*y/2.0 - 1.0)
                if zb > 0: continue
                z = (1.0 - (x/4.0)**2.0 - (y/2.0)**2.0)
                z = z**0.5
                # Longitude
                theta0 = 2.0*math.atan(z*x/(2.0 * (2.0*z*z-1.0)))
                # Latitude
                # We shift it into co-latitude
                phi0 = math.asin(z*y)
                # Now we just need to figure out which pixel contributes.
                # We do not have a fast search.
                if not (theta_p - dtheta_p <= theta0 <= theta_p + dtheta_p):
                    continue
                if not (phi_p - dphi_p <= phi0 <= phi_p + dphi_p):
                    continue
                img[i, j] = field[fi]
    return img

# Six faces, two vectors for each, two indices for each vector.  The function
# below unrolls how these are defined.  Order is bottom, front, right, left,
# back, top.
# This is [6][2][2]
cdef np.uint8_t ***face_defs = [
               [[0, 1], [2, 1]],
               [[0, 1], [5, 1]],
               [[2, 1], [5, 1]],
               [[4, 0], [3, 0]],
               [[6, 2], [3, 2]],
               [[4, 5], [6, 5]]
]

# This function accepts a set of eight vertices (for a hexahedron) that are
# assumed to be in order for bottom, then top, in the same clockwise or
# counterclockwise direction (i.e., like points 1-8 in Figure 4 of the ExodusII
# manual).  It will then either *match* or *fill* the results.  If it is
# matching, it will early terminate with a 0 or final-terminate with a 1 if the
# results match.  Otherwise, it will fill the signs with -1's and 1's to show
# the sign of the dot product of the point with the cross product of the face.
cdef int check_face_dot(np.float64_t point[3],
                        np.float64_t vertices[8][3],
                        np.int8_t signs[6],
                        int match):
    # Because of how we are doing this, we do not *care* what the signs are or
    # how the faces are ordered, we only care if they match between the point
    # and the centroid.
    # The faces are defined by vectors spanned (1-indexed):
    #   Bottom: 2->1, 2->3 (i.e., p1-p2, p3-p2)
    #   Front:  2->1, 2->6 (i.e., p1-p2, p6-p2)
    #   Right:  2->3, 2->6 (i.e., p3-p2, p6-p2)
    #   Left:   1->5, 1->4 (i.e., p5-p1, p4-p1)
    #   Back:   3->7, 3->4 (i.e., p7-p3, p4-p3)
    #   Top:    6->5, 6->7 (i.e., p5-p6, p7-p6)
    # So, let's compute these vectors.  See above where these are written out
    # for ease of use.
    cdef np.float64_t vec1[3], vec2[3], cp_vec[3], dp
    cdef int i, j, n, vi1a, vi1b, vi2a, vi2b
    for n in range(6):
        vi1a = face_defs[n][0][0]
        vi1b = face_defs[n][0][1]
        vi2a = face_defs[n][1][0]
        vi2b = face_defs[n][1][1]
        for i in range(3):
            vec1[i] = vertices[vi1b][i] - vertices[vi1a][i]
            vec2[i] = vertices[vi2b][i] - vertices[vi2a][i]
        # Now the cross product of vec1 x vec2
        cp_vec[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1]
        cp_vec[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2]
        cp_vec[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        dp = 0.0
        for j in range(3):
            dp += cp_vec[j] * point[j]
        if match == 0:
            if dp < 0:
                signs[n] = -1
            else:
                signs[n] = 1
        else:
            if dp < 0 and signs[n] < 0:
                continue
            elif dp > 0 and signs[n] > 0:
                continue
            else: # mismatch!
                return 0
    return 1

def pixelize_hex_mesh(np.ndarray[np.float64_t, ndim=2] coords,
                      np.ndarray[np.int64_t, ndim=2] conn,
                      buff_size,
                      np.ndarray[np.float64_t, ndim=1] field,
                      extents, int index_offset = 0):
    cdef np.ndarray[np.float64_t, ndim=3] img
    img = np.zeros(buff_size, dtype="float64")
    # Two steps:
    #  1. Is image point within the mesh bounding box?
    #  2. Is image point within the mesh element?
    # Second is more intensive.  It will require a bunch of dot and cross
    # products.  We are not guaranteed that the elements will be in the correct
    # order such that cross products are pointing in right direction, so we
    # compare against the centroid of the (assumed convex) element.
    # Note that we have to have a pseudo-3D pixel buffer.  One dimension will
    # always be 1.
    cdef np.float64_t pLE[3], pRE[3]
    cdef np.float64_t LE[3], RE[3]
    cdef int use
    cdef np.int8_t signs[6]
    cdef np.int64_t n, i, j, k, pi, pj, pk, ci, cj, ck
    cdef np.int64_t pstart[3], pend[3]
    cdef np.float64_t ppoint[3], centroid[3], idds[3], dds[3]
    cdef np.float64_t vertices[8][3]
    if conn.shape[1] != 8: # Hexahedral meshes must have 8 vertices
        raise RuntimeError
    for i in range(3):
        pLE[i] = extents[i][0]
        pRE[i] = extents[i][1]
        dds[i] = (pRE[i] - pLE[i])/buff_size[i]
        if dds[i] == 0.0:
            idds[i] = 0.0
        else:
            idds[i] = 1.0 / dds[i]
    for ci in range(conn.shape[0]):
        # Fill the vertices and compute the centroid
        centroid[0] = centroid[1] = centroid[2] = 0
        LE[0] = LE[1] = LE[2] = 1e60
        RE[0] = RE[1] = RE[2] = -1e60
        for n in range(conn.shape[1]): # 8
            cj = conn[ci, n] - index_offset
            for i in range(3):
                vertices[n][i] = coords[cj, i]
                centroid[i] += coords[cj, i]
                LE[i] = fmin(LE[i], vertices[n][i])
                RE[i] = fmax(RE[i], vertices[n][i])
        centroid[0] /= conn.shape[1]
        centroid[1] /= conn.shape[1]
        centroid[2] /= conn.shape[1]
        use = 1
        for i in range(3):
            if RE[i] < pLE[i] or LE[i] >= pRE[i]:
                use = 0
                break
            pstart[i] = i64max(<np.int64_t> ((LE[i] - pLE[i])*idds[i]), 0)
            pend[i] = i64min(<np.int64_t> ((RE[i] - pLE[i])*idds[i]), img.shape[i]-1)
        if use == 0:
            continue
        # Now our bounding box intersects, so we get the extents of our pixel
        # region which overlaps with the bounding box, and we'll check each
        # pixel in there.
        # First, we figure out the dot product of the centroid with all the
        # faces.
        check_face_dot(centroid, vertices, signs, 0)
        for pi in range(pstart[0], pend[0] + 1):
            ppoint[0] = pi * dds[0] + pLE[0]
            for pj in range(pstart[1], pend[1] + 1):
                ppoint[1] = pj * dds[1] + pLE[1]
                for pk in range(pstart[2], pend[2] + 1):
                    ppoint[2] = pk * dds[2] + pLE[2]
                    # Now we just need to figure out if our ppoint is within
                    # our set of vertices.
                    if check_face_dot(ppoint, vertices, signs, 1) == 0:
                        continue
                    # Else, we deposit!
                    img[pi, pj, pk] = field[ci]
    return img

#@cython.cdivision(True)
#@cython.boundscheck(False)
#@cython.wraparound(False)
def obtain_rvec(data):
    # This is just to let the pointers exist and whatnot.  We can't cdef them
    # inside conditionals.
    cdef np.ndarray[np.float64_t, ndim=1] xf
    cdef np.ndarray[np.float64_t, ndim=1] yf
    cdef np.ndarray[np.float64_t, ndim=1] zf
    cdef np.ndarray[np.float64_t, ndim=2] rf
    cdef np.ndarray[np.float64_t, ndim=3] xg
    cdef np.ndarray[np.float64_t, ndim=3] yg
    cdef np.ndarray[np.float64_t, ndim=3] zg
    cdef np.ndarray[np.float64_t, ndim=4] rg
    cdef np.float64_t c[3]
    cdef int i, j, k
    center = data.get_field_parameter("center")
    c[0] = center[0]; c[1] = center[1]; c[2] = center[2]
    if len(data['x'].shape) == 1:
        # One dimensional data
        xf = data['x']
        yf = data['y']
        zf = data['z']
        rf = YTArray(np.empty((3, xf.shape[0]), 'float64'), xf.units)
        for i in range(xf.shape[0]):
            rf[0, i] = xf[i] - c[0]
            rf[1, i] = yf[i] - c[1]
            rf[2, i] = zf[i] - c[2]
        return rf
    else:
        # Three dimensional data
        xg = data['x']
        yg = data['y']
        zg = data['z']
        shape = (3, xg.shape[0], xg.shape[1], xg.shape[2])
        rg = YTArray(np.empty(shape, 'float64'), xg.units)
        #rg = YTArray(rg, xg.units)
        for i in range(xg.shape[0]):
            for j in range(xg.shape[1]):
                for k in range(xg.shape[2]):
                    rg[0,i,j,k] = xg[i,j,k] - c[0]
                    rg[1,i,j,k] = yg[i,j,k] - c[1]
                    rg[2,i,j,k] = zg[i,j,k] - c[2]
        return rg

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def obtain_rv_vec(data, field_names = ("velocity_x",
                                       "velocity_y",
                                       "velocity_z"),
                  bulk_vector = "bulk_velocity"):
    # This is just to let the pointers exist and whatnot.  We can't cdef them
    # inside conditionals.
    cdef np.ndarray[np.float64_t, ndim=1] vxf
    cdef np.ndarray[np.float64_t, ndim=1] vyf
    cdef np.ndarray[np.float64_t, ndim=1] vzf
    cdef np.ndarray[np.float64_t, ndim=2] rvf
    cdef np.ndarray[np.float64_t, ndim=3] vxg
    cdef np.ndarray[np.float64_t, ndim=3] vyg
    cdef np.ndarray[np.float64_t, ndim=3] vzg
    cdef np.ndarray[np.float64_t, ndim=4] rvg
    cdef np.float64_t bv[3]
    cdef int i, j, k
    bulk_vector = data.get_field_parameter(bulk_vector)
    if len(data[field_names[0]].shape) == 1:
        # One dimensional data
        vxf = data[field_names[0]].astype("float64")
        vyf = data[field_names[1]].astype("float64")
        vzf = data[field_names[2]].astype("float64")
        vyf.convert_to_units(vxf.units)
        vzf.convert_to_units(vxf.units)
        rvf = YTArray(np.empty((3, vxf.shape[0]), 'float64'), vxf.units)
        if bulk_vector is None:
            bv[0] = bv[1] = bv[2] = 0.0
        else:
            bulk_vector = bulk_vector.in_units(vxf.units)
            bv[0] = bulk_vector[0]
            bv[1] = bulk_vector[1]
            bv[2] = bulk_vector[2]
        for i in range(vxf.shape[0]):
            rvf[0, i] = vxf[i] - bv[0]
            rvf[1, i] = vyf[i] - bv[1]
            rvf[2, i] = vzf[i] - bv[2]
        return rvf
    else:
        # Three dimensional data
        vxg = data[field_names[0]].astype("float64")
        vyg = data[field_names[1]].astype("float64")
        vzg = data[field_names[2]].astype("float64")
        vyg.convert_to_units(vxg.units)
        vzg.convert_to_units(vxg.units)
        shape = (3, vxg.shape[0], vxg.shape[1], vxg.shape[2])
        rvg = YTArray(np.empty(shape, 'float64'), vxg.units)
        if bulk_vector is None:
            bv[0] = bv[1] = bv[2] = 0.0
        else:
            bulk_vector = bulk_vector.in_units(vxg.units)
            bv[0] = bulk_vector[0]
            bv[1] = bulk_vector[1]
            bv[2] = bulk_vector[2]
        for i in range(vxg.shape[0]):
            for j in range(vxg.shape[1]):
                for k in range(vxg.shape[2]):
                    rvg[0,i,j,k] = vxg[i,j,k] - bv[0]
                    rvg[1,i,j,k] = vyg[i,j,k] - bv[1]
                    rvg[2,i,j,k] = vzg[i,j,k] - bv[2]
        return rvg

def grow_flagging_field(oofield):
    cdef np.ndarray[np.uint8_t, ndim=3] ofield = oofield.astype("uint8")
    cdef np.ndarray[np.uint8_t, ndim=3] nfield
    nfield = np.zeros_like(ofield)
    cdef int i, j, k, ni, nj, nk
    cdef int oi, oj, ok
    for ni in range(ofield.shape[0]):
        for nj in range(ofield.shape[1]):
            for nk in range(ofield.shape[2]):
                for oi in range(3):
                    i = ni + (oi - 1)
                    if i < 0 or i >= ofield.shape[0]: continue
                    for oj in range(3):
                        j = nj + (oj - 1)
                        if j < 0 or j >= ofield.shape[1]: continue
                        for ok in range(3):
                            k = nk + (ok - 1)
                            if k < 0 or k >= ofield.shape[2]: continue
                            if ofield[i, j, k] == 1:
                                nfield[ni, nj, nk] = 1
    return nfield.astype("bool")

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def fill_region(input_fields, output_fields,
                np.int32_t output_level,
                np.ndarray[np.int64_t, ndim=1] left_index,
                np.ndarray[np.int64_t, ndim=2] ipos,
                np.ndarray[np.int64_t, ndim=1] ires,
                np.ndarray[np.int64_t, ndim=1] level_dims,
                np.int64_t refine_by = 2
                ):
    cdef int i, n
    cdef np.int64_t tot
    cdef np.int64_t iind[3], oind[3], dim[3], oi, oj, ok, rf
    cdef np.ndarray[np.float64_t, ndim=3] ofield
    cdef np.ndarray[np.float64_t, ndim=1] ifield
    nf = len(input_fields)
    # The variable offsets governs for each dimension and each possible
    # wrapping if we do it.  Then the wi, wj, wk indices check into each
    # [dim][wrap] inside the loops.
    cdef int offsets[3][3], wi, wj, wk
    cdef np.int64_t off
    for i in range(3):
        dim[i] = output_fields[0].shape[i]
        offsets[i][0] = offsets[i][2] = 0
        offsets[i][1] = 1
        if left_index[i] < 0:
            offsets[i][2] = 1
        if left_index[i] + dim[i] >= level_dims[i]:
            offsets[i][0] = 1
    for n in range(nf):
        tot = 0
        ofield = output_fields[n]
        ifield = input_fields[n]
        for i in range(ipos.shape[0]):
            rf = refine_by**(output_level - ires[i]) 
            for wi in range(3):
                if offsets[0][wi] == 0: continue
                off = (left_index[0] + level_dims[0]*(wi-1))
                iind[0] = ipos[i, 0] * rf - off
                for oi in range(rf):
                    # Now we need to apply our offset
                    oind[0] = oi + iind[0]
                    if oind[0] < 0 or oind[0] >= dim[0]:
                        continue
                    for wj in range(3):
                        if offsets[1][wj] == 0: continue
                        off = (left_index[1] + level_dims[1]*(wj-1))
                        iind[1] = ipos[i, 1] * rf - off
                        for oj in range(rf):
                            oind[1] = oj + iind[1]
                            if oind[1] < 0 or oind[1] >= dim[1]:
                                continue
                            for wk in range(3):
                                if offsets[2][wk] == 0: continue
                                off = (left_index[2] + level_dims[2]*(wk-1))
                                iind[2] = ipos[i, 2] * rf - off
                                for ok in range(rf):
                                    oind[2] = ok + iind[2]
                                    if oind[2] < 0 or oind[2] >= dim[2]:
                                        continue
                                    ofield[oind[0], oind[1], oind[2]] = \
                                        ifield[i]
                                    tot += 1
    return tot
