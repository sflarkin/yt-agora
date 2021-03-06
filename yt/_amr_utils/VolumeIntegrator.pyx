"""
Simple integrators for the radiative transfer equation

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2009 Matthew Turk.  All Rights Reserved.

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

import numpy as np
cimport numpy as np
cimport cython
from stdlib cimport malloc, free, abs

cdef inline int imax(int i0, int i1):
    if i0 > i1: return i0
    return i1

cdef inline np.float64_t fmax(np.float64_t f0, np.float64_t f1):
    if f0 > f1: return f0
    return f1

cdef inline int imin(int i0, int i1):
    if i0 < i1: return i0
    return i1

cdef inline np.float64_t fmin(np.float64_t f0, np.float64_t f1):
    if f0 < f1: return f0
    return f1

cdef inline int iclip(int i, int a, int b):
    if i < a: return a
    if i > b: return b
    return i

cdef inline np.float64_t fclip(np.float64_t f,
                      np.float64_t a, np.float64_t b):
    return fmin(fmax(f, a), b)

cdef extern from "math.h":
    double exp(double x)
    float expf(float x)
    double floor(double x)
    double ceil(double x)
    double fmod(double x, double y)
    double log2(double x)
    long int lrint(double x)

cdef extern from "FixedInterpolator.h":
    np.float64_t fast_interpolate(int *ds, int *ci, np.float64_t *dp,
                                  np.float64_t *data)
cdef extern from "FixedInterpolator.h":
    np.float64_t trilinear_interpolate(int *ds, int *ci, np.float64_t *dp,
                                       np.float64_t *data)
    np.float64_t eval_gradient(int *ds, int *ci, np.float64_t *dp,
                                       np.float64_t *data, np.float64_t *grad)

cdef class VectorPlane

cdef class TransferFunctionProxy:
    cdef np.float64_t x_bounds[2]
    cdef np.float64_t *vs[4]
    cdef int nbins
    cdef public int ns
    cdef np.float64_t dbin, idbin
    cdef np.float64_t light_color[3]
    cdef np.float64_t light_dir[3]
    cdef int use_light
    cdef public object tf_obj
    def __cinit__(self, tf_obj):
        self.tf_obj = tf_obj
        cdef np.ndarray[np.float64_t, ndim=1] temp
        temp = tf_obj.red.y
        self.vs[0] = <np.float64_t *> temp.data
        temp = tf_obj.green.y
        self.vs[1] = <np.float64_t *> temp.data
        temp = tf_obj.blue.y
        self.vs[2] = <np.float64_t *> temp.data
        temp = tf_obj.alpha.y
        self.vs[3] = <np.float64_t *> temp.data
        self.x_bounds[0] = tf_obj.x_bounds[0]
        self.x_bounds[1] = tf_obj.x_bounds[1]
        self.nbins = tf_obj.nbins
        self.dbin = (self.x_bounds[1] - self.x_bounds[0])/self.nbins
        self.idbin = 1.0/self.dbin
        self.light_color[0] = tf_obj.light_color[0]
        self.light_color[1] = tf_obj.light_color[1]
        self.light_color[2] = tf_obj.light_color[2]
        self.light_dir[0] = tf_obj.light_dir[0]
        self.light_dir[1] = tf_obj.light_dir[1]
        self.light_dir[2] = tf_obj.light_dir[2]
        cdef np.float64_t normval = 0.0
        for i in range(3): normval += self.light_dir[i]**2
        normval = normval**0.5
        for i in range(3): self.light_dir[i] /= normval
        self.use_light = tf_obj.use_light

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void interpolate(self, np.float64_t dv, np.float64_t *trgba):
        cdef int bin_id, channel
        cdef np.float64_t bv, dy, dd, tf
        bin_id = <int> ((dv - self.x_bounds[0]) * self.idbin)
        # Recall that linear interpolation is y0 + (x-x0) * dx/dy
        dd = dv-(self.x_bounds[0] + bin_id * self.dbin) # x - x0
        if bin_id > self.nbins - 2 or bin_id < 0:
            for channel in range(4): trgba[channel] = 0.0
            return
        for channel in range(4):
            bv = self.vs[channel][bin_id] # This is x0
            dy = self.vs[channel][bin_id+1]-bv # dy
                # This is our final value for transfer function on the entering face
            trgba[channel] = bv+dd*dy*self.idbin

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void eval_transfer(self, np.float64_t dt, np.float64_t dv,
                                    np.float64_t *rgba, np.float64_t *grad):
        cdef int i
        cdef np.float64_t ta, tf, trgba[4], dot_prod
        self.interpolate(dv, trgba) 
        # get source alpha first
        # First locate our points
        dot_prod = 0.0
        if self.use_light:
            for i in range(3):
                dot_prod += self.light_dir[i] * grad[i]
            dot_prod = fmax(0.0, dot_prod)
            for i in range(3):
                trgba[i] += dot_prod*self.light_color[i]
        # alpha blending
        ta = (1.0 - rgba[3])*dt*trgba[3]
        for i in range(4):
            rgba[i] += ta*trgba[i]

cdef class VectorPlane:
    cdef public object avp_pos, avp_dir, acenter, aimage
    cdef np.float64_t *vp_pos, *vp_dir, *center, *image,
    cdef np.float64_t pdx, pdy, bounds[4]
    cdef int nv[2]
    cdef public object ax_vec, ay_vec
    cdef np.float64_t *x_vec, *y_vec

    def __cinit__(self, 
                  np.ndarray[np.float64_t, ndim=3] vp_pos,
                  np.ndarray[np.float64_t, ndim=1] vp_dir,
                  np.ndarray[np.float64_t, ndim=1] center,
                  bounds,
                  np.ndarray[np.float64_t, ndim=3] image,
                  np.ndarray[np.float64_t, ndim=1] x_vec,
                  np.ndarray[np.float64_t, ndim=1] y_vec):
        cdef int i, j
        self.avp_pos = vp_pos
        self.avp_dir = vp_dir
        self.acenter = center
        self.aimage = image
        self.ax_vec = x_vec
        self.ay_vec = y_vec
        self.vp_pos = <np.float64_t *> vp_pos.data
        self.vp_dir = <np.float64_t *> vp_dir.data
        self.center = <np.float64_t *> center.data
        self.image = <np.float64_t *> image.data
        self.x_vec = <np.float64_t *> x_vec.data
        self.y_vec = <np.float64_t *> y_vec.data
        self.nv[0] = vp_pos.shape[0]
        self.nv[1] = vp_pos.shape[1]
        for i in range(4): self.bounds[i] = bounds[i]
        self.pdx = (self.bounds[1] - self.bounds[0])/self.nv[0]
        self.pdy = (self.bounds[3] - self.bounds[2])/self.nv[1]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void get_start_stop(self, np.float64_t *ex, int *rv):
        # Extrema need to be re-centered
        cdef np.float64_t cx, cy
        cdef int i
        cx = cy = 0.0
        for i in range(3):
            cx += self.center[i] * self.x_vec[i]
            cy += self.center[i] * self.y_vec[i]
        rv[0] = lrint((ex[0] - cx - self.bounds[0])/self.pdx)
        rv[1] = rv[0] + lrint((ex[1] - ex[0])/self.pdx)
        rv[2] = lrint((ex[2] - cy - self.bounds[2])/self.pdy)
        rv[3] = rv[2] + lrint((ex[3] - ex[2])/self.pdy)

    cdef inline void copy_into(self, np.float64_t *fv, np.float64_t *tv,
                        int i, int j, int nk):
        # We know the first two dimensions of our from-vector, and our
        # to-vector is flat and 'ni' long
        cdef int k
        for k in range(nk):
            tv[k] = fv[(((k*self.nv[1])+j)*self.nv[0]+i)]

    cdef inline void copy_back(self, np.float64_t *fv, np.float64_t *tv,
                        int i, int j, int nk):
        cdef int k
        for k in range(nk):
            tv[(((k*self.nv[1])+j)*self.nv[0]+i)] = fv[k]

cdef class PartitionedGrid:
    cdef public object my_data
    cdef public object LeftEdge
    cdef public object RightEdge
    cdef np.float64_t *data
    cdef np.float64_t left_edge[3]
    cdef np.float64_t right_edge[3]
    cdef np.float64_t dds[3]
    cdef np.float64_t idds[3]
    cdef int dims[3]
    cdef public int parent_grid_id

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self,
                  int parent_grid_id,
                  np.ndarray[np.float64_t, ndim=3] data,
                  np.ndarray[np.float64_t, ndim=1] left_edge,
                  np.ndarray[np.float64_t, ndim=1] right_edge,
                  np.ndarray[np.int64_t, ndim=1] dims):
        # The data is likely brought in via a slice, so we copy it
        cdef int i, j, k, size
        self.parent_grid_id = parent_grid_id
        self.LeftEdge = left_edge
        self.RightEdge = right_edge
        for i in range(3):
            self.left_edge[i] = left_edge[i]
            self.right_edge[i] = right_edge[i]
            self.dims[i] = dims[i]
            self.dds[i] = (self.right_edge[i] - self.left_edge[i])/dims[i]
            self.idds[i] = 1.0/self.dds[i]
        self.my_data = data
        self.data = <np.float64_t*> data.data

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def cast_plane(self, TransferFunctionProxy tf, VectorPlane vp):
        # This routine will iterate over all of the vectors and cast each in
        # turn.  Might benefit from a more sophisticated intersection check,
        # like http://courses.csusm.edu/cs697exz/ray_box.htm
        cdef int vi, vj, hit, i, ni, nj, nn
        cdef int iter[4]
        cdef np.float64_t v_pos[3], v_dir[3], rgba[4], extrema[4]
        self.calculate_extent(vp, extrema)
        vp.get_start_stop(extrema, iter)
        iter[0] = iclip(iter[0], 0, vp.nv[0])
        iter[1] = iclip(iter[1], 0, vp.nv[0])
        iter[2] = iclip(iter[2], 0, vp.nv[1])
        iter[3] = iclip(iter[3], 0, vp.nv[1])
        hit = 0
        for vi in range(iter[0], iter[1]):
            for vj in range(iter[2], iter[3]):
                vp.copy_into(vp.vp_pos, v_pos, vi, vj, 3)
                vp.copy_into(vp.image, rgba, vi, vj, 4)
                self.integrate_ray(v_pos, vp.vp_dir, rgba, tf)
                vp.copy_back(rgba, vp.image, vi, vj, 4)
        return hit

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void calculate_extent(self, VectorPlane vp,
                               np.float64_t extrema[4]):
        # We do this for all eight corners
        cdef np.float64_t *edges[2], temp
        edges[0] = self.left_edge
        edges[1] = self.right_edge
        extrema[0] = extrema[2] = 1e300; extrema[1] = extrema[3] = -1e300
        cdef int i, j, k
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # This should rotate it into the vector plane
                    temp  = edges[i][0] * vp.x_vec[0]
                    temp += edges[j][1] * vp.x_vec[1]
                    temp += edges[k][2] * vp.x_vec[2]
                    if temp < extrema[0]: extrema[0] = temp
                    if temp > extrema[1]: extrema[1] = temp
                    temp  = edges[i][0] * vp.y_vec[0]
                    temp += edges[j][1] * vp.y_vec[1]
                    temp += edges[k][2] * vp.y_vec[2]
                    if temp < extrema[2]: extrema[2] = temp
                    if temp > extrema[3]: extrema[3] = temp
        #print extrema[0], extrema[1], extrema[2], extrema[3]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int integrate_ray(self, np.float64_t v_pos[3],
                                 np.float64_t v_dir[3],
                                 np.float64_t rgba[4],
                                 TransferFunctionProxy tf):
        cdef int cur_ind[3], step[3], x, y, i, n, flat_ind, hit, direction
        cdef np.float64_t intersect_t = 1.0
        cdef np.float64_t intersect[3], tmax[3], tdelta[3]
        cdef np.float64_t enter_t, dist, alpha, dt, exit_t
        cdef np.float64_t tr, tl, temp_x, temp_y, dv
        for i in range(3):
            if (v_dir[i] < 0):
                step[i] = -1
            else:
                step[i] = 1
            x = (i+1) % 3
            y = (i+2) % 3
            tl = (self.left_edge[i] - v_pos[i])/v_dir[i]
            tr = (self.right_edge[i] - v_pos[i])/v_dir[i]
            temp_x = (v_pos[x] + tl*v_dir[x])
            temp_y = (v_pos[y] + tl*v_dir[y])
            if self.left_edge[x] <= temp_x and temp_x <= self.right_edge[x] and \
               self.left_edge[y] <= temp_y and temp_y <= self.right_edge[y] and \
               0.0 <= tl and tl < intersect_t:
                direction = i
                intersect_t = tl
            temp_x = (v_pos[x] + tr*v_dir[x])
            temp_y = (v_pos[y] + tr*v_dir[y])
            if self.left_edge[x] <= temp_x and temp_x <= self.right_edge[x] and \
               self.left_edge[y] <= temp_y and temp_y <= self.right_edge[y] and \
               0.0 <= tr and tr < intersect_t:
                direction = i
                intersect_t = tr
        if self.left_edge[0] <= v_pos[0] and v_pos[0] <= self.right_edge[0] and \
           self.left_edge[1] <= v_pos[1] and v_pos[1] <= self.right_edge[1] and \
           self.left_edge[2] <= v_pos[2] and v_pos[2] <= self.right_edge[2]:
            intersect_t = 0.0
        if not ((0.0 <= intersect_t) and (intersect_t < 1.0)): return 0
        for i in range(3):
            intersect[i] = v_pos[i] + intersect_t * v_dir[i]
            cur_ind[i] = <int> floor((intersect[i] +
                                      step[i]*1e-8*self.dds[i] -
                                      self.left_edge[i])*self.idds[i])
            tmax[i] = (((cur_ind[i]+step[i])*self.dds[i])+
                        self.left_edge[i]-v_pos[i])/v_dir[i]
            # This deals with the asymmetry in having our indices refer to the
            # left edge of a cell, but the right edge of the brick being one
            # extra zone out.
            if cur_ind[i] == self.dims[i] and step[i] < 0:
                cur_ind[i] = self.dims[i] - 1
            if cur_ind[i] < 0 or cur_ind[i] >= self.dims[i]: return 0
            if step[i] > 0:
                tmax[i] = (((cur_ind[i]+1)*self.dds[i])
                            +self.left_edge[i]-v_pos[i])/v_dir[i]
            if step[i] < 0:
                tmax[i] = (((cur_ind[i]+0)*self.dds[i])
                            +self.left_edge[i]-v_pos[i])/v_dir[i]
            tdelta[i] = (self.dds[i]/v_dir[i])
            if tdelta[i] < 0: tdelta[i] *= -1
        # We have to jumpstart our calculation
        enter_t = intersect_t
        while 1:
            # dims here is one less than the dimensions of the data,
            # but we are tracing on the grid, not on the data...
            if (not (0 <= cur_ind[0] < self.dims[0])) or \
               (not (0 <= cur_ind[1] < self.dims[1])) or \
               (not (0 <= cur_ind[2] < self.dims[2])):
                break
            hit += 1
            if tmax[0] < tmax[1]:
                if tmax[0] < tmax[2]:
                    exit_t = fmin(tmax[0], 1.0)
                    self.sample_values(v_pos, v_dir, enter_t, exit_t, cur_ind,
                                       rgba, tf)
                    cur_ind[0] += step[0]
                    enter_t = tmax[0]
                    tmax[0] += tdelta[0]
                else:
                    exit_t = fmin(tmax[2], 1.0)
                    self.sample_values(v_pos, v_dir, enter_t, exit_t, cur_ind,
                                       rgba, tf)
                    cur_ind[2] += step[2]
                    enter_t = tmax[2]
                    tmax[2] += tdelta[2]
            else:
                if tmax[1] < tmax[2]:
                    exit_t = fmin(tmax[1], 1.0)
                    self.sample_values(v_pos, v_dir, enter_t, exit_t, cur_ind,
                                       rgba, tf)
                    cur_ind[1] += step[1]
                    enter_t = tmax[1]
                    tmax[1] += tdelta[1]
                else:
                    exit_t = fmin(tmax[2], 1.0)
                    self.sample_values(v_pos, v_dir, enter_t, exit_t, cur_ind,
                                       rgba, tf)
                    cur_ind[2] += step[2]
                    enter_t = tmax[2]
                    tmax[2] += tdelta[2]
            if enter_t > 1.0: break
        return hit

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void sample_values(self,
                            np.float64_t v_pos[3],
                            np.float64_t v_dir[3],
                            np.float64_t enter_t,
                            np.float64_t exit_t,
                            int ci[3],
                            np.float64_t *rgba,
                            TransferFunctionProxy tf):
        cdef np.float64_t cp[3], dp[3], temp, dt, t, dv
        cdef np.float64_t grad[3], ds[3]
        grad[0] = grad[1] = grad[2] = 0.0
        cdef int dti, i
        dt = (exit_t - enter_t) / (tf.ns) # 4 samples should be dt=0.25
        for i in range(3):
            temp = ci[i] * self.dds[i] + self.left_edge[i]
            dp[i] = (enter_t + 0.5 * dt) * v_dir[i] + v_pos[i] - temp
            dp[i] *= self.idds[i]
            ds[i] = v_dir[i] * self.idds[i] * dt
        for dti in range(tf.ns): 
            for i in range(3):
                dp[i] += ds[i]
            dv = trilinear_interpolate(self.dims, ci, dp, self.data)
            if (dv < tf.x_bounds[0]) or (dv > tf.x_bounds[1]):
                continue
            if tf.use_light == 1:
                eval_gradient(self.dims, ci, dp, self.data, grad)
            tf.eval_transfer(dt, dv, rgba, grad)

cdef class GridFace:
    cdef int direction
    cdef public np.float64_t coord
    cdef np.float64_t left_edge[3]
    cdef np.float64_t right_edge[3]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __init__(self, grid, int direction, int left):
        self.direction = direction
        if left == 1:
            self.coord = grid.LeftEdge[direction]
        else:
            self.coord = grid.RightEdge[direction]
        cdef int i
        for i in range(3):
            self.left_edge[i] = grid.LeftEdge[i]
            self.right_edge[i] = grid.RightEdge[i]
        self.left_edge[direction] = self.right_edge[direction] = self.coord

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int proj_overlap(self, np.float64_t *left_edge, np.float64_t *right_edge):
        cdef int xax, yax
        xax = (self.direction + 1) % 3
        yax = (self.direction + 2) % 3
        if left_edge[xax] >= self.right_edge[xax]: return 0
        if right_edge[xax] <= self.left_edge[xax]: return 0
        if left_edge[yax] >= self.right_edge[yax]: return 0
        if right_edge[yax] <= self.left_edge[yax]: return 0
        return 1

cdef class ProtoPrism:
    cdef np.float64_t left_edge[3]
    cdef np.float64_t right_edge[3]
    cdef public object LeftEdge
    cdef public object RightEdge
    cdef public object subgrid_faces
    cdef public int parent_grid_id
    def __cinit__(self, int parent_grid_id,
                  np.ndarray[np.float64_t, ndim=1] left_edge,
                  np.ndarray[np.float64_t, ndim=1] right_edge,
                  subgrid_faces):
        self.parent_grid_id = parent_grid_id
        cdef int i
        self.LeftEdge = left_edge
        self.RightEdge = right_edge
        for i in range(3):
            self.left_edge[i] = left_edge[i]
            self.right_edge[i] = right_edge[i]
        self.subgrid_faces = subgrid_faces

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def sweep(self, int direction = 0, int stack = 0):
        cdef int i
        cdef GridFace face
        cdef np.float64_t proto_split[3]
        for i in range(3): proto_split[i] = self.right_edge[i]
        for face in self.subgrid_faces[direction]:
            proto_split[direction] = face.coord
            if proto_split[direction] <= self.left_edge[direction]:
                continue
            if proto_split[direction] == self.right_edge[direction]:
                if stack == 2: return [self]
                return self.sweep((direction + 1) % 3, stack + 1)
            if face.proj_overlap(self.left_edge, proto_split) == 1:
                left, right = self.split(proto_split, direction)
                LC = left.sweep((direction + 1) % 3)
                RC = right.sweep(direction)
                return LC + RC
        raise RuntimeError

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef object split(self, np.float64_t *sp, int direction):
        cdef int i
        cdef np.ndarray split_left = self.LeftEdge.copy()
        cdef np.ndarray split_right = self.RightEdge.copy()

        for i in range(3): split_left[i] = self.right_edge[i]
        split_left[direction] = sp[direction]
        left = ProtoPrism(self.parent_grid_id, self.LeftEdge, split_left,
                          self.subgrid_faces)

        for i in range(3): split_right[i] = self.left_edge[i]
        split_right[direction] = sp[direction]
        right = ProtoPrism(self.parent_grid_id, split_right, self.RightEdge,
                           self.subgrid_faces)

        return (left, right)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_brick(self, np.ndarray[np.float64_t, ndim=1] grid_left_edge,
                        np.ndarray[np.float64_t, ndim=1] grid_dds,
                        np.ndarray[np.float64_t, ndim=3] data,
                        child_mask):
        # We get passed in the left edge, the dds (which gives dimensions) and
        # the data, which is already vertex-centered.
        cdef PartitionedGrid PG
        cdef int li[3], ri[3], idims[3], i
        for i in range(3):
            li[i] = lrint((self.left_edge[i] - grid_left_edge[i])/grid_dds[i])
            ri[i] = lrint((self.right_edge[i] - grid_left_edge[i])/grid_dds[i])
            idims[i] = ri[i] - li[i]
        if child_mask[li[0], li[1], li[2]] == 0: return []
        cdef np.ndarray[np.int64_t, ndim=1] dims = np.empty(3, dtype='int64')
        for i in range(3):
            dims[i] = idims[i]
        cdef np.ndarray[np.float64_t, ndim=3] new_data
        new_data = data[li[0]:ri[0]+1,li[1]:ri[1]+1,li[2]:ri[2]+1].copy()
        PG = PartitionedGrid(self.parent_grid_id, new_data,
                             self.LeftEdge, self.RightEdge, dims)
        return [PG]
