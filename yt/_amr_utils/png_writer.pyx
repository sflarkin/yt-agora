"""
A light interface to libpng

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: UCSD
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2010 Matthew Turk.  All Rights Reserved.

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

from stdio cimport fopen, fclose, FILE

cdef extern from "stdlib.h":
    # NOTE that size_t might not be int
    void *alloca(int)

cdef extern from "png.h":
    ctypedef unsigned long png_uint_32
    ctypedef long png_int_32
    ctypedef unsigned short png_uint_16
    ctypedef short png_int_16
    ctypedef unsigned char png_byte
    ctypedef void            *png_voidp
    ctypedef png_byte        *png_bytep
    ctypedef png_uint_32     *png_uint_32p
    ctypedef png_int_32      *png_int_32p
    ctypedef png_uint_16     *png_uint_16p
    ctypedef png_int_16      *png_int_16p
    ctypedef char            *png_charp
    ctypedef char            *png_const_charp
    ctypedef FILE            *png_FILE_p

    ctypedef struct png_struct:
        pass
    ctypedef png_struct      *png_structp

    ctypedef struct png_info:
        pass
    ctypedef png_info        *png_infop

    ctypedef struct png_color_8:
        png_byte red
        png_byte green
        png_byte blue
        png_byte gray
        png_byte alpha
    ctypedef png_color_8 *png_color_8p

    cdef png_const_charp PNG_LIBPNG_VER_STRING

    # Note that we don't support error or warning functions
    png_structp png_create_write_struct(
        png_const_charp user_png_ver, png_voidp error_ptr,
        void *error_fn, void *warn_fn)
    
    png_infop png_create_info_struct(png_structp png_ptr)

    void png_init_io(png_structp png_ptr, png_FILE_p fp)

    void png_set_IHDR(png_structp png_ptr, png_infop info_ptr,
        png_uint_32 width, png_uint_32 height, int bit_depth,
        int color_type, int interlace_method, int compression_method,
        int filter_method)

    cdef int PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE
    cdef int PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE

    void png_set_pHYs(png_structp png_ptr, png_infop info_ptr,
        png_uint_32 res_x, png_uint_32 res_y, int unit_type)

    cdef int PNG_RESOLUTION_METER

    void png_set_sBIT(png_structp png_ptr, png_infop info_ptr,
        png_color_8p sig_bit)

    void png_write_info(png_structp png_ptr, png_infop info_ptr)
    void png_write_image(png_structp png_ptr, png_bytep *image)
    void png_write_end(png_structp png_ptr, png_infop info_ptr)

    void png_destroy_write_struct(
        png_structp *png_ptr_ptr, png_infop *info_ptr_ptr)

def write_png(np.ndarray[np.uint8_t, ndim=3] buffer,
              char *filename, int dpi=100):

    # This is something of a translation of the matplotlib _png module
    cdef png_byte *pix_buffer = <png_byte *> buffer.data
    cdef int width = buffer.shape[0]
    cdef int height = buffer.shape[1]

    cdef FILE* fileobj = fopen(filename, "wb")
    cdef png_bytep *row_pointers
    cdef png_structp png_ptr
    cdef png_infop info_ptr

    cdef png_color_8 sig_bit
    cdef png_uint_32 row

    row_pointers = <png_bytep *> alloca(sizeof(png_bytep) * height)

    for row in range(height):
        row_pointers[row] = pix_buffer + row * width * 4
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL)
    info_ptr = png_create_info_struct(png_ptr)
    
    # Um we are ignoring setjmp sorry guys

    png_init_io(png_ptr, fileobj)

    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE)

    cdef size_t dots_per_meter = <size_t> (dpi / (2.54 / 100.0))
    png_set_pHYs(png_ptr, info_ptr, dots_per_meter, dots_per_meter,
                 PNG_RESOLUTION_METER)

    sig_bit.gray = 0
    sig_bit.red = sig_bit.green = sig_bit.blue = sig_bit.alpha = 8

    png_set_sBIT(png_ptr, info_ptr, &sig_bit)

    png_write_info(png_ptr, info_ptr)
    png_write_image(png_ptr, row_pointers)
    png_write_end(png_ptr, info_ptr)

    fclose(fileobj)
    png_destroy_write_struct(&png_ptr, &info_ptr)

def add_points_to_image(
        np.ndarray[np.uint8_t, ndim=3] buffer,
        np.ndarray[np.float64_t, ndim=1] px,
        np.ndarray[np.float64_t, ndim=1] py,
        np.float64_t pv):
    cdef int i, j, k, pi
    cdef int np = px.shape[0]
    cdef int xs = buffer.shape[0]
    cdef int ys = buffer.shape[1]
    cdef int v 
    v = iclip(<int>(pv * 255), 0, 255)
    print "VALUE CONTRIBUTION", v
    for pi in range(np):
        j = <int> (xs * px[pi])
        i = <int> (ys * py[pi])
        for k in range(3):
            buffer[i, j, k] = 0
    return
    for i in range(xs):
        for j in range(ys):
            for k in range(3):
                v = buffer[i, j, k]
                buffer[i, j, k] = iclip(v, 0, 255)
