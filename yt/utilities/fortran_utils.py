"""
Utilities for reading Fortran files.

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

import struct
import numpy as np
import os

def read_attrs(f, attrs,endian='='):
    r"""This function accepts a file pointer and reads from that file pointer
    according to a definition of attributes, returning a dictionary.

    Fortran unformatted files provide total bytesize at the beginning and end
    of a record.  By correlating the components of that record with attribute
    names, we construct a dictionary that gets returned.  Note that this
    function is used for reading sequentially-written records.  If you have
    many written that were written simultaneously, see read_record.

    Parameters
    ----------
    f : File object
        An open file object.  Should have been opened in mode rb.
    attrs : iterable of iterables
        This object should be an iterable of one of the formats: 
        [ (attr_name, count, struct type), ... ].
        [ ((name1,name2,name3),count, vector type]
        [ ((name1,name2,name3),count, 'type type type']
    endian : str
        '=' is native, '>' is big, '<' is little endian

    Returns
    -------
    values : dict
        This will return a dict of iterables of the components of the values in
        the file.

    Examples
    --------

    >>> header = [ ("ncpu", 1, "i"), ("nfiles", 2, "i") ]
    >>> f = open("fort.3", "rb")
    >>> rv = read_attrs(f, header)
    """
    vv = {}
    net_format = endian
    for a, n, t in attrs:
        for end in '@=<>':
            t = t.replace(end,'')
        net_format += "".join(["I"] + ([t] * n) + ["I"])
    size = struct.calcsize(net_format)
    vals = list(struct.unpack(net_format, f.read(size)))
    vv = {}
    for a, n, t in attrs:
        for end in '@=<>':
            t = t.replace(end,'')
        if type(a)==tuple:
            n = len(a)
        s1 = vals.pop(0)
        v = [vals.pop(0) for i in range(n)]
        s2 = vals.pop(0)
        if s1 != s2:
            size = struct.calcsize(endian + "I" + "".join(n*[t]) + "I")
        assert(s1 == s2)
        if n == 1: v = v[0]
        if type(a)==tuple:
            assert len(a) == len(v)
            for k,val in zip(a,v):
                vv[k]=val
        else:
            vv[a] = v
    return vv

def read_vector(f, d, endian='='):
    r"""This function accepts a file pointer and reads from that file pointer
    a vector of values.

    Parameters
    ----------
    f : File object
        An open file object.  Should have been opened in mode rb.
    d : data type
        This is the datatype (from the struct module) that we should read.
    endian : str
        '=' is native, '>' is big, '<' is little endian

    Returns
    -------
    tr : numpy.ndarray
        This is the vector of values read from the file.

    Examples
    --------

    >>> f = open("fort.3", "rb")
    >>> rv = read_vector(f, 'd')
    """
    pad_fmt = "%sI" % (endian)
    pad_size = struct.calcsize(pad_fmt)
    vec_len = struct.unpack(pad_fmt,f.read(pad_size))[0] # bytes
    vec_fmt = "%s%s" % (endian, d)
    vec_size = struct.calcsize(vec_fmt)
    if vec_len % vec_size != 0:
        print "fmt = '%s' ; length = %s ; size= %s" % (fmt, length, size)
        raise RuntimeError
    vec_num = vec_len / vec_size
    if isinstance(f, file): # Needs to be explicitly a file
        tr = np.fromfile(f, vec_fmt, count=vec_num)
    else:
        tr = np.fromstring(f.read(vec_len), vec_fmt, count=vec_num)
    vec_len2 = struct.unpack(pad_fmt,f.read(pad_size))[0]
    assert(vec_len == vec_len2)
    return tr

def skip(f, n=1, endian='='):
    r"""This function accepts a file pointer and skips a Fortran unformatted
    record. Optionally check that the skip was done correctly by checking 
    the pad bytes.

    Parameters
    ----------
    f : File object
        An open file object.  Should have been opened in mode rb.
    n : int
        Number of records to skip.
    check : bool
        Assert that the pad bytes are equal
    endian : str
        '=' is native, '>' is big, '<' is little endian

    Returns
    -------
    skipped: The number of elements in the skipped array

    Examples
    --------

    >>> f = open("fort.3", "rb")
    >>> skip(f, 3)
    """
    skipped = []
    pos = f.tell()
    for i in range(n):
        fmt = endian+"I"
        size = f.read(struct.calcsize(fmt))
        s1= struct.unpack(fmt, size)[0]
        f.seek(s1+ struct.calcsize(fmt), os.SEEK_CUR)
        s2= struct.unpack(fmt, size)[0]
        assert s1==s2 
        skipped.append(s1/struct.calcsize(fmt))
    return skipped

def peek_record_size(f,endian='='):
    r""" This function accept the file handle and returns
    the size of the next record and then rewinds the file
    to the previous position.

    Parameters
    ----------
    f : File object
        An open file object.  Should have been opened in mode rb.
    endian : str
        '=' is native, '>' is big, '<' is little endian

    Returns
    -------
    Number of bytes in the next record
    """
    pos = f.tell()
    s = struct.unpack('>i', f.read(struct.calcsize('>i')))
    f.seek(pos)
    return s[0]

def read_record(f, rspec, endian='='):
    r"""This function accepts a file pointer and reads from that file pointer
    a single "record" with different components.

    Fortran unformatted files provide total bytesize at the beginning and end
    of a record.  By correlating the components of that record with attribute
    names, we construct a dictionary that gets returned.

    Parameters
    ----------
    f : File object
        An open file object.  Should have been opened in mode rb.
    rspec : iterable of iterables
        This object should be an iterable of the format [ (attr_name, count,
        struct type), ... ].
    endian : str
        '=' is native, '>' is big, '<' is little endian

    Returns
    -------
    values : dict
        This will return a dict of iterables of the components of the values in
        the file.

    Examples
    --------

    >>> header = [ ("ncpu", 1, "i"), ("nfiles", 2, "i") ]
    >>> f = open("fort.3", "rb")
    >>> rv = read_record(f, header)
    """
    vv = {}
    net_format = endian + "I"
    for a, n, t in rspec:
        t = t if len(t)==1 else t[-1]
        net_format += "%s%s"%(n, t)
    net_format += "I"
    size = struct.calcsize(net_format)
    vals = list(struct.unpack(net_format, f.read(size)))
    vvv = vals[:]
    s1, s2 = vals.pop(0), vals.pop(-1)
    if s1 != s2:
        print "S1 = %s ; S2 = %s ; SIZE = %s"
        raise RuntimeError
    pos = 0
    for a, n, t in rspec:
        vv[a] = vals[pos:pos+n]
        pos += n
    return vv

