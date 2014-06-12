import cStringIO
import re
import os
import numpy as np
from thingking.httpmmap import HTTPArray
from thingking.arbitrary_page import PageCacheURL
from yt.funcs import mylog

_types = {
    'int': 'int32',
    'int64_t': 'int64',
    'float': 'float32',
    'double': 'float64',
    'unsigned int': 'I',
    'unsigned char': 'B',
}

def get_type(vtype, len=None):
    try:
        t = _types[vtype]
        if len is not None:
            t = np.dtype((t, len))
        else:
            t = np.dtype(t)
    except KeyError:
        t = eval("np."+vtype)
    return t

def lstrip(text_list):
    return [t.strip() for t in text_list]

def get_struct_vars(line):
    spl = lstrip(line.split(";"))
    multiv = lstrip(spl[0].split(","))
    ret = lstrip(multiv[0].split())
    ctype = ret[0]
    vnames = [ret[-1]] + multiv[1:]
    vnames = [v.strip() for v in vnames]
    for vtype in ret[1:-1]:
        ctype += ' ' + vtype
    num = None
    if len(vnames) == 1:
        if '[' in vnames[0]:
            num = int(vnames[0].split('[')[-1].strip(']'))
            #num = int(re.sub("\D", "", vnames[0]))
    ctype = get_type(ctype, len=num)
    return ctype, vnames

def bbox_filter(left, right, domain_width):

    def myfilter(chunk, mask=None):
        pos = np.array([chunk['x'], chunk['y'], chunk['z']]).T

        # This hurts, but is useful for periodicity. Probably should check first
        # if it is even needed for a given left/right
        for i in range(3):
            pos[:,i] = np.mod(pos[:,i] - left[i], domain_width[i]) + left[i]

        # Now get all particles that are within the bbox
        if mask is None:
            mask = np.all(pos >= left, axis=1) * np.all(pos < right, axis=1)
        else:
            np.multiply(mask, np.all(pos >= left, axis=1), mask)
            np.multiply(mask, np.all(pos < right, axis=1), mask)
        return mask

    return myfilter

def sphere_filter(center, radius, domain_width):

    def myfilter(chunk, mask=None):
        pos = np.array([chunk['x'], chunk['y'], chunk['z']]).T

        # This hurts, but is useful for periodicity. Probably should check first
        # if it is even needed for a given left/right
        for i in range(3):
            pos[:,i] = np.mod(pos[:,i] - left[i], domain_width[i]) + left[i]

        # Now get all particles that are within the radius
        if mask is None:
            mask = ((pos-center)**2).sum(axis=1)**0.5 < radius
        else:
            np.multiply(mask, np.linalg.norm(pos - center, 2) < radius, mask)
        return mask

    return myfilter


class DataStruct(object):
    """docstring for DataStruct"""

    _offset = 0

    def __init__(self, dtypes, num, filename):
        self.filename = filename
        self.dtype = np.dtype(dtypes)
        self.size = num
        self.itemsize = self.dtype.itemsize
        self.data = {}
        self.handle = None

    def set_offset(self, offset):
        self._offset = offset
        if self.size == -1:
            file_size = os.path.getsize(self.filename)
            file_size -= offset
            self.size = float(file_size) / self.itemsize
            assert(int(self.size) == self.size)

    def build_memmap(self):
        assert(self.size != -1)
        self.handle = np.memmap(self.filename, dtype=self.dtype,
                        mode='r', shape=self.size, offset=self._offset)
        for k in self.dtype.names:
            self.data[k] = self.handle[k]

    def __getitem__(self, key):
        mask = None
        kt = type(key)
        if kt == int or kt == np.int64 or kt == np.int32 or kt == np.int:
            if key == -1:
                key = slice(-1, None)
            else:
                key = slice(key, key+1)
        elif type(key) == np.ndarray:
            mask = key
            key = slice(None, None)
        if not isinstance(key, slice):
            raise NotImplementedError
        if key.start is None:
            key = slice(0, key.stop)
        if key.stop is None:
            key = slice(key.start, self.shape)
        if key.start < 0:
            key = slice(self.size + key.start, key.stop)
        if key.stop < 0:
            key = slice(key.start, self.size + key.stop)
        arr = self.handle[key.start:key.stop]
        if mask is None:
            return arr
        else:
            return arr[mask]

class RedirectArray(object):
    """docstring for RedirectArray"""
    def __init__(self, http_array, key):
        self.http_array = http_array
        self.key = key
        self.size = http_array.shape
        self.dtype = http_array.dtype[key]

    def __getitem__(self, sl):
        if isinstance(sl, int):
            return self.http_array[sl][self.key][0]
        return self.http_array[sl][self.key]

class HTTPDataStruct(DataStruct):
    """docstring for HTTPDataStruct"""

    def __init__(self, *args, **kwargs):
        super(HTTPDataStruct, self).__init__(*args, **kwargs)
        self.pcu = PageCacheURL(self.filename)

    def set_offset(self, offset):
        self._offset = offset
        if self.size == -1:
            # Read small piece:
            file_size = self.pcu.total_size
            file_size -= offset
            self.size = float(file_size) / self.itemsize
            assert(int(self.size) == self.size)

    def build_memmap(self):
        assert(self.size != -1)
        print 'Building memmap with offset: %i' % self._offset
        self.handle = HTTPArray(self.filename, dtype=self.dtype,
                        shape=self.size, offset=self._offset)
        for k in self.dtype.names:
            self.data[k] = RedirectArray(self.handle, k)


class SDFRead(dict):

    """docstring for SDFRead"""

    _eof = 'SDF-EOH'
    _data_struct = DataStruct

    def __init__(self, filename, header=None):
        self.filename = filename
        if header is None:
            header = filename
        self.header = header
        self.parameters = {}
        self.structs = []
        self.comments = []
        self.parse_header()
        self.set_offsets()
        self.load_memmaps()

    def parse_header(self):
        """docstring for parse_header"""
        # Pre-process
        ascfile = open(self.header, 'r')
        while True:
            l = ascfile.readline()
            if self._eof in l: break

            self.parse_line(l, ascfile)

        hoff = ascfile.tell()
        ascfile.close()
        if self.header != self.filename:
            hoff = 0
        self.parameters['header_offset'] = hoff

    def parse_line(self, line, ascfile):
        """Parse a line of sdf"""


        if 'struct' in line:
            self.parse_struct(line, ascfile)
            return

        if "#" in line:
            self.comments.append(line)
            return

        spl = lstrip(line.split("="))
        vtype, vname = lstrip(spl[0].split())
        vname = vname.strip("[]")
        vval = spl[-1].strip(";")
        if vtype == 'parameter':
            self.parameters[vname] = vval
            return
        elif vtype == "char":
            vtype = "str"

        try:
            vval = eval("np."+vtype+"(%s)" % vval)
        except AttributeError:
            vval = eval("np."+_types[vtype]+"(%s)" % vval)

        self.parameters[vname] = vval

    def parse_struct(self, line, ascfile):
        assert 'struct' in line

        str_types = []
        comments = []
        str_lines = []
        l = ascfile.readline()
        while "}" not in l:
            vtype, vnames = get_struct_vars(l)
            for v in vnames:
                str_types.append((v, vtype))
            l = ascfile.readline()
        num = l.strip("}[]")
        num = num.strip("\;\\\n]")
        if len(num) == 0:
            # We need to compute the number of records.  The DataStruct will
            # handle this.
            num = '-1'
        num = int(num)
        struct = self._data_struct(str_types, num, self.filename)
        self.structs.append(struct)
        return

    def set_offsets(self):
        running_off = self.parameters['header_offset']
        for struct in self.structs:
            struct.set_offset(running_off)
            running_off += struct.size * struct.itemsize
        return

    def load_memmaps(self):
        for struct in self.structs:
            struct.build_memmap()
            self.update(struct.data)


class HTTPSDFRead(SDFRead):

    """docstring for SDFRead"""

    _data_struct = HTTPDataStruct
    _eof = 'SDF-EOH'

    def parse_header(self):
        """docstring for parse_header"""
        # Pre-process
        ascfile = HTTPArray(self.header)
        max_header_size = 1024*1024
        lines = cStringIO.StringIO(ascfile[:max_header_size].data[:])
        while True:
            l = lines.readline()
            if self._eof in l: break

            self.parse_line(l, lines)

        hoff = lines.tell()
        if self.header != self.filename:
            hoff = 0
        self.parameters['header_offset'] = hoff


class SDFIndex(object):

    """docstring for SDFIndex

    This provides an index mechanism into the full SDF Dataset.

    Most useful class methods:
        get_cell_data(level, cell_iarr, fields)
        iter_bbox_data(left, right, fields)
        iter_bbox_data(left, right, fields)

    """
    def __init__(self, sdfdata, indexdata, level=9):
        super(SDFIndex, self).__init__()
        self.sdfdata = sdfdata
        self.indexdata = indexdata
        self.level = level
        idlevel = self.indexdata.parameters.get('level',None)
        if idlevel and idlevel != level:
            mylog.warning("Overriding index level to %i" % idlevel)
            self.level = idlevel

        self.rmin = None
        self.rmax = None
        self.domain_width = None
        self.domain_buffer = 0
        self.domain_dims = 0
        self.domain_active_dims = 0
        self.wandering_particles = True
        self.valid_indexdata = True
        self.masks = {
            "p" : int("011"*level, 2),
            "t" : int("101"*level, 2),
            "r" : int("110"*level, 2),
            "z" : int("011"*level, 2),
            "y" : int("101"*level, 2),
            "x" : int("110"*level, 2),
            2 : int("011"*level, 2),
            1 : int("101"*level, 2),
            0 : int("110"*level, 2),
        }
        self.dim_slices = {
            "p" : slice(0, None, 3),
            "t" : slice(1, None, 3),
            "r" : slice(2, None, 3),
            "z" : slice(0, None, 3),
            "y" : slice(1, None, 3),
            "x" : slice(2, None, 3),
            2 : slice(0, None, 3),
            1 : slice(1, None, 3),
            0 : slice(2, None, 3),
        }
        self.set_bounds()
        if self._midx_version >= 1.0:
            max_key = self.get_key(np.array([2**self.level - 1]*3))
        else:
            max_key = self.indexdata['index'][-1]
        self._max_key = max_key


    def _fix_rexact(self, rmin, rmax):

        center = 0.5*(rmax+rmin)
        mysize = (rmax-rmin)
        mysize *= (1.0 + 4.0*np.finfo(np.float32).eps)
        self.rmin = center - 0.5*mysize
        self.rmax = center + 0.5*mysize

    def set_bounds(self):
        sorted_rtp = self.sdfdata.parameters.get("sorted_rtp", False)
        sorted_xyz = self.sdfdata.parameters.get("sorted_xyz", False)
        morton_xyz = self.sdfdata.parameters.get("morton_xyz", False)

        self.rmin = np.zeros(3)
        self.rmax = np.zeros(3)
        r_0 = self.sdfdata.parameters['R0']
        try:
            r_0 = self.sdfdata.parameters['L0']
        except:
            pass

        if sorted_rtp:
            mylog.debug("Setting up Sorted RTP data")
            rtp_min = np.array([0.0, 0.0, -np.pi])
            rtp_max = np.array([r_0*1.01, 2*np.pi, np.pi])
            self._fix_rexact(rtp_min, rtp_max)
            self.true_domain_left = self.rmin.copy()
            self.true_domain_right = self.rmax.copy()
            self.true_domain_width = self.rmax - self.rmin
        elif sorted_xyz:
            mylog.debug("Setting up Sorted XYZ data")
            offset_center = self.sdfdata.parameters.get("offset_center", False)
            if offset_center:
                rmin = np.zeros(3)
                rmax = np.array([2.0*r_0]*3)
            else:
                rmin = -1.01*np.array([r_0]*3)
                rmax = 1.01*np.array([r_0]*3)
            self._fix_rexact(rmin, rmax)
            self.true_domain_left = self.rmin.copy()
            self.true_domain_right = self.rmax.copy()
            self.true_domain_width = self.rmax - self.rmin
        else:
            mylog.debug("Setting up regular data")
            rx = self.sdfdata.parameters.get('Rx')
            ry = self.sdfdata.parameters.get('Ry')
            rz = self.sdfdata.parameters.get('Rz')
            a =  self.sdfdata.parameters.get("a", 1.0)
            r = np.array([rx, ry, rz])
            rmin = -a*r
            rmax = a*r
            print rmin, rmax

            #/* expand root for non-power-of-two */
            expand_root = 0.0
            ic_Nmesh = self.sdfdata.parameters.get('ic_Nmesh',0)
            if ic_Nmesh != 0:
                f2 = 1<<int(np.log2(ic_Nmesh-1)+1)
                if (f2 != ic_Nmesh):
                    expand_root = 1.0*f2/ic_Nmesh - 1.0;
                mylog.debug("Expanding: %s, %s, %s" % (f2, ic_Nmesh, expand_root))
            #self._fix_rexact(rmin, rmax)
            self.true_domain_left = rmin.copy()
            self.true_domain_right = rmax.copy()
            self.true_domain_width = rmax - rmin
            rmin *= (1.0 + expand_root)
            rmax *= (1.0 + expand_root)
            self._fix_rexact(rmin, rmax)
        print self.rmin, self.rmax

        self._midx_version = self.indexdata.parameters.get("midx_version", 0)
        if self._midx_version == 1.0:
            rmin = np.zeros(3)
            rmax = np.zeros(3)
            rmin[0] = self.indexdata.parameters['x_min']
            rmin[1] = self.indexdata.parameters['y_min']
            rmin[2] = self.indexdata.parameters['z_min']
            rmax[0] = self.indexdata.parameters['x_max']
            rmax[1] = self.indexdata.parameters['y_max']
            rmax[2] = self.indexdata.parameters['z_max']
            self._fix_rexact(rmin, rmax)

        self.domain_width = self.rmax - self.rmin
        self.domain_dims = 1 << self.level
        self.domain_buffer = (self.domain_dims - int(self.domain_dims/(1.0 + expand_root)))/2
        self.domain_active_dims = self.domain_dims - 2*self.domain_buffer
        mylog.debug("SINDEX: %s, %s, %s " % (self.domain_width, self.domain_dims, self.domain_active_dims))

    def spread_bits(self, ival, level=None):
        if level is None:
            level = self.level
        res = 0
        for i in range(level):
            res |= ((ival>>i)&1)<<(i*3);
        return res

    def get_key(self, iarr, level=None):
        if level is None:
            level = self.level
        i1, i2, i3 = iarr
        return self.spread_bits(i1, level) | self.spread_bits(i2, level) << 1 | self.spread_bits(i3, level) << 2

    def spread_bitsv(self, ival, level=None):
        if level is None:
            level = self.level
        res = np.zeros_like(ival, dtype='int64')
        for i in range(level):
            res |= np.bitwise_and((ival>>i), 1)<<(i*3);
        return res

    def get_keyv(self, iarr, level=None):
        if level is None:
            level = self.level
        i1, i2, i3 = iarr
        return np.bitwise_or(
            np.bitwise_or(self.spread_bitsv(i1, level) , self.spread_bitsv(i2, level) << 1 ),
            self.spread_bitsv(i3, level) << 2)

    def get_key_slow(self, iarr, level=None):
        if level is None:
            level = self.level
        i1, i2, i3 = iarr
        rep1 = np.binary_repr(i1, width=self.level)
        rep2 = np.binary_repr(i2, width=self.level)
        rep3 = np.binary_repr(i3, width=self.level)
        inter = np.zeros(self.level*3, dtype='c')
        inter[self.dim_slices[0]] = rep1
        inter[self.dim_slices[1]] = rep2
        inter[self.dim_slices[2]] = rep3
        return int(inter.tostring(), 2)

    def get_key_ijk(self, i1, i2, i3, level=None):
        return self.get_key(np.array([i1, i2, i3]), level=level)

    def get_slice_key(self, ind, dim='r'):
        slb = np.binary_repr(ind, width=self.level)
        expanded = np.array([0]*self.level*3, dtype='c')
        expanded[self.dim_slices[dim]] = slb
        return int(expanded.tostring(), 2)

    def get_ind_from_key(self, key, dim='r'):
        ind = [0,0,0]
        br = np.binary_repr(key, width=self.level*3)
        for dim in range(3):
            ind[dim] = int(br[self.dim_slices[dim]],2)
        return ind

    def get_slice_chunks(self, slice_dim, slice_index):
        sl_key = self.get_slice_key(slice_index, dim=slice_dim)
        mask = (self.indexdata['index'] & ~self.masks[slice_dim]) == sl_key
        offsets = self.indexdata['base'][mask]
        lengths = self.indexdata['len'][mask]
        return mask, offsets, lengths

    def get_ibbox_slow(self, ileft, iright):
        """
        Given left and right indicies, return a mask and
        set of offsets+lengths into the sdf data.
        """
        mask = np.zeros(self.indexdata['index'].shape, dtype='bool')
        ileft = np.array(ileft)
        iright = np.array(iright)
        for i in range(3):
            left_key = self.get_slice_key(ileft[i], dim=i)
            right_key= self.get_slice_key(iright[i], dim=i)
            dim_inds = (self.indexdata['index'] & ~self.masks[i])
            mask *= (dim_inds >= left_key) * (dim_inds <= right_key)
            del dim_inds

        offsets = self.indexdata['base'][mask]
        lengths = self.indexdata['len'][mask]
        return mask, offsets, lengths

    def get_ibbox(self, ileft, iright):
        """
        Given left and right indicies, return a mask and
        set of offsets+lengths into the sdf data.
        """
        #print 'Getting data from ileft to iright:',  ileft, iright

        ix, iy, iz = (iright-ileft)*1j
        #print 'IBBOX:', ileft, iright, ix, iy, iz

        # plus 1 that is sliced, plus a bit since mgrid is not inclusive
        Z, Y, X = np.mgrid[ileft[2]:iright[2]+1.01,
                           ileft[1]:iright[1]+1.01,
                           ileft[0]:iright[0]+1.01]

        mask = slice(0, -1, None)
        X = X[mask, mask, mask].astype('int32').ravel()
        Y = Y[mask, mask, mask].astype('int32').ravel()
        Z = Z[mask, mask, mask].astype('int32').ravel()

        if self.wandering_particles:
            # Need to get padded bbox around the border to catch
            # wandering particles.
            dmask = X < self.domain_buffer
            dmask += Y < self.domain_buffer
            dmask += Z < self.domain_buffer
            dmask += X >= self.domain_dims
            dmask += Y >= self.domain_dims
            dmask += Z >= self.domain_dims
            dinds = self.get_keyv([X[dmask], Y[dmask], Z[dmask]])
            dinds = dinds[dinds < self._max_key]
            dinds = dinds[self.indexdata['len'][dinds] > 0]
            #print 'Getting boundary layers for wanderers, cells: %i' % dinds.size

        # Correct For periodicity
        X[X < self.domain_buffer] += self.domain_active_dims
        Y[Y < self.domain_buffer] += self.domain_active_dims
        Z[Z < self.domain_buffer] += self.domain_active_dims
        X[X >= self.domain_buffer + self.domain_active_dims] -= self.domain_active_dims
        Y[Y >= self.domain_buffer + self.domain_active_dims] -= self.domain_active_dims
        Z[Z >= self.domain_buffer + self.domain_active_dims] -= self.domain_active_dims

        #print 'periodic:',  X.min(), X.max(), Y.min(), Y.max(), Z.min(), Z.max()

        indices = self.get_keyv([X, Y, Z])
        # Only mask out if we are actually getting data rather than getting indices into
        # a space.
        if self.valid_indexdata:
            indices = indices[indices < self._max_key]
            indices = indices[self.indexdata['len'][indices] > 0]

        #indices = np.array([self.get_key_ijk(x, y, z) for x, y, z in zip(X, Y, Z)])
        # Here we sort the indices to batch consecutive reads together.
        if self.wandering_particles:
            indices = np.sort(np.append(indices, dinds))
        else:
            indices = np.sort(indices)
        return indices

    def get_bbox(self, left, right):
        """
        Given left and right indicies, return a mask and
        set of offsets+lengths into the sdf data.
        """
        ileft = np.floor((left - self.rmin) / self.domain_width *  self.domain_dims)
        iright = np.floor((right - self.rmin) / self.domain_width * self.domain_dims)
        #iright[iright <= ileft+1] += 1

        return self.get_ibbox(ileft, iright)

    def get_nparticles_bbox(self, left, right):
        """
        Given left and right edges, return total
        number of particles present.
        """
        ileft = np.floor((left - self.rmin) / self.domain_width *  self.domain_dims)
        iright = np.floor((right - self.rmin) / self.domain_width * self.domain_dims)
        indices = self.get_ibbox(ileft, iright)
        npart = 0
        for ind in indices:
            npart += self.indexdata['len'][ind]
        return npart

    def get_data(self, chunk, fields):
        data = {}
        for field in fields:
            data[field] = self.sdfdata[field][chunk]
        return data

    def get_next_nonzero_chunk(self, key, stop=None):
        # These next two while loops are to squeeze the keys if they are empty. Would be better
        # to go through and set base equal to the last non-zero base, i think.
        if stop is None:
            stop = self._max_key
        while key < stop:
            if self.indexdata['len'][key] == 0:
                #print 'Squeezing keys, incrementing'
                key += 1
            else:
                break
        return key

    def get_previous_nonzero_chunk(self, key, stop=None):
        # These next two while loops are to squeeze the keys if they are empty. Would be better
        # to go through and set base equal to the last non-zero base, i think.
        if stop is None:
            stop = self.indexdata['index'][0]
        while key > stop:
            if self.indexdata['len'][key] == 0:
                #print 'Squeezing keys, decrementing'
                key -= 1
            else:
                break
        return key

    def iter_data(self, inds, fields):
        num_inds = len(inds)
        num_reads = 0
        mylog.debug('SINDEX Reading %i chunks' % num_inds)
        i = 0
        while (i < num_inds):
            ind = inds[i]
            base = self.indexdata['base'][ind]
            length = self.indexdata['len'][ind]
            # Concatenate aligned reads
            nexti = i+1
            combined = 0
            while nexti < num_inds:
                nextind = inds[nexti]
                #        print 'b: %i l: %i end: %i  next: %i' % ( base, length, base + length, self.indexdata['base'][nextind] )
                if combined < 1024 and base + length == self.indexdata['base'][nextind]:
                    length += self.indexdata['len'][nextind]
                    i += 1
                    nexti += 1
                    combined += 1
                else:
                    break

            chunk = slice(base, base+length)
            mylog.debug('Reading chunk %i of length %i after catting %i starting at %i' % (i, length, combined, ind))
            num_reads += 1
            if length > 0:
                data = self.get_data(chunk, fields)
                yield data
                del data
            i += 1
        mylog.debug('Read %i chunks, batched into %i reads' % (num_inds, num_reads))

    def filter_particles(self, myiter, myfilter):
        for data in myiter:
            mask = myfilter(data)

            if mask.sum() == 0:
                continue
            filtered = {}
            for f in data.keys():
                filtered[f] = data[f][mask]

            yield filtered

    def filter_bbox(self, left, right, myiter):
        """
        Filter data by masking out data outside of a bbox defined
        by left/right. Account for periodicity of data, allowing left/right
        to be outside of the domain.
        """

        for data in myiter:
            #mask = np.zeros_like(data, dtype='bool')
            pos = np.array([data['x'].copy(), data['y'].copy(), data['z'].copy()]).T

            DW = self.true_domain_width
            # This hurts, but is useful for periodicity. Probably should check first
            # if it is even needed for a given left/right
            for i in range(3):
                #pos[:,i] = np.mod(pos[:,i] - left[i],
                #                  self.true_domain_width[i]) + left[i]
                mask = pos[:,i] >= left[i] + DW[i]
                pos[mask, i] -= DW[i]
                mask = pos[:,i] < right[i] - DW[i]
                pos[mask, i] += DW[i]
                #del mask

            # Now get all particles that are within the bbox
            mask = np.all(pos >= left, axis=1) * np.all(pos < right, axis=1)
            #print 'Mask shape, sum:', mask.shape, mask.sum()

            mylog.debug("Filtering particles, returning %i out of %i" % (mask.sum(), mask.shape[0]))

            if not np.any(mask):
                continue

            filtered = {ax: pos[:, i][mask] for i, ax in enumerate('xyz')}
            for f in data.keys():
                if f in 'xyz':
                    continue
                filtered[f] = data[f][mask]

            #for i, ax in enumerate('xyz'):
            #    #print left, right
            #    assert np.all(filtered[ax] >= left[i])
            #    assert np.all(filtered[ax] < right[i])

            yield filtered

    def iter_filtered_bbox_fields(self, left, right, data,
                                  pos_fields, fields):
        """
        This function should be destroyed, as it will only work with units.
        """

        kpcuq = left.in_units('kpccm').uq
        mpcuq = left.in_units('Mpc/h').uq
        DW = (self.true_domain_width * kpcuq).in_units('Mpc/h')
        if pos_fields is None:
            pos_fields = 'x','y','z'
        xf, yf, zf = pos_fields
        print pos_fields

        # I'm sorry.
        pos = mpcuq * np.array([data[xf].in_units('Mpc/h'), data[yf].in_units('Mpc/h'), data[zf].in_units('Mpc/h')]).T

        # This hurts, but is useful for periodicity. Probably should check first
        # if it is even needed for a given left/right
        for i in range(3):
            mask = pos[:,i] >= DW[i] + left[i]
            pos[mask, i] -= DW[i]
            mask = pos[:,i] < right[i] - DW[i]
            pos[mask, i] += DW[i]

        print left, right, pos.min(axis=0), pos.max(axis=0)
        # Now get all particles that are within the bbox
        mask = np.all(pos >= left, axis=1) * np.all(pos < right, axis=1)

        mylog.debug("Filtering particles, returning %i out of %i" % (mask.sum(), mask.shape[0]))

        if np.any(mask):
            for i,f in enumerate(pos_fields):
                yield f, pos[:, i][mask]

            for f in fields:
                if f in pos_fields:
                    continue
                # print 'yielding nonpos field', f
                yield f, data[f][mask]

    def iter_bbox_data(self, left, right, fields):
        mylog.debug('SINDEX Loading region from %s to %s' %(left, right))
        inds = self.get_bbox(left, right)
        # Need to put left/right in float32 to avoid fp roundoff errors
        # in the bbox later.
        #left = left.astype('float32')
        #right = right.astype('float32')

        #my_filter = bbox_filter(left, right, self.true_domain_width)
        data = []
        for dd in self.filter_bbox(
            left, right,
            self.iter_data(inds, fields)):
            yield dd
        #for dd in self.filter_particles(
        #    self.iter_data(inds, fields),
        #    my_filter):
        #    yield dd

    def iter_sphere_data(self, center, radius, fields):
        mylog.debug('SINDEX Loading spherical region %s to %s' %(center, radius))
        inds = self.get_bbox(center-radius, center+radius)

        my_filter = sphere_filter(center, radius, self.true_domain_width)

        for dd in self.filter_particles(
            self.iter_data(inds, fields),
            my_filter):
            yield dd

    def iter_ibbox_data(self, left, right, fields):
        mylog.debug('SINDEX Loading region from %s to %s' %(left, right))
        inds = self.get_ibbox(left, right)
        return self.iter_data(inds, fields)

    def get_contiguous_chunk(self, left_key, right_key, fields):
        liarr = self.get_ind_from_key(left_key)
        riarr = self.get_ind_from_key(right_key)

        lbase=0
        llen = 0
        if left_key > self._max_key:
            raise RuntimeError("Left key is too large. Key: %i Max Key: %i" % \
                               (left_key, self._max_key))
        right_key = min(right_key, self._max_key)

        left_key = self.get_next_nonzero_chunk(left_key)
        right_key = self.get_previous_nonzero_chunk(right_key, left_key)

        lbase = self.indexdata['base'][left_key]
        llen = self.indexdata['len'][left_key]

        rbase = self.indexdata['base'][right_key]
        rlen = self.indexdata['len'][right_key]

        length = rbase + rlen - lbase
        if length > 0:
            mylog.debug('Getting contiguous chunk of size %i starting at %i' % (length, lbase))
        return self.get_data(slice(lbase, lbase + length), fields)

    def get_key_data(self, key, fields):
        if key > self._max_key:
            raise RuntimeError("Left key is too large. Key: %i Max Key: %i" % (key, self._max_key))
        base = self.indexdata['base'][key]
        length = self.indexdata['len'][key] - base
        if length > 0:
            mylog.debug('Getting contiguous chunk of size %i starting at %i' % (length, base))
        return self.get_data(slice(base, base + length), fields)

    def iter_slice_data(self, slice_dim, slice_index, fields):
        mask, offsets, lengths = self.get_slice_chunks(slice_dim, slice_index)
        for off, l in zip(offsets, lengths):
            data = {}
            chunk = slice(off, off+l)
            for field in fields:
                data[field] = self.sdfdata[field][chunk]
            yield data
            del data

    def get_key_bounds(self, level, cell_iarr):
        """
        Get index keys for index file supplied.

        level: int
            Requested level
        cell_iarr: array-like, length 3
            Requested cell from given level.

        Returns:
            lmax_lk, lmax_rk
        """
        shift = self.level-level
        level_buff = 0
        level_lk = self.get_key(cell_iarr + level_buff)
        level_rk = self.get_key(cell_iarr + level_buff) + 1
        lmax_lk = (level_lk << shift*3)
        lmax_rk = (((level_rk) << shift*3) -1)
        #print "Level ", level, np.binary_repr(level_lk, width=self.level*3), np.binary_repr(level_rk, width=self.level*3)
        #print "Level ", self.level, np.binary_repr(lmax_lk, width=self.level*3), np.binary_repr(lmax_rk, width=self.level*3)
        return lmax_lk, lmax_rk

    def get_cell_data(self, level, cell_iarr, fields):
        """
        Get data from requested cell

        This uses the raw cell index, and doesn't account for periodicity or
        an expanded domain (non-power of 2).

        level: int
            Requested level
        cell_iarr: array-like, length 3
            Requested cell from given level.         fields: list
            Requested fields

        Returns:
            cell_data: dict
                Dictionary of field_name, field_data
        """
        cell_iarr = np.array(cell_iarr)
        lk, rk =self.get_key_bounds(level, cell_iarr)
        print 'Reading from ', lk, rk
        return self.get_contiguous_chunk(lk, rk, fields)

    def get_cell_bbox(self, level, cell_iarr):
        """Get floating point bounding box for a given sindex cell

        Returns:
            bbox: array-like, shape (3,2)

        """
        cell_iarr = np.array(cell_iarr)
        cell_width = self.get_cell_width(level)
        le = self.rmin + cell_iarr*cell_width
        re = le+cell_width
        bbox = np.array([le, re]).T
        assert bbox.shape == (3, 2)
        return bbox


    def iter_padded_bbox_data(self, level, cell_iarr, pad, fields):
        """
        Yields data chunks for a cell on the given level
        plus a padding around the cell, for a list of fields.

        Yields:
            dd: A dictionaries of data.

        Example:

        for chunk in sindex.iter_padded_bbox_data(
            6, np.array([128]*3), 8.0, ['x','y','z','ident']):

            print chunk['x'].max()

        """

        bbox = self.get_cell_bbox(level, cell_iarr)
        filter_left = bbox[:, 0] - pad
        filter_right = bbox[:, 1] + pad

        # Center cell
        for dd in self.filter_bbox(
            filter_left, filter_right,
            [self.get_cell_data(level, cell_iarr, fields)]):
            yield dd
            del dd

        # Bottom & Top
        pbox = bbox.copy()
        pbox[0, 0] -= pad[0]
        pbox[0, 1] += pad[0]
        pbox[1, 0] -= pad[1]
        pbox[1, 1] += pad[1]
        pbox[2, 0] -= pad[2]
        pbox[2, 1] = bbox[2, 0]
        for dd in self.filter_bbox(
            filter_left, filter_right,
            self.iter_bbox_data(pbox[:,0], pbox[:,1], fields)):
            yield dd
            del dd

        pbox[2, 0] = bbox[2, 1]
        pbox[2, 1] = pbox[2, 0] + pad[2]
        for dd in self.filter_bbox(
            filter_left, filter_right,
            self.iter_bbox_data(pbox[:,0], pbox[:,1], fields)):
            yield dd
            del dd

        # Front & Back
        pbox = bbox.copy()
        pbox[0, 0] -= pad[0]
        pbox[0, 1] += pad[0]
        pbox[1, 0] -= pad[1]
        pbox[1, 1] = bbox[1, 0]
        for dd in self.filter_bbox(
            filter_left, filter_right,
            self.iter_bbox_data(pbox[:,0], pbox[:,1], fields)):
            yield dd
            del dd

        pbox[1, 0] = bbox[1, 1]
        pbox[1, 1] = pbox[1, 0] + pad[1]
        for dd in self.filter_bbox(
            filter_left, filter_right,
            self.iter_bbox_data(pbox[:,0], pbox[:,1], fields)):
            yield dd
            del dd

        # Left & Right
        pbox = bbox.copy()
        pbox[0, 0] -= pad[0]
        pbox[0, 1] = bbox[0, 0]
        for dd in self.filter_bbox(
            filter_left, filter_right,
            self.iter_bbox_data(pbox[:,0], pbox[:,1], fields)):
            yield dd
            del dd

        pbox[0, 0] = bbox[0, 1]
        pbox[0, 1] = pbox[0, 0] + pad[0]
        for dd in self.filter_bbox(
            filter_left, filter_right,
            self.iter_bbox_data(pbox[:,0], pbox[:,1], fields)):
            yield dd
            del dd

    def get_padded_bbox_data(self, level, cell_iarr, pad, fields):
        """
        Return list of data chunks for a cell on the given level
        plus a padding around the cell, for a list of fields.

        Returns:
            data: A list of dictionaries of data.

        chunks = sindex.get_padded_bbox_data(6, np.array([128]*3),
                                             8.0, ['x','y','z','ident'])

        """
        bbox = self.get_cell_bbox(level, cell_iarr)
        filter_left = bbox[:, 0] - pad
        filter_right = bbox[:, 1] + pad

        data = []
        for dd in self.iter_padded_bbox_data(self, level, cell_iarr, pad, fields):
            data.append(dd)
        return data

    def get_cell_width(self, level):
        return self.domain_width / 2**level

    def iter_padded_bbox_keys(self, level, cell_iarr, pad):
        """

        Returns:
            bbox: array-like, shape (3,2)

        """
        bbox = self.get_cell_bbox(level, cell_iarr)
        filter_left = bbox[:, 0] - pad
        filter_right = bbox[:, 1] + pad

        # Need to get all of these
        low_key, high_key = self.get_key_bounds(level, cell_iarr)
        for k in xrange(low_key, high_key):
            yield k

        # Bottom & Top
        pbox = bbox.copy()
        pbox[0, 0] -= pad[0]
        pbox[0, 1] += pad[0]
        pbox[1, 0] -= pad[1]
        pbox[1, 1] += pad[1]
        pbox[2, 0] -= pad[2]
        pbox[2, 1] = bbox[2, 0]
        for k in self.get_bbox(pbox[:,0], pbox[:,1]):
            yield k

        pbox[2, 0] = bbox[2, 1]
        pbox[2, 1] = pbox[2, 0] + pad[2]
        for k in self.get_bbox(pbox[:,0], pbox[:,1]):
            yield k

        # Front & Back
        pbox = bbox.copy()
        pbox[0, 0] -= pad[0]
        pbox[0, 1] += pad[0]
        pbox[1, 0] -= pad[1]
        pbox[1, 1] = bbox[1, 0]
        for k in self.get_bbox(pbox[:,0], pbox[:,1]):
            yield k
        pbox[1, 0] = bbox[1, 1]
        pbox[1, 1] = pbox[1, 0] + pad[1]
        for k in self.get_bbox(pbox[:,0], pbox[:,1]):
            yield k

        # Left & Right
        pbox = bbox.copy()
        pbox[0, 0] -= pad[0]
        pbox[0, 1] = bbox[0, 0]
        for k in self.get_bbox(pbox[:,0], pbox[:,1]):
            yield k
        pbox[0, 0] = bbox[0, 1]
        pbox[0, 1] = pbox[0, 0] + pad[0]
        for k in self.get_bbox(pbox[:,0], pbox[:,1]):
            yield k

