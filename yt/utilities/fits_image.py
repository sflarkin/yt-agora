"""
FITSImageBuffer Class
"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
from yt.funcs import mylog, iterable
from yt.visualization.fixed_resolution import FixedResolutionBuffer
from yt.data_objects.construction_data_containers import YTCoveringGridBase

try:
    from astropy.io.fits import HDUList, ImageHDU
    from astropy import wcs as pywcs
except ImportError:
    pass

class FITSImageBuffer(HDUList):

    def __init__(self, data, fields=None, units="cm",
                 center=None, scale=None):
        r""" Initialize a FITSImageBuffer object.

        FITSImageBuffer contains a list of FITS ImageHDU instances, and optionally includes
        WCS information. It inherits from HDUList, so operations such as `writeto` are
        enabled. Images can be constructed from ImageArrays, NumPy arrays, dicts of such
        arrays, FixedResolutionBuffers, and YTCoveringGrids. The latter
        two are the most powerful because WCS information can be constructed from their coordinates.

        Parameters
        ----------
        data : FixedResolutionBuffer or a YTCoveringGrid. Or, an
            ImageArray, an numpy.ndarray, or dict of such arrays
            The data to be made into a FITS image or images.
        fields : single string or list of strings, optional
            The field names for the data. If *fields* is none and *data* has keys,
            it will use these for the fields. If *data* is just a single array one field name
            must be specified.
        units : string
            The units of the WCS coordinates, default "cm". 
        center : array_like, optional
            The coordinates [xctr,yctr] of the images in units
            *units*. If *units* is not specified, defaults to the origin. 
        scale : tuple of floats, optional
            Pixel scale in unit *units*. Will be ignored if *data* is
            a FixedResolutionBuffer or a YTCoveringGrid. Must be
            specified otherwise, or if *units* is "deg".

        Examples
        --------

        >>> ds = load("sloshing_nomag2_hdf5_plt_cnt_0150")
        >>> prj = ds.h.proj(2, "TempkeV", weight_field="Density")
        >>> frb = prj.to_frb((0.5, "mpc"), 800)
        >>> # This example just uses the FRB and puts the coords in kpc.
        >>> f_kpc = FITSImageBuffer(frb, fields="TempkeV", units="kpc")
        >>> # This example specifies sky coordinates.
        >>> scale = [1./3600.]*2 # One arcsec per pixel
        >>> f_deg = FITSImageBuffer(frb, fields="TempkeV", units="deg",
                                    scale=scale, center=(30., 45.))
        >>> f_deg.writeto("temp.fits")
        """
        
        super(HDUList, self).__init__()

        if isinstance(fields, basestring): fields = [fields]
            
        exclude_fields = ['x','y','z','px','py','pz',
                          'pdx','pdy','pdz','weight_field']
        
        if hasattr(data, 'keys'):
            img_data = data
        else:
            img_data = {}
            if fields is None:
                mylog.error("Please specify a field name for this array.")
                raise KeyError
            img_data[fields[0]] = data

        if fields is None: fields = img_data.keys()
        if len(fields) == 0:
            mylog.error("Please specify one or more fields to write.")
            raise KeyError

        first = False
    
        for key in fields:
            if key not in exclude_fields:
                mylog.info("Making a FITS image of field %s" % (key))
                if first:
                    hdu = PrimaryHDU(np.array(img_data[key]))
                    hdu.name = key
                else:
                    hdu = ImageHDU(np.array(img_data[key]), name=key)
                self.append(hdu)

        self.dimensionality = len(self[0].data.shape)
        
        if self.dimensionality == 2:
            self.nx, self.ny = self[0].data.shape
        elif self.dimensionality == 3:
            self.nx, self.ny, self.nz = self[0].data.shape

        has_coords = (isinstance(img_data, FixedResolutionBuffer) or
                      isinstance(img_data, YTCoveringGridBase))
        
        if center is None:
            if units == "deg":
                mylog.error("Please specify center=(RA, Dec) in degrees.")
                raise ValueError
            elif not has_coords:
                mylog.warning("Setting center to the origin.")
                center = [0.0]*self.dimensionality

        if scale is None:
            if units == "deg" or not has_coords:
                mylog.error("Please specify scale=(dx,dy[,dz]) in %s." % (units))
                raise ValueError

        w = pywcs.WCS(header=self[0].header, naxis=self.dimensionality)
        w.wcs.crpix = 0.5*(np.array(self.shape)+1)

        proj_type = ["linear"]*self.dimensionality

        if isinstance(img_data, FixedResolutionBuffer) and units != "deg":
            # FRBs are a special case where we have coordinate
            # information, so we take advantage of this and
            # construct the WCS object
            dx = (img_data.bounds[1]-img_data.bounds[0])/self.nx
            dy = (img_data.bounds[3]-img_data.bounds[2])/self.ny
            dx *= img_data.pf.units[units]
            dy *= img_data.pf.units[units]
            xctr = 0.5*(img_data.bounds[1]+img_data.bounds[0])
            yctr = 0.5*(img_data.bounds[3]+img_data.bounds[2])
            xctr *= img_data.pf.units[units]
            yctr *= img_data.pf.units[units]
            center = [xctr, yctr]
        elif isinstance(img_data, YTCoveringGridBase):
            dx, dy, dz = img_data.dds
            dx *= img_data.pf.units[units]
            dy *= img_data.pf.units[units]
            dz *= img_data.pf.units[units]
            center = 0.5*(img_data.left_edge+img_data.right_edge)
            center *= img_data.pf.units[units]
        elif units == "deg" and self.dimensionality == 2:
            dx = -scale[0]
            dy = scale[1]
            proj_type = ["RA---TAN","DEC--TAN"]
        else:
            dx = scale[0]
            dy = scale[1]
            if self.dimensionality == 3: dz = scale[2]
            
        w.wcs.crval = center
        w.wcs.cunit = [units]*self.dimensionality
        w.wcs.ctype = proj_type
        
        if self.dimensionality == 2:
            w.wcs.cdelt = [dx,dy]
        elif self.dimensionality == 3:
            w.wcs.cdelt = [dx,dy,dz]

        self._set_wcs(w)
            
    def _set_wcs(self, wcs):
        """
        Set the WCS coordinate information for all images
        with a WCS object *wcs*.
        """
        self.wcs = wcs
        h = self.wcs.to_header()
        for img in self:
            for k, v in h.items():
                img.header.update(k,v)

    def update_all_headers(self, key, value):
        """
        Update the FITS headers for all images with the
        same *key*, *value* pair.
        """
        for img in self: img.header.update(key,value)
            
    def keys(self):
        return [f.name for f in self]

    def has_key(self, key):
        return key in self.keys()

    def values(self):
        return [self[k] for k in self.keys()]

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def __add__(self, other):
        if len(set(self.keys()).intersection(set(other.keys()))) > 0:
            mylog.error("There are duplicate extension names! Don't know which ones you want to keep!")
            raise KeyError
        new_buffer = {}
        for im1 in self:
            new_buffer[im1.name] = im1.data
        for im2 in other:
            new_buffer[im2.name] = im2.data
        new_wcs = self.wcs
        return FITSImageBuffer(new_buffer, wcs=new_wcs)

    def writeto(self, fileobj, **kwargs):
        HDUList(self).writeto(fileobj, **kwargs)
        
    @property
    def shape(self):
        if self.dimensionality == 2:
            return self.nx, self.ny
        elif self.dimensionality == 3:
            return self.nx, self.ny, self.nz

    def to_glue(self, label="yt"):
        from glue.core import DataCollection, Data, Component
        from glue.core.coordinates import coordinates_from_header
        from glue.qt.glue_application import GlueApplication

        field_dict = dict((key,self[key].data) for key in self.keys())
        
        image = Data(label=label)
        image.coords = coordinates_from_header(self.wcs.to_header())
        for component_name in field_dict:
            comp = Component(field_dict[component_name])
            image.add_component(comp, component_name)
        dc = DataCollection([image])

        app = GlueApplication(dc)
        app.start()

        

    
