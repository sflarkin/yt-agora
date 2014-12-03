"""
Generating PPV FITS cubes
"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
from yt.utilities.on_demand_imports import _astropy
from yt.utilities.orientation import Orientation
from yt.utilities.fits_image import FITSImageBuffer, sanitize_fits_unit, \
    create_sky_wcs
from yt.visualization.volume_rendering.camera import off_axis_projection
from yt.funcs import get_pbar
from yt.utilities.physical_constants import clight, mh
import yt.units.dimensions as ytdims
from yt.units.yt_array import YTQuantity
from yt.funcs import iterable
from yt.utilities.parallel_tools.parallel_analysis_interface import \
    parallel_root_only, parallel_objects
import re
import ppv_utils
from yt.funcs import is_root

def create_vlos(normal, no_shifting):
    if no_shifting:
        def _v_los(field, data):
            return data.ds.arr(data["zeros"], "cm/s")
    elif isinstance(normal, basestring):
        def _v_los(field, data): 
            return -data["velocity_%s" % normal]
    else:
        orient = Orientation(normal)
        los_vec = orient.unit_vectors[2]
        def _v_los(field, data):
            vz = data["velocity_x"]*los_vec[0] + \
                data["velocity_y"]*los_vec[1] + \
                data["velocity_z"]*los_vec[2]
            return -vz
    return _v_los

fits_info = {"velocity":("m/s","VELOCITY","v"),
             "frequency":("Hz","FREQUENCY","f"),
             "energy":("eV","ENERGY","E"),
             "wavelength":("angstrom","WAVELENG","lambda")}

class PPVCube(object):
    def __init__(self, ds, normal, field, center="c", width=(1.0,"unitary"),
                 dims=(100,100,100), velocity_bounds=None, thermal_broad=False,
                 atomic_weight=56., method="integrate", no_shifting=False):
        r""" Initialize a PPVCube object.

        Parameters
        ----------
        ds : dataset
            The dataset.
        normal : array_like or string
            The normal vector along with to make the projections. If an array, it
            will be normalized. If a string, it will be assumed to be along one of the
            principal axes of the domain ("x","y", or "z").
        field : string
            The field to project.
        center : A sequence of floats, a string, or a tuple.
            The coordinate of the center of the image. If set to 'c', 'center' or
            left blank, the plot is centered on the middle of the domain. If set to
            'max' or 'm', the center will be located at the maximum of the
            ('gas', 'density') field. Centering on the max or min of a specific
            field is supported by providing a tuple such as ("min","temperature") or
            ("max","dark_matter_density"). Units can be specified by passing in *center*
            as a tuple containing a coordinate and string unit name or by passing
            in a YTArray. If a list or unitless array is supplied, code units are
            assumed.
        width : float, tuple, or YTQuantity.
            The width of the projection. A float will assume the width is in code units.
            A (value, unit) tuple or YTQuantity allows for the units of the width to be
            specified.
        dims : tuple, optional
            A 3-tuple of dimensions (nx,ny,nv) for the cube.
        velocity_bounds : tuple, optional
            A 3-tuple of (vmin, vmax, units) for the velocity bounds to
            integrate over. If None, the largest velocity of the
            dataset will be used, e.g. velocity_bounds = (-v.max(), v.max())
        atomic_weight : float, optional
            Set this value to the atomic weight of the particle that is emitting the line
            if *thermal_broad* is True. Defaults to 56 (Fe).
        method : string, optional
            Set the projection method to be used.
            "integrate" : line of sight integration over the line element.
            "sum" : straight summation over the line of sight.
        no_shifting : boolean, optional
            If set, no shifting due to velocity will occur but only thermal broadening.
            Should not be set when *thermal_broad* is False, otherwise nothing happens!

        Examples
        --------
        >>> i = 60*np.pi/180.
        >>> L = [0.0,np.sin(i),np.cos(i)]
        >>> cube = PPVCube(ds, L, "density", width=(10.,"kpc"),
        ...                velocity_bounds=(-5.,4.,"km/s"))
        """

        self.ds = ds
        self.field = field
        self.width = width
        self.particle_mass = atomic_weight*mh
        self.thermal_broad = thermal_broad
        self.no_shifting = no_shifting

        if no_shifting and not thermal_broad:
            raise RuntimeError("no_shifting cannot be True when thermal_broad is False!")

        self.center = ds.coordinates.sanitize_center(center, normal)[0]

        self.nx = dims[0]
        self.ny = dims[1]
        self.nv = dims[2]

        if method not in ["integrate","sum"]:
            raise RuntimeError("Only the 'integrate' and 'sum' projection +"
                               "methods are supported in PPVCube.")

        dd = ds.all_data()

        fd = dd._determine_fields(field)[0]

        self.field_units = ds._get_field_info(fd).units

        if velocity_bounds is None:
            vmin, vmax = dd.quantities.extrema("velocity_magnitude")
            self.v_bnd = -vmax, vmax
        else:
            self.v_bnd = (ds.quan(velocity_bounds[0], velocity_bounds[2]),
                          ds.quan(velocity_bounds[1], velocity_bounds[2]))

        self.vbins = np.linspace(self.v_bnd[0], self.v_bnd[1], num=self.nv+1)
        self._vbins = self.vbins.copy()
        self.vmid = 0.5*(self.vbins[1:]+self.vbins[:-1])
        self.vmid_cgs = self.vmid.in_cgs().v
        self.dv = self.vbins[1]-self.vbins[0]
        self.dv_cgs = self.dv.in_cgs().v

        self.current_v = 0.0

        _vlos = create_vlos(normal, self.no_shifting)
        self.ds.add_field(("gas","v_los"), function=_vlos, units="cm/s")

        _intensity = self.create_intensity()
        self.ds.add_field(("gas","intensity"), function=_intensity, units=self.field_units)

        if method == "integrate":
            self.proj_units = str(ds.quan(1.0, self.field_units+"*cm").units)
        elif method == "sum":
            self.proj_units = self.field_units

        self.data = ds.arr(np.zeros((self.nx,self.ny,self.nv)), self.proj_units)
        storage = {}
        pbar = get_pbar("Generating cube.", self.nv)
        for sto, i in parallel_objects(xrange(self.nv), storage=storage):
            self.current_v = self.vmid_cgs[i]
            if isinstance(normal, basestring):
                prj = ds.proj("intensity", ds.coordinates.axis_id[normal], method=method)
                buf = prj.to_frb(width, self.nx, center=self.center)["intensity"]
            else:
                buf = off_axis_projection(ds, self.center, normal, width,
                                          (self.nx, self.ny), "intensity",
                                          no_ghost=True, method=method)[::-1]
            sto.result_id = i
            sto.result = buf
            pbar.update(i)
        pbar.finish()

        self.data = ds.arr(np.zeros((self.nx,self.ny,self.nv)), self.proj_units)
        if is_root():
            for i, buf in sorted(storage.items()):
                self.data[:,:,i] = buf[:,:]

        self.axis_type = "velocity"

        # Now fix the width
        if iterable(self.width):
            self.width = ds.quan(self.width[0], self.width[1])
        elif isinstance(self.width, YTQuantity):
            self.width = width
        else:
            self.width = ds.quan(self.width, "code_length")

        self.ds.field_info.pop(("gas","intensity"))
        self.ds.field_info.pop(("gas","v_los"))

    def create_intensity(self):
        def _intensity(field, data):
            v = self.current_v-data["v_los"].v
            T = data["temperature"].v
            w = ppv_utils.compute_weight(self.thermal_broad, self.dv_cgs,
                                         self.particle_mass, v.flatten(), T.flatten())
            w[np.isnan(w)] = 0.0                                                                                                                        
            return data[self.field]*w.reshape(v.shape)                                                                                                  
        return _intensity

    def transform_spectral_axis(self, rest_value, units):
        """
        Change the units of the spectral axis to some equivalent unit, such
        as energy, wavelength, or frequency, by providing a *rest_value* and the
        *units* of the new spectral axis. This corresponds to the Doppler-shifting
        of lines due to gas motions and thermal broadening.
        """
        if self.axis_type != "velocity":
            self.reset_spectral_axis()
        x0 = self.ds.quan(rest_value, units)
        if x0.units.dimensions == ytdims.rate or x0.units.dimensions == ytdims.energy:
            self.vbins = x0*(1.-self.vbins.in_cgs()/clight)
        elif x0.units.dimensions == ytdims.length:
            self.vbins = x0/(1.-self.vbins.in_cgs()/clight)
        self.vmid = 0.5*(self.vbins[1:]+self.vbins[:-1])
        self.dv = self.vbins[1]-self.vbins[0]
        dims = self.dv.units.dimensions
        if dims == ytdims.rate:
            self.axis_type = "frequency"
        elif dims == ytdims.length:
            self.axis_type = "wavelength"
        elif dims == ytdims.energy:
            self.axis_type = "energy"
        elif dims == ytdims.velocity:
            self.axis_type = "velocity"

    def reset_spectral_axis(self):
        """
        Reset the spectral axis to the original velocity range and units.
        """
        self.vbins = self._vbins.copy()
        self.vmid = 0.5*(self.vbins[1:]+self.vbins[:-1])
        self.dv = self.vbins[1]-self.vbins[0]

    @parallel_root_only
    def write_fits(self, filename, clobber=True, length_unit=None,
                   sky_scale=None, sky_center=None):
        r""" Write the PPVCube to a FITS file.

        Parameters
        ----------
        filename : string
            The name of the file to write.
        clobber : boolean
            Whether or not to clobber an existing file with the same name.
        length_unit : string
            The units to convert the coordinates to in the file.
        sky_scale : tuple, optional
            Conversion between an angle unit and a length unit, if sky
            coordinates are desired, e.g. (1.0, "arcsec/kpc")
        sky_center : tuple, optional
            The (RA, Dec) coordinate in degrees of the central pixel. Must
            be specified with *sky_scale*.

        Examples
        --------
        >>> cube.write_fits("my_cube.fits", clobber=False, sky_scale=(1.0,"arcsec/kpc"))
        """
        vunit = fits_info[self.axis_type][0]
        vtype = fits_info[self.axis_type][1]

        v_center = 0.5*(self.vbins[0]+self.vbins[-1]).in_units(vunit).value

        if length_unit is None:
            units = str(self.ds.get_smallest_appropriate_unit(self.width))
        else:
            units = length_unit
        units = sanitize_fits_unit(units)
        dx = self.width.in_units(units).v/self.nx
        dy = self.width.in_units(units).v/self.ny
        dv = self.dv.in_units(vunit).v

        w = _astropy.pywcs.WCS(naxis=3)
        w.wcs.crpix = [0.5*(self.nx+1), 0.5*(self.ny+1), 0.5*(self.nv+1)]
        w.wcs.cdelt = [dx,dy,dv]
        w.wcs.crval = [0.0,0.0,v_center]
        w.wcs.cunit = [units,units,vunit]
        w.wcs.ctype = ["LINEAR","LINEAR",vtype]

        if sky_scale is not None and sky_center is not None:
            w = create_sky_wcs(w, sky_center, sky_scale)

        fib = FITSImageBuffer(self.data.transpose(2,0,1), fields=self.field, wcs=w)
        fib[0].header["bunit"] = re.sub('()', '', str(self.proj_units))
        fib[0].header["btype"] = self.field

        fib.writeto(filename, clobber=clobber)

    def __repr__(self):
        return "PPVCube [%d %d %d] (%s < %s < %s)" % (self.nx, self.ny, self.nv,
                                                      self.vbins[0],
                                                      fits_info[self.axis_type][2],
                                                      self.vbins[-1])

    def __getitem__(self, item):
        return self.data[item]
