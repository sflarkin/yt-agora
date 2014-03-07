"""
Data structures for HaloCatalog frontend.




"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import h5py
import numpy as np
import stat
import weakref
import struct
import glob
import time
import os

from .fields import \
    HaloCatalogFieldInfo

from yt.utilities.cosmology import Cosmology
from yt.geometry.particle_geometry_handler import \
    ParticleIndex
from yt.data_objects.static_output import \
    Dataset, \
    ParticleFile
import yt.utilities.fortran_utils as fpu
from yt.units.yt_array import \
    YTArray, \
    YTQuantity
    
class HaloCatalogHDF5File(ParticleFile):
    def __init__(self, pf, io, filename, file_id):
        with h5py.File(filename, "r") as f:
            self.header = dict((field, f.attrs[field]) \
                               for field in f.attrs.keys())

        super(HaloCatalogHDF5File, self).__init__(pf, io, filename, file_id)
    
class HaloCatalogDataset(Dataset):
    _index_class = ParticleIndex
    _file_class = HaloCatalogHDF5File
    _field_info_class = HaloCatalogFieldInfo
    _particle_mass_name = "particle_mass"
    _particle_coordinates_name = "Coordinates"
    _suffix = ".h5"

    def __init__(self, filename, dataset_type="halocatalog_hdf5",
                 n_ref = 16, over_refine_factor = 1):
        self.n_ref = n_ref
        self.over_refine_factor = over_refine_factor
        super(HaloCatalogDataset, self).__init__(filename, dataset_type)

    def _parse_parameter_file(self):
        with h5py.File(self.parameter_filename, "r") as f:
            hvals = dict((key, f.attrs[key]) for key in f.attrs.keys())
        self.dimensionality = 3
        self.refine_by = 2
        self.unique_identifier = \
            int(os.stat(self.parameter_filename)[stat.ST_CTIME])
        prefix = ".".join(self.parameter_filename.rsplit(".", 2)[:-2])
        self.filename_template = "%s.%%(num)s%s" % (prefix, self._suffix)
        self.file_count = len(glob.glob(prefix + "*" + self._suffix))

        for attr in ["cosmological_simulation", "current_time", "current_redshift",
                     "hubble_constant", "omega_matter", "omega_lambda",
                     "domain_left_edge", "domain_right_edge"]:
            setattr(self, attr, hvals[attr])
        self.periodicity = (True, True, True)
        self.particle_types = ("halos")
        self.particle_types_raw = ("halos")

        nz = 1 << self.over_refine_factor
        self.domain_dimensions = np.ones(3, "int32") * nz
        self.parameters.update(hvals)

    def _set_code_unit_attributes(self):
        self.length_unit = self.quan(1.0, "cm")
        self.mass_unit = self.quan(1.0, "g")
        self.velocity_unit = self.quan(1.0, "cm / s")
        self.time_unit = self.quan(1.0, "s")

    @classmethod
    def _is_valid(self, *args, **kwargs):
        if not args[0].endswith(".h5"): return False
        with h5py.File(args[0], "r") as f:
            if "data_type" in f.attrs and \
              f.attrs["data_type"] == "halo_catalog":
                return True
        return False
