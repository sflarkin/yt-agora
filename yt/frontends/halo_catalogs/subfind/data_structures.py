"""
Data structures for Subfind frontend.




"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

from collections import defaultdict
import h5py
import numpy as np
import stat
import weakref
import struct
import glob
import time
import os

from .fields import \
    SubfindFieldInfo

from yt.utilities.cosmology import Cosmology
from yt.utilities.definitions import \
    mpc_conversion, sec_conversion
from yt.utilities.exceptions import \
     YTException
from yt.geometry.particle_geometry_handler import \
    ParticleIndex
from yt.data_objects.static_output import \
    Dataset, \
    ParticleFile
from yt.frontends.sph.data_structures import \
    _fix_unit_ordering
import yt.utilities.fortran_utils as fpu
from yt.units.yt_array import \
    YTArray, \
    YTQuantity

class SubfindParticleIndex(ParticleIndex):
    def __init__(self, pf, dataset_type):
        super(SubfindParticleIndex, self).__init__(pf, dataset_type)

    def _calculate_particle_index_offsets(self):
        particle_count = defaultdict(int)
        for data_file in self.data_files:
            data_file.index_offset = dict([(ptype, particle_count[ptype]) for
                                           ptype in data_file.particle_count])
            for ptype in data_file.particle_count:
                particle_count[ptype] += data_file.particle_count[ptype]
        
    def _setup_geometry(self):
        super(SubfindParticleIndex, self)._setup_geometry()
        self._calculate_particle_index_offsets()
    
class SubfindHDF5File(ParticleFile):
    def __init__(self, pf, io, filename, file_id):
        with h5py.File(filename, "r") as f:
            self.header = dict((field, f.attrs[field]) \
                               for field in f.attrs.keys())

        super(SubfindHDF5File, self).__init__(pf, io, filename, file_id)
    
class SubfindDataset(Dataset):
    _index_class = SubfindParticleIndex
    _file_class = SubfindHDF5File
    _field_info_class = SubfindFieldInfo
    _suffix = ".hdf5"

    def __init__(self, filename, dataset_type="subfind_hdf5",
                 n_ref = 16, over_refine_factor = 1):
        self.n_ref = n_ref
        self.over_refine_factor = over_refine_factor
        super(SubfindDataset, self).__init__(filename, dataset_type)

    def _parse_parameter_file(self):
        handle = h5py.File(self.parameter_filename, mode="r")
        hvals = {}
        hvals.update((str(k), v) for k, v in handle["/Header"].attrs.items())
        hvals["NumFiles"] = hvals["NumFilesPerSnapshot"]
        hvals["Massarr"] = hvals["MassTable"]

        self.dimensionality = 3
        self.refine_by = 2
        self.unique_identifier = \
            int(os.stat(self.parameter_filename)[stat.ST_CTIME])

        # Set standard values
        self.current_time = self.quan(hvals["Time_GYR"] * sec_conversion["Gyr"], "s")
        self.domain_left_edge = np.zeros(3, "float64")
        self.domain_right_edge = np.ones(3, "float64") * hvals["BoxSize"]
        nz = 1 << self.over_refine_factor
        self.domain_dimensions = np.ones(3, "int32") * nz
        self.cosmological_simulation = 1
        self.periodicity = (True, True, True)
        self.current_redshift = hvals["Redshift"]
        self.omega_lambda = hvals["OmegaLambda"]
        self.omega_matter = hvals["Omega0"]
        self.hubble_constant = hvals["HubbleParam"]
        self.parameters = hvals
        prefix = os.path.abspath(
            os.path.join(os.path.dirname(self.parameter_filename), 
                         os.path.basename(self.parameter_filename).split(".", 1)[0]))
        
        suffix = self.parameter_filename.rsplit(".", 1)[-1]
        self.filename_template = "%s.%%(num)i.%s" % (prefix, suffix)
        self.file_count = len(glob.glob(prefix + "*" + self._suffix))
        if self.file_count == 0:
            raise YTException(message="No data files found.", pf=self)
        self.particle_types = ("FOF", "SUBFIND")
        self.particle_types_raw = ("FOF", "SUBFIND")
        
        # To avoid having to open files twice
        self._unit_base = {}
        self._unit_base.update(
            (str(k), v) for k, v in handle["/Units"].attrs.items())
        # Comoving cm is given in the Units
        self._unit_base['cmcm'] = 1.0 / self._unit_base["UnitLength_in_cm"]
        handle.close()

    def _set_code_unit_attributes(self):
        # Set a sane default for cosmological simulations.
        if self._unit_base is None and self.cosmological_simulation == 1:
            mylog.info("Assuming length units are in Mpc/h (comoving)")
            self._unit_base = dict(length = (1.0, "Mpccm/h"))
        # The other same defaults we will use from the standard Gadget
        # defaults.
        unit_base = self._unit_base or {}
        if "length" in unit_base:
            length_unit = unit_base["length"]
        elif "UnitLength_in_cm" in unit_base:
            if self.cosmological_simulation == 0:
                length_unit = (unit_base["UnitLength_in_cm"], "cm")
            else:
                length_unit = (unit_base["UnitLength_in_cm"], "cmcm/h")
        else:
            raise RuntimeError
        length_unit = _fix_unit_ordering(length_unit)
        self.length_unit = self.quan(length_unit[0], length_unit[1])

        unit_base = self._unit_base or {}
        if "velocity" in unit_base:
            velocity_unit = unit_base["velocity"]
        elif "UnitVelocity_in_cm_per_s" in unit_base:
            velocity_unit = (unit_base["UnitVelocity_in_cm_per_s"], "cm/s")
        else:
            velocity_unit = (1e5, "cm/s")
        velocity_unit = _fix_unit_ordering(velocity_unit)
        self.velocity_unit = self.quan(velocity_unit[0], velocity_unit[1])
        # We set hubble_constant = 1.0 for non-cosmology, so this is safe.
        # Default to 1e10 Msun/h if mass is not specified.
        if "mass" in unit_base:
            mass_unit = unit_base["mass"]
        elif "UnitMass_in_g" in unit_base:
            if self.cosmological_simulation == 0:
                mass_unit = (unit_base["UnitMass_in_g"], "g")
            else:
                mass_unit = (unit_base["UnitMass_in_g"], "g/h")
        else:
            # Sane default
            mass_unit = (1.0, "1e10*Msun/h")
        mass_unit = _fix_unit_ordering(mass_unit)
        self.mass_unit = self.quan(mass_unit[0], mass_unit[1])
        self.time_unit = self.quan(unit_base["UnitTime_in_s"], "s")

    @classmethod
    def _is_valid(self, *args, **kwargs):
        try:
            fileh = h5py.File(args[0], mode='r')
            if "Constants" in fileh["/"].keys() and \
               "Header" in fileh["/"].keys() and \
               "SUBFIND" in fileh["/"].keys():
                fileh.close()
                return True
            fileh.close()
        except:
            pass
        return False
