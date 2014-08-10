"""
ARTIO-specific fields




"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np

from yt.funcs import mylog
from yt.fields.field_info_container import \
    FieldInfoContainer
from yt.fields.field_detector import \
    FieldDetector
from yt.units.yt_array import \
    YTArray

from yt.utilities.physical_constants import \
    mh, \
    mass_sun_cgs, \
    boltzmann_constant_cgs, \
    amu_cgs

b_units = "code_magnetic"
ra_units = "code_length / code_time**2"
rho_units = "code_mass / code_length**3"
vel_units = "code_velocity"
# NOTE: ARTIO uses momentum density.
mom_units = "code_mass / (code_length**2 * code_time)"
en_units = "code_mass*code_velocity**2/code_length**3"

class ARTIOFieldInfo(FieldInfoContainer):
    known_other_fields = (
        ("HVAR_GAS_DENSITY", (rho_units, ["density"], None)),
        ("HVAR_GAS_ENERGY", (en_units, ["total_energy"], None)),
        ("HVAR_INTERNAL_ENERGY", (en_units, ["thermal_energy"], None)),
        ("HVAR_PRESSURE", ("", ["pressure"], None)), # Unused
        ("HVAR_MOMENTUM_X", (mom_units, ["momentum_x"], None)),
        ("HVAR_MOMENTUM_Y", (mom_units, ["momentum_y"], None)),
        ("HVAR_MOMENTUM_Z", (mom_units, ["momentum_z"], None)),
        ("HVAR_GAMMA", ("", ["gamma"], None)),
        ("HVAR_METAL_DENSITY_Ia", (rho_units, ["metal_ia_density"], None)),
        ("HVAR_METAL_DENSITY_II", (rho_units, ["metal_ii_density"], None)),
        ("VAR_POTENTIAL", ("", ["potential"], None)),
        ("VAR_POTENTIAL_HYDRO", ("", ["gas_potential"], None)),
    )

    known_particle_fields = (
        ("POSITION_X", ("code_length", ["particle_position_x"], None)),
        ("POSITION_Y", ("code_length", ["particle_position_y"], None)),
        ("POSITION_Z", ("code_length", ["particle_position_z"], None)),
        ("VELOCITY_X", (vel_units, ["particle_velocity_x"], None)),
        ("VELOCITY_Y", (vel_units, ["particle_velocity_y"], None)),
        ("VELOCITY_Z", (vel_units, ["particle_velocity_z"], None)),
        ("MASS", ("code_mass", ["particle_mass"], None)),
        ("PID", ("", ["particle_index"], None)),
        ("SPECIES", ("", ["particle_type"], None)),
        ("BIRTH_TIME", ("", [], None)),  # code-units defined as dimensionless to 
                                         # avoid incorrect conversion 
        ("INITIAL_MASS", ("code_mass", ["initial_mass"], None)),
        ("METALLICITY_SNIa", ("", ["metallicity_snia"], None)),
        ("METALLICITY_SNII", ("", ["metallicity_snii"], None)),
    )

    def setup_fluid_fields(self):
        def _get_vel(axis):
            def velocity(field, data):
                return data["momentum_%s" % axis]/data["density"]
            return velocity
        for ax in 'xyz':
            self.add_field(("gas", "velocity_%s" % ax),
                           function = _get_vel(ax),
                           units = "cm/s")

        def _temperature(field, data):
            tr = data["thermal_energy"]/data["density"]
            # We want this to match *exactly* what ARTIO would compute
            # internally.  We therefore use the exact values that are internal
            # to ARTIO, rather than yt's own internal constants.
            mH  = 1.007825*amu_cgs
            mHe = 4.002602*amu_cgs
            Yp    = 0.24
            XH    = 1.0 - Yp
            XHe   = 0.25*Yp
            mb = XH*mH + XHe*mHe
            wmu   = 4.0/(8.0-5.0*Yp)
            # Note that we have gamma = 5.0/3.0 here
            tr *= (data["gamma"] - 1.0)
            tr *= wmu
            tr *= mb/boltzmann_constant_cgs
            return tr
        # TODO: The conversion factor here needs to be addressed, as previously
        # it was set as:
        # unit_T = unit_v**2.0*mb / constants.k
        self.add_field(("gas", "temperature"), function = _temperature,
                       units = "K")

        def _metal_density(field, data):
            tr = data["metal_ia_density"]
            tr += data["metal_ii_density"]
            return tr
        self.add_field(("gas","metal_density"),
                       function=_metal_density,
                       units="g/cm**3",
                       take_log=True)

    def setup_particle_fields(self, ptype):
        def _creation_time(field,data):
            if isinstance(data,FieldDetector):
                return data["BIRTH_TIME"]
            if any(data["BIRTH_TIME"] > 0.0):
                raise ValueError("Invalid value for BIRTH_TIME found in ARTIO frontend")
            return YTArray(data.ds._handle.tphys_from_tcode_array(data["BIRTH_TIME"]),"yr")
        

        def _creation_redshift(field,data):
            if isinstance(data,FieldDetector):
                return data["BIRTH_TIME"]
            if any(data["BIRTH_TIME"] > 0.0):
                raise ValueError("Invalid value for BIRTH_TIME found in ARTIO frontend")
            return 1.0/data.ds._handle.auni_from_tcode_array(data["BIRTH_TIME"]) - 1.0

        def _age(field, data):
            if isinstance(data,FieldDetector):
                return data["BIRTH_TIME"]
            return data.ds.current_time - data["creation_time"]

        self.add_field((ptype, "creation_time"), function=_creation_time, units="yr",
                    particle_type=True)
        self.add_field((ptype, "creation_redshift"), function=_creation_redshift,
                    particle_type=True)
        self.add_field((ptype, "age"), function=_age, units="yr",
                    particle_type=True)

        super(ARTIOFieldInfo, self).setup_particle_fields(ptype)
