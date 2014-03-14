"""
Fields specific to Enzo



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
from yt.units.yt_array import \
    YTArray
from yt.fields.species_fields import \
    add_species_field_by_density
from yt.utilities.physical_constants import \
    mh, me, mp, \
    mass_sun_cgs

b_units = "code_magnetic"
ra_units = "code_length / code_time**2"
rho_units = "code_mass / code_length**3"
vel_units = "code_velocity"

known_species_masses = dict(
  (sp, mh * v) for sp, v in [
                ("HI", 1.0),
                ("HII", 1.0),
                ("De", 1.0),
                ("HeI", 4.0),
                ("HeII", 4.0),
                ("HeIII", 4.0),
                ("H2I", 2.0),
                ("H2II", 2.0),
                ("HM", 1.0),
                ("DI", 2.0),
                ("DII", 2.0),
                ("HDI", 3.0),
    ])

known_species_names = {
    'HI'      : 'H',
    'HII'     : 'H_p1',
    'HeI'     : 'He',
    'HeII'    : 'He_p1',
    'HeIII'   : 'He_p2',
    'H2I'     : 'H2',
    'H2II'    : 'H2_p1',
    'HM'      : 'H_m1',
    'DI'      : 'D',
    'DII'     : 'D_p1',
    'HD'      : 'HD',
    'Electron': 'De'
}

class EnzoFieldInfo(FieldInfoContainer):
    known_other_fields = (
        ("Cooling_Time", ("code_time", ["cooling_time"], None)),
        ("HI_kph", ("1/code_time", [], None)),
        ("HeI_kph", ("1/code_time", [], None)),
        ("HeII_kph", ("1/code_time", [], None)),
        ("H2I_kdiss", ("1/code_time", [], None)),
        ("Bx", (b_units, ["magnetic_field_x"], None)),
        ("By", (b_units, ["magnetic_field_y"], None)),
        ("Bz", (b_units, ["magnetic_field_z"], None)),
        ("RadAccel1", (ra_units, ["radiation_acceleration_x"], None)),
        ("RadAccel2", (ra_units, ["radiation_acceleration_y"], None)),
        ("RadAccel3", (ra_units, ["radiation_acceleration_z"], None)),
        ("Dark_Matter_Density", (rho_units, ["dark_matter_density"], None)),
        ("Temperature", ("K", ["temperature"], None)),
        ("Dust_Temperature", ("K", ["dust_temperature"], None)),
        ("x-velocity", (vel_units, ["velocity_x"], None)),
        ("y-velocity", (vel_units, ["velocity_y"], None)),
        ("z-velocity", (vel_units, ["velocity_z"], None)),
        ("RaySegments", ("", ["ray_segments"], None)),
        ("PhotoGamma", (ra_units, ["photo_gamma"], None)),
        ("Density", (rho_units, ["density"], None)),
        ("Metal_Density", (rho_units, ["metal_density"], None)),
        # Note: we do not alias Electron_Density to anything
        ("Electron_Density", (rho_units, [], None)),
    )

    known_particle_fields = (
        ("particle_position_x", ("code_length", [], None)),
        ("particle_position_y", ("code_length", [], None)),
        ("particle_position_z", ("code_length", [], None)),
        ("particle_velocity_x", (vel_units, [], None)),
        ("particle_velocity_y", (vel_units, [], None)),
        ("particle_velocity_z", (vel_units, [], None)),
        ("creation_time", ("code_time", [], None)),
        ("dynamical_time", ("code_time", [], None)),
        ("metallicity_fraction", ("code_metallicity", [], None)),
        ("metallicity", ("", [], None)),
        ("particle_type", ("", [], None)),
        ("particle_index", ("", [], None)),
        ("particle_mass", ("code_mass", [], None)),
        ("GridID", ("", [], None)),
        ("identifier", ("", ["particle_index"], None)),
        ("level", ("", [], None)),
    )

    def __init__(self, pf, field_list):
        if pf.parameters["HydroMethod"] == 2:
            sl_left = slice(None,-2,None)
            sl_right = slice(1,-1,None)
            div_fac = 1.0
        else:
            sl_left = slice(None,-2,None)
            sl_right = slice(2,None,None)
            div_fac = 2.0
        slice_info = (sl_left, sl_right, div_fac)
        super(EnzoFieldInfo, self).__init__(pf, field_list, slice_info)

    def add_species_field(self, species):
        # This is currently specific to Enzo.  Hopefully in the future we will
        # have deeper integration with other systems, such as Dengo, to provide
        # better understanding of ionization and molecular states.
        #
        # We have several fields to add based on a given species field.  First
        # off, we add the species field itself.  Then we'll add a few more
        # items...
        #
        self.add_output_field(("enzo", "%s_Density" % species),
                           take_log=True,
                           units="code_mass/code_length**3")
        yt_name = known_species_names[species]
        self.alias(("gas", "%s_density" % yt_name),
                   ("enzo", "%s_Density" % species))
        add_species_field_by_density(self, "gas", yt_name)

    def setup_species_fields(self):
        species_names = [fn.rsplit("_Density")[0] for ft, fn in 
                         self.field_list if fn.endswith("_Density")]
        species_names = [sp for sp in species_names
                         if sp in known_species_names]
        def _electron_density(field, data):
            return data["Electron_Density"] * (me/mp)
        self.add_field(("gas", "De_density"),
                       function = _electron_density,
                       units = "g/cm**3")
        for sp in species_names:
            self.add_species_field(sp)


    def setup_fluid_fields(self):
        # Now we conditionally load a few other things.
        if self.pf.parameters["MultiSpecies"] > 0:
            self.setup_species_fields()
        self.setup_energy_field()

    def setup_energy_field(self):
        # We check which type of field we need, and then we add it.
        ge_name = None
        te_name = None
        if ("enzo", "Gas_Energy") in self.field_list:
            ge_name = "Gas_Energy"
        elif ("enzo", "GasEnergy") in self.field_list:
            ge_name = "GasEnergy"
        if ("enzo", "Total_Energy") in self.field_list:
            te_name = "Total_Energy"
        elif ("enzo", "TotalEnergy") in self.field_list:
            te_name = "TotalEnergy"

        if self.pf.parameters["HydroMethod"] == 2:
            self.add_output_field(("enzo", te_name),
                units="code_length**2/code_time**2")
            self.alias(("gas", "thermal_energy"), ("enzo", te_name))

        elif self.pf.parameters["DualEnergyFormalism"] == 1:
            self.add_output_field(
                ("enzo", ge_name),
                units="code_length**2/code_time**2")
            self.alias(
                ("gas", "thermal_energy"),
                ("enzo", ge_name),
                units = "erg/g")
        elif self.pf.parameters["HydroMethod"] in (4, 6):
            self.add_output_field(
                ("enzo", te_name),
                units="code_length**2/code_time**2")
            # Subtract off B-field energy
            def _sub_b(field, data):
                return data[te_name] - 0.5*(
                    data["x-velocity"]**2.0
                    + data["y-velocity"]**2.0
                    + data["z-velocity"]**2.0 ) \
                    - data["MagneticEnergy"]/data["Density"]
            self.add_field(
                ("gas", "thermal_energy"),
                function=_sub_b, units = "erg/g")
        else: # Otherwise, we assume TotalEnergy is kinetic+thermal
            self.add_output_field(
                ("enzo", te_name),
                units = "code_length**2/code_time**2")
            def _tot_minus_kin(field, data):
                return data[te_name] - 0.5*(
                    data["x-velocity"]**2.0
                    + data["y-velocity"]**2.0
                    + data["z-velocity"]**2.0 )
            self.add_field(
                ("gas", "thermal_energy"),
                units = "erg/g")

    def setup_particle_fields(self, ptype):

        def _age(field, data):
            return data.pf.current_time - data["creation_time"]
        self.add_field((ptype, "age"), function = _age,
                           particle_type = True,
                           units = "yr")

        super(EnzoFieldInfo, self).setup_particle_fields(ptype)
