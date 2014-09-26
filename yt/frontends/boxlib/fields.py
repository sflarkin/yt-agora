"""
Orion-specific fields



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
import string
import re

from yt.utilities.physical_constants import \
    mh, boltzmann_constant_cgs, amu_cgs
from yt.fields.field_info_container import \
    FieldInfoContainer

rho_units = "code_mass / code_length**3"
mom_units = "code_mass / (code_time * code_length**2)"
eden_units = "code_mass / (code_time**2 * code_length)" # erg / cm^3

spec_finder = re.compile(r'.*\((\D*)(\d*)\).*')

def _thermal_energy_density(field, data):
    # What we've got here is UEINT:
    # u here is velocity
    # E is energy density from the file
    #   rho e = rho E - rho * u * u / 2
    ke = 0.5 * ( data["momentum_x"]**2
               + data["momentum_y"]**2
               + data["momentum_z"]**2) / data["density"]
    return data["eden"] - ke

def _thermal_energy(field, data):
    # This is little e, so we take thermal_energy_density and divide by density
    return data["thermal_energy_density"] / data["density"]

def _temperature(field,data):
    mu = data.ds.parameters["mu"]
    gamma = data.ds.parameters["gamma"]
    tr  = data["thermal_energy_density"] / data["density"]
    tr *= mu * amu_cgs / boltzmann_constant_cgs
    tr *= (gamma - 1.0)
    return tr

class BoxlibFieldInfo(FieldInfoContainer):
    known_other_fields = (
        ("density", (rho_units, ["density"], None)),
        ("eden", (eden_units, ["energy_density"], None)),
        ("xmom", (mom_units, ["momentum_x"], None)),
        ("ymom", (mom_units, ["momentum_y"], None)),
        ("zmom", (mom_units, ["momentum_z"], None)),
        ("temperature", ("K", ["temperature"], None)),
        ("Temp", ("K", ["temperature"], None)),
        ("x_velocity", ("cm/s", ["velocity_x"], None)),
        ("y_velocity", ("cm/s", ["velocity_y"], None)),
        ("z_velocity", ("cm/s", ["velocity_z"], None)),
        ("xvel", ("cm/s", ["velocity_x"], None)),
        ("yvel", ("cm/s", ["velocity_y"], None)),
        ("zvel", ("cm/s", ["velocity_z"], None)),
    )

    known_particle_fields = (
        ("particle_mass", ("code_mass", [], None)),
        ("particle_position_x", ("code_length", [], None)),
        ("particle_position_y", ("code_length", [], None)),
        ("particle_position_z", ("code_length", [], None)),
        ("particle_momentum_x", (mom_units, [], None)),
        ("particle_momentum_y", (mom_units, [], None)),
        ("particle_momentum_z", (mom_units, [], None)),
        # Note that these are *internal* agmomen
        ("particle_angmomen_x", ("code_length**2/code_time", [], None)),
        ("particle_angmomen_y", ("code_length**2/code_time", [], None)),
        ("particle_angmomen_z", ("code_length**2/code_time", [], None)),
        ("particle_id", ("", ["particle_index"], None)),
        ("particle_mdot", ("code_mass/code_time", [], None)),
        # "mlast",
        # "r",
        # "mdeut",
        # "n",
        # "burnstate",
        # "luminosity",
    )

    def setup_fluid_fields(self):
        # Now, let's figure out what fields are included.
        if any(f[1] == "xmom" for f in self.field_list):
            self.setup_momentum_to_velocity()
        self.add_field(("gas", "thermal_energy"),
                       function = _thermal_energy,
                       units = "erg/g")
        self.add_field(("gas", "thermal_energy_density"),
                       function = _thermal_energy_density,
                       units = "erg/cm**3")
        if ("gas", "temperature") not in self.field_aliases:
            self.add_field(("gas", "temperature"),
                           function=_temperature,
                           units="K")

    def setup_momentum_to_velocity(self):
        def _get_vel(axis):
            def velocity(field, data):
                return data["%smom" % axis]/data["density"]
        for ax in 'xyz':
            self.add_field(("gas", "velocity_%s" % ax),
                           function = _get_vel(ax),
                           units = "cm/s")

class CastroFieldInfo(FieldInfoContainer):

    known_other_fields = (
        ("density", ("g/cm**3", ["density"], r"\rho")),
        ("xmom", ("g/(cm**2 * s)", ["momentum_x"], r"\rho u")),
        ("ymom", ("g/(cm**2 * s)", ["momentum_y"], r"\rho v")),
        ("zmom", ("g/(cm**2 * s)", ["momentum_z"], r"\rho w")),
        # velocity components are not always present
        ("x_velocity", ("cm/s", ["velocity_x"], r"u")),
        ("y_velocity", ("cm/s", ["velocity_y"], r"v")),
        ("z_velocity", ("cm/s", ["velocity_z"], r"w")),
        ("rho_E", ("erg/cm**3", ["energy_density"], r"\rho E")),
        # internal energy density (not just thermal)
        ("rho_e", ("erg/cm**3", [], r"\rho e")),
        ("Temp", ("K", ["temperature"], r"T")),
        ("grav_x", ("cm/s**2", [], r"g\cdot e_x")),
        ("grav_y", ("cm/s**2", [], r"g\cdot e_y")),
        ("grav_z", ("cm/s**2", [], r"g\cdot e_z")),
        ("pressure", ("dyne/cm**2", [], r"p")),
        ("kineng", ("erg/cm**3", [], r"\frac{1}{2}\rho|U|**2")),
        ("soundspeed", ("cm/s", ["sound_speed"], None)),
        ("Machnumber", ("", ["mach_number"], None)),
        ("entropy", ("erg/(g*K)", ["entropy"], r"s")),
        ("magvort", ("1/s", ["vorticity_magnitude"], r"|\nabla \times U|")),
        ("divu", ("1/s", [], r"\nabla \cdot U")),
        ("eint_E", ("erg/g", [], r"e(E,U)")),
        ("eint_e", ("erg/g", [], r"e")),
        ("magvel", ("cm/s", ["velocity_magnitude"], r"|U|")),
        ("radvel", ("cm/s", [], r"U\cdot e_r")),
        ("magmom", ("g*cm/s", ["momentum_magnitude"], r"|\rho U|")),
        ("maggrav", ("cm/s**2", [], r"|g|")),
        ("phiGrav", ("erg/g", [], r"|\Phi|")),
    )

    def setup_fluid_fields(self):
        # add X's
        for _, field in self.ds.field_list:
            if field.startswith("X("):
                # We have a fraction
                nice_name = field[2:-1]
                self.alias(("gas", "%s_fraction" % nice_name), ("boxlib", field),
                           units = "")
                def _create_density_func(field_name):
                    def _func(field, data):
                        return data[field_name] * data["gas", "density"]
                    return _func
                func = _create_density_func(("gas", "%s_fraction" % nice_name))
                self.add_field(name = ("gas", "%s_density" % nice_name),
                               function = func,
                               units = "g/cm**3")
                # We know this will either have one letter, or two.
                if field[3] in string.letters:
                    element, weight = field[2:4], field[4:-1]
                else:
                    element, weight = field[2:3], field[3:-1]
                weight = int(weight)
                # Here we can, later, add number density.

class MaestroFieldInfo(FieldInfoContainer):

    known_other_fields = (
        ("density", ("g/cm**3", ["density"], None)),
        ("x_vel", ("cm/s", ["velocity_x"], r"\tilde{u}")),
        ("y_vel", ("cm/s", ["velocity_y"], r"\tilde{v}")),
        ("z_vel", ("cm/s", ["velocity_z"], r"\tilde{w}")),
        ("magvel", ("cm/s", ["velocity_magnitude"],
                    r"|\tilde{\mathbf{U}} + w_0 \mathbf{e}_r|")),
        ("radial_velocity", ("cm/s", [], r"U\cdot e_r")),
        ("tfromp", ("K", [], "T(\\rho,p,X)")),
        ("tfromh", ("K", [], "T(\\rho,h,X)")),
        ("Machnumber", ("", ["mach_number"], None)),
        ("S", ("1/s", [], None)),
        ("ad_excess", ("", [], "Adiabatic Excess")),
        ("deltaT", ("", [], "\delta T")),
        ("deltagamma", ("", [], "\Gamma_1^ - \bar \Gamma_1")),
        ("deltap", ("", [], "\delta p")),
        ("divw0", ("1/s", [], "\nabla w_0")),
        # Specific entropy
        ("entropy", ("erg/(g*K)", ["entropy"], "s")),
        ("entropypert", ("", [], "\delta s")),
        ("enucdot", ("erg/(g*s)", [], "\dot{\epsilon_{nuc}}")),
        ("Hext", ("erg/(g*s)", [], "H_{ext}")),
        # Perturbational pressure grad
        ("gpi_x", ("dyne/cm**3", [], "\left(\nabla\pi\right)_x")),
        ("gpi_y", ("dyne/cm**3", [], "\left(\nabla\pi\right)_y")),
        ("gpi_z", ("dyne/cm**3", [], "\left(\nabla\pi\right)_z")),
        ("h", ("erg/g", [], "h")),
        ("h0", ("erg/g", [], "h_0")),
        # Momentum cannot be computed because we need to include base and
        # full state.
        ("momentum", ("g*cm/s", ["momentum_magnitude"], None)),
        ("p0", ("erg/cm**3", [], "p_0")),
        ("p0pluspi", ("erg/cm**3", [], "p_0 + \pi")),
        ("pi", ("erg/cm**3", [], "\pi")),
        ("pioverp0", ("", [], "\pi/p_0")),
        # Base state density
        ("rho0", ("g/cm**3", [], "\\rho_0")),
        ("rhoh", ("erg/cm**3", ["enthalpy_density"], "(\\rho h)")),
        # Base state enthalpy density
        ("rhoh0", ("erg/cm**3", [], "(\\rho h)_0")),
        ("rhohpert", ("erg/cm**3", [], "(\\rho h)^\prime")),
        ("rhopert", ("g/cm**3", [], "\\rho^\prime")),
        ("soundspeed", ("cm/s", ["sound_speed"], None)),
        ("sponge", ("", [], None)),
        ("tpert", ("K", [], "T - \bar T")),
        # Again, base state -- so we can't compute ourselves.
        ("vort", ("1/s", ["vorticity_magnitude"], "|\nabla\times\tilde{U}|")),
        # Base state
        ("w0_x", ("cm/s", [], "(w_0)_x")),
        ("w0_y", ("cm/s", [], "(w_0)_y")),
        ("w0_z", ("cm/s", [], "(w_0)_z")),
    )

    def setup_fluid_fields(self):
        # pick the correct temperature field
        if self.ds.parameters["use_tfromp"]:
            self.alias(("gas", "temperature"), ("boxlib", "tfromp"),
                       units="K")
        else:
            self.alias(("gas", "temperature"), ("boxlib", "tfromh"),
                       units="K")

        # Add X's and omegadots, units of 1/s
        for _, field in self.ds.field_list:
            if field.startswith("X("):
                # We have a fraction
                nice_name, tex_label = _nice_species_name(field)
                print nice_name, tex_label
                self.alias(("gas", "%s_fraction" % nice_name),
                           ("boxlib", field),
                           units="")
                def _create_density_func(field_name):
                    def _func(field, data):
                        return data[field_name] * data["gas", "density"]
                    return _func
                func = _create_density_func(("gas", "%s_fraction" % nice_name))
                self.add_field(name=("gas", "%s_density" % nice_name),
                               function=func,
                               units="g/cm**3",
                               display_name=r'\\rho%s' % tex_label)

                # Most of the time our species will be of the form
                # element name + atomic weight (e.g. C12), but
                # sometimes we make up descriptive names (e.g. ash)
                if any(char.isdigit() for char in field):
                    # We know this will either have one letter, or two.
                    if field[3] in string.letters:
                        element, weight = field[2:4], field[4:-1]
                    else:
                        element, weight = field[2:3], field[3:-1]
                    weight = int(weight)

                # Here we can, later, add number density.
            if field.startswith("omegadot("):
                nice_name, tex_label = _nice_species_name(field)
                print nice_name, tex_label
                display_name = r'\dot{\omega}\left[%s\right]' % tex_label
                self.add_output_field(("boxlib", field), units="1/s",
                                      display_name=display_name)
                self.alias(("gas", "%s_creation_rate" % nice_name),
                           ("boxlib", field), units="1/s")


def _nice_species_name(field):
    nice_name = None
    tex_label = None
    spec_match = spec_finder.search(field)
    if spec_match is not None:
        nice_name = ''.join(spec_match.groups())
        tex_label = r"X\left(^{%s}%s\right)" % spec_match.groups()[::-1]
    return nice_name, tex_label
