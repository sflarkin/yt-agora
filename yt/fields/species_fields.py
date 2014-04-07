"""
Fields based on species of molecules or atoms.



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
import re

from yt.utilities.physical_constants import \
    mh, \
    mass_sun_cgs, \
    amu_cgs
from yt.funcs import *
from yt.utilities.chemical_formulas import \
    ChemicalFormula

# See YTEP-0003 for details, but we want to ensure these fields are all
# populated:
#
#   * _mass
#   * _density
#   * _fraction
#   * _number_density
#

def _create_fraction_func(ftype, species):
    def _frac(field, data):
        return data[ftype, "%s_density" % species] \
             / data[ftype, "density"]
    return _frac

def _create_mass_func(ftype, species):
    def _mass(field, data):
        return data[ftype, "%s_density" % species] \
             * data["index", "cell_volume"]
    return _mass

def _create_number_density_func(ftype, species):
    formula = ChemicalFormula(species)
    weight = formula.weight # This is in AMU
    weight *= amu_cgs
    def _number_density(field, data):
        return data[ftype, "%s_density" % species] \
             / weight
    return _number_density

def _create_density_func(ftype, species):
    def _density(field, data):
        return data[ftype, "%s_fraction" % species] \
            * data[ftype,'density']
    return _density

def add_species_field_by_density(registry, ftype, species, 
                                 particle_type = False):
    """
    This takes a field registry, a fluid type, and a species name and then
    adds the other fluids based on that.  This assumes that the field
    "SPECIES_density" already exists and refers to mass density.
    """
    registry.add_field((ftype, "%s_fraction" % species), 
                       function = _create_fraction_func(ftype, species),
                       particle_type = particle_type,
                       units = "")

    registry.add_field((ftype, "%s_mass" % species),
                       function = _create_mass_func(ftype, species),
                       particle_type = particle_type,
                       units = "g")

    registry.add_field((ftype, "%s_number_density" % species),
                       function = _create_number_density_func(ftype, species),
                       particle_type = particle_type,
                       units = "cm**-3")

def add_species_field_by_fraction(registry, ftype, species, 
                                  particle_type = False):
    """
    This takes a field registry, a fluid type, and a species name and then
    adds the other fluids based on that.  This assumes that the field
    "SPECIES_fraction" already exists and refers to mass fraction.
    """
    registry.add_field((ftype, "%s_density" % species), 
                       function = _create_density_func(ftype, species),
                       particle_type = particle_type,
                       units = "g/cm**3")

    registry.add_field((ftype, "%s_mass" % species),
                       function = _create_mass_func(ftype, species),
                       particle_type = particle_type,
                       units = "g")

    registry.add_field((ftype, "%s_number_density" % species),
                       function = _create_number_density_func(ftype, species),
                       particle_type = particle_type,
                       units = "cm**-3")

def add_nuclei_density_fields(registry, ftype,
                              particle_type = False):
    elements = _get_all_elements(registry.species_names)
    for element in elements:
        registry.add_field((ftype, "%s_nuclei_density" % element),
                           function = _nuclei_density,
                           particle_type = particle_type,
                           units = "cm**-3")

def _nuclei_density(field, data):
    element = field.name[1][:field.name[1].find("_")]
    field_data = np.zeros_like(data["gas", "%s_number_density" % 
                                    data.pf.field_info.species_names[0]])
    for species in data.pf.field_info.species_names:
        nucleus = species
        if "_" in species:
            nucleus = species[:species.find("_")]
        num = _get_element_multiple(nucleus, element)
        field_data += num * data["gas", "%s_number_density" % species]
    return field_data

def _get_all_elements(species_list):
    elements = []
    for species in species_list:
        for item in re.findall('[A-Z][a-z]?|[0-9]+', species):
            if not item.isdigit() and item not in elements \
              and item != "El":
                elements.append(item)
    return elements
    
def _get_element_multiple(compound, element):
    my_split = re.findall('[A-Z][a-z]?|[0-9]+', compound)
    if element not in my_split:
        return 0
    loc = my_split.index(element)
    if loc == len(my_split) - 1 or not my_split[loc + 1].isdigit():
        return 1
    return int(my_split[loc + 1])
