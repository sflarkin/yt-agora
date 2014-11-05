"""
Equivalencies between different kinds of units

"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import yt.utilities.physical_constants as pc
from yt.units.dimensions import temperature, mass, energy, length, rate, \
    velocity, dimensionless, density, number_density, flux
from yt.extern.six import add_metaclass
import numpy as np

equivalence_registry = {}

class RegisteredEquivalence(type):
    def __init__(cls, name, b, d):
        type.__init__(cls, name, b, d)
        if hasattr(cls, "_type_name") and not cls._skip_add:
            equivalence_registry[cls._type_name] = cls

@add_metaclass(RegisteredEquivalence)
class Equivalence(object):
    _skip_add = False

class NumberDensityEquivalence(Equivalence):
    _type_name = "number_density"
    dims = (density,number_density,)

    def convert(self, x, new_dims, mu=0.6):
        if new_dims == number_density:
            return x/(mu*pc.mh)
        elif new_dims == density:
            return x*mu*pc.mh

    def __str__(self):
        return "number density: density <-> number density"

class ThermalEquivalence(Equivalence):
    _type_name = "thermal"
    dims = (temperature,energy,)

    def convert(self, x, new_dims):
        if new_dims == energy:
            return pc.kboltz*x
        elif new_dims == temperature:
            return x/pc.kboltz

    def __str__(self):
        return "thermal: temperature <-> energy"

class MassEnergyEquivalence(Equivalence):
    _type_name = "mass_energy"
    dims = (mass,energy,)

    def convert(self, x, new_dims):
        if new_dims == energy:
            return x*pc.clight*pc.clight
        elif new_dims == mass:
            return x/(pc.clight*pc.clight)

    def __str__(self):
        return "mass_energy: mass <-> energy"

class SpectralEquivalence(Equivalence):
    _type_name = "spectral"
    dims = (length,rate,energy,)

    def convert(self, x, new_dims):
        if new_dims == energy:
            if x.units.dimensions == length:
                nu = pc.clight/x
            elif x.units.dimensions == rate:
                nu = x
            return pc.hcgs*nu
        elif new_dims == length:
            if x.units.dimensions == rate:
                return pc.clight/x
            elif x.units.dimensions == energy:
                return pc.hcgs*pc.clight/x
        elif new_dims == rate:
            if x.units.dimensions == length:
                return pc.clight/x
            elif x.units.dimensions == energy:
                return x/pc.hcgs

    def __str__(self):
        return "spectral: length <-> rate <-> energy"

class SoundSpeedEquivalence(Equivalence):
    _type_name = "sound_speed"
    dims = (velocity,temperature,energy,)

    def convert(self, x, new_dims, mu=0.6, gamma=5./3.):
        if new_dims == velocity:
            if x.units.dimensions == temperature:
                kT = pc.kboltz*x
            elif x.units.dimensions == energy:
                kT = x
            return np.sqrt(gamma*kT/(mu*pc.mh))
        else:
            kT = x*x*mu*pc.mh/gamma
            if new_dims == temperature:
                return kT/pc.kboltz
            else:
                return kT

    def __str__(self):
        return "sound_speed (ideal gas): velocity <-> temperature <-> energy"

class LorentzEquivalence(Equivalence):
    _type_name = "lorentz"
    dims = (dimensionless,velocity,)

    def convert(self, x, new_dims):
        if new_dims == dimensionless:
            beta = x.in_cgs()/pc.clight
            return 1./np.sqrt(1.-beta**2)
        elif new_dims == velocity:
            return pc.clight*np.sqrt(1.-1./(x*x))

    def __str__(self):
        return "lorentz: velocity <-> dimensionless"

class SchwarzschildEquivalence(Equivalence):
    _type_name = "schwarzschild"
    dims = (mass,length,)

    def convert(self, x, new_dims):
        if new_dims == length:
            return 2.*pc.G*x/(pc.clight*pc.clight)
        elif new_dims == mass:
            return 0.5*x*pc.clight*pc.clight/pc.G

    def __str__(self):
        return "schwarzschild: mass <-> length"

class ComptonEquivalence(Equivalence):
    _type_name = "compton"
    dims = (mass,length,)

    def convert(self, x, new_dims):
        return pc.hcgs/(x*pc.clight)

    def __str__(self):
        return "compton: mass <-> length"

class EffectiveTemperature(Equivalence):
    _type_name = "effective_temperature"
    dims = (flux,temperature,)

    def convert(self, x, new_dims):
        if new_dims == flux:
            return pc.stefan_boltzmann_constant_cgs*x**4
        elif new_dims == temperature:
            return (x/pc.stefan_boltzmann_constant_cgs)**0.25

    def __str__(self):
        return "effective_temperature: flux <-> temperature"

