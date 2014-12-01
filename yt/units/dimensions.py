"""
Base dimensions


"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

from sympy import Symbol, sympify, Rational

mass = Symbol("(mass)", positive=True)
length = Symbol("(length)", positive=True)
time = Symbol("(time)", positive=True)
temperature = Symbol("(temperature)", positive=True)
angle = Symbol("(angle)", positive=True)
current_mks = Symbol("(current_mks)", positive=True)
dimensionless = sympify(1)

base_dimensions = [mass, length, time, temperature, angle, current_mks,
                   dimensionless]

#
# Derived dimensions
#

rate = 1 / time

velocity     = length / time
acceleration = length / time**2
jerk         = length / time**3
snap         = length / time**4
crackle      = length / time**5
pop          = length / time**6

area     = length * length
volume   = area * length
momentum = mass * velocity
force    = mass * acceleration
energy   = force * length
power    = energy / time
flux     = power / area
specific_flux = flux / rate
number_density = 1/(length*length*length)
density = mass * number_density

# Gaussian electromagnetic units
charge_cgs  = (energy * length)**Rational(1, 2)  # proper 1/2 power
current_cgs = charge_cgs / time
electric_field_cgs = charge_cgs / length**2
magnetic_field_cgs = electric_field_cgs

# SI electromagnetic units
charge_mks = current_mks * time
electric_field_mks = force / charge_mks
magnetic_field_mks = electric_field_mks / velocity

# Since cgs is our default, I'm adding these aliases for backwards-compatibility
charge = charge_cgs
electric_field = electric_field_cgs
magnetic_field = magnetic_field_cgs

solid_angle = angle * angle

derived_dimensions = [rate, velocity, acceleration, jerk, snap, crackle, pop, 
                      momentum, force, energy, power, charge_cgs, electric_field_cgs,
                      magnetic_field_cgs, solid_angle, flux, specific_flux, volume,
                      area, current_cgs, charge_mks, electric_field_mks,
                      magnetic_field_mks]

dimensions = base_dimensions + derived_dimensions
