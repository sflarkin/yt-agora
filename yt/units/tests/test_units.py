"""
Test symbolic unit handling.




"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import nose
import numpy as np
from numpy.testing import \
    assert_approx_equal, assert_array_almost_equal_nulp, \
    assert_allclose, assert_raises
from nose.tools import assert_true
from sympy import Symbol
from yt.testing import fake_random_pf

# dimensions
from yt.units.dimensions import \
    mass, length, time, temperature, energy, magnetic_field, power, rate
# functions
from yt.units.unit_object import get_conversion_factor
# classes
from yt.units.unit_object import Unit, UnitParseError
# objects
from yt.units.unit_lookup_table import \
    default_unit_symbol_lut, unit_prefixes, prefixable_units
# unit definitions
from yt.utilities.physical_constants import \
    cm_per_pc, sec_per_year, cm_per_km, cm_per_mpc, \
    mass_sun_grams

def test_no_conflicting_symbols():
    """
    Check unit symbol definitions for conflicts.

    """
    full_set = set(default_unit_symbol_lut.keys())

    # go through all possible prefix combos
    for symbol in default_unit_symbol_lut.keys():
        if symbol in prefixable_units:
            keys = unit_prefixes.keys()
        else:
            keys = [symbol]
        for prefix in keys:
            new_symbol = "%s%s" % (prefix, symbol)

            # test if we have seen this symbol
            if new_symbol in full_set:
                print "Duplicate symbol: %s" % new_symbol
                raise RuntimeError

            full_set.add(new_symbol)
    yield assert_true, True

def test_dimensionless():
    """
    Create dimensionless unit and check attributes.

    """
    u1 = Unit()

    yield assert_true, u1.is_dimensionless
    yield assert_true, u1.expr == 1
    yield assert_true, u1.cgs_value == 1
    yield assert_true, u1.dimensions == 1

    u2 = Unit("")

    yield assert_true, u2.is_dimensionless
    yield assert_true, u2.expr == 1
    yield assert_true, u2.cgs_value == 1
    yield assert_true, u2.dimensions == 1

#
# Start init tests
#

def test_create_from_string():
    """
    Create units with strings and check attributes.

    """

    u1 = Unit("g * cm**2 * s**-2")
    yield assert_true, u1.dimensions == energy
    yield assert_true, u1.cgs_value == 1.0

    # make sure order doesn't matter
    u2 = Unit("cm**2 * s**-2 * g")
    yield assert_true, u2.dimensions == energy
    yield assert_true, u2.cgs_value == 1.0

    # Test rationals
    u3 = Unit("g**0.5 * cm**-0.5 * s**-1")
    yield assert_true, u3.dimensions == magnetic_field
    yield assert_true, u3.cgs_value == 1.0

    # sqrt functions
    u4 = Unit("sqrt(g)/sqrt(cm)/s")
    yield assert_true, u4.dimensions == magnetic_field
    yield assert_true, u4.cgs_value == 1.0

    # commutative sqrt function
    u5 = Unit("sqrt(g/cm)/s")
    yield assert_true, u5.dimensions == magnetic_field
    yield assert_true, u5.cgs_value == 1.0

    # nonzero CGS conversion factor
    u6 = Unit("Msun/pc**3")
    yield assert_true, u6.dimensions == mass/length**3
    yield assert_array_almost_equal_nulp, np.array([u6.cgs_value]), \
        np.array([mass_sun_grams/cm_per_pc**3])

    yield assert_raises, UnitParseError, Unit, 'm**m'
    yield assert_raises, UnitParseError, Unit, 'm**g'
    yield assert_raises, UnitParseError, Unit, 'm+g'
    yield assert_raises, UnitParseError, Unit, 'm-g'


def test_create_from_expr():
    """
    Create units from sympy Exprs and check attributes.

    """
    pc_cgs = cm_per_pc
    yr_cgs = sec_per_year

    # Symbol expr
    s1 = Symbol("pc", positive=True)
    s2 = Symbol("yr", positive=True)
    # Mul expr
    s3 = s1 * s2
    # Pow expr
    s4  = s1**2 * s2**(-1)

    u1 = Unit(s1)
    u2 = Unit(s2)
    u3 = Unit(s3)
    u4 = Unit(s4)

    yield assert_true, u1.expr == s1
    yield assert_true, u2.expr == s2
    yield assert_true, u3.expr == s3
    yield assert_true, u4.expr == s4

    yield assert_allclose, u1.cgs_value, pc_cgs, 1e-12
    yield assert_allclose, u2.cgs_value, yr_cgs, 1e-12
    yield assert_allclose, u3.cgs_value, pc_cgs * yr_cgs, 1e-12
    yield assert_allclose, u4.cgs_value, pc_cgs**2 / yr_cgs, 1e-12

    yield assert_true, u1.dimensions == length
    yield assert_true, u2.dimensions == time
    yield assert_true, u3.dimensions == length * time
    yield assert_true, u4.dimensions == length**2 / time


def test_create_with_duplicate_dimensions():
    """
    Create units with overlapping dimensions. Ex: km/Mpc.

    """

    u1 = Unit("erg * s**-1")
    u2 = Unit("km/s/Mpc")
    km_cgs = cm_per_km
    Mpc_cgs = cm_per_mpc

    yield assert_true, u1.cgs_value == 1
    yield assert_true, u1.dimensions == power

    yield assert_allclose, u2.cgs_value, km_cgs / Mpc_cgs, 1e-12
    yield assert_true, u2.dimensions == rate

def test_create_new_symbol():
    """
    Create unit with unknown symbol.

    """
    u1 = Unit("abc", cgs_value=42, dimensions=(mass/time))

    yield assert_true, u1.expr == Symbol("abc", positive=True)
    yield assert_true, u1.cgs_value == 42
    yield assert_true, u1.dimensions == mass / time

    u1 = Unit("abc", cgs_value=42, dimensions=length**3)

    yield assert_true, u1.expr == Symbol("abc", positive=True)
    yield assert_true, u1.cgs_value == 42
    yield assert_true, u1.dimensions == length**3

    u1 = Unit("abc", cgs_value=42, dimensions=length*(mass*length))

    yield assert_true, u1.expr == Symbol("abc", positive=True)
    yield assert_true, u1.cgs_value == 42
    yield assert_true,  u1.dimensions == length**2*mass

    yield assert_raises, UnitParseError, Unit, 'abc', \
        {'cgs_value':42, 'dimensions':length**length}
    yield assert_raises, UnitParseError, Unit, 'abc', \
        {'cgs_value':42, 'dimensions':length**(length*length)}
    yield assert_raises, UnitParseError, Unit, 'abc', \
        {'cgs_value':42, 'dimensions':length-mass}
    yield assert_raises, UnitParseError, Unit, 'abc', \
        {'cgs_value':42, 'dimensions':length+mass}

def test_create_fail_on_unknown_symbol():
    """
    Fail to create unit with unknown symbol, without cgs_value and dimensions.

    """
    try:
        u1 = Unit(Symbol("jigawatts"))
    except UnitParseError:
        yield assert_true, True
    else:
        yield assert_true, False

def test_create_fail_on_bad_symbol_type():
    """
    Fail to create unit with bad symbol type.

    """
    try:
        u1 = Unit([1])  # something other than Expr and str
    except UnitParseError:
        yield assert_true, True
    else:
        yield assert_true, False

def test_create_fail_on_bad_dimensions_type():
    """
    Fail to create unit with bad dimensions type.

    """
    try:
        u1 = Unit("a", cgs_value=1, dimensions="(mass)")
    except UnitParseError:
        yield assert_true, True
    else:
        yield assert_true, False


def test_create_fail_on_dimensions_content():
    """
    Fail to create unit with bad dimensions expr.

    """
    a = Symbol("a")

    try:
        u1 = Unit("a", cgs_value=1, dimensions=a)
    except UnitParseError:
        pass
    else:
        yield asser_true, False


def test_create_fail_on_cgs_value_type():
    """
    Fail to create unit with bad cgs_value type.

    """
    try:
        u1 = Unit("a", cgs_value="a", dimensions=(mass/time))
    except UnitParseError:
        yield assert_true, True
    else:
        yield assert_true, False

#
# End init tests
#

def test_string_representation():
    """
    Check unit string representation.

    """
    pc = Unit("pc")
    Myr = Unit("Myr")
    speed = pc / Myr
    dimensionless = Unit()

    yield assert_true, str(pc) == "pc"
    yield assert_true, str(Myr) == "Myr"
    yield assert_true, str(speed) == "pc/Myr"
    yield assert_true, repr(speed) == "pc/Myr"
    yield assert_true, str(dimensionless) == "dimensionless"

#
# Start operation tests
#

def test_multiplication():
    """
    Multiply two units.

    """
    msun_cgs = mass_sun_grams
    pc_cgs = cm_per_pc

    # Create symbols
    msun_sym = Symbol("Msun", positive=True)
    pc_sym = Symbol("pc", positive=True)
    s_sym = Symbol("s", positive=True)

    # Create units
    u1 = Unit("Msun")
    u2 = Unit("pc")

    # Mul operation
    u3 = u1 * u2

    yield assert_true, u3.expr == msun_sym * pc_sym
    yield assert_allclose, u3.cgs_value, msun_cgs * pc_cgs, 1e-12
    yield assert_true, u3.dimensions == mass * length

    # Pow and Mul operations
    u4 = Unit("pc**2")
    u5 = Unit("Msun * s")

    u6 = u4 * u5

    yield assert_true, u6.expr == pc_sym**2 * msun_sym * s_sym
    yield assert_allclose, u6.cgs_value, pc_cgs**2 * msun_cgs, 1e-12
    yield assert_true, u6.dimensions == length**2 * mass * time


def test_division():
    """
    Divide two units.

    """
    pc_cgs = cm_per_pc
    km_cgs = cm_per_km

    # Create symbols
    pc_sym = Symbol("pc", positive=True)
    km_sym = Symbol("km", positive=True)
    s_sym = Symbol("s", positive=True)

    # Create units
    u1 = Unit("pc")
    u2 = Unit("km * s")

    u3 = u1 / u2

    yield assert_true, u3.expr == pc_sym / (km_sym * s_sym)
    yield assert_allclose, u3.cgs_value, pc_cgs / km_cgs, 1e-12
    yield assert_true, u3.dimensions == 1 / time


def test_power():
    """
    Take units to some power.

    """
    from sympy import nsimplify

    pc_cgs = cm_per_pc
    mK_cgs = 1e-3
    u1_dims = mass * length**2 * time**-3 * temperature**4
    u1 = Unit("g * pc**2 * s**-3 * mK**4")

    u2 = u1**2

    yield assert_true, u2.dimensions == u1_dims**2
    yield assert_allclose, u2.cgs_value, (pc_cgs**2 * mK_cgs**4)**2, 1e-12

    u3 = u1**(-1.0/3)

    yield assert_true, u3.dimensions == nsimplify(u1_dims**(-1.0/3))
    yield assert_allclose, u3.cgs_value, (pc_cgs**2 * mK_cgs**4)**(-1.0/3), 1e-12


def test_equality():
    """
    Check unit equality with different symbols, but same dimensions and cgs_value.

    """
    u1 = Unit("km * s**-1")
    u2 = Unit("m * ms**-1")

    yield assert_true, u1 == u2

#
# End operation tests.
#

def test_cgs_equivalent():
    """
    Check cgs equivalent of a unit.

    """
    Msun_cgs = mass_sun_grams
    Mpc_cgs = cm_per_mpc

    u1 = Unit("Msun * Mpc**-3")
    u2 = Unit("g * cm**-3")
    u3 = u1.get_cgs_equivalent()

    yield assert_true, u2.expr == u3.expr
    yield assert_true, u2 == u3

    yield assert_allclose, u1.cgs_value, Msun_cgs / Mpc_cgs**3, 1e-12
    yield assert_true, u2.cgs_value == 1
    yield assert_true, u3.cgs_value == 1

    mass_density = mass / length**3

    yield assert_true, u1.dimensions == mass_density
    yield assert_true, u2.dimensions == mass_density
    yield assert_true, u3.dimensions == mass_density

    yield assert_allclose, get_conversion_factor(u1, u3)[0], \
        Msun_cgs / Mpc_cgs**3, 1e-12

def test_is_code_unit():
    pf = fake_random_pf(64, nprocs=1)
    u1 = Unit('code_mass', registry=pf.unit_registry)
    u2 = Unit('code_mass/code_length', registry=pf.unit_registry)
    u3 = Unit('code_velocity*code_mass**2', registry=pf.unit_registry)
    u4 = Unit('code_time*code_mass**0.5', registry=pf.unit_registry)
    u5 = Unit('code_mass*g', registry=pf.unit_registry)
    u6 = Unit('g/cm**3')

    yield assert_true, u1.is_code_unit
    yield assert_true, u2.is_code_unit
    yield assert_true, u3.is_code_unit
    yield assert_true, u4.is_code_unit
    yield assert_true, not u5.is_code_unit
    yield assert_true, not u6.is_code_unit
