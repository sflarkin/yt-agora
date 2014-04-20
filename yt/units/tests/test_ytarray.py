"""
Test ndarray subclass that handles symbolic units.




"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

from nose.tools import assert_true
from numpy.testing import \
    assert_approx_equal, assert_array_equal, \
    assert_equal, assert_raises, \
    assert_array_almost_equal_nulp
from numpy import array
from yt.units.yt_array import \
    YTArray, YTQuantity, \
    unary_operators, binary_operators
from yt.units.unit_object import Unit
from yt.utilities.exceptions import \
    YTUnitOperationError, YTUfuncUnitError
from yt.testing import fake_random_pf, requires_module
from yt.funcs import fix_length
import numpy as np
import copy
import operator
import cPickle as pickle
import tempfile
import itertools

def operate_and_compare(a, b, op, answer):
    # Test generator for YTArrays tests
    assert_array_equal(op(a, b), answer)

def assert_isinstance(a, type):
    assert isinstance(a, type)

def test_addition():
    """
    Test addition of two YTArrays

    """

    # Same units
    a1 = YTArray([1, 2, 3], 'cm')
    a2 = YTArray([4, 5, 6], 'cm')
    answer = YTArray([5, 7, 9], 'cm')

    yield operate_and_compare, a1, a2, operator.add, answer
    yield operate_and_compare, a2, a1, operator.add, answer
    yield operate_and_compare, a1, a2, np.add, answer
    yield operate_and_compare, a2, a1, np.add, answer

    # different units
    a1 = YTArray([1, 2, 3], 'cm')
    a2 = YTArray([4, 5, 6], 'm')
    answer1 = YTArray([401, 502, 603], 'cm')
    answer2 = YTArray([4.01, 5.02, 6.03], 'm')

    yield operate_and_compare, a1, a2, operator.add, answer1
    yield operate_and_compare, a2, a1, operator.add, answer2
    yield assert_raises, YTUfuncUnitError, np.add, a1, a2

    # Test dimensionless quantities
    a1 = YTArray([1,2,3])
    a2 = array([4,5,6])
    answer = YTArray([5, 7, 9])

    yield operate_and_compare, a1, a2, operator.add, answer
    yield operate_and_compare, a2, a1, operator.add, answer
    yield operate_and_compare, a1, a2, np.add, answer
    yield operate_and_compare, a2, a1, np.add, answer

    # Catch the different dimensions error
    a1 = YTArray([1, 2, 3], 'm')
    a2 = YTArray([4, 5, 6], 'kg')

    yield assert_raises, YTUnitOperationError, operator.add, a1, a2
    yield assert_raises, YTUnitOperationError, operator.iadd, a1, a2

def test_subtraction():
    """
    Test subtraction of two YTArrays

    """

    # Same units
    a1 = YTArray([1, 2, 3], 'cm')
    a2 = YTArray([4, 5, 6], 'cm')
    answer1 = YTArray([-3, -3, -3], 'cm')
    answer2 = YTArray([3, 3, 3], 'cm')

    yield operate_and_compare, a1, a2, operator.sub, answer1
    yield operate_and_compare, a2, a1, operator.sub, answer2
    yield operate_and_compare, a1, a2, np.subtract, answer1
    yield operate_and_compare, a2, a1, np.subtract, answer2

    # different units
    a1 = YTArray([1, 2, 3], 'cm')
    a2 = YTArray([4, 5, 6], 'm')
    answer1 = YTArray([-399, -498, -597], 'cm')
    answer2 = YTArray([3.99, 4.98, 5.97], 'm')

    yield operate_and_compare, a1, a2, operator.sub, answer1
    yield operate_and_compare, a2, a1, operator.sub, answer2
    yield assert_raises, YTUfuncUnitError, np.subtract, a1, a2

    # Test dimensionless quantities
    a1 = YTArray([1,2,3])
    a2 = array([4,5,6])
    answer1 = YTArray([-3, -3, -3])
    answer2 = YTArray([3, 3, 3])

    yield operate_and_compare, a1, a2, operator.sub, answer1
    yield operate_and_compare, a2, a1, operator.sub, answer2
    yield operate_and_compare, a1, a2, np.subtract, answer1
    yield operate_and_compare, a2, a1, np.subtract, answer2

    # Catch the different dimensions error
    a1 = YTArray([1, 2, 3], 'm')
    a2 = YTArray([4, 5, 6], 'kg')

    yield assert_raises, YTUnitOperationError, operator.sub, a1, a2
    yield assert_raises, YTUnitOperationError, operator.isub, a1, a2

def test_multiplication():
    """
    Test multiplication of two YTArrays

    """

    # Same units
    a1 = YTArray([1, 2, 3], 'cm')
    a2 = YTArray([4, 5, 6], 'cm')
    answer = YTArray([4, 10, 18], 'cm**2')

    yield operate_and_compare, a1, a2, operator.mul, answer
    yield operate_and_compare, a2, a1, operator.mul, answer
    yield operate_and_compare, a1, a2, np.multiply, answer
    yield operate_and_compare, a2, a1, np.multiply, answer

    # different units, same dimension
    a1 = YTArray([1, 2, 3], 'cm')
    a2 = YTArray([4, 5, 6], 'm')
    answer1 = YTArray([400, 1000, 1800], 'cm**2')
    answer2 = YTArray([.04, .10, .18], 'm**2')
    answer3 = YTArray([4, 10, 18], 'cm*m')

    yield operate_and_compare, a1, a2, operator.mul, answer1
    yield operate_and_compare, a2, a1, operator.mul, answer2
    yield operate_and_compare, a1, a2, np.multiply, answer3
    yield operate_and_compare, a2, a1, np.multiply, answer3

    # different dimensions
    a1 = YTArray([1, 2, 3], 'cm')
    a2 = YTArray([4, 5, 6], 'g')
    answer = YTArray([4, 10, 18], 'cm*g')

    yield operate_and_compare, a1, a2, operator.mul, answer
    yield operate_and_compare, a2, a1, operator.mul, answer
    yield operate_and_compare, a1, a2, np.multiply, answer
    yield operate_and_compare, a2, a1, np.multiply, answer

    # One dimensionless, one unitful
    a1 = YTArray([1,2,3], 'cm')
    a2 = array([4,5,6])
    answer = YTArray([4, 10, 18], 'cm')

    yield operate_and_compare, a1, a2, operator.mul, answer
    yield operate_and_compare, a2, a1, operator.mul, answer
    yield operate_and_compare, a1, a2, np.multiply, answer
    yield operate_and_compare, a2, a1, np.multiply, answer

    # Both dimensionless quantities
    a1 = YTArray([1,2,3])
    a2 = array([4,5,6])
    answer = YTArray([4, 10, 18])

    yield operate_and_compare, a1, a2, operator.mul, answer
    yield operate_and_compare, a2, a1, operator.mul, answer
    yield operate_and_compare, a1, a2, np.multiply, answer
    yield operate_and_compare, a2, a1, np.multiply, answer

def test_division():
    """
    Test multiplication of two YTArrays

    """

    # Same units
    a1 = YTArray([1., 2., 3.], 'cm')
    a2 = YTArray([4., 5., 6.], 'cm')
    answer1 = YTArray([0.25, 0.4, 0.5])
    answer2 = YTArray([4, 2.5, 2])

    yield operate_and_compare, a1, a2, operator.div, answer1
    yield operate_and_compare, a2, a1, operator.div, answer2
    yield operate_and_compare, a1, a2, np.divide, answer1
    yield operate_and_compare, a2, a1, np.divide, answer2

    # different units, same dimension
    a1 = YTArray([1., 2., 3.], 'cm')
    a2 = YTArray([4., 5., 6.], 'm')
    answer1 = YTArray([.0025, .004, .005])
    answer2 = YTArray([400, 250, 200])
    answer3 = YTArray([0.25, 0.4, 0.5], 'cm/m')
    answer4 = YTArray([4.0, 2.5, 2.0], 'm/cm')

    yield operate_and_compare, a1, a2, operator.div, answer1
    yield operate_and_compare, a2, a1, operator.div, answer2
    yield operate_and_compare, a1, a2, np.divide, answer3
    yield operate_and_compare, a2, a1, np.divide, answer4

    # different dimensions
    a1 = YTArray([1., 2., 3.], 'cm')
    a2 = YTArray([4., 5., 6.], 'g')
    answer1 = YTArray([0.25, 0.4, 0.5], 'cm/g')
    answer2 = YTArray([4, 2.5, 2], 'g/cm')

    yield operate_and_compare, a1, a2, operator.div, answer1
    yield operate_and_compare, a2, a1, operator.div, answer2
    yield operate_and_compare, a1, a2, np.divide, answer1
    yield operate_and_compare, a2, a1, np.divide, answer2

    # One dimensionless, one unitful
    a1 = YTArray([1., 2., 3.], 'cm')
    a2 = array([4., 5., 6.])
    answer1 = YTArray([0.25, 0.4, 0.5], 'cm')
    answer2 = YTArray([4, 2.5, 2], '1/cm')

    yield operate_and_compare, a1, a2, operator.div, answer1
    yield operate_and_compare, a2, a1, operator.div, answer2
    yield operate_and_compare, a1, a2, np.divide, answer1
    yield operate_and_compare, a2, a1, np.divide, answer2

    # Both dimensionless quantities
    a1 = YTArray([1., 2., 3.])
    a2 = array([4., 5., 6.])
    answer1 = YTArray([0.25, 0.4, 0.5])
    answer2 = YTArray([4, 2.5, 2])

    yield operate_and_compare, a1, a2, operator.div, answer1
    yield operate_and_compare, a2, a1, operator.div, answer2
    yield operate_and_compare, a1, a2, np.divide, answer1
    yield operate_and_compare, a2, a1, np.divide, answer2

def test_power():
    """
    Test power operator ensure units are correct.

    """

    from yt.units import cm

    cm_arr = np.array([1.0, 1.0]) * cm

    assert_equal, cm**3, YTQuantity(1, 'cm**3')
    assert_equal, np.power(cm, 3), YTQuantity(1, 'cm**3')
    assert_equal, cm**YTQuantity(3), YTQuantity(1, 'cm**3')
    assert_raises, YTUnitOperationError, np.power, cm, YTQuantity(3, 'g')

    assert_equal, cm_arr**3, YTArray([1,1], 'cm**3')
    assert_equal, np.power(cm_arr, 3), YTArray([1,1], 'cm**3')
    assert_equal, cm_arr**YTQuantity(3), YTArray([1,1], 'cm**3')
    assert_raises, YTUnitOperationError, np.power, cm_arr, YTQuantity(3, 'g')

def test_comparisons():
    """
    Test numpy ufunc comparison operators for unit consistency.

    """
    from yt.units.yt_array import YTArray, YTQuantity

    a1 = YTArray([1,2,3], 'cm')
    a2 = YTArray([2,1,3], 'cm')
    a3 = YTArray([.02, .01, .03], 'm')

    ops = (
        np.less,
        np.less_equal,
        np.greater,
        np.greater_equal,
        np.equal,
        np.not_equal
    )

    answers = (
        [True, False, False],
        [True, False, True],
        [False, True, False],
        [False, True, True],
        [False, False, True],
        [True, True, False],
    )

    for op, answer in zip(ops, answers):
        yield operate_and_compare, a1, a2, op, answer

    for op in ops:
        yield assert_raises, YTUfuncUnitError, op, a1, a3

    for op, answer in zip(ops, answers):
        yield operate_and_compare, a1, a3.in_units('cm'), op, answer

def test_unit_conversions():
    """
    Test operations that convert to different units or cast to ndarray

    """
    from yt.units.yt_array import YTQuantity
    from yt.units.unit_object import Unit

    km = YTQuantity(1, 'km')
    km_in_cm = km.in_units('cm')
    km_unit = Unit('km')
    cm_unit = Unit('cm')
    kpc_unit = Unit('kpc')

    yield assert_equal, km_in_cm, km
    yield assert_equal, km_in_cm.in_cgs(), 1e5
    yield assert_equal, km_in_cm.in_mks(), 1e3
    yield assert_equal, km_in_cm.units, cm_unit

    km.convert_to_units('cm')

    yield assert_equal, km, YTQuantity(1, 'km')
    yield assert_equal, km.in_cgs(), 1e5
    yield assert_equal, km.in_mks(), 1e3
    yield assert_equal, km.units, cm_unit

    km.convert_to_units('kpc')

    yield assert_array_almost_equal_nulp, km, YTQuantity(1, 'km')
    yield assert_array_almost_equal_nulp, km.in_cgs(), YTQuantity(1e5, 'cm')
    yield assert_array_almost_equal_nulp, km.in_mks(), YTQuantity(1e3, 'm')
    yield assert_equal, km.units, kpc_unit

    yield assert_isinstance, km.to_ndarray(), np.ndarray
    yield assert_isinstance, km.ndarray_view(), np.ndarray

    dyne = YTQuantity(1.0, 'dyne')

    yield assert_equal, dyne.in_cgs(), dyne
    yield assert_equal, dyne.in_cgs(), 1.0
    yield assert_equal, dyne.in_mks(), dyne
    yield assert_equal, dyne.in_mks(), 1e-5
    yield assert_equal, str(dyne.in_mks().units), 'kg*m/s**2'
    yield assert_equal, str(dyne.in_cgs().units), 'cm*g/s**2'

    em3 = YTQuantity(1.0, 'erg/m**3')

    yield assert_equal, em3.in_cgs(), em3
    yield assert_equal, em3.in_cgs(), 1e-6
    yield assert_equal, em3.in_mks(), em3
    yield assert_equal, em3.in_mks(), 1e-7
    yield assert_equal, str(em3.in_mks().units), 'kg/(m*s**2)'
    yield assert_equal, str(em3.in_cgs().units), 'g/(cm*s**2)'


def test_yt_array_yt_quantity_ops():
    """
    Test operations that combine YTArray and YTQuantity
    """
    a = YTArray(range(10), 'cm')
    b = YTQuantity(5, 'g')

    yield assert_isinstance, a*b, YTArray
    yield assert_isinstance, b*a, YTArray

    yield assert_isinstance, a/b, YTArray
    yield assert_isinstance, b/a, YTArray

    yield assert_isinstance, a*a, YTArray
    yield assert_isinstance, a/a, YTArray

    yield assert_isinstance, b*b, YTQuantity
    yield assert_isinstance, b/b, YTQuantity

def test_selecting():
    """
    Test slicing of two YTArrays

    """
    a = YTArray(range(10), 'cm')
    a_slice = a[:3]
    a_fancy_index = a[[1,1,3,5]]
    a_array_fancy_index = a[array([[1,1], [3,5]])]
    a_boolean_index = a[a > 5]
    a_selection = a[0]

    yield assert_array_equal, a_slice, YTArray([0, 1, 2], 'cm')
    yield assert_array_equal, a_fancy_index, YTArray([1,1,3,5], 'cm')
    yield assert_array_equal, a_array_fancy_index, YTArray([[1, 1,], [3,5]], 'cm')
    yield assert_array_equal, a_boolean_index, YTArray([6,7,8,9], 'cm')
    yield assert_isinstance, a_selection, YTQuantity

    # .base points to the original array for a numpy view.  If it is not a
    # view, .base is None.
    yield assert_true, a_slice.base is a
    yield assert_true, a_fancy_index.base is None
    yield assert_true, a_array_fancy_index.base is None
    yield assert_true, a_boolean_index.base is None

def test_fix_length():
    """
    Test fixing the length of an array. Used in spheres and other data objects
    """
    pf = fake_random_pf(64, nprocs=1, length_unit=10)
    length = pf.quan(1.0,'code_length')
    new_length = fix_length(length, pf=pf)
    yield assert_equal, YTQuantity(10, 'cm'), new_length

def test_ytarray_pickle():
    pf = fake_random_pf(64, nprocs=1)
    test_data = [pf.quan(12.0, 'code_length'), pf.arr([1,2,3], 'code_length')]

    for data in test_data:
        tempf = tempfile.NamedTemporaryFile(delete=False)
        pickle.dump(data, tempf)
        tempf.close()

        loaded_data = pickle.load(open(tempf.name, "rb"))

        yield assert_array_equal, data, loaded_data
        yield assert_equal, data.units, loaded_data.units
        yield assert_array_equal, array(data.in_cgs()), array(loaded_data.in_cgs())
        yield assert_equal, float(data.units.cgs_value), float(loaded_data.units.cgs_value)

def test_copy():
    quan = YTQuantity(1, 'g')
    arr = YTArray([1,2,3], 'cm')

    yield assert_equal, copy.copy(quan), quan
    yield assert_array_equal, copy.copy(arr), arr

    yield assert_equal,  copy.deepcopy(quan), quan
    yield assert_array_equal, copy.deepcopy(arr), arr

    yield assert_equal, quan.copy(), quan
    yield assert_array_equal, arr.copy(), arr

    yield assert_equal, np.copy(quan), quan
    yield assert_array_equal, np.copy(arr), arr

def unary_ufunc_comparison(ufunc, a):
    out = a.copy()
    a_array = a.to_ndarray()
    if ufunc in (np.isreal, np.iscomplex, ):
        # According to the numpy docs, these two explicitly do not do
        # in-place copies.
        ret = ufunc(a)
        assert_true(not hasattr(ret, 'units'))
        assert_array_equal(ret, ufunc(a))
    elif ufunc in (np.exp, np.exp2, np.log, np.log2, np.log10, np.expm1,
                   np.log1p, np.sin, np.cos, np.tan, np.arcsin, np.arccos,
                   np.arctan, np.sinh, np.cosh, np.tanh, np.arccosh,
                   np.arcsinh, np.arctanh, np.deg2rad, np.rad2deg,
                   np.isfinite, np.isinf, np.isnan, np.signbit, np.sign,
                   np.rint, np.logical_not):
        # These operations should return identical results compared to numpy.

        try:
            ret = ufunc(a, out=out)
        except YTUnitOperationError:
            assert_true(ufunc in (np.deg2rad, np.rad2deg))
            ret = ufunc(YTArray(a, '1'))

        assert_array_equal(ret, out)
        assert_array_equal(ret, ufunc(a_array))
        # In-place copies do not drop units.
        assert_true(hasattr(out, 'units'))
        assert_true(not hasattr(ret, 'units'))
    elif ufunc in (np.absolute, np.conjugate, np.floor, np.ceil,
                   np.trunc, np.negative):
        ret = ufunc(a, out=out)

        assert_array_equal(ret, out)
        assert_array_equal(ret.to_ndarray(), ufunc(a_array))
        assert_true(ret.units == out.units)
    elif ufunc in (np.ones_like, np.square, np.sqrt, np.reciprocal):
        if ufunc is np.ones_like:
            ret = ufunc(a)
        else:
            ret = ufunc(a, out=out)
            assert_array_equal(ret, out)

        assert_array_equal(ret.to_ndarray(), ufunc(a_array))
        if ufunc is np.square:
            assert_true(out.units == a.units**2)
            assert_true(ret.units == a.units**2)
        elif ufunc is np.sqrt:
            assert_true(out.units == a.units**0.5)
            assert_true(ret.units == a.units**0.5)
        elif ufunc is np.reciprocal:
            assert_true(out.units == a.units**-1)
            assert_true(ret.units == a.units**-1)
    elif ufunc is np.modf:
        ret1, ret2 = ufunc(a)
        npret1, npret2 = ufunc(a_array)

        assert_array_equal(ret1.to_ndarray(), npret1)
        assert_array_equal(ret2.to_ndarray(), npret2)
    elif ufunc is np.frexp:
        ret1, ret2 = ufunc(a)
        npret1, npret2 = ufunc(a_array)

        assert_array_equal(ret1, npret1)
        assert_array_equal(ret2, npret2)
    else:
        # There shouldn't be any untested ufuncs.
        assert_true(False)

def binary_ufunc_comparison(ufunc, a, b):
    out = a.copy()
    if ufunc in (np.add, np.subtract, np.remainder, np.fmod, np.mod, np.arctan2,
                 np.hypot, np.greater, np.greater_equal, np.less, np.less_equal,
                 np.equal, np.not_equal, np.logical_and, np.logical_or,
                 np.logical_xor, np.maximum, np.minimum, np.fmax, np.fmin,
                 np.nextafter):
        if a.units != b.units and a.units.dimensions == b.units.dimensions:
            assert_raises(YTUfuncUnitError, ufunc, a, b)
            return
        elif a.units != b.units:
            assert_raises(YTUnitOperationError, ufunc, a, b)
            return

    ret = ufunc(a, b, out=out)

    if ufunc is np.multiply:
        assert_true(ret.units == a.units*b.units)
    elif ufunc in (np.divide, np.true_divide, np.arctan2):
        assert_true(ret.units.dimensions == (a.units/b.units).dimensions)
    elif ufunc in (np.greater, np.greater_equal, np.less, np.less_equal,
                   np.not_equal, np.equal, np.logical_and, np.logical_or,
                   np.logical_xor):
        assert_true(not isinstance(ret, YTArray) and isinstance(ret, np.ndarray))
    assert_array_equal(ret, out)
    assert_array_equal(ret, ufunc(np.array(a), np.array(b)))

def test_ufuncs():
    for ufunc in unary_operators:
        yield unary_ufunc_comparison, ufunc, YTArray([.3, .4, .5], 'cm')
        yield unary_ufunc_comparison, ufunc, YTArray([12, 23, 47], 'g')
        yield unary_ufunc_comparison, ufunc, YTArray([2, 4, -6], 'erg/m**3')

    for ufunc in binary_operators:

        # arr**arr is undefined for arrays with units because
        # each element of the result would have different units.
        if ufunc is np.power:
            a = YTArray([.3, .4, .5], 'cm')
            b = YTArray([.1, .2, .3], 'dimensionless')
            c = np.array(b)
            yield binary_ufunc_comparison, ufunc, a, b
            yield binary_ufunc_comparison, ufunc, a, c
            continue

        a = YTArray([.3, .4, .5], 'cm')
        b = YTArray([.1, .2, .3], 'cm')
        c = YTArray([.1, .2, .3], 'm')
        d = YTArray([.1, .2, .3], 'g')
        e = YTArray([.1, .2, .3], 'erg/m**3')

        for pair in itertools.product([a,b,c,d,e], repeat=2):
            yield binary_ufunc_comparison, ufunc, pair[0], pair[1]

def test_convenience():

    arr = YTArray([1, 2, 3], 'cm')

    yield assert_equal, arr.unit_quantity, YTQuantity(1, 'cm')
    yield assert_equal, arr.uq, YTQuantity(1, 'cm')
    yield assert_isinstance, arr.unit_quantity, YTQuantity
    yield assert_isinstance, arr.uq, YTQuantity

    yield assert_array_equal, arr.unit_array, YTArray(np.ones_like(arr), 'cm')
    yield assert_array_equal, arr.ua, YTArray(np.ones_like(arr), 'cm')
    yield assert_isinstance, arr.unit_array, YTArray
    yield assert_isinstance, arr.ua, YTArray

    yield assert_array_equal, arr.ndview, arr.view(np.ndarray)
    yield assert_array_equal, arr.d, arr.view(np.ndarray)
    yield assert_true, arr.ndview.base is arr.base
    yield assert_true, arr.d.base is arr.base

    yield assert_array_equal, arr.value, np.array(arr)
    yield assert_array_equal, arr.v, np.array(arr)

@requires_module("astropy")
def test_astropy():
    from yt.utilities.on_demand_imports import ap

    ap_arr = np.arange(10)*ap.units.km/ap.units.hr
    yt_arr = YTArray(np.arange(10), "km/hr")

    ap_quan = 10.*ap.units.Msun**0.5/(ap.units.kpc**3)
    yt_quan = YTQuantity(10.,"sqrt(Msun)/kpc**3")

    yield assert_array_equal, ap_arr, yt_arr.to_astropy()
    yield assert_array_equal, yt_arr, YTArray(ap_arr)

    yield assert_equal, ap_quan, yt_quan.to_astropy()
    yield assert_equal, yt_quan, YTQuantity(ap_quan)

    yield assert_array_equal, yt_arr, YTArray(yt_arr.to_astropy())
    yield assert_equal, yt_quan, YTQuantity(yt_quan.to_astropy())


