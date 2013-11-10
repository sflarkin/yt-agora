"""



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import itertools as it
import numpy as np
import importlib
import os
from yt.funcs import *
from yt.config import ytcfg
from numpy.testing import assert_array_equal, assert_almost_equal, \
    assert_approx_equal, assert_array_almost_equal, assert_equal, \
    assert_array_less, assert_string_equal, assert_array_almost_equal_nulp,\
    assert_allclose, assert_raises

def assert_rel_equal(a1, a2, decimals, err_msg='', verbose=True):
    # We have nan checks in here because occasionally we have fields that get
    # weighted without non-zero weights.  I'm looking at you, particle fields!
    if isinstance(a1, np.ndarray):
        assert(a1.size == a2.size)
        # Mask out NaNs
        a1[np.isnan(a1)] = 1.0
        a2[np.isnan(a2)] = 1.0
    elif np.any(np.isnan(a1)) and np.any(np.isnan(a2)):
        return True
    return assert_almost_equal(np.array(a1)/np.array(a2), 1.0, decimals, err_msg=err_msg,
                               verbose=verbose)

def amrspace(extent, levels=7, cells=8):
    """Creates two numpy arrays representing the left and right bounds of
    an AMR grid as well as an array for the AMR level of each cell.

    Parameters
    ----------
    extent : array-like
        This a sequence of length 2*ndims that is the bounds of each dimension.
        For example, the 2D unit square would be given by [0.0, 1.0, 0.0, 1.0].
        A 3D cylindrical grid may look like [0.0, 2.0, -1.0, 1.0, 0.0, 2*np.pi].
    levels : int or sequence of ints, optional
        This is the number of AMR refinement levels.  If given as a sequence (of
        length ndims), then each dimension will be refined down to this level.
        All values in this array must be the same or zero.  A zero valued dimension
        indicates that this dim should not be refined.  Taking the 3D cylindrical
        example above if we don't want refine theta but want r and z at 5 we would
        set levels=(5, 5, 0).
    cells : int, optional
        This is the number of cells per refinement level.

    Returns
    -------
    left : float ndarray, shape=(npoints, ndims)
        The left AMR grid points.
    right : float ndarray, shape=(npoints, ndims)
        The right AMR grid points.
    level : int ndarray, shape=(npoints,)
        The AMR level for each point.

    Examples
    --------
    >>> l, r, lvl = amrspace([0.0, 2.0, 1.0, 2.0, 0.0, 3.14], levels=(3,3,0), cells=2)
    >>> print l
    [[ 0.     1.     0.   ]
     [ 0.25   1.     0.   ]
     [ 0.     1.125  0.   ]
     [ 0.25   1.125  0.   ]
     [ 0.5    1.     0.   ]
     [ 0.     1.25   0.   ]
     [ 0.5    1.25   0.   ]
     [ 1.     1.     0.   ]
     [ 0.     1.5    0.   ]
     [ 1.     1.5    0.   ]]

    """
    extent = np.asarray(extent, dtype='f8')
    dextent = extent[1::2] - extent[::2]
    ndims = len(dextent)

    if isinstance(levels, int):
        minlvl = maxlvl = levels
        levels = np.array([levels]*ndims, dtype='int32')
    else:
        levels = np.asarray(levels, dtype='int32')
        minlvl = levels.min()
        maxlvl = levels.max()
        if minlvl != maxlvl and (minlvl != 0 or set([minlvl, maxlvl]) != set(levels)):
            raise ValueError("all levels must have the same value or zero.")
    dims_zero = (levels == 0)
    dims_nonzero = ~dims_zero
    ndims_nonzero = dims_nonzero.sum()

    npoints = (cells**ndims_nonzero - 1)*maxlvl + 1
    left = np.empty((npoints, ndims), dtype='float64')
    right = np.empty((npoints, ndims), dtype='float64')
    level = np.empty(npoints, dtype='int32')

    # fill zero dims
    left[:,dims_zero] = extent[::2][dims_zero]
    right[:,dims_zero] = extent[1::2][dims_zero]

    # fill non-zero dims
    dcell = 1.0 / cells
    left_slice =  tuple([slice(extent[2*n], extent[2*n+1], extent[2*n+1]) if \
        dims_zero[n] else slice(0.0,1.0,dcell) for n in range(ndims)])
    right_slice = tuple([slice(extent[2*n+1], extent[2*n], -extent[2*n+1]) if \
        dims_zero[n] else slice(dcell,1.0+dcell,dcell) for n in range(ndims)])
    left_norm_grid = np.reshape(np.mgrid[left_slice].T.flat[ndims:], (-1, ndims))
    lng_zero = left_norm_grid[:,dims_zero]
    lng_nonzero = left_norm_grid[:,dims_nonzero]

    right_norm_grid = np.reshape(np.mgrid[right_slice].T.flat[ndims:], (-1, ndims))
    rng_zero = right_norm_grid[:,dims_zero]
    rng_nonzero = right_norm_grid[:,dims_nonzero]

    level[0] = maxlvl
    left[0,:] = extent[::2]
    right[0,dims_zero] = extent[1::2][dims_zero]
    right[0,dims_nonzero] = (dcell**maxlvl)*dextent[dims_nonzero] + extent[::2][dims_nonzero]
    for i, lvl in enumerate(range(maxlvl, 0, -1)):
        start = (cells**ndims_nonzero - 1)*i + 1
        stop = (cells**ndims_nonzero - 1)*(i+1) + 1
        dsize = dcell**(lvl-1) * dextent[dims_nonzero]
        level[start:stop] = lvl
        left[start:stop,dims_zero] = lng_zero
        left[start:stop,dims_nonzero] = lng_nonzero*dsize + extent[::2][dims_nonzero]
        right[start:stop,dims_zero] = rng_zero
        right[start:stop,dims_nonzero] = rng_nonzero*dsize + extent[::2][dims_nonzero]

    return left, right, level

def fake_random_pf(ndims, peak_value = 1.0, fields = ("Density",),
                   negative = False, nprocs = 1):
    from yt.frontends.stream.api import load_uniform_grid
    if not iterable(ndims):
        ndims = [ndims, ndims, ndims]
    else:
        assert(len(ndims) == 3)
    if not iterable(negative):
        negative = [negative for f in fields]
    assert(len(fields) == len(negative))
    offsets = []
    for n in negative:
        if n:
            offsets.append(0.5)
        else:
            offsets.append(0.0)
    data = dict((field, (np.random.random(ndims) - offset) * peak_value)
                 for field,offset in zip(fields,offsets))
    ug = load_uniform_grid(data, ndims, 1.0, nprocs = nprocs)
    return ug

def expand_keywords(keywords, full=False):
    """
    expand_keywords is a means for testing all possible keyword
    arguments in the nosetests.  Simply pass it a dictionary of all the
    keyword arguments and all of the values for these arguments that you
    want to test.

    It will return a list of kwargs dicts containing combinations of
    the various kwarg values you passed it.  These can then be passed
    to the appropriate function in nosetests. 

    If full=True, then every possible combination of keywords is produced,
    otherwise, every keyword option is included at least once in the output
    list.  Be careful, by using full=True, you may be in for an exponentially
    larger number of tests! 

    keywords : dict
        a dictionary where the keys are the keywords for the function,
        and the values of each key are the possible values that this key
        can take in the function

   full : bool
        if set to True, every possible combination of given keywords is 
        returned

    Returns
    -------
    array of dicts
        An array of dictionaries to be individually passed to the appropriate
        function matching these kwargs.

    Examples
    --------
    >>> keywords = {}
    >>> keywords['dpi'] = (50, 100, 200)
    >>> keywords['cmap'] = ('algae', 'jet')
    >>> list_of_kwargs = expand_keywords(keywords)
    >>> print list_of_kwargs

    array([{'cmap': 'algae', 'dpi': 50}, 
           {'cmap': 'jet', 'dpi': 100},
           {'cmap': 'algae', 'dpi': 200}], dtype=object)

    >>> list_of_kwargs = expand_keywords(keywords, full=True)
    >>> print list_of_kwargs

    array([{'cmap': 'algae', 'dpi': 50}, 
           {'cmap': 'algae', 'dpi': 100},
           {'cmap': 'algae', 'dpi': 200}, 
           {'cmap': 'jet', 'dpi': 50},
           {'cmap': 'jet', 'dpi': 100}, 
           {'cmap': 'jet', 'dpi': 200}], dtype=object)

    >>> for kwargs in list_of_kwargs:
    ...     write_projection(*args, **kwargs)
    """

    # if we want every possible combination of keywords, use iter magic
    if full:
        keys = sorted(keywords)
        list_of_kwarg_dicts = np.array([dict(zip(keys, prod)) for prod in \
                              it.product(*(keywords[key] for key in keys))])
            
    # if we just want to probe each keyword, but not necessarily every 
    # combination
    else:
        # Determine the maximum number of values any of the keywords has
        num_lists = 0
        for val in keywords.values():
            if isinstance(val, str):
                num_lists = max(1.0, num_lists)
            else:
                num_lists = max(len(val), num_lists)
    
        # Construct array of kwargs dicts, each element of the list is a different
        # **kwargs dict.  each kwargs dict gives a different combination of
        # the possible values of the kwargs
    
        # initialize array
        list_of_kwarg_dicts = np.array([dict() for x in range(num_lists)])
    
        # fill in array
        for i in np.arange(num_lists):
            list_of_kwarg_dicts[i] = {}
            for key in keywords.keys():
                # if it's a string, use it (there's only one)
                if isinstance(keywords[key], str):
                    list_of_kwarg_dicts[i][key] = keywords[key]
                # if there are more options, use the i'th val
                elif i < len(keywords[key]):
                    list_of_kwarg_dicts[i][key] = keywords[key][i]
                # if there are not more options, use the 0'th val
                else:
                    list_of_kwarg_dicts[i][key] = keywords[key][0]

    return list_of_kwarg_dicts

def requires_module(module):
    """
    Decorator that takes a module name as an argument and tries to import it.
    If the module imports without issue, the function is returned, but if not, 
    a null function is returned. This is so tests that depend on certain modules
    being imported will not fail if the module is not installed on the testing
    platform.
    """
    def ffalse(func):
        return lambda: None
    def ftrue(func):
        return func
    try:
        importlib.import_module(module)
    except ImportError:
        return ffalse
    else:
        return ftrue
    
def requires_file(req_file):
    path = ytcfg.get("yt", "test_data_dir")
    def ffalse(func):
        return lambda: None
    def ftrue(func):
        return func
    if os.path.exists(req_file):
        return ftrue
    else:
        if os.path.exists(os.path.join(path,req_file)):
            return ftrue
        else:
            return ffalse
                                        
