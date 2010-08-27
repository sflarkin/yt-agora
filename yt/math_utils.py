"""
Commonly used mathematical functions.

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: UCSD Physics/CASS
Author: Stephen Skory <stephenskory@yahoo.com>
Affiliation: UCSD Physics/CASS
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2008-2009 Matthew Turk.  All Rights Reserved.

  This file is part of yt.

  yt is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as na
import math

def periodic_dist(a, b, period):
    r"""Find the Euclidian periodic distance between two points.
    
    Parameters
    ----------
    a : array or list
        An array or list of floats.
    
    b : array of list
        An array or list of floats.
    
    period : float or array or list
        If the volume is symmetrically periodic, this can be a single float,
        otherwise an array or list of floats giving the periodic size of the
        volume for each dimension.

    Examples
    --------
    >>> a = na.array([0.1, 0.1, 0.1])
    >>> b = na.array([0.9, 0,9, 0.9])
    >>> period = 1.
    >>> dist = periodic_dist(a, b, 1.)
    >>> dist
    0.3464102
    """
    a = na.array(a)
    b = na.array(b)
    if a.size != b.size: RunTimeError("Arrays must be the same shape.")
    c = na.empty((2, a.size), dtype="float64")
    c[0,:] = abs(a - b)
    c[1,:] = period - abs(a - b)
    d = na.amin(c, axis=0)**2
    return math.sqrt(d.sum())

def rotate_vector_3D(a, dim, angle):
    r"""Rotates the elements of an array around an axis by some angle.
    
    Given an array of 3D vectors a, this rotates them around a coordinate axis
    by a clockwise angle. An alternative way to think about it is the
    coordinate axes are rotated counterclockwise, which changes the directions
    of the vectors accordingly.
    
    Parameters
    ----------
    a : array
        An array of 3D vectors with dimension Nx3.
    
    dim : integer
        A integer giving the axis around which the vectors will be rotated.
        (x, y, z) = (0, 1, 2).
    
    angle : float
        The angle in radians through which the vectors will be rotated
        clockwise.
    
    Examples
    --------
    >>> a = na.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [3, 4, 5]])
    >>> b = rotate_vector_3D(a, 2, na.pi/2)
    >>> print b
    [[  1.00000000e+00  -1.00000000e+00   0.00000000e+00]
    [  6.12323400e-17  -1.00000000e+00   1.00000000e+00]
    [  1.00000000e+00   6.12323400e-17   1.00000000e+00]
    [  1.00000000e+00  -1.00000000e+00   1.00000000e+00]
    [  4.00000000e+00  -3.00000000e+00   5.00000000e+00]]
    
    """
    mod = False
    if len(a.shape) == 1:
        mod = True
        a = na.array([a])
    if a.shape[1] !=3:
        raise SyntaxError("The second dimension of the array a must be == 3!")
    if dim == 0:
        R = na.array([[1, 0,0],
            [0, na.cos(angle), na.sin(angle)],
            [0, -na.sin(angle), na.cos(angle)]])
    elif dim == 1:
        R = na.array([[na.cos(angle), 0, -na.sin(angle)],
            [0, 1, 0],
            [na.sin(angle), 0, na.cos(angle)]])
    elif dim == 2:
        R = na.array([[na.cos(angle), na.sin(angle), 0],
            [-na.sin(angle), na.cos(angle), 0],
            [0, 0, 1]])
    else:
        raise SyntaxError("dim must be 0, 1, or 2!")
    if mod:
        return na.dot(R, a.T).T[0]
    else:
        return na.dot(R, a.T).T
    

def modify_reference_frame(CoM, L, P, V):
    r"""Rotates and translates data into a new reference frame to make
    calculations easier.
    
    This is primarily useful for calculations of halo data.
    The data is translated into the center of mass frame.
    Next, it is rotated such that the angular momentum vector for the data
    is aligned with the z-axis. Put another way, if one calculates the angular
    momentum vector on the data that comes out of this function, it will
    always be along the positive z-axis.
    If the center of mass is re-calculated, it will be at the origin.
    
    Parameters
    ----------
    CoM : array
        The center of mass in 3D.
    
    L : array
        The angular momentum vector.
    
    P : array
        The positions of the data to be modified (i.e. particle or grid cell
        postions). The array should be Nx3.
    
    V : array
        The velocities of the data to be modified (i.e. particle or grid cell
        velocities). The array should be Nx3.
    
    Returns
    -------
    L : array
        The angular momentum vector equal to [0, 0, 1] modulo machine error.
    
    P : array
        The modified positional data.
    
    V : array
        The modified velocity data.
    
    Examples
    --------
    >>> CoM = na.array([0.5, 0.5, 0.5])
    >>> L = na.array([1, 0, 0])
    >>> P = na.array([[1, 0.5, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0.5], [0, 0, 0]])
    >>> V = p.copy()
    >>> LL, PP, VV = modify_reference_frame(CoM, L, P, V)
    >>> LL
    array([  6.12323400e-17,   0.00000000e+00,   1.00000000e+00])
    >>> PP
    array([[  3.06161700e-17,   0.00000000e+00,   5.00000000e-01],
           [ -3.06161700e-17,   0.00000000e+00,  -5.00000000e-01],
           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
           [  5.00000000e-01,  -5.00000000e-01,  -5.00000000e-01]])
    >>> VV
    array([[ -5.00000000e-01,   5.00000000e-01,   1.00000000e+00],
           [ -5.00000000e-01,   5.00000000e-01,   3.06161700e-17],
           [ -5.00000000e-01,   5.00000000e-01,   5.00000000e-01],
           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00]])

    """
    if (L == na.array([0, 0, 1.])).all():
        # Whew! Nothing to do!
        return L, P, V
    # First translate the positions to center of mass reference frame.
    P = P - CoM
    # Now find the angle between modified L and the x-axis.
    LL = L.copy()
    LL[2] = 0.
    theta = na.arccos(na.inner(LL, [1.,0,0])/na.inner(LL,LL)**.5)
    if L[1] < 0:
        theta = -theta
    # Now rotate all the position, velocity, and L vectors by this much around
    # the z axis.
    P = rotate_vector_3D(P, 2, theta)
    V = rotate_vector_3D(V, 2, theta)
    L = rotate_vector_3D(L, 2, theta)
    # Now find the angle between L and the z-axis.
    theta = na.arccos(na.inner(L, [0,0,1])/na.inner(L,L)**.5)
    # This time we rotate around the y axis.
    P = rotate_vector_3D(P, 1, theta)
    V = rotate_vector_3D(V, 1, theta)
    L = rotate_vector_3D(L, 1, theta)
    return L, P, V

def compute_rotational_velocity(CoM, L, P, V):
    r"""Computes the rotational velocity for some data around an axis.
    
    This is primarily for halo computations.
    Given some data, this computes the circular rotational velocity of each
    point (particle) in reference to the axis defined by the angular momentum
    vector.
    This is accomplished by converting the reference frame of the center of
    mass of the halo.
    
    Parameters
    ----------
    CoM : array
        The center of mass in 3D.
    
    L : array
        The angular momentum vector.
    
    P : array
        The positions of the data to be modified (i.e. particle or grid cell
        postions). The array should be Nx3.
    
    V : array
        The velocities of the data to be modified (i.e. particle or grid cell
        velocities). The array should be Nx3.
    
    Returns
    -------
    v : array
        An array N elements long that gives the circular rotational velocity
        for each datum (particle).
    
    Examples
    --------
    >>> CoM = na.array([0, 0, 0])
    >>> L = na.array([0, 0, 1])
    >>> P = na.array([[1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 0]])
    >>> V = na.array([[0, 1, 10], [-1, -1, -1], [1, 1, 1], [1, -1, -1]])
    >>> circV = compute_rotational_velocity(CoM, L, P, V)
    >>> circV
    array([ 1.        ,  0.        ,  0.        ,  1.41421356])

    """
    # First we translate into the simple coordinates.
    L, P, V = modify_reference_frame(CoM, L, P, V)
    # Find the vector in the plane of the galaxy for each position point
    # that is perpendicular to the radial vector.
    radperp = na.cross([0, 0, 1], P)
    # Find the component of the velocity along the radperp vector.
    # Unf., I don't think there's a better way to do this.
    res = na.empty(V.shape[0], dtype='float64')
    for i, rp in enumerate(radperp):
        temp = na.dot(rp, V[i]) / na.dot(rp, rp) * rp
        res[i] = na.dot(temp, temp)**0.5
    return res
    
def compute_parallel_velocity(CoM, L, P, V):
    r"""Computes the parallel velocity for some data around an axis.
    
    This is primarily for halo computations.
    Given some data, this computes the velocity component along the angular
    momentum vector.
    This is accomplished by converting the reference frame of the center of
    mass of the halo.
    
    Parameters
    ----------
    CoM : array
        The center of mass in 3D.
    
    L : array
        The angular momentum vector.
    
    P : array
        The positions of the data to be modified (i.e. particle or grid cell
        postions). The array should be Nx3.
    
    V : array
        The velocities of the data to be modified (i.e. particle or grid cell
        velocities). The array should be Nx3.
    
    Returns
    -------
    v : array
        An array N elements long that gives the parallel velocity for
        each datum (particle).
    
    Examples
    --------
    >>> CoM = na.array([0, 0, 0])
    >>> L = na.array([0, 0, 1])
    >>> P = na.array([[1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 0]])
    >>> V = na.array([[0, 1, 10], [-1, -1, -1], [1, 1, 1], [1, -1, -1]])
    >>> paraV = compute_parallel_velocity(CoM, L, P, V)
    >>> paraV
    array([10, -1,  1, -1])
    
    """
    # First we translate into the simple coordinates.
    L, P, V = modify_reference_frame(CoM, L, P, V)
    # And return just the z-axis velocities.
    return V[:,2]

def compute_radial_velocity(CoM, L, P, V):
    r"""Computes the radial velocity for some data around an axis.
    
    This is primarily for halo computations.
    Given some data, this computes the radial velocity component for the data.
    This is accomplished by converting the reference frame of the center of
    mass of the halo.
    
    Parameters
    ----------
    CoM : array
        The center of mass in 3D.
    
    L : array
        The angular momentum vector.
    
    P : array
        The positions of the data to be modified (i.e. particle or grid cell
        postions). The array should be Nx3.
    
    V : array
        The velocities of the data to be modified (i.e. particle or grid cell
        velocities). The array should be Nx3.
    
    Returns
    -------
    v : array
        An array N elements long that gives the radial velocity for
        each datum (particle).
    
    Examples
    --------
    >>> CoM = na.array([0, 0, 0])
    >>> L = na.array([0, 0, 1])
    >>> P = na.array([[1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 0]])
    >>> V = na.array([[0, 1, 10], [-1, -1, -1], [1, 1, 1], [1, -1, -1]])
    >>> radV = compute_radial_velocity(CoM, L, P, V)
    >>> radV
    array([ 1.        ,  1.41421356 ,  0.        ,  0.])
    
    """
    # First we translate into the simple coordinates.
    L, P, V = modify_reference_frame(CoM, L, P, V)
    # We find the tangential velocity by dotting the velocity vector
    # with the cylindrical radial vector for this point.
    # Unf., I don't think there's a better way to do this.
    P[:,2] = 0
    res = na.empty(V.shape[0], dtype='float64')
    for i, rad in enumerate(P):
        temp = na.dot(rad, V[i]) / na.dot(rad, rad) * rad
        res[i] = na.dot(temp, temp)**0.5
    return res

def compute_cylindrical_radius(CoM, L, P, V):
    r"""Compute the radius for some data around an axis in cylindrical
    coordinates.
    
    This is primarily for halo computations.
    Given some data, this computes the cylindrical radius for each point.
    This is accomplished by converting the reference frame of the center of
    mass of the halo.
    
    Parameters
    ----------
    CoM : array
        The center of mass in 3D.
    
    L : array
        The angular momentum vector.
    
    P : array
        The positions of the data to be modified (i.e. particle or grid cell
        postions). The array should be Nx3.
    
    V : array
        The velocities of the data to be modified (i.e. particle or grid cell
        velocities). The array should be Nx3.
    
    Returns
    -------
    cyl_r : array
        An array N elements long that gives the radial velocity for
        each datum (particle).
    
    Examples
    --------
    >>> CoM = na.array([0, 0, 0])
    >>> L = na.array([0, 0, 1])
    >>> P = na.array([[1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 0]])
    >>> V = na.array([[0, 1, 10], [-1, -1, -1], [1, 1, 1], [1, -1, -1]])
    >>> cyl_r = compute_cylindrical_radius(CoM, L, P, V)
    >>> cyl_r
    array([ 1.        ,  1.41421356,  0.        ,  1.41421356])
    """
    # First we translate into the simple coordinates.
    L, P, V = modify_reference_frame(CoM, L, P, V)
    # Demote all the positions to the z=0 plane, which makes the distance
    # calculation very easy.
    P[:,2] = 0
    return na.sqrt((P * P).sum(axis=1))
    
