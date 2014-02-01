"""
Function to calculate volume in common between two n-cubes, with optional
periodic boundary conditions.



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np

def common_volume(n_cube_1, n_cube_2, periodic=None):
    "Return the n-volume in common between the two n-cubes."

    # Check for proper args.
    if ((len(np.shape(n_cube_1)) != 2) or
        (np.shape(n_cube_1)[1] != 2) or
        (np.shape(n_cube_1) != np.shape(n_cube_2))):
        print "Arguments must be 2 (n, 2) numpy array."
        return 0

    if ((periodic is not None) and
        (np.shape(n_cube_1) != np.shape(periodic))):
        print "periodic argument must be (n, 2) numpy array."
        return 0

    nCommon = 1.0
    for q in range(np.shape(n_cube_1)[0]):
        if (periodic is None):
            nCommon *= common_segment(n_cube_1[q], n_cube_2[q])
        else:
            nCommon *= common_segment(n_cube_1[q], n_cube_2[q],
                                      periodic=periodic[q])

    return nCommon

def common_segment(seg1, seg2, periodic=None):
    "Return the length of the common segment."

    # Check for proper args.
    if ((len(seg1) != 2) or (len(seg2) != 2)):
        print "Arguments must be arrays of size 2."
        return 0

    # If not periodic, then this is very easy.
    if periodic is None:
        seg1.sort()
        len1 = seg1[1] - seg1[0]
        seg2.sort()
        len2 = seg2[1] - seg2[0]

        common = 0.0

        add = seg1[1] - seg2[0]
        if ((add > 0) and (add <= max(len1, len2))):
            common += add
        add = seg2[1] - seg1[0]
        if ((add > 0) and (add <= max(len1, len2))):
            common += add
        common = min(common, len1, len2)
        return common

    # If periodic, it's a little more complicated.
    else:
        if len(periodic) != 2:
            print "periodic array must be of size 2."
            return 0

        seg1.sort()
        flen1 = seg1[1] - seg1[0]
        len1 = flen1 - int(flen1)
        seg2.sort()
        flen2 = seg2[1] - seg2[0]
        len2 = flen2 - int(flen2)

        periodic.sort()
        scale = periodic[1] - periodic[0]

        if (abs(int(flen1)-int(flen2)) >= scale):
            return min(flen1, flen2)

        # Adjust for periodicity
        seg1[0] = np.mod(seg1[0], scale) + periodic[0]
        seg1[1] = seg1[0] + len1
        if (seg1[1] > periodic[1]): seg1[1] -= scale
        seg2[0] = np.mod(seg2[0], scale) + periodic[0]
        seg2[1] = seg2[0] + len2
        if (seg2[1] > periodic[1]): seg2[1] -= scale

        # create list of non-periodic segments
        pseg1 = []
        if (seg1[0] >= seg1[1]):
            pseg1.append([seg1[0], periodic[1]])
            pseg1.append([periodic[0], seg1[1]])
        else:
            pseg1.append(seg1)
        pseg2 = []
        if (seg2[0] >= seg2[1]):
            pseg2.append([seg2[0], periodic[1]])
            pseg2.append([periodic[0], seg2[1]])
        else:
            pseg2.append(seg2)

        # Add up common segments.
        common = min(int(flen1), int(flen2))

        for subseg1 in pseg1:
            for subseg2 in pseg2:
                common += common_segment(subseg1, subseg2)

        return common
