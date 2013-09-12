"""
Clump tools for use with the yt Clump object



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
nar = np.array

counter = 0
def recursive_all_clumps(clump,list,level,parentnumber):
    """A recursive function to flatten the hierarchy in *clump*.
    Not to be called directly: please call return_all_clumps, below."""
    global counter
    counter += 1
    clump.number = counter
    clump.parentnumber = parentnumber
    counter += 1
    list.append(clump)
    clump.level = level
    if clump.children != None:
        for child in clump.children:
            x = recursive_all_clumps(child,list,level+1,clump.number)
    return list

def return_all_clumps(clump):
    """Flatten the hierarchy defined by *clump*.
    Additionally adds three variables to the clump:
    level        = depth of hierarchy
    number       = index of clump in the final array
    parentnumber = index of this clumps parent

    """
    global counter
    counter = 0
    list = []
    level = 0
    clump.level = level
    parentnumber=-1
    recursive_all_clumps(clump,list,level,parentnumber)
    return list

def return_bottom_clumps(clump,dbg=0):
    """Recursively return clumps at the bottom of the hierarchy.
    This gives a list of clumps similar to what one would get from a CLUMPFIND routine"""
    global counter
    counter = 0
    list = []
    level = 0
    recursive_bottom_clumps(clump,list,dbg,level)
    return list
def recursive_bottom_clumps(clump,clump_list, dbg = 0,level=0):
    """Loops over a list of clumps (clumps) and fills clump_list with the bottom most.
    Recursive. Prints the level and the number of cores to the screen."""

    global counter
    if dbg > 0:
        print tabs(level), "l =",level, "n_core",counter

    if ((clump.children is None) or (len(clump.children) == 0)):
        counter += 1
        clump_list.append( clump )
    else:
        for child in clump.children:
            recursive_bottom_clumps(child,clump_list,dbg=dbg,level=level+1)

def clump_list_sort(clump_list):
    """Returns a copy of clump_list, sorted by ascending minimum density.  This
    eliminates overlap when passing to
    yt.visualization.plot_modification.ClumpContourCallback"""
    minDensity = [c['Density'].min() for c in clump_list]
    
    args = np.argsort(minDensity)
    list = nar(clump_list)[args]
    reverse = range(list.size-1,-1,-1)
    return list[reverse]
