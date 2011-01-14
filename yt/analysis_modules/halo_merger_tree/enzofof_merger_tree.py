"""
A very simple, purely-serial, merger tree script that knows how to parse FOF
catalogs output by Enzo and then compare parent/child relationships.

Author: Matthew J. Turk <matthewturk@gmail.com>
Affiliation: NSF / Columbia
Author: John H. Wise <jwise@astro.princeton.edu>
Affiliation: Princeton
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2010-2011 Matthew Turk.  All Rights Reserved.

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

# First pass at a simplified merger tree
#
# Basic outline:
#
# 1. Halo find inline, obtaining particle catalogs
# 2. Load dataset at time t
# 3. Load dataset at time t+1
# 4. Parse catalogs for t and t+1
# 5. Place halos for t+1 in kD-tree
# 6. For every halo in t, execute ball-query with some linking length
# 7. For every halo in ball-query result, execute numpy's intersect1d on
#    particle IDs
# 8. Parentage is described by a fraction of particles that pass from one to
#    the other; we have both descendent fractions and ancestory fractions. 

import numpy as na
import h5py
import time
import pdb
import cPickle
import glob

from yt.funcs import *
from yt.utilities.pykdtree import KDTree

# We don't currently use this, but we may again find a use for it in the
# future.
class MaxLengthDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.order = [None] * 50

    def __setitem__(self, key, val):
        if key not in self.order:
            to_remove = self.order.pop(0)
            self.pop(to_remove, None)
        self.order.append(key)
        dict.__setitem__(self, key, val)

    def __getitem__(self, key):
        if key in self.order:
            self.order.pop(self.order.index(key))
            self.order.append(key)
        return dict.__getitem__(self, key)

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        self.order.pop(self.order.index(key))
        self.order.insert(0, None)

class HaloCatalog(object):
    cache = None
    def __init__(self, output_id, cache = True):
        r"""A catalog of halos, parsed from EnzoFOF outputs.

        This class will read in catalogs output by the Enzo FOF halo finder and
        make available their positions, radii, etc.  Enzo FOF was provided
        starting with 2.0, and can be run either inline (with the correct
        options) or as a postprocessing step using the `-F` command line
        option.  This class is mostly useful when calculating a merger tree,
        and when the particle IDs for members of a given halo are output as
        well.

        Parameters
        ----------
        output_id : int
            This is the integer output id of the halo catalog to parse and
            load.
        cache : bool
            Should we store, in between accesses, the particle IDs?  If set to
            true, the correct particle files must exist.
        """
        self.output_id = output_id
        self.redshift = 0.0
        self.particle_file = h5py.File("FOF/particles_%05i.h5" % output_id, "r")
        self.parse_halo_catalog()
        if cache: self.cache = dict()#MaxLengthDict()

    def parse_halo_catalog(self):
        hp = []
        for line in open("FOF/groups_%05i.dat" % self.output_id):
            if line.strip() == "": continue # empty
            if line.startswith("# Red"):
                self.redshift = float(line.split("=")[1])
            if line[0] == "#": continue # comment
            if line[0] == "d": continue # datavar
            x,y,z = [float(f) for f in line.split(None, 3)[:-1]]
            hp.append([x,y,z])
        if hp != []:
            self.halo_positions = na.array(hp)
            self.halo_kdtree = KDTree(self.halo_positions)
        else:
            self.halo_positions = None
            self.halo_kdtree = None
        return hp

    def read_particle_ids(self, halo_id):
        if self.cache is not None:
            if halo_id not in self.cache:
                self.cache[halo_id] = self.particle_file["/Halo%08i/Particle ID" % halo_id][:]
            ids = self.cache[halo_id]
        else:
            ids = self.particle_file["/Halo%08i/Particle ID" % halo_id][:]
        return HaloParticleList(halo_id, self.halo_positions[halo_id,:], ids)

    def calculate_parentage_fractions(self, other_catalog, radius = 0.10):
        parentage_fractions = {}
        if self.halo_positions == None or other_catalog.halo_positions == None:
            return parentage_fractions
        mylog.debug("Ball-tree query with radius %0.3e", radius)
        all_nearest = self.halo_kdtree.query_ball_tree(
            other_catalog.halo_kdtree, radius)
        pbar = get_pbar("Halo Mergers", self.halo_positions.shape[0])
        for hid1, nearest in enumerate(all_nearest):
            pbar.update(hid1)
            parentage_fractions[hid1] = {}
            HPL1 = self.read_particle_ids(hid1)
            for hid2 in nearest:
                HPL2 = other_catalog.read_particle_ids(hid2)
                p1, p2 = HPL1.find_relative_parentage(HPL2)
                parentage_fractions[hid1][hid2] = (p1, p2)
            parentage_fractions[hid1]["NumberOfParticles"] = HPL1.number_of_particles
        pbar.finish()
        return parentage_fractions

class HaloParticleList(object):
    def __init__(self, halo_id, position, particle_ids):
        self.halo_id = halo_id
        self.position = na.array(position)
        self.particle_ids = particle_ids
        self.number_of_particles = particle_ids.size

    def find_nearest(self, other_tree, radius = 0.10):
        return other_tree.query_ball_point(self.position, radius)

    def find_relative_parentage(self, child):
        # Return two values: percent this halo gave to the other, and percent
        # of the other that comes from this halo
        overlap = na.intersect1d(self.particle_ids, child.particle_ids).size
        of_child_from_me = float(overlap)/child.particle_ids.size
        of_mine_from_me = float(overlap)/self.particle_ids.size
        return of_child_from_me, of_mine_from_me

class EnzoFOFMergerBranch(object):
    def __init__(self, tree, output_num, halo_id):
        self.output_num = output_num
        self.halo_id = halo_id
        self.npart = tree.relationships[output_num][halo_id]["NumberOfParticles"]
        self.children = []
        self.progenitor = -1
        max_relationship = 0.0
        for k,v in tree.relationships[output_num][halo_id].items():
            if not str(k).isdigit(): continue
            if v[1] != 0.0:
                self.children.append((k,v[1]))
                if v[1] > max_relationship:
                    self.progenitor = k
                    max_relationship = v[1]

class EnzoFOFMergerTree(object):
    r"""Calculates the parentage relationships for halos for a series of
    outputs, using the framework provided in enzofof_merger_tree.
    """

    def __init__(self, zrange=None, cycle_range=None, output=False):
        r"""
        Parameters
        ----------
        zrange : tuple
            This is the redshift range (min, max) to calculate the
            merger tree.
        cycle_range : tuple, optional
            This is the cycle number range (min, max) to caluclate the
            merger tree.  If both zrange and cycle_number given,
            ignore zrange.
        output : bool, optional
            If provided, both .cpkl and .txt files containing the parentage
            relationships will be output.
        
        Examples
        --------
        mt = EnzoFOFMergerTree((0.0, 6.0))
        mt.build_tree(0)  # Create tree for halo 0
        mt.print_tree()
        mt.write_dot()
        """
        self.relationships = {}
        self.redshifts = {}
        self.find_outputs(zrange, cycle_range, output)
        self.run_merger_tree(output)

    def clear_data(self):
        r"""Deletes previous merger tree, but keeps parentage
        relationships.
        """
        del self.levels

    def find_outputs(self, zrange, cycle_range, output):
        self.numbers = []
        files = glob.glob("FOF/groups_*.dat")
        # If using redshift range, load redshifts only
        for f in files:
            num = int(f[-9:-4])
            if cycle_range == None:
                HC = HaloCatalog(num)
                # Allow for some epsilon
                diff1 = (HC.redshift - zrange[0]) / zrange[0]
                diff2 = (HC.redshift - zrange[1]) / zrange[1]
                if diff1 >= -1e-3 and diff2 <= 1e-3:
                    self.numbers.append(num)
                del HC
            else:
                if num >= cycle_range[0] and num <= cycle_range[1]:
                    self.numbers.append(num)
        self.numbers.sort()

    def run_merger_tree(self, output):
        # Run merger tree for all outputs, starting with the last output
        for i in range(len(self.numbers)-1, 0, -1):
            if output:
                output = "tree-%5.5d-%5.5d" % (self.numbers[i], self.numbers[i-1])
            else:
                output = None
            z0, z1, fr = find_halo_relationships(self.numbers[i], self.numbers[i-1],
                                                 output_basename=output)
            self.relationships[self.numbers[i]] = fr
            self.redshifts[self.numbers[i]] = z0
        # Fill in last redshift
        self.redshifts[self.numbers[0]] = z1

    def build_tree(self, halonum):
        r"""Builds a merger tree, starting at the last output.

        Parameters
        ----------
        halonum : int
            Halo number in the last output to analyze.
        """
        self.halonum = halonum
        self.output_numbers = sorted(self.relationships, reverse=True)
        self.levels = {}
        trunk = self.output_numbers[0]
        self.levels[trunk] = [EnzoFOFMergerBranch(self, trunk, halonum)]
        self.generate_tree()

    def generate_tree(self):
        for i in range(1,len(self.output_numbers)):
            prev = self.output_numbers[i-1]
            this = self.output_numbers[i]
            self.levels[this] = []
            this_halos = []  # To check for duplicates
            for h in self.levels[prev]:
                for c in h.children:
                    if c[0] in this_halos: continue
                    if self.relationships[this] == {}: continue
                    self.levels[this].append(EnzoFOFMergerBranch(self, this, c[0]))
                    this_halos.append(c[0])

    def print_tree(self):
        r"""Prints the merger tree to stdout.
        """
        for lvl in sorted(self.levels, reverse=True):
            print "========== Cycle %5.5d (z=%f) ==========" % \
                  (lvl, self.redshifts[lvl])
            for br in self.levels[lvl]:
                print "Parent halo = %d" % br.halo_id
                print "--> Most massive progenitor == Halo %d" % \
                      (br.progenitor)
                for c in br.children:
                    print "-->    Halo %8.8d :: fraction = %g" % (c[0], c[1])

    def write_dot(self, filename=None):
        r"""Writes merger tree to a GraphViz file.

        User is responsible for creating an image file from it.

        Parameters
        ----------
        filename : str, optional
            Filename to write the GraphViz file.  Default will be
            tree_halo%05i.dat.
        """
        if filename == None: filename = "tree_halo%5.5d.dot" % self.halonum
        fp = open(filename, "w")
        fp.write("digraph G {\n")
        fp.write("    node [shape=rect];\n")
        sorted_lvl = sorted(self.levels, reverse=True)
        for ii,lvl in enumerate(sorted_lvl):
            # Since we get the cycle number from the key, it won't
            # exist for the last level, i.e. children of last level.
            # Get it from self.numbers.
            if ii < len(sorted_lvl)-1:
                next_lvl = sorted_lvl[ii+1]
            else:
                next_lvl = self.numbers[0]
            for br in self.levels[lvl]:
                for c in br.children:
                    color = "red" if c[0] == br.progenitor else "black"
                    line = "    C%d_H%d -> C%d_H%d [color=%s];\n" % \
                           (lvl, br.halo_id, next_lvl, c[0], color)
                    fp.write(line)
                    last_level = (ii,lvl)
        for ii,lvl in enumerate(sorted_lvl):
            for br in self.levels[lvl]:
                line = "C%d_H%d [label=\"Halo %d\\n%d particles\"]\n" % \
                       (lvl, br.halo_id, br.halo_id, br.npart)
                fp.write(line)
        # Last level, annotate children because they have no associated branches
        for br in self.levels[last_level[1]]:
            for c in br.children:
                npart = self.relationships[last_level[1]][c[0]]["NumberOfParticles"]
                lvl = self.numbers[0]
                line = "C%d_H%d [label=\"Halo %d\\n%d particles\"]\n" % \
                       (lvl, c[0], c[0], npart)
                fp.write(line)
        # Output redshifts
        fp.write("\n")
        fp.write("node [shape=plaintext]\n")
        fp.write("edge [style=invis]\n")
        line = ""
        for k in sorted(self.redshifts, reverse=True):
            line = line + "\"z = %0.3f\"" % (self.redshifts[k]) + " -> "
            if k == self.numbers[0]: break
        line = line[:-4]  # Remove last arrow
        fp.write("\n%s\n" % line)
        
        fp.write("}\n")
        fp.close()

def find_halo_relationships(output1_id, output2_id, output_basename = None,
                            radius = 0.10):
    r"""Calculate the parentage and child relationships between two EnzoFOF
    halo catalogs.

    This function performs a very simple merger tree calculation between two
    sets of halos.  For every halo in the second halo catalog, it looks to the
    first halo catalog to find the parents by looking at particle IDs.  The
    particle IDs from the child halos are identified in potential parents, and
    then both percent-of-parent and percent-to-child values are recorded.

    Note that this works only with catalogs constructed by Enzo's FOF halo
    finder, not with catalogs constructed by yt.

    Parameters
    ----------
    output1_id : int
        This is the integer output id of the (first) halo catalog to parse and
        load.
    output2_id : int
        This is the integer output id of the (second) halo catalog to parse and
        load.
    output_basename : string
        If provided, both .cpkl and .txt files containing the parentage
        relationships will be output.
    radius : float, default to 0.10
        In absolute units, the radius to examine when guessing possible
        parent/child relationships.  If this value is too small, you will miss
        possible relationships.

    Returns
    -------
    pfrac : dict
        This is a dict of dicts.  The first key is the parent halo id, the
        second is the child halo id.  The values are the percent contributed
        from parent to child and the percent of a child that came from the
        parent.
    """
    mylog.info("Parsing Halo Catalog %04i", output1_id)
    HC1 = HaloCatalog(output1_id, False)
    mylog.info("Parsing Halo Catalog %04i", output2_id)
    HC2 = HaloCatalog(output2_id, True)
    mylog.info("Calculating fractions")
    pfrac = HC1.calculate_parentage_fractions(HC2)

    if output_basename is not None and pfrac != {}:
        f = open("%s.txt" % (output_basename), "w")
        for hid1 in sorted(pfrac):
            for hid2 in sorted(pfrac[hid1]):
                if not str(hid2).isdigit(): continue
                p1, p2 = pfrac[hid1][hid2]
                if p1 == 0.0: continue
                f.write( "Halo %s (%s) contributed %0.3e of its particles to %s (%s), which makes up %0.3e of that halo\n" % (
                            hid1, output1_id, p2, hid2, output2_id, p1))
        f.close()

        cPickle.dump(pfrac, open("%s.cpkl" % (output_basename), "wb"))

    return HC1.redshift, HC2.redshift, pfrac
