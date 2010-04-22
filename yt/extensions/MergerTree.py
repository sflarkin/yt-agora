"""
MergerTree class and member functions.

Author: Stephen Skory <sskory@physics.ucsd.edu>
Affiliation: CASS/UC San Diego, CA
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2008-2010 Stephen Skory.  All Rights Reserved.

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

import yt.lagos as lagos
from yt.extensions.kdtree import *
from yt.lagos.HaloFinding import HaloFinder
from yt.logger import lagosLogger as mylog
import yt.extensions.HaloProfiler as HP

import numpy as na
import os, glob, md5, time, gc, sys
import h5py
import types
import sqlite3 as sql
from collections import defaultdict

column_types = {
"GlobalHaloID":"INTEGER",
"SnapCurrentTimeIdentifier":"INTEGER",
"SnapZ":"FLOAT",
"SnapHaloID":"INTEGER",
"HaloMass":"FLOAT",
"NumPart":"INTEGER",
"CenMassX":"FLOAT",
"CenMassY":"FLOAT",
"CenMassZ":"FLOAT",
"BulkVelX":"FLOAT",
"BulkVelY":"FLOAT",
"BulkVelZ":"FLOAT",
"MaxRad":"FLOAT",
"ChildHaloID0":"INTEGER",
"ChildHaloFrac0":"FLOAT",
"ChildHaloID1":"INTEGER",
"ChildHaloFrac1":"FLOAT",
"ChildHaloID2":"INTEGER",
"ChildHaloFrac2":"FLOAT",
"ChildHaloID3":"INTEGER",
"ChildHaloFrac3":"FLOAT",
"ChildHaloID4":"INTEGER", 
"ChildHaloFrac4":"FLOAT"}

# In order.
columns = ["GlobalHaloID", "SnapCurrentTimeIdentifier", "SnapZ", 
"SnapHaloID", "HaloMass", "NumPart", "CenMassX", "CenMassY",
"CenMassZ", "BulkVelX", "BulkVelY", "BulkVelZ", "MaxRad",
"ChildHaloID0", "ChildHaloFrac0",
"ChildHaloID1", "ChildHaloFrac1",
"ChildHaloID2", "ChildHaloFrac2",
"ChildHaloID3", "ChildHaloFrac3",
"ChildHaloID4", "ChildHaloFrac4"]

class DatabaseFunctions(object):
    # Common database functions so it doesn't have to be repeated.
    def _open_database(self):
        # open the database. Check to make sure the database file exists.
        if not os.path.exists(self.database):
            mylog.error("The database file %s cannot be found. Exiting." % \
                self.database)
            return False
        self.conn = sql.connect(self.database)
        self.cursor = self.conn.cursor()
        return True

    def _close_database(self):
        # close the database cleanly.
        self.cursor.close()
        self.conn.close()

class MergerTree(DatabaseFunctions, lagos.ParallelAnalysisInterface):
    def __init__(self, restart_files=[], database='halos.db',
            halo_finder_function=HaloFinder, halo_finder_threshold=80.0,
            FOF_link_length=0.2, dm_only=False, refresh=False, sleep=1,
            index=True):
        self.restart_files = restart_files # list of enzo restart files
        self.with_halos = na.ones(len(restart_files), dtype='bool')
        self.database = database # the sqlite database of haloes.
        self.halo_finder_function = halo_finder_function # which halo finder to use
        self.halo_finder_threshold = halo_finder_threshold # overdensity threshold
        self.FOF_link_length= FOF_link_length # For FOF
        self.dm_only = dm_only
        self.refresh = refresh
        self.sleep = sleep # How long to wait between db sync checks.
        if self.sleep <= 0.:
            self.sleep = 5
        # MPI stuff
        self.mine = self._mpi_get_rank()
        if self.mine is None:
            self.mine = 0
        self.size = self._mpi_get_size()
        if self.size is None:
            self.size = 1
        # Get to work.
        if self.refresh and self.mine == 0:
            try:
                os.unlink(self.database)
            except:
                pass
        self._barrier()
        self._open_create_database()
        self._create_halo_table()
        self._run_halo_finder_add_to_db()
        # Find the h5 file names for all the halos.
        for snap in self.restart_files:
            self._build_h5_refs(snap)
        # Loop over the pairs of snapshots to locate likely neighbors, and
        # then use those likely neighbors to compute fractional contributions.
        last = None
        for snap, pair in enumerate(zip(self.restart_files[:-1], self.restart_files[1:])):
            if not self.with_halos[snap] or not self.with_halos[snap+1]:
                continue
            self._find_likely_children(pair[0], pair[1])
            # last is the data for the parent dataset, which can be supplied
            # as the child from the previous round for all but the first loop.
            last = self._compute_child_fraction(pair[0], pair[1], last)
        del last
        if self.mine == 0 and index:
            mylog.info("Creating database index.")
            line = "CREATE INDEX IF NOT EXISTS HalosIndex ON Halos ("
            for name in columns:
                line += name +","
            line = line[:-1] + ");"
            self.cursor.execute(line)
        self._barrier()
        self._close_database()
        mylog.info("Done!")
        
    def _read_halo_lists(self):
        self.halo_lists = []
        for i,file in enumerate(self.halo_files):
            hp = HP.HaloProfiler(self.restart_files[i], halo_list_file=file)
            self.halo_lists.append(hp.all_halos)

    def _run_halo_finder_add_to_db(self):
        for cycle, file in enumerate(self.restart_files):
            gc.collect()
            pf = lagos.EnzoStaticOutput(file)
            # If the halos are already found, skip this data step, unless
            # refresh is True.
            dir = os.path.dirname(file)
            if os.path.exists(os.path.join(dir, 'MergerHalos.out')) and \
                    os.path.exists(os.path.join(dir, 'MergerHalos.txt')) and \
                    glob.glob(os.path.join(dir, 'MergerHalos*h5')) is not [] and \
                    not self.refresh:
                pass
            else:
                # Run the halo finder.
                if self.halo_finder_function == yt.lagos.HaloFinding.FOFHaloFinder:
                    halos = self.halo_finder_function(pf,
                        link=self.FOF_link_length, dm_only=self.dm_only)
                else:
                    halos = self.halo_finder_function(pf,
                        threshold=self.halo_finder_threshold, dm_only=self.dm_only)
                halos.write_out(os.path.join(dir, 'MergerHalos.out'))
                halos.write_particle_lists(os.path.join(dir, 'MergerHalos'))
                halos.write_particle_lists_txt(os.path.join(dir, 'MergerHalos'))
                if len(halos) == 0:
                    mylog.info("Dataset %s has no halos." % file)
                    self.with_halos[cycle] = False
                    continue
                del halos
            # Now add halo data to the db if it isn't already there by
            # checking the first halo.
            currt = pf['CurrentTimeIdentifier']
            line = "SELECT GlobalHaloID from Halos where SnapHaloID=0\
            and SnapCurrentTimeIdentifier=%d;" % currt
            self.cursor.execute(line)
            result = self.cursor.fetchone()
            if result != None:
                continue
            red = pf['CosmologyCurrentRedshift']
            # Read the halos off the disk using the Halo Profiler tools.
            hp = HP.HaloProfiler(file, halo_list_file='MergerHalos.out',
            halo_list_format={'id':0, 'mass':1, 'numpart':2, 'center':[7, 8, 9], 'velocity':[10, 11, 12], 'r_max':13})
            if len(hp.all_halos) == 0:
                mylog.info("Dataset %s has no halos." % file)
                self.with_halos[cycle] = False
                continue
            mylog.info("Entering halos into database for z=%f" % red)
            if self.mine == 0:
                for ID,halo in enumerate(hp.all_halos):
                    numpart = int(halo['numpart'])
                    values = (None, currt, red, ID, halo['mass'], numpart,
                    halo['center'][0], halo['center'][1], halo['center'][2],
                    halo['velocity'][0], halo['velocity'][1], halo['velocity'][2],
                    halo['r_max'] / pf['mpc'],
                    -1,0.,-1,0.,-1,0.,-1,0.,-1,0.)
                    # 23 question marks for 23 data columns.
                    line = ''
                    for i in range(23):
                        line += '?,'
                    # Pull off the last comma.
                    line = 'INSERT into Halos VALUES (' + line[:-1] + ')'
                    self.cursor.execute(line, values)
                self.conn.commit()
            self._barrier()
            del hp
    
    def _open_create_database(self):
        # open the database. This creates the database file on disk if it
        # doesn't already exist. Open it first on root, and then on the others.
        if self.mine == 0:
            self.conn = sql.connect(self.database)
        self._barrier()
        self._ensure_db_sync()
        if self.mine != 0:
            self.conn = sql.connect(self.database)
        self.cursor = self.conn.cursor()

    def _ensure_db_sync(self):
        # If the database becomes out of sync for each task, ostensibly due to
        # parallel file system funniness, things will go bad very quickly.
        # Therefore, just to be very, very careful, we will ensure that the
        # md5 hash of the file is identical across all tasks before proceeding.
        self._barrier()
        for i in range(5):
            try:
                file = open(self.database)
            except IOError:
                # This is to give a little bit of time for the database creation
                # to replicate across the file system.
                time.sleep(self.sleep)
                file = open(self.database)
            hash = md5.md5(file.read()).hexdigest()
            file.close()
            ignore, hashes = self._mpi_info_dict(hash)
            hashes = set(hashes.values())
            if len(hashes) == 1:
                break
            else:
                # Wait a little bit for the file system to (hopefully) sync up.
                time.sleep(self.sleep)
        if len(hashes) == 1:
            return
        else:
            mylog.error("The file system is not properly synchronizing the database.")
            raise RunTimeError("Fatal error. Exiting.")

    def _create_halo_table(self):
        if self.mine == 0:
            # Handle the error if it already exists.
            try:
                # Create the table that will store the halo data.
                line = "CREATE TABLE Halos (GlobalHaloID INTEGER PRIMARY KEY,\
                    SnapCurrentTimeIdentifier INTEGER, SnapZ FLOAT, SnapHaloID INTEGER, \
                    HaloMass FLOAT,\
                    NumPart INTEGER, CenMassX FLOAT, CenMassY FLOAT,\
                    CenMassZ FLOAT, BulkVelX FLOAT, BulkVelY FLOAT, BulkVelZ FLOAT,\
                    MaxRad FLOAT,\
                    ChildHaloID0 INTEGER, ChildHaloFrac0 FLOAT, \
                    ChildHaloID1 INTEGER, ChildHaloFrac1 FLOAT, \
                    ChildHaloID2 INTEGER, ChildHaloFrac2 FLOAT, \
                    ChildHaloID3 INTEGER, ChildHaloFrac3 FLOAT, \
                    ChildHaloID4 INTEGER, ChildHaloFrac4 FLOAT);"
                self.cursor.execute(line)
                self.conn.commit()
            except sql.OperationalError:
                pass
        self._barrier()
    
    def _find_likely_children(self, parentfile, childfile):
        # For each halo in the parent list, identify likely children in the 
        # list of children.
        
        # First, read in the locations of the child halos.
        child_pf = lagos.EnzoStaticOutput(childfile)
        child_t = child_pf['CurrentTimeIdentifier']
        line = "SELECT SnapHaloID, CenMassX, CenMassY, CenMassZ FROM \
        Halos WHERE SnapCurrentTimeIdentifier = %d" % child_t
        self.cursor.execute(line)
        
        mylog.info("Finding likely parents for z=%1.5f child halos." % \
            child_pf["CosmologyCurrentRedshift"])
        
        # Build the kdtree for the children by looping over the fetched rows.
        child_points = []
        for row in self.cursor:
            child_points.append([row[1], row[2], row[3]])
        # Turn it into fortran.
        child_points = na.array(child_points)
        fKD.pos = na.asfortranarray(child_points.T)
        fKD.qv = na.empty(3, dtype='float64')
        fKD.dist = na.empty(5, dtype='float64')
        fKD.tags = na.empty(5, dtype='int64')
        fKD.nn = 5
        fKD.sort = True
        fKD.rearrange = True
        create_tree()

        # Find the parent points from the database.
        parent_pf = lagos.EnzoStaticOutput(parentfile)
        parent_t = parent_pf['CurrentTimeIdentifier']
        line = "SELECT SnapHaloID, CenMassX, CenMassY, CenMassZ FROM \
        Halos WHERE SnapCurrentTimeIdentifier = %d" % parent_t
        self.cursor.execute(line)

        # Loop over the returned rows, and find the likely neighbors for the
        # parents.
        candidates = {}
        for row in self.cursor:
            fKD.qv = na.array([row[1], row[2], row[3]])
            find_nn_nearest_neighbors()
            NNtags = fKD.tags[:] - 1
            nIDs = []
            for n in NNtags:
                nIDs.append(n)
            if len(nIDs) < 5:
                # We need to fill in fake halos if there aren't enough halos,
                # which can happen at high redshifts.
                while len(nIDs) < 5:
                    nIDs.append(-1)
            candidates[row[0]] = nIDs
        
        del fKD.pos, fKD.tags, fKD.dist
        free_tree() # Frees the kdtree object.
        
        self.candidates = candidates
        
        # This stores the masses contributed to each child candidate.
        self.child_mass_arr = na.zeros(len(candidates)*5, dtype='float64')
        # Records where to put the entries in the above array.
        self.child_mass_loc = defaultdict(dict)
        for i,halo in enumerate(sorted(candidates)):
            for j, child in enumerate(candidates[halo]):
                self.child_mass_loc[halo][child] = i*5 + j

    def _build_h5_refs(self, filename):
        # For this snapshot, add lists of file names that contain the
        # particle info for each halo.
        if not hasattr(self, 'h5files'):
            self.h5files = defaultdict(dict)
        if not hasattr(self, 'names'):
            self.names = defaultdict(set)
        file_pf = lagos.EnzoStaticOutput(filename)
        currt = file_pf['CurrentTimeIdentifier']
        dir = os.path.dirname(filename)
        h5txt = os.path.join(dir, 'MergerHalos.txt')
        lines = file(h5txt)
        names = set([])
        for i,line in enumerate(lines):
            # Get rid of the carriage returns and turn it into a list.
            line = line.strip().split()
            self.h5files[currt][i] = line[1:]
            names.update(line[1:])
            self.names[currt].update(line[1:])
        lines.close()

    def _compute_child_fraction(self, parentfile, childfile, last):
        # Given a parent and child snapshot, and a list of child candidates,
        # compute what fraction of the parent halo goes to each of the children.
        
        parent_pf = lagos.EnzoStaticOutput(parentfile)
        child_pf = lagos.EnzoStaticOutput(childfile)
        parent_currt = parent_pf['CurrentTimeIdentifier']
        child_currt = child_pf['CurrentTimeIdentifier']
        
        mylog.info("Computing fractional contribututions of particles to z=%1.5f halos." % \
            child_pf['CosmologyCurrentRedshift'])
        
        if last == None:
            # First we're going to read in the particles, haloIDs and masses from
            # the parent dataset.
            parent_names = list(self.names[parent_currt])
            parent_names.sort()
            parent_IDs = na.array([], dtype='int64')
            parent_masses = na.array([], dtype='float64')
            parent_halos = na.array([], dtype='int32')
            for i,pname in enumerate(parent_names):
                if i>=self.mine and i%self.size==self.mine:
                    h5fp = h5py.File(pname)
                    for group in h5fp:
                        gID = int(group[4:])
                        thisIDs = h5fp[group]['particle_index'][:]
                        thisMasses = h5fp[group]['ParticleMassMsun'][:]
                        parent_IDs = na.concatenate((parent_IDs, thisIDs))
                        parent_masses = na.concatenate((parent_masses, thisMasses))
                        parent_halos = na.concatenate((parent_halos, 
                            na.ones(thisIDs.size, dtype='int32') * gID))
                    h5fp.close()
            
            # Sort the arrays by particle index in ascending order.
            sort = parent_IDs.argsort()
            parent_IDs = parent_IDs[sort]
            parent_masses = parent_masses[sort]
            parent_halos = parent_halos[sort]
            del sort
        else:
            # We can use old data and save disk reading.
            (parent_IDs, parent_masses, parent_halos) = last
        # Used to communicate un-matched particles.
        parent_send = na.ones(parent_IDs.size, dtype='bool')
        
        # Now get the child halo data.
        child_names = list(self.names[child_currt])
        child_names.sort()
        child_IDs = na.array([], dtype='int64')
        child_masses = na.array([], dtype='float64')
        child_halos = na.array([], dtype='int32')
        for i,cname in enumerate(child_names):
            if i>=self.mine and i%self.size==self.mine:
                h5fp = h5py.File(cname)
                for group in h5fp:
                    gID = int(group[4:])
                    thisIDs = h5fp[group]['particle_index'][:]
                    thisMasses = h5fp[group]['ParticleMassMsun'][:]
                    child_IDs = na.concatenate((child_IDs, thisIDs))
                    child_masses = na.concatenate((child_masses, thisMasses))
                    child_halos = na.concatenate((child_halos, 
                        na.ones(thisIDs.size, dtype='int32') * gID))
                h5fp.close()
        
        # Sort the arrays by particle index.
        sort = child_IDs.argsort()
        child_IDs = child_IDs[sort]
        child_masses = child_masses[sort]
        child_halos = child_halos[sort]
        child_send = na.ones(child_IDs.size, dtype='bool')
        del sort
        
        # Parent IDs on the left, child IDs on the right. We skip down both
        # columns matching IDs. If they are out of synch, the index(es) is/are
        # advanced until they match up again.
        left = 0
        right = 0
        while left < parent_IDs.size and right < child_IDs.size:
            if parent_IDs[left] == child_IDs[right]:
                # They match up, add this relationship.
                try:
                    loc = self.child_mass_loc[parent_halos[left]][child_halos[right]]
                except KeyError:
                    # This happens when a child halo contains a particle from
                    # a parent halo, but the child is not identified as a 
                    # candidate child halo. So we do nothing and move on with
                    # our lives.
                    left += 1
                    right += 1
                    continue
                self.child_mass_arr[loc] += parent_masses[left]
                # Mark this pair so we don't send them later.
                parent_send[left] = False
                child_send[right] = False
                left += 1
                right += 1
                continue
            if parent_IDs[left] < child_IDs[right]:
                # The left is too small, so we need to increase it.
                left += 1
                continue
            if parent_IDs[left] > child_IDs[right]:
                # Right too small.
                right += 1
                continue

        # Now we send all the un-matched particles to the root task for one more
        # pass. This depends on the assumption that most of the particles do
        # not move very much between data dumps, so that not too many particles
        # will be dumped on the single task.
        parent_IDs_tosend = parent_IDs[parent_send]
        parent_masses_tosend = parent_masses[parent_send]
        parent_halos_tosend = parent_halos[parent_send]
        child_IDs_tosend = child_IDs[child_send]
        child_halos_tosend = child_halos[child_send]
        
        parent_IDs_tosend = self._mpi_concatenate_array_on_root_long(parent_IDs_tosend)
        parent_masses_tosend = self._mpi_concatenate_array_on_root_double(parent_masses_tosend)
        parent_halos_tosend = self._mpi_concatenate_array_on_root_int(parent_halos_tosend)
        child_IDs_tosend = self._mpi_concatenate_array_on_root_long(child_IDs_tosend)
        child_halos_tosend = self._mpi_concatenate_array_on_root_int(child_halos_tosend)

        # Resort the received particles.
        Psort = parent_IDs_tosend.argsort()
        parent_IDs_tosend = parent_IDs_tosend[Psort]
        parent_masses_tosend = parent_masses_tosend[Psort]
        parent_halos_tosend = parent_halos_tosend[Psort]
        Csort = child_IDs_tosend.argsort()
        child_IDs_tosend = child_IDs_tosend[Csort]
        child_halos_tosend = child_halos_tosend[Csort]
        del Psort, Csort

        # Now Again.
        if self.mine == 0:
            matched = 0
            left = 0
            right = 0
            while left < parent_IDs_tosend.size and right < child_IDs_tosend.size:
                if parent_IDs_tosend[left] == child_IDs_tosend[right]:
                    # They match up, add this relationship.
                    try:
                        loc = self.child_mass_loc[parent_halos_tosend[left]][child_halos_tosend[right]]
                    except KeyError:
                        # This happens when a child halo contains a particle from
                        # a parent halo, but the child is not identified as a 
                        # candidate child halo. So we do nothing and move on with
                        # our lives.
                        left += 1
                        right += 1
                        continue
                    self.child_mass_arr[loc] += parent_masses_tosend[left]
                    matched += 1
                    left += 1
                    right += 1
                    continue
                if parent_IDs_tosend[left] < child_IDs_tosend[right]:
                    # The left is too small, so we need to increase it.
                    left += 1
                    continue
                if parent_IDs_tosend[left] > child_IDs_tosend[right]:
                    # Right too small.
                    right += 1
                    continue
            mylog.info("Clean-up round matched %d of %d parents and %d children." % \
            (matched, parent_IDs_tosend.size, child_IDs_tosend.size))

        # Now we sum up the contributions globally.
        self.child_mass_arr = self._mpi_Allsum_double(self.child_mass_arr)
        
        # Turn these Msol masses into percentages of the parent.
        line = "SELECT HaloMass FROM Halos WHERE SnapCurrentTimeIdentifier=%d \
        ORDER BY SnapHaloID ASC;" % parent_currt
        self.cursor.execute(line)
        mark = 0
        result = self.cursor.fetchone()
        while result:
            mass = result[0]
            self.child_mass_arr[mark:mark+5] /= mass
            mark += 5
            result = self.cursor.fetchone()
        
        # Get the global ID for the SnapHaloID=0 from the child, this will
        # be used to prevent unnecessary SQL reads.
        line = "SELECT GlobalHaloID FROM Halos WHERE SnapCurrentTimeIdentifier=%d \
        AND SnapHaloID=0;" % child_currt
        self.cursor.execute(line)
        baseChildID = self.cursor.fetchone()[0]
        
        # Now we prepare a big list of writes to put in the database.
        write_values = []
        for i,parent_halo in enumerate(sorted(self.candidates)):
            child_indexes = []
            child_per = []
            for j,child in enumerate(self.candidates[parent_halo]):
                if child == -1:
                    # Account for fake children.
                    child_indexes.append(-1)
                    child_per.append(0.)
                    continue
                # We need to get the GlobalHaloID for this child.
                child_globalID = baseChildID + child
                child_indexes.append(child_globalID)
                child_per.append(self.child_mass_arr[i*5 + j])
            # Sort by percentages, desending.
            child_per, child_indexes = zip(*sorted(zip(child_per, child_indexes), reverse=True))
            values = []
            for pair in zip(child_indexes, child_per):
                values.extend([int(pair[0]), float(pair[1])])
            values.extend([parent_currt, parent_halo])
            # This has the child ID, child percent listed five times, followed
            # by the currt and this parent halo ID (SnapHaloID).
            values = tuple(values)
            write_values.append(values)
        
        # Now we do the actual writing, but only by task 0.
        line = 'UPDATE Halos SET ChildHaloID0=?, ChildHaloFrac0=?,\
        ChildHaloID1=?, ChildHaloFrac1=?,\
        ChildHaloID2=?, ChildHaloFrac2=?,\
        ChildHaloID3=?, ChildHaloFrac3=?,\
        ChildHaloID4=?, ChildHaloFrac4=?\
        WHERE SnapCurrentTimeIdentifier=? AND SnapHaloID=?;'
        if self.mine == 0:
            for values in write_values:
                self.cursor.execute(line, values)
            self.conn.commit()
        
        # This has a barrier in it, which ensures the disk isn't lagging.
        self._ensure_db_sync()
        
        return (child_IDs, child_masses, child_halos)

class MergerTreeConnect(DatabaseFunctions):
    def __init__(self, database='halos.db'):
        self.database = database
        result = self._open_database()
        if not result:
            return None
    
    def close(self):
        # To be more like typical Python open/close.
        self._close_database()
    
    def query(self, string):
        # Query the database and return a list of tuples.
        if string is None:
            mylog.error("You must enter a SQL query.")
            return None
        items = []
        self.cursor.execute(string)
        results = self.cursor.fetchone()
        while results:
            items.append(results)
            results = self.cursor.fetchone()
        return items

class Node(object):
    def __init__(self, CoM, mass, parentIDs, z, color):
        self.CoM = CoM
        self.mass = mass
        self.parentIDs = parentIDs # In descending order of contribution
        self.z = z
        self.color = color

class Link(object):
    def __init__(self):
        self.childIDs = []
        self.fractions = []

class MergerTreeDotOutput(DatabaseFunctions, lagos.ParallelAnalysisInterface):
    def __init__(self, halos=None, database='halos.db',
            dotfile='MergerTree.gv', current_time=None, link_min=0.2):
        self.database = database
        self.link_min = link_min
        if halos is None:
            mylog.error("Please provide at least one halo to start the tree. Exiting.")
            return None
        result = self._open_database()
        if not result:
            return None
        if type(halos) == types.IntType:
            halos = [halos]
        if current_time is not None:
            halos = self._translate_haloIDs(halos, current_time)
        newhalos = set(halos)
        # A key is the GlobalHaloID for this halo, and the content is a
        # Node object.
        self.nodes = {}
        # A key is the GlobalHaloID for the parent in the relationship,
        # and the content is a Link ojbect.
        self.links = defaultdict(Link)
        # Record which halos are at the same z level for convenience.
        # They key is a z value, and the content a list of co-leveled halo IDs.
        self.levels = defaultdict(list)
        # For the first set of halos.
        self._add_nodes(newhalos)
        # Recurse over parents.
        while len(newhalos) > 0:
            mylog.info("Finding parents for %d children." % len(newhalos))
            newhalos = self._find_parents(newhalos)
            self._add_nodes(newhalos)
        mylog.info("Writing out %s to disk." % dotfile)
        self._open_dot(dotfile)
        self._write_nodes()
        self._write_links()
        self._write_levels()
        self._close_dot()
        self._close_database()
        return None

    def _translate_haloIDs(self, halos, current_time):
        # If the input is in the haloID equivalent to SnapHaloID, translate them
        # to GlobalHaloIDs.
        new_haloIDs=[]
        for halo in halos:
            line = "SELECT GlobalHaloID FROM Halos WHERE SnapHaloID=? AND \
            SnapCurrentTimeIdentifier=? limit 1;"
            values = (halo, current_time)
            self.cursor.execute(line, values)
            new_haloIDs.append(self.cursor.fetchone()[0])
        return new_haloIDs
        
    def _find_parents(self, halos):
        # Given a set of halos, find their parents and add that to each of their
        # node records. At the same time, make a link record for that
        # relationship.
        # This stores the newly discovered parent halos.
        newhalos = set([])
        for halo in halos:
            line = "SELECT GlobalHaloID, ChildHaloFrac0,\
                ChildHaloFrac1, ChildHaloFrac2,ChildHaloFrac3, ChildHaloFrac4,\
                ChildHaloID0, ChildHaloID1, ChildHaloID2, \
                ChildHaloID3, ChildHaloID4 \
                FROM Halos WHERE\
                ChildHaloID0=? or ChildHaloID1=? or ChildHaloID2=? or\
                ChildHaloID3=? or ChildHaloID4=?;"
            values = (halo, halo, halo, halo, halo)
            self.cursor.execute(line, values)
            result = self.cursor.fetchone()
            while result:
                res = list(result)
                pID = result[0]
                pfracs = res[1:6]
                cIDs = res[6:11]
                for pair in zip(cIDs, pfracs):
                    if pair[1] <= self.link_min or pair[0] != halo:
                        continue
                    else:
                        self.nodes[halo].parentIDs.append(pID)
                        self.links[pID].childIDs.append(halo)
                        self.links[pID].fractions.append(pair[1])
                        newhalos.add(pID)
                result = self.cursor.fetchone()
        return newhalos
    
    def _add_nodes(self, newhalos):
        # Each call of this function always happens for a set of newhalos that
        # are at the same z. To give the halos color we will figure out how
        # many halos total were found this z.
        # There's probably a way to do this with only one SQL operation.
        if len(newhalos) == 0:
            return
        ahalo = list(newhalos)[0]
        line = 'SELECT SnapCurrentTimeIdentifier FROM Halos WHERE GlobalHaloID=?;'
        values = (ahalo,)
        self.cursor.execute(line, values)
        result = self.cursor.fetchone()
        # Use currt to get the number.
        line = 'SELECT max(SnapHaloID) FROM Halos where SnapCurrentTimeIdentifier=?;'
        values = (result[0],)
        self.cursor.execute(line, values)
        maxID = self.cursor.fetchone()[0]
        # For the new halos, create nodes for them.
        for halo in newhalos:
            line = 'SELECT SnapZ, HaloMass, CenMassX, CenMassY, CenMassZ,\
            SnapHaloID FROM Halos WHERE GlobalHaloID=? limit 1;'
            value = (halo,)
            self.cursor.execute(line, value)
            result = self.cursor.fetchone()
            self.nodes[halo] = Node(na.array([result[2],result[3],result[4]]),
                result[1], [], result[0], 1. - float(result[5])/maxID)
            self.levels[result[0]].append(halo)

    def _open_dot(self, dotfile):
        # Write out the opening stuff in the dotfile.
        self.dotfile=self._write_on_root(dotfile)
        line = 'digraph galaxy {size="10, 10";\n'
        line += 'node [style=bold, shape=record];\n'
        self.dotfile.write(line)
    
    def _close_dot(self):
        self.dotfile.write("\n};\n")
        self.dotfile.close()
    
    def _write_nodes(self):
        # Write out the nodes to the dot file.
        self.dotfile.write("{\n")
        for halo in self.nodes:
            this = self.nodes[halo]
            line = '"%d" [label="{%1.3e\\n(%1.3f,%1.3f,%1.3f)}", shape="record",' \
                % (halo, this.mass, this.CoM[0], this.CoM[1], this.CoM[2])
            line += ' color="%0.3f 1. %0.3f"];\n' % (this.color, this.color)
            self.dotfile.write(line)
        self.dotfile.write("};\n")
    
    def _write_links(self):
        # Write out the links to the dot file.
        self.dotfile.write("{\n")
        for parent in self.links:
            this = self.links[parent]
            for child,frac in zip(this.childIDs, this.fractions):
                if frac > self.link_min:
                    line = '"%d"->"%d" [label="%3.2f%%", color="blue", fontsize=10];\n' \
                        % (parent, child, frac*100.)
                    self.dotfile.write(line)
        self.dotfile.write("};\n")

    def _write_levels(self):
        # Write out the co-leveled halos to the dot file.
        for z in self.levels:
            this = self.levels[z]
            self.dotfile.write("{ rank = same;\n")
            line = '"%1.5f"; ' % z
            for halo in this:
                line += '"%d"; ' % halo
            line += "\n};\n"
            self.dotfile.write(line)
        # Also write out the unlinked boxes for the redshifts.
        line = '{"%1.5f" [label="{%1.5f}", shape="record" color="green"];}\n' \
            % (z, z)

class MergerTreeTextOutput(DatabaseFunctions, lagos.ParallelAnalysisInterface):
    def __init__(self, database='halos.db', outfile='MergerTreeDB.txt'):
        self.database = database
        self.outfile = outfile
        result = self._open_database()
        if not result:
            return None
        self._write_out()
        self._close_database()
        return None
    
    def _write_out(self):
        # Essentially dump the contents of the database into a text file.
        fp = open(self.outfile, "w")
        # Make the header line.
        spacing = {}
        for column in columns:
            spacing[column] = (max(15,len(column)+1))
        line = "# "
        for column in columns:
            line += "%s" % column.ljust(spacing[column])
        line += "\n"
        fp.write(line)
        # Get the data.
        line = "SELECT * FROM Halos ORDER BY SnapZ DESC, SnapHaloID ASC;"
        self.cursor.execute(line)
        results = self.cursor.fetchone()
        # Write out the columns.
        while results:
            line = "  "
            for i,column in enumerate(columns):
                if column_types[column] == "FLOAT":
                    this = "%1.6e" % results[i]
                    line += this.ljust(spacing[column])
                if column_types[column] == "INTEGER":
                    this = "%d" % results[i]
                    line += this.ljust(spacing[column])
            line += "\n"
            fp.write(line)
            results = self.cursor.fetchone()
        fp.close()
        
