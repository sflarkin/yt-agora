"""
HOP-output data handling

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
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

from yt.lagos import *
from yt.math_utils import *
from yt.lagos.hop.EnzoHop import RunHOP
try:
    from yt.lagos.parallelHOP.parallelHOP import *
except ImportError:
    mylog.debug("ParallelHOP not imported.")

try:
    from yt.lagos.fof.EnzoFOF import RunFOF
except ImportError:
    pass
from yt.performance_counters import yt_counters, time_function

from kd import *
from yt.funcs import *
import math, sys, itertools, gc, random
from collections import defaultdict

TINY = 1.e-40

class Halo(object):
    """
    A data source that returns particle information about the members of a
    HOP-identified halo.
    """
    __metaclass__ = ParallelDummy # This will proxy up our methods
    _distributed = False
    _processing = False
    _owner = 0
    indices = None
    dont_wrap = ["get_sphere", "write_particle_list"]
    extra_wrap = ["__getitem__"]

    def __init__(self, halo_list, id, indices = None, size=None, CoM=None,
        max_dens_point=None, group_total_mass=None, max_radius=None, bulk_vel=None,
        tasks=None, rms_vel=None):
        self._max_dens = halo_list._max_dens
        self.id = id
        self.data = halo_list._data_source
        if indices is not None:
            self.indices = halo_list._base_indices[indices]
        else:
            self.indices = None
        # We assume that if indices = None, the instantiator has OTHER plans
        # for us -- i.e., setting it somehow else
        self.size = size
        self.CoM = CoM
        self.max_dens_point = max_dens_point
        self.group_total_mass = group_total_mass
        self.max_radius = max_radius
        self.bulk_vel = bulk_vel
        self.tasks = tasks
        self.rms_vel = rms_vel
        self.bin_count = None
        self.overdensity = None

    def center_of_mass(self):
        """
        Calculate and return the center of mass.
        """
        c_vec = self.maximum_density_location() - na.array([0.5,0.5,0.5])
        pm = self["ParticleMassMsun"]
        cx = (self["particle_position_x"] - c_vec[0])
        cy = (self["particle_position_y"] - c_vec[1])
        cz = (self["particle_position_z"] - c_vec[2])
        com = na.array([v-na.floor(v) for v in [cx,cy,cz]])
        return (com*pm).sum(axis=1)/pm.sum() + c_vec

    def maximum_density(self):
        """
        Return the HOP-identified maximum density.
        """
        return self._max_dens[self.id][0]

    def maximum_density_location(self):
        """
        Return the location HOP identified as maximally dense.
        """
        return na.array([
                self._max_dens[self.id][1],
                self._max_dens[self.id][2],
                self._max_dens[self.id][3]])

    def total_mass(self):
        """
        Returns the total mass in solar masses of the halo.
        """
        return self["ParticleMassMsun"].sum()

    def bulk_velocity(self):
        """
        Returns the mass-weighted average velocity.
        """
        pm = self["ParticleMassMsun"]
        vx = (self["particle_velocity_x"] * pm).sum()
        vy = (self["particle_velocity_y"] * pm).sum()
        vz = (self["particle_velocity_z"] * pm).sum()
        return na.array([vx,vy,vz])/pm.sum()

    def rms_velocity(self):
        """
        Returns the mass-weighted RMS velocity for the halo
        particles in cgs units.
        """
        bv = self.bulk_velocity()
        pm = self["ParticleMassMsun"]
        sm = pm.sum()
        vx = (self["particle_velocity_x"] - bv[0]) * pm/sm
        vy = (self["particle_velocity_y"] - bv[1]) * pm/sm
        vz = (self["particle_velocity_z"] - bv[2]) * pm/sm
        s = vx**2. + vy**2. + vz**2.
        ms = na.mean(s)
        return na.sqrt(ms) * pm.size

    def maximum_radius(self, center_of_mass=True):
        """
        Returns the maximum radius in the halo for all particles,
        either from the point of maximum density or from the (default)
        *center_of_mass*.
        """
        if center_of_mass: center = self.center_of_mass()
        else: center = self.maximum_density_location()
        rx = na.abs(self["particle_position_x"]-center[0])
        ry = na.abs(self["particle_position_y"]-center[1])
        rz = na.abs(self["particle_position_z"]-center[2])
        r = na.sqrt(na.minimum(rx, 1.0-rx)**2.0
                +   na.minimum(ry, 1.0-ry)**2.0
                +   na.minimum(rz, 1.0-rz)**2.0)
        return r.max()

    def __getitem__(self, key):
        if ytcfg.getboolean("yt","inline") == False:
            return self.data[key][self.indices]
        else:
            return self.data[key][self.indices]

    def get_sphere(self, center_of_mass=True):
        """
        Returns an EnzoSphere centered on either the point of maximum density
        or the *center_of_mass*, with the maximum radius of the halo.
        """
        if center_of_mass: center = self.center_of_mass()
        else: center = self.maximum_density_location()
        radius = self.maximum_radius()
        # A bit of a long-reach here...
        sphere = self.data.hierarchy.sphere(
                        center, radius=radius)
        return sphere

    def get_size(self):
        return self.indices.size

    def write_particle_list(self, handle):
        self._processing = True
        gn = "Halo%08i" % (self.id)
        handle.create_group("/%s" % gn)
        for field in ["particle_position_%s" % ax for ax in 'xyz'] \
                   + ["particle_velocity_%s" % ax for ax in 'xyz'] \
                   + ["particle_index"] + ["ParticleMassMsun"]:
            handle.create_dataset("/%s/%s" % (gn, field), data=self[field])
        n = handle["/%s" % gn]
        # set attributes on n
        self._processing = False

    def virial_mass(self, virial_overdensity=200., bins=300):
        """
        Return the virial mass of the halo in Msun, using only the particles
        in the halo (no baryonic information used).
        Calculate using *bins* number of bins and *virial_overdensity* density
        threshold. Returns -1 if the halo is not virialized.
        """
        self.virial_info(bins=bins)
        vir_bin = self.virial_bin(virial_overdensity=virial_overdensity, bins=bins)
        if vir_bin != -1:
            return self.mass_bins[vir_bin]
        else:
            return -1
        
    
    def virial_radius(self, virial_overdensity=200., bins=300):
        """
        Return the virial radius of the halo in code units, using only the
        particles in the halo (no baryonic information used).
        Calculate using *bins* number of bins and *virial_overdensity* density
        threshold. Returns -1 if the halo is not virialized.
        """
        self.virial_info(bins=bins)
        vir_bin = self.virial_bin(virial_overdensity=virial_overdensity, bins=bins)
        if vir_bin != -1:
            return self.radial_bins[vir_bin]
        else:
            return -1

    def virial_bin(self, virial_overdensity=200., bins=300):
        """
        Return the bin index for the virial radius for the given halo.
        Returns -1 if the halo is not virialized to the set
        *virial_overdensity*. 
        """
        self.virial_info(bins=bins)
        over = (self.overdensity > virial_overdensity)
        if (over == True).any():
            vir_bin = max(na.arange(bins+1)[over])
            return vir_bin
        else:
            return -1
    
    def virial_info(self, bins=300):
        """
        Calculate the virial profile bins for this halo, using only the particles
        in the halo (no baryonic information used).
        Calculate using *bins* number of bins.
        """
        # Skip if we've already calculated for this number of bins.
        if self.bin_count == bins and self.overdensity is not None:
            return None
        self.bin_count = bins
        # Cosmology
        h = self.data.pf['CosmologyHubbleConstantNow']
        Om_matter = self.data.pf['CosmologyOmegaMatterNow']
        z = self.data.pf['CosmologyCurrentRedshift']
        rho_crit_now = 1.8788e-29 * h**2.0 * Om_matter # g cm^-3
        Msun2g = 1.989e33
        rho_crit = rho_crit_now * ((1.0 + z)**3.0)
        
        # Get some pertinent information about the halo.
        self.mass_bins = na.zeros(self.bin_count+1, dtype='float64')
        dist = na.empty(self.indices.size, dtype='float64')
        cen = self.center_of_mass()
        period = self.data.pf["DomainRightEdge"] - \
            self.data.pf["DomainLeftEdge"]
        mark = 0
        # Find the distances to the particles. I don't like this much, but I
        # can't see a way to eliminate a loop like this, either here or in
        # yt.math.
        for pos in izip(self["particle_position_x"], self["particle_position_y"],
                self["particle_position_z"]):
            dist[mark] = periodic_dist(cen, pos, period)
            mark += 1
        # Set up the radial bins.
        # Multiply min and max to prevent issues with digitize below.
        self.radial_bins = na.logspace(math.log10(min(dist)*.99 + TINY), 
            math.log10(max(dist)*1.01 + 2*TINY), num=self.bin_count+1)
        # Find out which bin each particle goes into, and add the particle
        # mass to that bin.
        inds = na.digitize(dist, self.radial_bins) - 1
        if self["particle_position_x"].size > 1:
            for index in na.unique(inds):
                self.mass_bins[index] += sum(self["ParticleMassMsun"][inds==index])
        # Now forward sum the masses in the bins.
        for i in xrange(self.bin_count):
            self.mass_bins[i+1] += self.mass_bins[i]
        # Calculate the over densities in the bins.
        self.overdensity = self.mass_bins * Msun2g / \
        (4./3. * math.pi * rho_crit * \
        (self.radial_bins * self.data.pf["cm"])**3.0)
        

class HOPHalo(Halo):
    pass

class parallelHOPHalo(Halo,ParallelAnalysisInterface):
    dont_wrap = ["maximum_density","maximum_density_location",
        "center_of_mass","total_mass","bulk_velocity","maximum_radius",
        "get_size","get_sphere", "write_particle_list","__getitem__", 
        "virial_info", "virial_bin", "virial_mass", "virial_radius",
        "rms_velocity"]

    def maximum_density(self):
        """
        Return the HOP-identified maximum density.
        """
        if self.max_dens_point is not None:
            return self._max_dens[self.id][0]
        max = self._mpi_allmax(self._max_dens[self.id][0])
        return max

    def maximum_density_location(self):
        """
        Return the location HOP identified as maximally dense.
        """
        if self.max_dens_point is not None:
            return self.max_dens_point[1:]
        # If I own the maximum density, my location is globally correct.
        max_dens = self.maximum_density()
        if self._max_dens[self.id][0] == max_dens:
            value = na.array([
                self._max_dens[self.id][1],
                self._max_dens[self.id][2],
                self._max_dens[self.id][3]])
        else:
            value = na.array([0,0,0])
        # This works, and isn't appropriate but for now will be fine...
        value = self._mpi_allsum(value)
        return value

    def center_of_mass(self):
        """
        Calculate and return the center of mass.
        """
        # If it's precomputed, we save time!
        if self.CoM is not None:
            return self.CoM
        # This need to be called by all tasks, but not all will end up using
        # it.
        c_vec = self.maximum_density_location() - na.array([0.5,0.5,0.5])
        if self.indices is not None:
            pm = self["ParticleMassMsun"]
            cx = (self["particle_position_x"] - c_vec[0])
            cy = (self["particle_position_y"] - c_vec[1])
            cz = (self["particle_position_z"] - c_vec[2])
            com = na.array([v-na.floor(v) for v in [cx,cy,cz]])
            my_mass = pm.sum()
            my_com = ((com*pm).sum(axis=1)/my_mass + c_vec) * my_mass
        else:
            my_mass = 0.
            my_com = na.array([0.,0.,0.])
        global_mass = self._mpi_allsum(my_mass)
        global_com = self._mpi_allsum(my_com)
        return global_com / global_mass

    def total_mass(self):
        """
        Returns the total mass in solar masses of the halo.
        """
        if self.group_total_mass is not None:
            return self.group_total_mass
        if self.indices is not None:
            my_mass = self["ParticleMassMsun"].sum()
        else:
            my_mass = 0.
        global_mass = self._mpi_allsum(float(my_mass))
        return global_mass

    def bulk_velocity(self):
        """
        Returns the mass-weighted average velocity.
        """
        if self.bulk_vel is not None:
            return self.bulk_vel
        # Unf. this cannot be reasonably computed inside of parallelHOP because
        # we don't pass velocities in.
        if self.indices is not None:
            pm = self["ParticleMassMsun"]
            vx = (self["particle_velocity_x"] * pm).sum()
            vy = (self["particle_velocity_y"] * pm).sum()
            vz = (self["particle_velocity_z"] * pm).sum()
            pm = pm.sum()
        else:
            pm = 0.
            vx = 0.
            vy = 0.
            vz = 0.
        bv = na.array([vx,vy,vz,pm])
        global_bv = self._mpi_allsum(bv)
        return global_bv[:3]/global_bv[3]

    def rms_velocity(self):
        """
        Returns the RMS velocity for the halo particles in cgs units.
        """
        if self.rms_vel is not None:
            return self.rms_vel
        bv = self.bulk_velocity()
        pm = self["ParticleMassMsun"]
        sm = pm.sum()
        if self.indices is not None:
            vx = (self["particle_velocity_x"] - bv[0]) * pm/sm
            vy = (self["particle_velocity_y"] - bv[1]) * pm/sm
            vz = (self["particle_velocity_z"] - bv[2]) * pm/sm
            s = vx**2 + vy**2 + vz**2
            s = na.sum(s)
            size = vx.size
            ss = na.array([s, float(size)])
        else:
            ss = na.array([0.,0.])
        global_ss = self._mpi_allsum(ss)
        ms = global_ss[0] / global_ss[1]
        return na.sqrt(ms) * global_ss[1]

    def maximum_radius(self, center_of_mass=True):
        """
        Returns the maximum radius in the halo for all particles,
        either from the point of maximum density or from the (default)
        *center_of_mass*.
        """
        if self.max_radius is not None:
            return self.max_radius
        if center_of_mass: center = self.center_of_mass()
        else: center = self.maximum_density_location()
        if self.indices is not None:
            rx = na.abs(self["particle_position_x"]-center[0])
            ry = na.abs(self["particle_position_y"]-center[1])
            rz = na.abs(self["particle_position_z"]-center[2])
            r = na.sqrt(na.minimum(rx, 1.0-rx)**2.0
                    +   na.minimum(ry, 1.0-ry)**2.0
                    +   na.minimum(rz, 1.0-rz)**2.0)
            my_max = r.max()
            
        else:
            my_max = 0.
        return self._mpi_allmax(my_max)

    def get_size(self):
        if self.size is not None:
            return self.size
        if self.indices is not None:
            my_size = self.indices.size
        else:
            my_size = 0
        global_size = self._mpi_allsum(my_size)
        return global_size

    def __getitem__(self, key):
        if ytcfg.getboolean("yt","inline") == False:
            return self.data[key][self.indices]
        else:
            return self.data[key][self.indices]

    def virial_mass(self, virial_overdensity=200., bins=300):
        """
        Return the virial mass of the halo in Msun, using only the particles
        in the halo (no baryonic information used).
        Calculate using *bins* number of bins and *virial_overdensity* density
        threshold. Returns -1 if the halo is not virialized.
        """
        self.virial_info(bins=bins)
        vir_bin = self.virial_bin(virial_overdensity=virial_overdensity, bins=bins)
        if vir_bin != -1:
            return self.mass_bins[vir_bin]
        else:
            return -1
        
    
    def virial_radius(self, virial_overdensity=200., bins=300):
        """
        Return the virial radius of the halo in code units, using only the
        particles in the halo (no baryonic information used).
        Calculate using *bins* number of bins and *virial_overdensity* density
        threshold. Returns -1 if the halo is not virialized.
        """
        self.virial_info(bins=bins)
        vir_bin = self.virial_bin(virial_overdensity=virial_overdensity, bins=bins)
        if vir_bin != -1:
            return self.radial_bins[vir_bin]
        else:
            return -1

    def virial_bin(self, virial_overdensity=200., bins=300):
        """
        Return the bin index for the virial radius for the given halo.
        Returns -1 if the halo is not virialized to the set
        *virial_overdensity*. 
        """
        self.virial_info(bins=bins)
        over = (self.overdensity > virial_overdensity)
        if (over == True).any():
            vir_bin = max(na.arange(bins+1)[over])
            return vir_bin
        else:
            return -1

    def virial_info(self, bins=300):
        """
        Calculate the virial profile bins for this halo, using only the particles
        in the halo (no baryonic information used).
        Calculate using *bins* number of bins.
        """
        # Skip if we've already calculated for this number of bins.
        if self.bin_count == bins and self.overdensity is not None:
            return None
        # Do this for all because all will use it.
        self.bin_count = bins
        period = self.data.pf["DomainRightEdge"] - \
            self.data.pf["DomainLeftEdge"]
        self.mass_bins = na.zeros(self.bin_count+1, dtype='float64')
        cen = self.center_of_mass()
        # Cosmology
        h = self.data.pf['CosmologyHubbleConstantNow']
        Om_matter = self.data.pf['CosmologyOmegaMatterNow']
        z = self.data.pf['CosmologyCurrentRedshift']
        rho_crit_now = 1.8788e-29 * h**2.0 * Om_matter # g cm^-3
        Msun2g = 1.989e33
        rho_crit = rho_crit_now * ((1.0 + z)**3.0)
        # If I own some of this halo operate on the particles.
        if self.indices is not None:
            # Get some pertinent information about the halo.
            dist = na.empty(self.indices.size, dtype='float64')
            mark = 0
            # Find the distances to the particles. I don't like this much, but I
            # can't see a way to eliminate a loop like this, either here or in
            # yt.math.
            for pos in izip(self["particle_position_x"], self["particle_position_y"],
                    self["particle_position_z"]):
                dist[mark] = periodic_dist(cen, pos, period)
                mark += 1
            dist_min, dist_max = min(dist), max(dist)
        # If I don't have this halo, make some dummy values.
        else:
            dist_min = max(period)
            dist_max = 0.0
        # In this parallel case, we're going to find the global dist extrema
        # and built identical bins on all tasks.
        dist_min = self._mpi_allmin(dist_min)
        dist_max = self._mpi_allmax(dist_max)
        # Set up the radial bins.
        # Multiply min and max to prevent issues with digitize below.
        self.radial_bins = na.logspace(math.log10(dist_min*.99 + TINY), 
            math.log10(dist_max*1.01 + 2*TINY), num=self.bin_count+1)
        if self.indices is not None and self.indices.size > 1:
            # Find out which bin each particle goes into, and add the particle
            # mass to that bin.
            inds = na.digitize(dist, self.radial_bins) - 1
            for index in na.unique(inds):
                self.mass_bins[index] += sum(self["ParticleMassMsun"][inds==index])
            # Now forward sum the masses in the bins.
            for i in xrange(self.bin_count):
                self.mass_bins[i+1] += self.mass_bins[i]
        # Sum up the mass_bins globally
        self.mass_bins = self._mpi_Allsum_double(self.mass_bins)
        # Calculate the over densities in the bins.
        self.overdensity = self.mass_bins * Msun2g / \
        (4./3. * math.pi * rho_crit * \
        (self.radial_bins * self.data.pf["cm"])**3.0)


class FOFHalo(Halo):

    def center_of_mass(self):
        """
        Calculate and return the center of mass.
        """
        pm = self["ParticleMassMsun"]
        cx = self["particle_position_x"]
        cy = self["particle_position_y"]
        cz = self["particle_position_z"]
        c_vec = na.array([cx[0],cy[0],cz[0]]) - na.array([0.5,0.5,0.5])
        cx = cx - c_vec[0]
        cy = cy - c_vec[1]
        cz = cz - c_vec[2]
        com = na.array([v-na.floor(v) for v in [cx,cy,cz]])
        com = (pm * com).sum(axis=1)/pm.sum() + c_vec
        return com

    def maximum_density(self):
        return -1

    def maximum_density_location(self):
        return self.center_of_mass()

class HaloList(object):

    _fields = ["particle_position_%s" % ax for ax in 'xyz']

    def __init__(self, data_source, dm_only = True):
        """
        Run hop on *data_source* with a given density *threshold*.  If
        *dm_only* is set, only run it on the dark matter particles, otherwise
        on all particles.  Returns an iterable collection of *HopGroup* items.
        """
        self._data_source = data_source
        self.dm_only = dm_only
        self._groups = []
        self._max_dens = {}
        self.__obtain_particles()
        self._run_finder()
        mylog.info("Parsing outputs")
        self._parse_output()
        mylog.debug("Finished. (%s)", len(self))

    def __obtain_particles(self):
        if self.dm_only: ii = self._get_dm_indices()
        else: ii = slice(None)
        self.particle_fields = {}
        for field in self._fields:
            if ytcfg.getboolean("yt","inline") == False:
                tot_part = self._data_source[field].size
                if field == "particle_index":
                    self.particle_fields[field] = self._data_source[field][ii].astype('int64')
                else:
                    self.particle_fields[field] = self._data_source[field][ii].astype('float64')
            else:
                tot_part = self._data_source[field].size
                if field == "particle_index":
                    self.particle_fields[field] = self._data_source[field][ii].astype('int64')
                else:
                    self.particle_fields[field] = self._data_source[field][ii].astype('float64')
        self._base_indices = na.arange(tot_part)[ii]

    def _get_dm_indices(self):
        if 'creation_time' in self._data_source.hierarchy.field_list:
            mylog.debug("Differentiating based on creation time")
            return (self._data_source["creation_time"] < 0)
        elif 'particle_type' in self._data_source.hierarchy.field_list:
            mylog.debug("Differentiating based on particle type")
            return (self._data_source["particle_type"] == 1)
        else:
            mylog.warning("No particle_type, no creation_time, so not distinguishing.")
            return slice(None)
    

    def _parse_output(self):
        unique_ids = na.unique(self.tags)
        counts = na.bincount(self.tags+1)
        sort_indices = na.argsort(self.tags)
        grab_indices = na.indices(self.tags.shape).ravel()[sort_indices]
        dens = self.densities[sort_indices]
        cp = 0
        for i in unique_ids:
            cp_c = cp + counts[i+1]
            if i == -1:
                cp += counts[i+1]
                continue
            group_indices = grab_indices[cp:cp_c]
            self._groups.append(self._halo_class(self, i, group_indices))
            md_i = na.argmax(dens[cp:cp_c])
            px, py, pz = [self.particle_fields['particle_position_%s'%ax][group_indices]
                                            for ax in 'xyz']
            self._max_dens[i] = (dens[cp:cp_c][md_i], px[md_i], py[md_i], pz[md_i])
            cp += counts[i+1]

    def __len__(self):
        return len(self._groups)
 
    def __iter__(self):
        for i in self._groups: yield i

    def __getitem__(self, key):
        return self._groups[key]

    def nearest_neighbors_3D(self, haloID, num_neighbors=7, search_radius=.2):
        """
        for halo *haloID*, find up to *num_neighbors* nearest neighbors in 3D
        using the kd tree. Search over *search_radius* in code units.
        Returns a list of the neighbors distances and ID with format
        [distance,haloID].
        """
        period = self.pf['DomainRightEdge'] - self.pf['DomainLeftEdge']
        # Initialize the dataset of points from all the haloes
        dataset = []
        for group in self:
            p = Point()
            p.data = group.center_of_mass().tolist()
            p.haloID = group.id
            dataset.append(p)
        mylog.info('Building kd tree...')
        kd = buildKdHyperRectTree(dataset[:],2*num_neighbors)
        # make the neighbors object
        neighbors = Neighbors()
        neighbors.k = num_neighbors
        neighbors.points = []
        neighbors.minDistanceSquared = search_radius * search_radius
        mylog.info('Finding nearest neighbors...')
        getKNN(self[haloID].center_of_mass().tolist(), kd, neighbors,0., period.tolist())
        # convert the data in order to return something less perverse than a
        # Neighbors object, also root the distances
        n_points = []
        for n in neighbors.points:
            n_points.append([math.sqrt(n[0]),n[1].haloID])
        return n_points

    def nearest_neighbors_2D(self, haloID, num_neighbors=7, search_radius=.2,
        proj_dim=0):
        """
        for halo *haloID*, find up to *num_neighbors* nearest neighbors in 2D
        using the kd tree. Search over *search_radius* in code units.
        The halo positions are projected along dimension *proj_dim*.
        Returns a list of the neighbors distances and ID with format
        [distance,haloID].
        """
        # Set up a vector to multiply other vectors by to project along proj_dim
        vec = na.array([1.,1.,1.])
        vec[proj_dim] = 0.
        period = self.pf['DomainRightEdge'] - self.pf['DomainLeftEdge']
        period = period * vec
        # Initialize the dataset of points from all the haloes
        dataset = []
        for group in self:
            p = Point()
            cm = group.center_of_mass() * vec
            p.data = cm.tolist()
            p.haloID = group.id
            dataset.append(p)
        mylog.info('Building kd tree...')
        kd = buildKdHyperRectTree(dataset[:],2*num_neighbors)
        # make the neighbors object
        neighbors = Neighbors()
        neighbors.k = num_neighbors
        neighbors.points = []
        neighbors.minDistanceSquared = search_radius * search_radius
        mylog.info('Finding nearest neighbors...')
        cm = self[haloID].center_of_mass() * vec
        getKNN(cm.tolist(), kd, neighbors,0., period.tolist())
        # convert the data in order to return something less perverse than a
        # Neighbors object, also root the distances
        n_points = []
        for n in neighbors.points:
            n_points.append([math.sqrt(n[0]),n[1].haloID])
        return n_points

    def write_out(self, filename):
        """
        Write out standard HOP information to *filename*.
        """
        if hasattr(filename, 'write'):
            f = filename
        else:
            f = open(filename,"w")
        f.write("# HALOS FOUND WITH %s\n" % (self._name))
        f.write("\t".join(["# Group","Mass","# part","max dens"
                           "x","y","z", "center-of-mass",
                           "x","y","z",
                           "vx","vy","vz","max_r","rms_v","\n"]))
        for group in self:
            f.write("%10i\t" % group.id)
            f.write("%0.9e\t" % group.total_mass())
            f.write("%10i\t" % group.get_size())
            f.write("%0.9e\t" % group.maximum_density())
            f.write("\t".join(["%0.9e" % v for v in group.maximum_density_location()]))
            f.write("\t")
            f.write("\t".join(["%0.9e" % v for v in group.center_of_mass()]))
            f.write("\t")
            f.write("\t".join(["%0.9e" % v for v in group.bulk_velocity()]))
            f.write("\t")
            f.write("%0.9e\t" % group.maximum_radius())
            f.write("%0.9e\t" % group.rms_velocity())
            f.write("\n")
            f.flush()
        f.close()

    def write_particle_lists_txt(self, prefix, fp=None):
        """
        Write out the location of halo data in hdf5 files to *prefix*.
        """
        if hasattr(fp, 'write'):
            f = fp
        else:
            f = open("%s.txt" % prefix,"w")
        for group in self:
            if group.tasks is not None:
                fn = ""
                for task in group.tasks:
                    fn += "%s.h5 " % self._get_filename(prefix, rank=task)
            elif self._distributed:
                fn = "%s.h5" % self._get_filename(prefix, rank=group._owner)
            else:
                fn = "%s.h5" % self._get_filename(prefix)
            gn = "Halo%08i" % (group.id)
            f.write("%s %s\n" % (gn, fn))
            f.flush()
        f.close()

class HOPHaloList(HaloList):

    _name = "HOP"
    _halo_class = HOPHalo
    _fields = ["particle_position_%s" % ax for ax in 'xyz'] + \
              ["ParticleMassMsun"]

    def __init__(self, data_source, threshold=160.0, dm_only=True):
        """
        Run hop on *data_source* with a given density *threshold*.  If
        *dm_only* is set, only run it on the dark matter particles, otherwise
        on all particles.  Returns an iterable collection of *HopGroup* items.
        """
        self.threshold = threshold
        mylog.info("Initializing HOP")
        HaloList.__init__(self, data_source, dm_only)

    def _run_finder(self):
        self.densities, self.tags = \
            RunHOP(self.particle_fields["particle_position_x"],
                   self.particle_fields["particle_position_y"],
                   self.particle_fields["particle_position_z"],
                   self.particle_fields["ParticleMassMsun"],
                   self.threshold)
        self.particle_fields["densities"] = self.densities
        self.particle_fields["tags"] = self.tags

    def write_out(self, filename="HopAnalysis.out"):
        HaloList.write_out(self, filename)

class FOFHaloList(HaloList):
    _name = "FOF"
    _halo_class = FOFHalo

    def __init__(self, data_source, link=0.2, dm_only=True):
        self.link = link
        mylog.info("Initializing FOF")
        HaloList.__init__(self, data_source, dm_only)

    def _run_finder(self):
        self.tags = \
            RunFOF(self.particle_fields["particle_position_x"],
                   self.particle_fields["particle_position_y"],
                   self.particle_fields["particle_position_z"],
                   self.link)
        self.densities = na.ones(self.tags.size, dtype='float64') * -1
        self.particle_fields["densities"] = self.densities
        self.particle_fields["tags"] = self.tags

    def write_out(self, filename="FOFAnalysis.out"):
        HaloList.write_out(self, filename)

class parallelHOPHaloList(HaloList,ParallelAnalysisInterface):
    _name = "parallelHOP"
    _halo_class = parallelHOPHalo
    _fields = ["particle_position_%s" % ax for ax in 'xyz'] + \
              ["ParticleMassMsun", "particle_index"]

    def __init__(self, data_source, padding, num_neighbors, bounds, total_mass,
        period, threshold=160.0, dm_only=True, rearrange=True, premerge=True):
        """
        Run hop on *data_source* with a given density *threshold*.  If
        *dm_only* is set, only run it on the dark matter particles, otherwise
        on all particles.  Returns an iterable collection of *HopGroup* items.
        """
        self.threshold = threshold
        self.num_neighbors = num_neighbors
        self.bounds = bounds
        self.total_mass = total_mass
        self.rearrange = rearrange
        self.period = period
        self._data_source = data_source
        self.premerge = premerge
        mylog.info("Initializing HOP")
        HaloList.__init__(self, data_source, dm_only)

    def _run_finder(self):
        yt_counters("Reading Data")
        # Test to make sure the particle IDs aren't suspicious.
        exit = False
        if (self.particle_fields["particle_index"] < 0).any():
            mylog.error("Negative values in particle_index field. Parallel HOP will fail.")
            exit = True
        if na.unique(self.particle_fields["particle_index"]).size != \
                self.particle_fields["particle_index"].size:
            mylog.error("Non-unique values in particle_index field. Parallel HOP will fail.")
            exit = True
        self._mpi_exit_test(exit)
        obj = RunParallelHOP(self.period, self.padding,
            self.num_neighbors, self.bounds,
            self.particle_fields["particle_position_x"],
            self.particle_fields["particle_position_y"],
            self.particle_fields["particle_position_z"],
            self.particle_fields["particle_index"],
            self.particle_fields["ParticleMassMsun"]/self.total_mass,
            self.threshold, rearrange=self.rearrange, premerge=self.premerge)
        self.densities, self.tags = obj.density, obj.chainID
        # I'm going to go ahead and delete self.densities because it's not
        # actually being used. I'm not going to remove it altogether because
        # it may be useful to someone someday.
        del self.densities
        self.group_count = obj.group_count
        self.group_sizes = obj.group_sizes
        if self.group_count == 0:
            mylog.info("There are no halos found.")
            return
        self.CoM = obj.CoM
        self.Tot_M = obj.Tot_M * self.total_mass
        self.max_dens_point = obj.max_dens_point
        self.max_radius = obj.max_radius
        # Precompute the bulk velocity in parallel.
        yt_counters("Precomp bulk vel.")
        self.bulk_vel = na.zeros((self.group_count, 3), dtype='float64')
        yt_counters("bulk vel. reading data")
        pm = self.particle_fields["ParticleMassMsun"]
        if ytcfg.getboolean("yt","inline") == False:
            xv = self._data_source["particle_velocity_x"][self._base_indices]
            yv = self._data_source["particle_velocity_y"][self._base_indices]
            zv = self._data_source["particle_velocity_z"][self._base_indices]
        else:
            xv = self._data_source["particle_velocity_x"][self._base_indices]
            yv = self._data_source["particle_velocity_y"][self._base_indices]
            zv = self._data_source["particle_velocity_z"][self._base_indices]
        yt_counters("bulk vel. reading data")
        yt_counters("bulk vel. computing")
        select = (self.tags >= 0)
        calc = len(na.where(select == True)[0])
        if calc:
            vel = na.empty((calc, 3), dtype='float64')
            ms = pm[select]
            vel[:,0] = xv[select] * ms
            vel[:,1] = yv[select] * ms
            vel[:,2] = zv[select] * ms
            subchain = self.tags[select]
            sort = subchain.argsort()
            vel = vel[sort]
            sort_subchain = subchain[sort]
            uniq_subchain = na.unique(sort_subchain)
            diff_subchain = na.ediff1d(sort_subchain)
            marks = (diff_subchain > 0)
            marks = na.arange(calc)[marks] + 1
            marks = na.concatenate(([0], marks, [calc]))
            for i, u in enumerate(uniq_subchain):
                self.bulk_vel[u] = na.sum(vel[marks[i]:marks[i+1]], axis=0)
            del vel, subchain, sort_subchain
            del diff_subchain
        # Bring it together, and divide by the previously computed total mass
        # of each halo.
        self.bulk_vel = self._mpi_Allsum_double(self.bulk_vel)
        for groupID in xrange(self.group_count):
            self.bulk_vel[groupID] = self.bulk_vel[groupID] / self.Tot_M[groupID]
        yt_counters("bulk vel. computing")
        # Now calculate the RMS velocity of the groups in parallel, very
        # similarly to the bulk velocity and re-using some of the arrays.
        yt_counters("rms vel computing")
        rms_vel_temp = na.zeros((self.group_count,2), dtype='float64')
        if calc:
            vel = na.empty((calc, 3), dtype='float64')
            vel[:,0] = xv[select] * ms
            vel[:,1] = yv[select] * ms
            vel[:,2] = zv[select] * ms
            vel = vel[sort]
            for i, u in enumerate(uniq_subchain):
                # This finds the sum locally.
                rms_vel_temp[u][0] = na.sum(((vel[marks[i]:marks[i+1]] - \
                    self.bulk_vel[u]) / self.Tot_M[u])**2.)
                # I could use self.group_sizes...
                rms_vel_temp[u][1] = marks[i+1] - marks[i]
            del vel, marks, uniq_subchain
        # Bring it together.
        rms_vel_temp = self._mpi_Allsum_double(rms_vel_temp)
        self.rms_vel = na.empty(self.group_count, dtype='float64')
        for groupID in xrange(self.group_count):
            # Here we do the Mean and the Root.
            self.rms_vel[groupID] = \
                na.sqrt(rms_vel_temp[groupID][0] / rms_vel_temp[groupID][1]) * \
                self.group_sizes[groupID]
        del rms_vel_temp
        yt_counters("rms vel computing")
        self.taskID = obj.mine
        self.halo_taskmap = obj.halo_taskmap # A defaultdict.
        del obj
        yt_counters("Precomp bulk vel.")

    def _parse_output(self):
        yt_counters("Final Grouping")
        """
        Each task will make an entry for all groups, but it may be empty.
        """
        unique_ids = na.unique(self.tags)
        counts = na.bincount((self.tags+1).tolist())
        sort_indices = na.argsort(self.tags)
        grab_indices = na.indices(self.tags.shape).ravel()[sort_indices]
        del sort_indices
        cp = 0
        index = 0
        # We want arrays for parallel HOP
        self._groups = na.empty(self.group_count, dtype='object')
        self._max_dens = na.empty((self.group_count, 4), dtype='float64')
        if self.group_count == 0:
            mylog.info("There are no halos found.")
            return
        for i in unique_ids:
            if i == -1:
                cp += counts[i+1]
                continue
            # If there is a gap in the unique_ids, make empty groups to 
            # fill it in.
            while index < i:
                self._groups[index] = self._halo_class(self, index, \
                    size=self.group_sizes[index], CoM=self.CoM[index], \
                    max_dens_point=self.max_dens_point[index], \
                    group_total_mass=self.Tot_M[index], max_radius=self.max_radius[index],
                    bulk_vel=self.bulk_vel[index], tasks=self.halo_taskmap[index],
                    rms_vel=self.rms_vel[index])
                # I don't own this halo
                self._do_not_claim_object(self._groups[index])
                self._max_dens[index] = [self.max_dens_point[index][0], self.max_dens_point[index][1], \
                    self.max_dens_point[index][2], self.max_dens_point[index][3]]
                index += 1
            cp_c = cp + counts[i+1]
            group_indices = grab_indices[cp:cp_c]
            self._groups[index] = self._halo_class(self, i, group_indices, \
                size=self.group_sizes[i], CoM=self.CoM[i], \
                max_dens_point=self.max_dens_point[i], \
                group_total_mass=self.Tot_M[i], max_radius=self.max_radius[i],
                bulk_vel=self.bulk_vel[i], tasks=self.halo_taskmap[index],
                rms_vel=self.rms_vel[i])
            # This halo may be owned by many, including this task
            self._claim_object(self._groups[index])
            self._max_dens[index] = [self.max_dens_point[i][0], self.max_dens_point[i][1], \
                self.max_dens_point[i][2], self.max_dens_point[i][3]]
            cp += counts[i+1]
            index += 1
        # If there are missing groups at the end, add them.
        while index < self.group_count:
            self._groups[index] = self._halo_class(self, index, \
                size=self.group_sizes[index], CoM=self.CoM[index], \
                max_dens_point=self.max_dens_point[i], \
                group_total_mass=self.Tot_M[index], max_radius=self.max_radius[index],
                bulk_vel=self.bulk_vel[index], tasks=self.halo_taskmap[index],
                rms_vel=self.rms_vel[index])
            self._do_not_claim_object(self._groups[index])
            self._max_dens[index] = [self.max_dens_point[index][0], self.max_dens_point[index][1], \
                self.max_dens_point[index][2], self.max_dens_point[index][3]]
            index += 1
        # Clean up
        del self.max_dens_point, self.max_radius, self.bulk_vel
        del self.halo_taskmap, self.tags, self.rms_vel
        del grab_indices, unique_ids, counts
        try:
            del group_indices
        except UnboundLocalError:
            pass

    def __len__(self):
        return self.group_count

    def write_out(self, filename="parallelHopAnalysis.out"):
        HaloList.write_out(self, filename)

class GenericHaloFinder(HaloList, ParallelAnalysisInterface):
    def __init__(self, pf, dm_only=True, padding=0.0):
        self.pf = pf
        self.hierarchy = pf.h
        self.center = (pf["DomainRightEdge"] + pf["DomainLeftEdge"])/2.0

    def _parse_halolist(self, threshold_adjustment):
        groups, max_dens, hi  = [], {}, 0
        LE, RE = self.bounds
        for halo in self._groups:
            this_max_dens = halo.maximum_density_location()
            # if the most dense particle is in the box, keep it
            if na.all((this_max_dens >= LE) & (this_max_dens <= RE)):
                # Now we add the halo information to OURSELVES, taken from the
                # self.hop_list
                # We need to mock up the HOPHaloList thingie, so we need to set:
                #     self._max_dens
                max_dens_temp = list(self._max_dens[halo.id])[0] / threshold_adjustment
                max_dens[hi] = [max_dens_temp] + list(self._max_dens[halo.id])[1:4]
                groups.append(self._halo_class(self, hi))
                groups[-1].indices = halo.indices
                self._claim_object(groups[-1])
                hi += 1
        del self._groups, self._max_dens # explicit >> implicit
        self._groups = groups
        self._max_dens = max_dens

    def _join_halolists(self):
        # First we get the total number of halos the entire collection
        # has identified
        # Note I have added a new method here to help us get information
        # about processors and ownership and so forth.
        # _mpi_info_dict returns a dict of {proc: whatever} where whatever is
        # what is fed in on each proc.
        mine, halo_info = self._mpi_info_dict(len(self))
        nhalos = sum(halo_info.values())
        # Figure out our offset
        my_first_id = sum([v for k,v in halo_info.items() if k < mine])
        # Fix our max_dens
        max_dens = {}
        for i,m in self._max_dens.items(): max_dens[i+my_first_id] = m
        self._max_dens = max_dens
        # sort the list by the size of the groups
        # Now we add ghost halos and reassign all the IDs
        # Note: we already know which halos we own!
        after = my_first_id + len(self._groups)
        # One single fake halo, not owned, does the trick
        self._groups = [self._halo_class(self, i) for i in range(my_first_id)] + \
                       self._groups + \
                       [self._halo_class(self, i) for i in range(after, nhalos)]
        id = 0
        for proc in sorted(halo_info.keys()):
            for halo in self._groups[id:id+halo_info[proc]]:
                halo.id = id
                halo._distributed = self._distributed
                halo._owner = proc
                id += 1
        def haloCmp(h1,h2):
            c = cmp(h1.total_mass(),h2.total_mass())
            if c != 0:
                return -1 * c
            if c == 0:
                return cmp(h1.center_of_mass()[0],h2.center_of_mass()[0])
        self._groups.sort(haloCmp)
        sorted_max_dens = {}
        for i, halo in enumerate(self._groups):
            if halo.id in self._max_dens:
                sorted_max_dens[i] = self._max_dens[halo.id]
            halo.id = i
        self._max_dens = sorted_max_dens
        for i, halo in enumerate(self._groups):
            halo._max_dens = self._max_dens
        
    def _reposition_particles(self, bounds):
        # This only does periodicity.  We do NOT want to deal with anything
        # else.  The only reason we even do periodicity is the 
        LE, RE = bounds
        dw = self.pf["DomainRightEdge"] - self.pf["DomainLeftEdge"]
        for i, ax in enumerate('xyz'):
            arr = self._data_source["particle_position_%s" % ax]
            arr[arr < LE[i]-self.padding] += dw[i]
            arr[arr > RE[i]+self.padding] -= dw[i]

    def write_out(self, filename):
        #self._data_source.get_data(["particle_velocity_%s" % ax for ax in 'xyz'])
        f = self._write_on_root(filename)
        HaloList.write_out(self, f)

    def write_particle_lists_txt(self, prefix):
        f = self._write_on_root("%s.txt" % prefix)
        HaloList.write_particle_lists_txt(self, prefix, fp=f)

    @parallel_blocking_call
    def write_particle_lists(self, prefix):
        fn = "%s.h5" % self._get_filename(prefix)
        f = h5py.File(fn, "w")
        for halo in self._groups:
            if not self._is_mine(halo): continue
            halo.write_particle_list(f)

class parallelHF(GenericHaloFinder, parallelHOPHaloList):
    def __init__(self, pf, threshold=160, dm_only=True, resize=True, rearrange=True,\
        fancy_padding=True, safety=1.5, premerge=True, sample=0.03):
        GenericHaloFinder.__init__(self, pf, dm_only, padding=0.0)
        self.padding = 0.0
        self.num_neighbors = 65
        self.safety = safety
        self.sample = sample
        period = pf["DomainRightEdge"] - pf["DomainLeftEdge"]
        topbounds = na.array([[0., 0., 0.], period])
        # Cut up the volume evenly initially, with no padding.
        padded, LE, RE, self._data_source = self._partition_hierarchy_3d(padding=self.padding)
        # also get the total mass of particles
        yt_counters("Reading Data")
        # Adaptive subregions by bisection.
        ds_names = ["particle_position_x","particle_position_y","particle_position_z"]
        if ytcfg.getboolean("yt","inline") == False and \
           resize and self._mpi_get_size() != 1:
            cut_list = self._partition_hierarchy_3d_bisection_list()
            root_points = self._subsample_points()
            self.bucket_bounds = []
            if self._mpi_get_rank() == 0:
                self._recursive_divide(root_points, topbounds, 0, cut_list)
            self.bucket_bounds = self._mpi_bcast_pickled(self.bucket_bounds)
            my_bounds = self.bucket_bounds[self._mpi_get_rank()]
            LE, RE = my_bounds[0], my_bounds[1]
            self._data_source = self.hierarchy.region_strict([0.]*3, LE, RE)
        # If this isn't parallel, define the region as an AMRRegionStrict so
        # particle IO works.
        if self._mpi_get_size() == 1:
            self._data_source = self.hierarchy.periodic_region_strict([0.5]*3, LE, RE)
        # get the average spacing between particles for this region
        # The except is for the serial case, where the full box is what we want.
        if ytcfg.getboolean("yt","inline") == False:
            data = self._data_source["particle_position_x"]
        else:
            data = self._data_source["particle_position_x"]
        try:
            l = self._data_source.right_edge - self._data_source.left_edge
        except AttributeError:
            l = pf["DomainRightEdge"] - pf["DomainLeftEdge"]
        vol = l[0] * l[1] * l[2]
        full_vol = vol
        if not fancy_padding:
            avg_spacing = (float(vol) / data.size)**(1./3.)
            # padding is a function of inter-particle spacing, this is an
            # approximation, but it's OK with the safety factor
            padding = (self.num_neighbors)**(1./3.) * self.safety * avg_spacing
            self.padding = (na.ones(3,dtype='float64')*padding, na.ones(3,dtype='float64')*padding)
            mylog.info('padding %s avg_spacing %f vol %f local_parts %d' % \
                (str(self.padding), avg_spacing, vol, data.size))
        # Another approach to padding, perhaps more accurate.
        elif fancy_padding and self._distributed:
            LE_padding, RE_padding = na.empty(3,dtype='float64'), na.empty(3,dtype='float64')
            for dim in xrange(3):
                if ytcfg.getboolean("yt","inline") == False:
                    data = self._data_source[ds_names[dim]]
                else:
                    data = self._data_source[ds_names[dim]]
                num_bins = 1000
                width = self._data_source.right_edge[dim] - self._data_source.left_edge[dim]
                area = (self._data_source.right_edge[(dim+1)%3] - self._data_source.left_edge[(dim+1)%3]) * \
                    (self._data_source.right_edge[(dim+2)%3] - self._data_source.left_edge[(dim+2)%3])
                bin_width = float(width)/float(num_bins)
                bins = na.arange(num_bins+1, dtype='float64') * bin_width + self._data_source.left_edge[dim]
                counts, bins = na.histogram(data, bins, new=True)
                # left side.
                start = 0
                count = counts[0]
                while count < self.num_neighbors:
                    start += 1
                    count += counts[start]
                # Get the avg spacing in just this boundary.
                vol = area * (bins[start+1] - bins[0])
                avg_spacing = (float(vol) / count)**(1./3.)
                LE_padding[dim] = (self.num_neighbors)**(1./3.) * self.safety * avg_spacing
                # right side.
                start = -1
                count = counts[-1]
                while count < self.num_neighbors:
                    start -= 1
                    count += counts[start]
                vol = area * (bins[-1] - bins[start-1])
                avg_spacing = (float(vol) / count)**(1./3.)
                RE_padding[dim] = (self.num_neighbors)**(1./3.) * self.safety * avg_spacing
            self.padding = (LE_padding, RE_padding)
            del bins, counts
            mylog.info('fancy_padding %s avg_spacing %f full_vol %f local_parts %d %s' % \
                (str(self.padding), avg_spacing, full_vol, data.size, str(self._data_source)))
        # Now we get the full box mass after we have the final composition of
        # subvolumes.
        if ytcfg.getboolean("yt","inline") == False:
            total_mass = self._mpi_allsum((self._data_source["ParticleMassMsun"].astype('float64')).sum())
        else:
            total_mass = self._mpi_allsum((self._data_source["ParticleMassMsun"].astype('float64')).sum())
        if not self._distributed:
            self.padding = (na.zeros(3,dtype='float64'), na.zeros(3,dtype='float64'))
        self.bounds = (LE, RE)
        (LE_padding, RE_padding) = self.padding
        parallelHOPHaloList.__init__(self, self._data_source, self.padding, \
        self.num_neighbors, self.bounds, total_mass, period, \
        threshold=threshold, dm_only=dm_only, rearrange=rearrange, premerge=premerge)
        self._join_halolists()
        yt_counters("Final Grouping")

    def _subsample_points(self):
        # Read in a random subset of the points in each domain, and then
        # collect them on the root task.
        xp = self._data_source["particle_position_x"]
        yp = self._data_source["particle_position_y"]
        zp = self._data_source["particle_position_z"]
        n_parts = self._mpi_allsum(xp.size)
        local_parts = xp.size
        random_points = int(self.sample * n_parts)
        # We want to get a representative selection of random particles in
        # each subvolume.
        adjust = float(local_parts) / ( float(n_parts) / self._mpi_get_size())
        n_random = int(adjust * float(random_points) / self._mpi_get_size())
        mylog.info("Reading in %d random particles." % n_random)
        # Get unique random particles.
        my_points = na.empty((n_random, 3), dtype='float64')
        uni = na.array(random.sample(xrange(xp.size), n_random))
        uni = uni[uni.argsort()]
        my_points[:,0] = xp[uni]
        my_points[:,1] = yp[uni]
        my_points[:,2] = zp[uni]
        # Collect them on the root task.
        mine, sizes = self._mpi_info_dict(n_random)
        if mine == 0:
            tot_random = sum(sizes.values())
            root_points = na.empty((tot_random, 3), dtype='float64')
            root_points.shape = (1, 3*tot_random)
        else:
            root_points = na.empty([])
        my_points.shape = (1, n_random*3)
        root_points = self._mpi_concatenate_array_on_root_double(my_points[0])
        if mine == 0:
            root_points.shape = (tot_random, 3)
        return root_points

    def _recursive_divide(self, points, bounds, level, cut_list):
        dim = cut_list[level][0]
        parts = points.shape[0]
        num_bins = 1000
        width = bounds[1][dim] - bounds[0][dim]
        bin_width = width / num_bins
        bins = na.arange(num_bins+1, dtype='float64') * bin_width + bounds[0][dim]
        counts, bins = na.histogram(points[:,dim], bins)
        # Find the bin that passes the cut points.
        midpoints = [bounds[0][dim]]
        sum = 0
        bin = 0
        for step in xrange(1,cut_list[level][1]):
            while sum < ((parts*step)/cut_list[level][1]):
                lastsum = sum
                sum += counts[bin]
                bin += 1
            # Bin edges
            left_edge = bins[bin-1]
            right_edge = bins[bin]
            # Find a better approx of the midpoint cut line using a linear approx.
            a = float(sum - lastsum) / (right_edge - left_edge)
            midpoints.append(left_edge + (0.5 - (float(lastsum) / parts / 2)) / a)
        midpoints.append(bounds[1][dim])

        # Split the points & update the bounds.
        subpoints = []
        subbounds = []
        for pair in zip(midpoints[:-1],midpoints[1:]):
            select = na.bitwise_and(points[:,dim] >= pair[0],
                points[:,dim] < pair[1])
            subpoints.append(points[select])
            nb = bounds.copy()
            nb[0][dim] = pair[0]
            nb[1][dim] = pair[1]
            subbounds.append(nb)
        # If we're at the maxlevel, make a bucket. Otherwise, recurse down.
        maxlevel = len(cut_list) - 1
        for pair in zip(subpoints, subbounds):
            if level == maxlevel:
                self.bucket_bounds.append(pair[1])
            else:
                self._recursive_divide(pair[0], pair[1], level+1, cut_list)

    def _join_halolists(self):
        if self.group_count == 0:
            mylog.info("There are no halos found.")
            return
        ms = -self.Tot_M.copy()
        del self.Tot_M
        Cx = self.CoM[:,0].copy()
        sorted = na.lexsort([Cx, ms])
        del Cx, ms
        self._groups = self._groups[sorted]
        self._max_dens = self._max_dens[sorted]
        for i in xrange(self.group_count):
            self._groups[i].id = i
        del sorted, self.group_sizes, self.CoM


class HOPHaloFinder(GenericHaloFinder, HOPHaloList):
    def __init__(self, pf, threshold=160, dm_only=True, padding=0.02):
        GenericHaloFinder.__init__(self, pf, dm_only, padding)
        
        # do it once with no padding so the total_mass is correct (no duplicated particles)
        self.padding = 0.0
        padded, LE, RE, self._data_source = self._partition_hierarchy_3d(padding=self.padding)
        # For scaling the threshold, note that it's a passthrough
        if dm_only:
            select = self._get_dm_indices()
            total_mass = self._mpi_allsum((self._data_source["ParticleMassMsun"][select]).sum())
        else:
            total_mass = self._mpi_allsum(self._data_source["ParticleMassMsun"].sum())
        # MJT: Note that instead of this, if we are assuming that the particles
        # are all on different processors, we should instead construct an
        # object representing the entire domain and sum it "lazily" with
        # Derived Quantities.
        self.padding = padding #* pf["unitary"] # This should be clevererer
        padded, LE, RE, self._data_source = self._partition_hierarchy_3d(padding=self.padding)
        self.bounds = (LE, RE)
        # reflect particles around the periodic boundary
        #self._reposition_particles((LE, RE))
        if dm_only:
            select = self._get_dm_indices()
            sub_mass = self._data_source["ParticleMassMsun"][select].sum()
        else:
            sub_mass = self._data_source["ParticleMassMsun"].sum()
        HOPHaloList.__init__(self, self._data_source, threshold*total_mass/sub_mass, dm_only)
        self._parse_halolist(total_mass/sub_mass)
        self._join_halolists()

class FOFHaloFinder(GenericHaloFinder, FOFHaloList):
    def __init__(self, pf, link=0.2, dm_only=True, padding=0.02):
        self.pf = pf
        self.hierarchy = pf.h
        self.center = (pf["DomainRightEdge"] + pf["DomainLeftEdge"])/2.0
        self.padding = 0.0 #* pf["unitary"] # This should be clevererer
        # get the total number of particles across all procs, with no padding
        padded, LE, RE, self._data_source = self._partition_hierarchy_3d(padding=self.padding)
        n_parts = self._mpi_allsum(self._data_source["particle_position_x"].size)
        # get the average spacing between particles
        l = pf["DomainRightEdge"] - pf["DomainLeftEdge"]
        vol = l[0] * l[1] * l[2]
        avg_spacing = (float(vol) / n_parts)**(1./3.)
        self.padding = padding
        padded, LE, RE, self._data_source = self._partition_hierarchy_3d(padding=self.padding)
        self.bounds = (LE, RE)
        # reflect particles around the periodic boundary
        #self._reposition_particles((LE, RE))
        # here is where the FOF halo finder is run
        FOFHaloList.__init__(self, self._data_source, link * avg_spacing, dm_only)
        self._parse_halolist(1.)
        self._join_halolists()

HaloFinder = HOPHaloFinder
