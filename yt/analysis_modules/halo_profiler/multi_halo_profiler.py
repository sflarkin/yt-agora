"""
HaloProfiler class and member functions.

Author: Britton Smith <brittons@origins.colorado.edu>
Affiliation: CASA/University of CO, Boulder
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2008-2009 Britton Smith.  All Rights Reserved.

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
import os
import h5py
import types

from yt.funcs import *
from yt.utilities.math_utils import periodic_dist

from yt.convenience import \
    load
from yt.data_objects.profiles import \
    BinnedProfile1D, EmptyProfileData
from yt.analysis_modules.halo_finding.api import *
from .halo_filters import \
    VirialFilter
from .centering_methods import \
    centering_registry
from yt.data_objects.field_info_container import \
    add_field

from yt.utilities.parallel_tools.parallel_analysis_interface import \
    ParallelAnalysisInterface, \
    parallel_blocking_call, \
    parallel_root_only
from yt.visualization.fixed_resolution import \
    FixedResolutionBuffer
from yt.visualization.image_writer import write_image

PROFILE_RADIUS_THRESHOLD = 2

class HaloProfiler(ParallelAnalysisInterface):
    "Radial profiling, filtering, and projections for halos in cosmological simulations."
    def __init__(self, dataset, output_dir=None,
                 halos='multiple', halo_list_file='HopAnalysis.out', 
                 halo_list_format='yt_hop', halo_finder_function=parallelHF, 
                 halo_finder_args=None, 
                 halo_finder_kwargs=dict(threshold=160.0, safety=1.5, 
                                         dm_only=False, resize=True, 
                                         fancy_padding=True, rearrange=True),
                 use_density_center=False, density_center_exponent=1.0,
                 halo_radius=0.1, radius_units='1', n_profile_bins=50,
                 recenter = None,
                 profile_output_dir='radial_profiles', projection_output_dir='projections',
                 projection_width=8.0, projection_width_units='mpc', project_at_level='max',
                 velocity_center=['bulk', 'halo'], filter_quantities=['id','center']):
        """
        Initialize a HaloProfiler object.
        :param output_dir (str): if specified, all output will be put into this path instead of 
               in the dataset directories.  Default: None.
        :param halos (str): "multiple" for profiling more than one halo.  In this mode halos are read in 
               from a list or identified with a halo finder.  In "single" mode, the one and only halo 
               center is identified automatically as the location of the peak in the density field.  
               Default: "multiple".
        :param halo_list_file (str): name of file containing the list of halos.  The HaloProfiler will 
               look for this file in the data directory.  Default: "HopAnalysis.out".
        :param halo_list_format (str or dict): the format of the halo list file.  "yt_hop" for the format 
               given by yt's halo finders.  "enzo_hop" for the format written by enzo_hop.  "p-groupfinder" 
               for P-Groupfinder.  This keyword 
               can also be given in the form of a dictionary specifying the column in which various 
               properties can be found.  For example, {"id": 0, "center": [1, 2, 3], "mass": 4, "radius": 5}.  
               Default: "yt_hop".
        :param halo_finder_function (function): If halos is set to multiple and the file given by 
               halo_list_file does not exit, the halo finding function specified here will be called.  
               Default: HaloFinder (yt_hop).
        :param halo_finder_args (tuple): args given with call to halo finder function.  Default: None.
        :param halo_finder_kwargs (dict): kwargs given with call to halo finder function. Default: None.
        :param recenter (str or function name): The name of a function that
               recenters the halo.
        :param halo_radius (float): if no halo radii are provided in the halo list file, this parameter is 
               used to specify the radius out to which radial profiles will be made.  This keyword is also 
               used when halos is set to single.  Default: 0.1.
        :param radius_units (str): the units of halo_radius.  Default: "1" (code units).
        :param n_profile_bins (int): the number of bins in the radial profiles.  Default: 50.
        :param profile_output_dir (str): the subdirectory, inside the data directory, in which radial profile 
               output files will be created.  The directory will be created if it does not exist.  
               Default: "radial_profiles".
        :param projection_output_dir (str): the subdirectory, inside the data directory, in which projection 
               output files will be created.  The directory will be created if it does not exist.  
               Default: "projections".
        :param projection_width (float): the width of halo projections.  Default: 8.0.
        :param projection_width_units (str): the units of projection_width. Default: "mpc".
        :param project_at_level (int or "max"): the maximum refinement level to be included in projections.  
               Default: "max" (maximum level within the dataset).
        :param velocity_center (list): the method in which the halo bulk velocity is calculated (used for 
               calculation of radial and tangential velocities.  Valid options are:
     	          - ["bulk", "halo"] (Default): the velocity provided in the halo list
                  - ["bulk", "sphere"]: the bulk velocity of the sphere centered on the halo center.
    	          - ["max", field]: the velocity of the cell that is the location of the maximum of the field 
                                    specified (used only when halos set to single).
        :param filter_quantities (list): quantities from the original halo list file to be written out in the 
               filtered list file.  Default: ['id','center'].
        """

        self.dataset = dataset
        self.output_dir = output_dir
        self.profile_output_dir = profile_output_dir
        self.projection_output_dir = projection_output_dir
        self.n_profile_bins = n_profile_bins
        self.projection_width = projection_width
        self.projection_width_units = projection_width_units
        self.project_at_level = project_at_level
        self.filter_quantities = filter_quantities
        if self.filter_quantities is None: self.filter_quantities = []

        self.profile_fields = []
        self.projection_fields = []

        self._halo_filters = []
        self.all_halos = []
        self.filtered_halos = []
        self._projection_halo_list = []

        # Create output directory if specified
        if self.output_dir is not None:
            self.__check_directory(self.output_dir)

        # Set halo finder function and parameters, if needed.
        self.halo_finder_function = halo_finder_function
        self.halo_finder_args = halo_finder_args
        if self.halo_finder_args is None: self.halo_finder_args = ()
        self.halo_finder_kwargs = halo_finder_kwargs
        if self.halo_finder_kwargs is None: self.halo_finder_kwargs = {}

        # Set option to get halos from hop or single halo at density maximum.
        # multiple: get halos from hop
        # single: get single halo from density maximum
        self.halos = halos
        if not(self.halos is 'multiple' or self.halos is 'single'):
            mylog.error("Keyword, halos, must be either 'single' or 'multiple'.")
            return None

        # Set halo list format.
        # 'yt_hop': yt hop output.
        # 'enzo_hop': enzo_hop output.
        # dictionary: a dictionary containing fields and their corresponding columns.
        self.halo_list_file = halo_list_file
        if halo_list_format == 'yt_hop':
            self.halo_list_format = {'id':0, 'mass':1, 'np': 2, 
                                     'center':[7, 8, 9], 'velocity':[10, 11, 12], 'r_max':13}
        elif halo_list_format == 'enzo_hop':
            self.halo_list_format = {'id':0, 'center':[4, 5, 6]}
        elif halo_list_format == 'p-groupfinder':
            self.halo_list_format = {'id':3, 'mass':5, 'center':[0, 1, 2], 'r200kpc':8}
        elif isinstance(halo_list_format, types.DictType):
            self.halo_list_format = halo_list_format
        else:
            mylog.error("Keyword, halo_list_format, must be 'yt_hop', 'enzo_hop', 'p-groupfinder', or a dictionary of custom settings.")
            return None

        # Option to recenter sphere someplace else.
        self.recenter = recenter

        # Look for any field that might need to have the bulk velocity set.
        self._need_bulk_velocity = False
        for field in [hp['field'] for hp in self.profile_fields]:
            if 'Velocity' in field or 'Mach' in field:
                self._need_bulk_velocity = True
                break

        # Check validity for VelocityCenter parameter which toggles how the 
        # velocity is zeroed out for radial velocity profiles.
        self.velocity_center = velocity_center[:]
        if self.velocity_center[0] == 'bulk':
            if self.velocity_center[1] == 'halo' and \
                    self.halos is 'single':
                mylog.error("Parameter, VelocityCenter, must be set to 'bulk sphere' or 'max <field>' with halos flag set to 'single'.")
                return None
            if self.velocity_center[1] == 'halo' and \
                    self.halo_list_format is 'enzo_hop':
                mylog.error("Parameter, VelocityCenter, must be 'bulk sphere' for old style hop output files.")
                return None
            if not(self.velocity_center[1] == 'halo' or 
                   self.velocity_center[1] == 'sphere'):
                mylog.error("Second value of VelocityCenter must be either 'halo' or 'sphere' if first value is 'bulk'.")
                return None
        elif self.velocity_center[0] == 'max':
            if self.halos is 'multiple':
                mylog.error("Getting velocity center from a max field value only works with halos='single'.")
                return None
        else:
            mylog.error("First value of parameter, VelocityCenter, must be either 'bulk' or 'max'.")
            return None

        # Create dataset object.
        self.pf = load(self.dataset)
        self.pf.h
        if self.halos is 'single' or not 'r_max' in self.halo_list_format:
            self.halo_radius = halo_radius / self.pf[radius_units]

        # Get halo(s).
        if self.halos is 'single':
            v, center = self.pf.h.find_max('Density')
            singleHalo = {}
            singleHalo['center'] = center
            singleHalo['r_max'] = self.halo_radius * self.pf.units['mpc']
            singleHalo['id'] = 0
            self.all_halos.append(singleHalo)
        elif self.halos is 'multiple':
            # Get hop data.
            self._load_halo_data()
            if len(self.all_halos) == 0:
                mylog.error("No halos loaded, there will be nothing to do.")
                return None
        else:
            mylog.error("I don't know whether to get halos from hop or from density maximum.  This should not have happened.")
            return None

    def add_halo_filter(self, function, *args, **kwargs):
        "Add a halo filter to the filter list."

        self._halo_filters.append({'function':function, 'args':args, 'kwargs':kwargs})

    def add_profile(self, field, weight_field=None, accumulation=False):
        "Add a field for profiling."

        self.profile_fields.append({'field':field, 'weight_field':weight_field, 'accumulation':accumulation})

    def add_projection(self, field, weight_field=None, cmap='algae'):
        "Add a field for projection."

        self.projection_fields.append({'field':field, 'weight_field':weight_field, 
                                       'cmap': cmap})

    @parallel_blocking_call
    def make_profiles(self, filename=None, prefilters=None, **kwargs):
        "Make radial profiles for all halos on the list."

        if len(self.all_halos) == 0:
            mylog.error("Halo list is empty, returning.")
            return None

        # Reset filtered halo list.
        self.filtered_halos = []

        # Check to see if the VirialFilter has been added to the filter list.
        # If a lower mass cutoff is being used, use it to make a pre-filter.
        if prefilters is None: prefilters = []
        virial_prefilter = None
        virial_prefilter_safety_factor = 0.5
        all_filter_functions = [hf['function'] for hf in self._halo_filters]
        virial_filter = VirialFilter in all_filter_functions
        if 'mass' in self.halo_list_format and VirialFilter in all_filter_functions:
            vFilter = self._halo_filters[all_filter_functions.index(VirialFilter)]
            if vFilter['kwargs'].has_key('virial_filters') and \
               vFilter['kwargs']['virial_filters'] is not None:
                all_vqFilters = [vqf[0] for vqf in vFilter['kwargs']['virial_filters']]
                if 'TotalMassMsun' in all_vqFilters:
                    mass_filter = vFilter['kwargs']['virial_filters'][all_vqFilters.index('TotalMassMsun')]
                    if '>' in mass_filter[1]:
                        virial_prefilter = "halo['mass'] %s %f * %s" % \
                            (mass_filter[1], virial_prefilter_safety_factor, mass_filter[2])
                        prefilters.append(virial_prefilter)
                    elif '<' in mass_filter[1]:
                        virial_prefilter = "halo['mass'] %s %f * %s" % \
                            (mass_filter[1], (1./virial_prefilter_safety_factor), mass_filter[2])
                        prefilters.append(virial_prefilter)

        # Add profile fields necessary for calculating virial quantities.
        if virial_filter: self._check_for_needed_profile_fields()

        # Create output directory.
        if self.output_dir is not None:
            self.__check_directory("%s/%s" % (self.output_dir, self.pf.directory))
            my_output_dir = "%s/%s/%s" % (self.output_dir, self.pf.directory, 
                                          self.profile_output_dir)
        else:
            my_output_dir = "%s/%s" % (self.pf.fullpath, self.profile_output_dir)
        self.__check_directory(my_output_dir)

        # Profile all halos.
        for halo in self._get_objs('all_halos', round_robin=True):

            # Apply prefilters to avoid profiling unwanted halos.
            filter_result = True
            haloQuantities = {}
            if prefilters is not None:
                for prefilter in prefilters:
                    if not eval(prefilter):
                        filter_result = False
                        break

            if filter_result and len(self.profile_fields) > 0:

                profile_filename = "%s/Halo_%04d_profile.dat" % (my_output_dir, halo['id'])

                profiledHalo = self._get_halo_profile(halo, profile_filename, virial_filter=virial_filter)

                if profiledHalo is None:
                    continue

                # Apply filter and keep track of the quantities that are returned.
                for hFilter in self._halo_filters:
                    filter_result, filterQuantities = hFilter['function'](profiledHalo, *hFilter['args'], 
                                                                          **hFilter['kwargs'])

                    if not filter_result: break

                    if filterQuantities is not None:
                        haloQuantities.update(filterQuantities)

            if filter_result:
                for quantity in self.filter_quantities:
                    if halo.has_key(quantity): haloQuantities[quantity] = halo[quantity]

                self.filtered_halos.append(haloQuantities)

        self.filtered_halos = self._mpi_catlist(self.filtered_halos)
        self.filtered_halos.sort(key = lambda a:a['id'])

        if filename is not None:
            self._write_filtered_halo_list(filename, **kwargs)

    def _get_halo_profile(self, halo, filename, virial_filter=True, force_write=False):
        """
        Profile a single halo and write profile data to a file.
        If file already exists, read profile data from file.
        Return a dictionary of id, center, and virial quantities if virial_filter is True.
        """

        # Read profile from file if it already exists.
        # If not, profile will be None.
        profile = self._read_profile(filename)

        # Make profile if necessary.
        newProfile = profile is None
        if newProfile:

            r_min = 2 * self.pf.h.get_smallest_dx() * self.pf['mpc']
            if (halo['r_max'] / r_min < PROFILE_RADIUS_THRESHOLD):
                mylog.error("Skipping halo with r_max / r_min = %f." % (halo['r_max']/r_min))
                return None

            sphere = self.pf.h.sphere(halo['center'], halo['r_max']/self.pf.units['mpc'])
            if len(sphere._grids) == 0: return None
            new_sphere = False

            if self.recenter:
                old = halo['center']
                if self.recenter in centering_registry:
                    new_x, new_y, new_z = \
                        centering_registry[self.recenter](sphere)
                else:
                    # user supplied function
                    new_x, new_y, new_z = self.recenter(sphere)
                halo['center'] = [new_x, new_y, new_z]
                d = periodic_dist(old, halo['center'],
                    self.pf.domain_right_edge - self.pf.domain_left_edge) * \
                    self.pf['kpc']
                mylog.info("Recentered %1.3e kpc away." % d)
                new_sphere = True

            if new_sphere:
                # Temporary solution to memory leak.
                for g in self.pf.h.grids:
                    g.clear_data()
                sphere.clear_data()
                del sphere
                sphere = self.pf.h.sphere(halo['center'], halo['r_max']/self.pf.units['mpc'])

            if self._need_bulk_velocity:
                # Set bulk velocity to zero out radial velocity profiles.
                if self.velocity_center[0] == 'bulk':
                    if self.velocity_center[1] == 'halo':
                        sphere.set_field_parameter('bulk_velocity', halo['velocity'])
                    elif self.velocity_center[1] == 'sphere':
                        sphere.set_field_parameter('bulk_velocity', sphere.quantities['BulkVelocity']())
                    else:
                        mylog.error("Invalid parameter: VelocityCenter.")
                elif self.velocity_center[0] == 'max':
                    max_grid, max_cell, max_value, max_location = \
                        self.pf.h.find_max_cell_location(self.velocity_center[1])
                    sphere.set_field_parameter('bulk_velocity', [max_grid['x-velocity'][max_cell],
                                                                 max_grid['y-velocity'][max_cell],
                                                                 max_grid['z-velocity'][max_cell]])

            try:
                profile = BinnedProfile1D(sphere, self.n_profile_bins, "RadiusMpc",
                                                r_min, halo['r_max'],
                                                log_space=True, lazy_reader=False)
            except EmptyProfileData:
                mylog.error("Caught EmptyProfileData exception, returning None for this halo.")
                return None
            for hp in self.profile_fields:
                profile.add_fields(hp['field'], weight=hp['weight_field'], accumulation=hp['accumulation'])

        if virial_filter:
            self._add_actual_overdensity(profile)

        if newProfile:
            mylog.info("Writing halo %d" % halo['id'])
            profile.write_out(filename, format='%0.6e')
        elif force_write:
            mylog.info("Re-writing halo %d" % halo['id'])
            self._write_profile(profile, filename, format='%0.6e')

        if newProfile:
            # Temporary solution to memory leak.
            for g in self.pf.h.grids:
                g.clear_data()
            sphere.clear_data()
            del sphere

        return profile

    @parallel_blocking_call
    def make_projections(self, axes=[0, 1, 2], halo_list='filtered', save_images=False, save_cube=True):
        "Make projections of all halos using specified fields."

        # Get list of halos for projecting.
        if halo_list == 'filtered':
            self._halo_projection_list = self.filtered_halos
        elif halo_list == 'all':
            self._halo_projection_list = self.all_halos
        elif isinstance(halo_list, types.StringType):
            self._halo_projection_list = self._read_halo_list(halo_list)
        elif isinstance(halo_list, types.ListType):
            self._halo_projection_list = halo_list
        else:
            mylog.error("Keyword, halo_list', must be 'filtered', 'all', a filename, or an actual list.")
            return

        if len(self._halo_projection_list) == 0:
            mylog.error("Halo list for projections is empty.")
            return

        # Set resolution for fixed resolution output.
        if self.project_at_level == 'max':
            proj_level = self.pf.h.max_level
        else:
            proj_level = int(self.project_at_level)
        proj_dx = self.pf.units[self.projection_width_units] / self.pf.parameters['TopGridDimensions'][0] / \
            (self.pf.parameters['RefineBy']**proj_level)
        projectionResolution = int(self.projection_width / proj_dx)

        # Create output directory.
        if self.output_dir is not None:
            self.__check_directory("%s/%s" % (self.output_dir, self.pf.directory))
            my_output_dir = "%s/%s/%s" % (self.output_dir, self.pf.directory, 
                                          self.projection_output_dir)
        else:
            my_output_dir = "%s/%s" % (self.pf.fullpath, self.projection_output_dir)
        self.__check_directory(my_output_dir)

        center = [0.5 * (self.pf.parameters['DomainLeftEdge'][w] + self.pf.parameters['DomainRightEdge'][w])
                  for w in range(self.pf.parameters['TopGridRank'])]

        for halo in self._get_objs('_halo_projection_list', round_robin=True):
            if halo is None:
                continue
            # Check if region will overlap domain edge.
            # Using non-periodic regions is faster than using periodic ones.
            leftEdge = [(halo['center'][w] - 0.5 * self.projection_width/self.pf.units[self.projection_width_units])
                        for w in range(len(halo['center']))]
            rightEdge = [(halo['center'][w] + 0.5 * self.projection_width/self.pf.units[self.projection_width_units])
                         for w in range(len(halo['center']))]

            mylog.info("Projecting halo %04d in region: [%f, %f, %f] to [%f, %f, %f]." %
                       (halo['id'], leftEdge[0], leftEdge[1], leftEdge[2], rightEdge[0], rightEdge[1], rightEdge[2]))

            need_per = False
            for w in range(len(halo['center'])):
                if ((leftEdge[w] < self.pf.parameters['DomainLeftEdge'][w]) or
                    (rightEdge[w] > self.pf.parameters['DomainRightEdge'][w])):
                    need_per = True
                    break

            if need_per:
                region = self.pf.h.periodic_region(halo['center'], leftEdge, rightEdge)
            else:
                region = self.pf.h.region(halo['center'], leftEdge, rightEdge)

            # Make projections.
            if not isinstance(axes, types.ListType): axes = list([axes])
            for w in axes:
                projections = []
                # YT projections do not follow the right-hand rule.
                coords = range(3)
                del coords[w]
                x_axis = coords[0]
                y_axis = coords[1]

                for hp in self.projection_fields:
                    projections.append(self.pf.h.proj(w, hp['field'], 
                                                      weight_field=hp['weight_field'], 
                                                      data_source=region, center=halo['center'],
                                                      serialize=False))
                
                # Set x and y limits, shift image if it overlaps domain boundary.
                if need_per:
                    pw = self.projection_width/self.pf.units[self.projection_width_units]
                    #shift_projections(self.pf, projections, halo['center'], center, w)
                    # Projection has now been shifted to center of box.
                    proj_left = [center[x_axis]-0.5*pw, center[y_axis]-0.5*pw]
                    proj_right = [center[x_axis]+0.5*pw, center[y_axis]+0.5*pw]
                else:
                    proj_left = [leftEdge[x_axis], leftEdge[y_axis]]
                    proj_right = [rightEdge[x_axis], rightEdge[y_axis]]

                # Save projection data to hdf5 file.
                if save_cube or save_images:
                    axis_labels = ['x', 'y', 'z']

                    if save_cube:
                        dataFilename = "%s/Halo_%04d_%s_data.h5" % \
                            (my_output_dir, halo['id'], axis_labels[w])
                        mylog.info("Saving projection data to %s." % dataFilename)
                        output = h5py.File(dataFilename, "a")

                    # Create fixed resolution buffer for each projection and write them out.
                    for e, hp in enumerate(self.projection_fields):
                        frb = FixedResolutionBuffer(projections[e], (proj_left[0], proj_right[0], 
                                                                     proj_left[1], proj_right[1]),
                                                    (projectionResolution, projectionResolution),
                                                    antialias=False)
                        dataset_name = "%s_%s" % (hp['field'], hp['weight_field'])
                        if save_cube:
                            if dataset_name in output.listnames(): del output[dataset_name]
                            output.create_dataset(dataset_name, data=frb[hp['field']])
                        if save_images:
                            filename = "%s/Halo_%04d_%s_%s.png" % (my_output_dir, halo['id'], 
                                                                   dataset_name, axis_labels[w])
                            write_image(na.log10(frb[hp['field']]), filename, cmap_name=hp['cmap'])
                    if save_cube: output.close()

            del region

    def _add_actual_overdensity(self, profile):
        "Calculate overdensity from TotalMassMsun and CellVolume fields."

        if 'ActualOverdensity' in profile.keys():
            return

        rho_crit_now = 1.8788e-29 * self.pf.hubble_constant**2.0 * \
            self.pf.omega_matter # g cm^-3
        Msun2g = 1.989e33
        rho_crit = rho_crit_now * ((1.0 + self.pf.current_redshift)**3.0)

        profile['ActualOverdensity'] = (Msun2g * profile['TotalMassMsun']) / \
            profile['CellVolume'] / rho_crit

    def _check_for_needed_profile_fields(self):
        "Make sure CellVolume and TotalMass fields are added so virial quantities can be calculated."
        all_profile_fields = [hp['field'] for hp in self.profile_fields]
        if not 'CellVolume' in all_profile_fields:
            mylog.info("Adding CellVolume field to so virial quantities can be calculated")
            self.add_profile('CellVolume', weight_field=None, accumulation=True)
        if not 'TotalMassMsun' in all_profile_fields:
            mylog.info("Adding TotalMassMsun field to so virial quantities can be calculated")
            self.add_profile('TotalMassMsun', weight_field=None, accumulation=True)

    def _load_halo_data(self, filename=None):
        "Read hop output file or run hop if it doesn't exist."

        # Don't run if hop data already loaded.
        if self.all_halos:
            return

        if filename is None:
            filename = self.halo_list_file

        if self.output_dir is not None:
            self.__check_directory("%s/%s" % (self.output_dir, self.pf.directory))
            hop_file = "%s/%s/%s" % (self.output_dir, self.pf.directory, filename)
        else:
            hop_file = "%s/%s" % (self.pf.fullpath, filename)

        if not(os.path.exists(hop_file)):
            mylog.info("Halo finder file not found, running halo finder to get halos.")
            self._run_hop(hop_file)

        self.all_halos = self._read_halo_list(hop_file)

    def _read_halo_list(self, listFile):
        """
        Read halo list from aue file.
        Allow for columnar data in varying formats.
        """

        def __isE(arg):
            parts = arg.lower().split('e')
            if len(parts) != 2: return False
            return not (True in [q.isalpha() for q in ''.join(parts)])

        def __get_num(arg):
            if __isE(arg):
                return float(arg)
            if arg != arg.swapcase():
                return arg
            return float(arg)

        mylog.info("Reading halo information from %s." % listFile)
        haloList = []
        listLines = file(listFile)

        fields = self.halo_list_format.keys()
        getID = not 'id' in fields
        has_rmax = 'r_max' in fields
        has_r200kpc = 'r200kpc' in fields

        for line in listLines:
            line = line.strip()
            if len(line) > 0 and not line.startswith('#') and not line[0].isalpha():
                halo = {}
                onLine = line.split()
                for field in fields:
                    if isinstance(self.halo_list_format[field], types.ListType):
                        halo[field] = [__get_num(onLine[q]) for q in self.halo_list_format[field]]
                    else:
                        halo[field] = __get_num(onLine[self.halo_list_format[field]])
                if getID: halo['id'] = len(haloList)
                if has_rmax:
                    halo['r_max'] *= self.pf.units['mpc']
                elif has_r200kpc:
                    # If P-Groupfinder used, r_200 [kpc] is calculated.
                    # set r_max as 50% past r_200.
                    halo['r_max'] = 1.5 * halo['r200kpc'] / 1000.
                else:
                    halo['r_max'] = self.halo_radius * self.pf.units['mpc']
                haloList.append(halo)

        mylog.info("Loaded %d halos." % (len(haloList)))
        return haloList

    def _read_profile(self, profileFile):
        "Read radial profile from file.  Return None if it doesn't have all the fields requested."

        # Check to see if file exists.
        if not os.path.exists(profileFile):
            return None

        f = open(profileFile, 'r')
        lines = f.readlines()
        f.close()

        # Get fields from header.
        header = lines.pop(0)
        header = header.strip()
        fields = header.split()
        # First string is '#'.
        fields.pop(0)

        profile = {}
        profile_obj = FakeProfile(self.pf)
        for field in fields:
            profile[field] = []

        # Check if all fields needed are present.
        all_profile_fields = [hp['field'] for hp in self.profile_fields]
        for field in all_profile_fields:
            if not field in profile:
                return None

        # Fill profile fields, skip bad values.
        for line in lines:
            line = line.strip()
            onLine = line.split()
            lineOK = True
            for value in onLine:
                if value.isalpha():
                    lineOK = False
                    break
            if lineOK:
                for q, field in enumerate(fields):
                    profile[field].append(float(onLine[q]))

        for field in fields:
            profile[field] = na.array(profile[field])

        profile_obj._data = profile

        if len(profile[fields[0]]) > 1:
            return profile_obj
        else:
            return None

    @parallel_blocking_call
    def _run_hop(self, hop_file):
        "Run hop to get halos."

        hop_results = self.halo_finder_function(self.pf, *self.halo_finder_args, **self.halo_finder_kwargs)
        hop_results.write_out(hop_file)

        del hop_results
        self.pf.h.clear_all_data()

    @parallel_root_only
    def _write_filtered_halo_list(self, filename, format="%s"):
        """
        Write out list of filtered halos along with any quantities 
        picked up during the filtering process.
        """

        if len(self.filtered_halos) == 0:
            mylog.error("No halos in filtered list.")
            return

        filename = "%s/%s" % (self.pf.fullpath, filename)
        mylog.info("Writing filtered halo list to %s." % filename)
        file = open(filename, "w")
        fields = [field for field in sorted(self.filtered_halos[0])]
        halo_fields = []
        for halo_field in self.filter_quantities:
            if halo_field in fields:
                fields.remove(halo_field)
                halo_fields.append(halo_field)
        # Make it so number of fields in header is same as number of data columns.
        header_fields = []
        for halo_field in halo_fields:
            if isinstance(self.filtered_halos[0][halo_field], types.ListType):
                header_fields.extend(["%s[%d]" % (halo_field, q) 
                                      for q in range(len(self.filtered_halos[0][halo_field]))])
            else:
                header_fields.append(halo_field)
        file.write("# ")
        file.write("\t".join(header_fields + fields + ["\n"]))

        for halo in self.filtered_halos:
            for halo_field in halo_fields:
                if isinstance(halo[halo_field], types.ListType):
                    field_data = na.array(halo[halo_field])
                    field_data.tofile(file, sep="\t", format=format)
                else:
                    if halo_field == 'id':
                        file.write("%04d" % halo[halo_field])
                    else:
                        file.write("%s" % halo[halo_field])
                file.write("\t")
            field_data = na.array([halo[field] for field in fields])
            field_data.tofile(file, sep="\t", format=format)
            file.write("\n")
        file.close()

    def _write_profile(self, profile, filename, format="%0.16e"):
        fid = open(filename, "w")
        fields = [field for field in sorted(profile.keys()) if field != "UsedBins"]
        fid.write("\t".join(["#"] + fields + ["\n"]))
        field_data = na.array([profile[field] for field in fields])
        for line in range(field_data.shape[1]):
            field_data[:, line].tofile(fid, sep="\t", format=format)
            fid.write("\n")
        fid.close()

    @parallel_root_only
    def __check_directory(self, my_output_dir):
        if (os.path.exists(my_output_dir)):
            if not(os.path.isdir(my_output_dir)):
                mylog.error("Output directory exists, but is not a directory: %s." % my_output_dir)
                raise IOError(my_output_dir)
        else:
            os.mkdir(my_output_dir)

def shift_projections(pf, projections, oldCenter, newCenter, axis):
    """
    Shift projection data around.
    This is necessary when projecting a preiodic region.
    """
    offset = [newCenter[q]-oldCenter[q] for q in range(len(oldCenter))]
    width = [pf.parameters['DomainRightEdge'][q]-pf.parameters['DomainLeftEdge'][q] \
                 for q in range(len(oldCenter))]

    del offset[axis]
    del width[axis]

    for plot in projections:
        # Get name of data field.
        other_fields = {'px':True, 'py':True, 'pdx':True, 'pdy':True, 'weight_field':True}
        for pfield in plot.data.keys():
            if not(other_fields.has_key(pfield)):
                field = pfield
                break

        # Shift x and y positions.
        plot['px'] += offset[0]
        plot['py'] += offset[1]

        # Wrap off-edge cells back around to other side (periodic boundary conditions).
        plot['px'][plot['px'] < 0] += width[0]
        plot['py'][plot['py'] < 0] += width[1]
        plot['px'][plot['px'] > width[0]] -= width[0]
        plot['py'][plot['py'] > width[1]] -= width[1]

        # After shifting, some cells have fractional coverage on both sides of the box.
        # Find those cells and make copies to be placed on the other side.

        # Cells hanging off the right edge.
        add_x_px = plot['px'][plot['px'] + 0.5 * plot['pdx'] > width[0]]
        add_x_px -= width[0]
        add_x_py = plot['py'][plot['px'] + 0.5 * plot['pdx'] > width[0]]
        add_x_pdx = plot['pdx'][plot['px'] + 0.5 * plot['pdx'] > width[0]]
        add_x_pdy = plot['pdy'][plot['px'] + 0.5 * plot['pdx'] > width[0]]
        add_x_field = plot[field][plot['px'] + 0.5 * plot['pdx'] > width[0]]
        add_x_weight_field = plot['weight_field'][plot['px'] + 0.5 * plot['pdx'] > width[0]]

        # Cells hanging off the left edge.
        add2_x_px = plot['px'][plot['px'] - 0.5 * plot['pdx'] < 0]
        add2_x_px += width[0]
        add2_x_py = plot['py'][plot['px'] - 0.5 * plot['pdx'] < 0]
        add2_x_pdx = plot['pdx'][plot['px'] - 0.5 * plot['pdx'] < 0]
        add2_x_pdy = plot['pdy'][plot['px'] - 0.5 * plot['pdx'] < 0]
        add2_x_field = plot[field][plot['px'] - 0.5 * plot['pdx'] < 0]
        add2_x_weight_field = plot['weight_field'][plot['px'] - 0.5 * plot['pdx'] < 0]

        # Cells hanging off the top edge.
        add_y_px = plot['px'][plot['py'] + 0.5 * plot['pdy'] > width[1]]
        add_y_py = plot['py'][plot['py'] + 0.5 * plot['pdy'] > width[1]]
        add_y_py -= width[1]
        add_y_pdx = plot['pdx'][plot['py'] + 0.5 * plot['pdy'] > width[1]]
        add_y_pdy = plot['pdy'][plot['py'] + 0.5 * plot['pdy'] > width[1]]
        add_y_field = plot[field][plot['py'] + 0.5 * plot['pdy'] > width[1]]
        add_y_weight_field = plot['weight_field'][plot['py'] + 0.5 * plot['pdy'] > width[1]]

        # Cells hanging off the bottom edge.
        add2_y_px = plot['px'][plot['py'] - 0.5 * plot['pdy'] < 0]
        add2_y_py = plot['py'][plot['py'] - 0.5 * plot['pdy'] < 0]
        add2_y_py += width[1]
        add2_y_pdx = plot['pdx'][plot['py'] - 0.5 * plot['pdy'] < 0]
        add2_y_pdy = plot['pdy'][plot['py'] - 0.5 * plot['pdy'] < 0]
        add2_y_field = plot[field][plot['py'] - 0.5 * plot['pdy'] < 0]
        add2_y_weight_field = plot['weight_field'][plot['py'] - 0.5 * plot['pdy'] < 0]

        # Add the hanging cells back to the projection data.
        plot.data['px'] = na.concatenate([plot['px'], add_x_px, add_y_px, add2_x_px, add2_y_px])
        plot.data['py'] = na.concatenate([plot['py'], add_x_py, add_y_py, add2_x_py, add2_y_py])
        plot.data['pdx'] = na.concatenate([plot['pdx'], add_x_pdx, add_y_pdx, add2_x_pdx, add2_y_pdx])
        plot.data['pdy'] = na.concatenate([plot['pdy'], add_x_pdy, add_y_pdy, add2_x_pdy, add2_y_pdy])
        plot.data[field] = na.concatenate([plot[field], add_x_field, add_y_field, add2_x_field, add2_y_field])
        plot.data['weight_field'] = na.concatenate([plot['weight_field'],
                                                    add_x_weight_field, add_y_weight_field, 
                                                    add2_x_weight_field, add2_y_weight_field])

        # Delete original copies of hanging cells.
        del add_x_px, add_y_px, add2_x_px, add2_y_px
        del add_x_py, add_y_py, add2_x_py, add2_y_py
        del add_x_pdx, add_y_pdx, add2_x_pdx, add2_y_pdx
        del add_x_pdy, add_y_pdy, add2_x_pdy, add2_y_pdy
        del add_x_field, add_y_field, add2_x_field, add2_y_field
        del add_x_weight_field, add_y_weight_field, add2_x_weight_field, add2_y_weight_field

class FakeProfile(ParallelAnalysisInterface):
    """
    This is used to mimic a profile object when reading profile data from disk.
    """
    def __init__(self, pf):
        self.pf = pf
        self._data = {}

    def __getitem__(self, key):
        return self._data[key]

    def keys(self):
        return self._data.keys()
