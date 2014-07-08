"""
Writing yt data to a GDF file.


"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import os
import sys
import h5py
import numpy as np

from yt import __version__ as yt_version
from yt.utilities.exceptions import YTGDFAlreadyExists

def write_to_gdf(pf, gdf_path, data_author=None, data_comment=None,
                 particle_type_name="dark_matter", clobber=False):
    r"""
    Write a parameter file to the given path in the Grid Data Format.

    Parameters
    ----------
    pf : Dataset object
        The yt data to write out.
    gdf_path : string
        The path of the file to output.
    data_author : string, optional
        The name of the author who wrote the data. Default: None.
    data_comment : string, optional
        A descriptive comment. Default: None.
    particle_type_name : string, optional
        The particle type of the particles in the dataset. Default: "dark_matter"
    clobber : boolean, optional
        Whether or not to clobber an already existing file. If False, attempting
        to overwrite an existing file will result in an exception.

    Examples
    --------
    >>> write_to_gdf(ds, "clumps.h5", data_author="Your Mom",
    ...              data_comment="All Your Base Are Belong To Us", clobber=True)
    """

    f = _create_new_gdf(pf, gdf_path, data_author, data_comment,
                        particle_type_name, clobber=clobber)

    # now add the fields one-by-one
    for field_name in pf.field_list:
        _write_field_to_gdf(pf, f, field_name, particle_type_name)

    # don't forget to close the file.
    f.close()


def save_field(pf, field_name, field_parameters=None):
    """
    Write a single field associated with the parameter file pf to the
    backup file.

    Parameters
    ----------
    pf : Dataset object
        The yt parameter file that the field is associated with.
    field_name : string
        The name of the field to save.
    field_parameters : dictionary
        A dictionary of field parameters to set.
    """

    if isinstance(field_name, tuple):
        field_name = field_name[1]
    field_obj = pf._get_field_info(field_name)
    if field_obj.particle_type:
        print("Saving particle fields currently not supported.")
        return

    backup_filename = pf.backup_filename
    if os.path.exists(backup_filename):
        # backup file already exists, open it
        f = h5py.File(backup_filename, "r+")
    else:
        # backup file does not exist, create it
        f = _create_new_gdf(pf, backup_filename, data_author=None,
                            data_comment=None,
                            particle_type_name="dark_matter")

    # now save the field
    _write_field_to_gdf(pf, f, field_name, particle_type_name="dark_matter",
                        field_parameters=field_parameters)

    # don't forget to close the file.
    f.close()


def _write_field_to_gdf(pf, fhandle, field_name, particle_type_name,
                        field_parameters=None):

    # add field info to field_types group
    g = fhandle["field_types"]
    # create the subgroup with the field's name
    if isinstance(field_name, tuple):
        field_name = field_name[1]
    fi = pf._get_field_info(field_name)
    try:
        sg = g.create_group(field_name)
    except ValueError:
        print "Error - File already contains field called " + field_name
        sys.exit(1)

    # grab the display name and units from the field info container.
    display_name = fi.display_name
    units = fi.get_units()

    # check that they actually contain something...
    if display_name:
        sg.attrs["field_name"] = display_name
    else:
        sg.attrs["field_name"] = field_name
    if units:
        sg.attrs["field_units"] = units
    else:
        sg.attrs["field_units"] = "None"
    # @todo: the values must be in CGS already right?
    sg.attrs["field_to_cgs"] = 1.0
    # @todo: is this always true?
    sg.attrs["staggering"] = 0

    # now add actual data, grid by grid
    g = fhandle["data"]
    for grid in pf.index.grids:

        # set field parameters, if specified
        if field_parameters is not None:
            for k, v in field_parameters.iteritems():
                grid.set_field_parameter(k, v)

        grid_group = g["grid_%010i" % (grid.id - grid._id_offset)]
        particles_group = grid_group["particles"]
        pt_group = particles_group[particle_type_name]
        # add the field data to the grid group
        # Check if this is a real field or particle data.
        grid.get_data(field_name)
        if fi.particle_type:  # particle data
            pt_group[field_name] = grid[field_name]
        else:  # a field
            grid_group[field_name] = grid[field_name]


def _create_new_gdf(pf, gdf_path, data_author=None, data_comment=None,
                    particle_type_name="dark_matter", clobber=False):
    # Make sure we have the absolute path to the file first
    gdf_path = os.path.abspath(gdf_path)

    # Stupid check -- is the file already there?
    # @todo: make this a specific exception/error.
    if os.path.exists(gdf_path) and not clobber:
        raise YTGDFAlreadyExists(gdf_path)

    ###
    # Create and open the file with h5py
    ###
    f = h5py.File(gdf_path, "w")

    ###
    # "gridded_data_format" group
    ###
    g = f.create_group("gridded_data_format")
    g.attrs["data_software"] = "yt"
    g.attrs["data_software_version"] = yt_version
    if data_author is not None:
        g.attrs["data_author"] = data_author
    if data_comment is not None:
        g.attrs["data_comment"] = data_comment

    ###
    # "simulation_parameters" group
    ###
    g = f.create_group("simulation_parameters")
    g.attrs["refine_by"] = pf.refine_by
    g.attrs["dimensionality"] = pf.dimensionality
    g.attrs["domain_dimensions"] = pf.domain_dimensions
    g.attrs["current_time"] = pf.current_time
    g.attrs["domain_left_edge"] = pf.domain_left_edge
    g.attrs["domain_right_edge"] = pf.domain_right_edge
    g.attrs["unique_identifier"] = pf.unique_identifier
    g.attrs["cosmological_simulation"] = pf.cosmological_simulation
    # @todo: Where is this in the yt API?
    g.attrs["num_ghost_zones"] = 0
    # @todo: Where is this in the yt API?
    g.attrs["field_ordering"] = 0
    # @todo: not yet supported by yt.
    g.attrs["boundary_conditions"] = np.array([0, 0, 0, 0, 0, 0], 'int32')

    if pf.cosmological_simulation:
        g.attrs["current_redshift"] = pf.current_redshift
        g.attrs["omega_matter"] = pf.omega_matter
        g.attrs["omega_lambda"] = pf.omega_lambda
        g.attrs["hubble_constant"] = pf.hubble_constant

    ###
    # "field_types" group
    ###
    g = f.create_group("field_types")

    ###
    # "particle_types" group
    ###
    g = f.create_group("particle_types")

    # @todo: Particle type iterator
    sg = g.create_group(particle_type_name)
    sg["particle_type_name"] = particle_type_name

    ###
    # root datasets -- info about the grids
    ###
    f["grid_dimensions"] = pf.index.grid_dimensions
    f["grid_left_index"] = np.array(
        [grid.get_global_startindex() for grid in pf.index.grids]
    ).reshape(pf.index.grid_dimensions.shape[0], 3)
    f["grid_level"] = pf.index.grid_levels.flat
    # @todo: Fill with proper values
    f["grid_parent_id"] = -np.ones(pf.index.grid_dimensions.shape[0])
    f["grid_particle_count"] = pf.index.grid_particle_count

    ###
    # "data" group -- where we should spend the most time
    ###

    g = f.create_group("data")
    for grid in pf.index.grids:
        # add group for this grid
        grid_group = g.create_group("grid_%010i" % (grid.id - grid._id_offset))
        # add group for the particles on this grid
        particles_group = grid_group.create_group("particles")
        pt_group = particles_group.create_group(particle_type_name)

    return f
