"""
Data structures for Rockstar



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np

BINARY_HEADER_SIZE=256
header_dt = (
    ("magic", 1, "Q"),
    ("snap", 1, "q"),
    ("chunk", 1, "q"),
    ("scale", 1, "f"),
    ("Om", 1, "f"),
    ("Ol", 1, "f"),
    ("h0", 1, "f"),
    ("bounds", 6, "f"),
    ("num_halos", 1, "q"),
    ("num_particles", 1, "q"),
    ("box_size", 1, "f"),
    ("particle_mass", 1, "f"),
    ("particle_type", 1, "q"),
    ("format_revision", 1, "i"),
    ("version", 12, "c"),
    ("unused", BINARY_HEADER_SIZE - 4*12 - 4 - 8*6 - 12, "c")
)

halo_dt = np.dtype([
    ('particle_identifier', np.int64),
    ('particle_position_x', np.float32),
    ('particle_position_y', np.float32),
    ('particle_position_z', np.float32),
    ('particle_velocity_x', np.float32),
    ('particle_velocity_y', np.float32),
    ('particle_velocity_z', np.float32),
    ('particle_corevel_x', np.float32),
    ('particle_corevel_y', np.float32),
    ('particle_corevel_z', np.float32),
    ('particle_bulkvel_x', np.float32),
    ('particle_bulkvel_y', np.float32),
    ('particle_bulkvel_z', np.float32),
    ('particle_mass', np.float32),
    ('radius', np.float32),
    ('child_r', np.float32),
    ('vmax_r', np.float32),
    ('mgrav', np.float32),
    ('vmax', np.float32),
    ('rvmax', np.float32),
    ('rs', np.float32),
    ('klypin_rs', np.float32),
    ('vrms', np.float32),
    ('Jx', np.float32),
    ('Jy', np.float32),
    ('Jz', np.float32),
    ('energy', np.float32),
    ('spin', np.float32),
    ('alt_m1', np.float32),
    ('alt_m2', np.float32),
    ('alt_m3', np.float32),
    ('alt_m4', np.float32),
    ('Xoff', np.float32),
    ('Voff', np.float32),
    ('b_to_a', np.float32),
    ('c_to_a', np.float32),
    ('Ax', np.float32),
    ('Ay', np.float32),
    ('Az', np.float32),
    ('b_to_a2', np.float32),
    ('c_to_a2', np.float32),
    ('A2x', np.float32),
    ('A2y', np.float32),
    ('A2z', np.float32),
    ('bullock_spin', np.float32),
    ('kin_to_pot', np.float32),
    ('m_pe_b', np.float32),
    ('m_pe_d', np.float32),
    ('num_p', np.int64),
    ('num_child_particles', np.int64),
    ('p_start', np.int64),
    ('desc', np.int64),
    ('flags', np.int64),
    ('n_core', np.int64),
    ('min_pos_err', np.float32),
    ('min_vel_err', np.float32),
    ('min_bulkvel_err', np.float32),
])

particle_dt = np.dtype([
    ('particle_identifier', np.int64),
    ('particle_position_x', np.float32),
    ('particle_position_y', np.float32),
    ('particle_position_z', np.float32),
    ('particle_velocity_x', np.float32),
    ('particle_velocity_y', np.float32),
    ('particle_velocity_z', np.float32),
])
