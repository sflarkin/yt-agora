"""
These are common particle deposition fields.




"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np

from yt.funcs import *
from yt.fields.derived_field import \
    ValidateParameter, \
    ValidateSpatial
from yt.utilities.physical_constants import \
    mass_hydrogen_cgs, \
    mass_sun_cgs, \
    mh

from yt.units.yt_array import \
    uconcatenate

from yt.utilities.math_utils import \
    get_sph_r_component, \
    get_sph_theta_component, \
    get_sph_phi_component, \
    get_cyl_r_component, \
    get_cyl_z_component, \
    get_cyl_theta_component, \
    get_cyl_r, get_cyl_theta, \
    get_cyl_z, get_sph_r, \
    get_sph_theta, get_sph_phi, \
    periodic_dist, euclidean_dist
     
def _field_concat(fname):
    def _AllFields(field, data):
        v = []
        for ptype in data.pf.particle_types:
            data.pf._last_freq = (ptype, None)
            if ptype == "all" or \
                ptype in data.pf.known_filters:
                  continue
            v.append(data[ptype, fname].copy())
        rv = uconcatenate(v, axis=0)
        return rv
    return _AllFields

def _field_concat_slice(fname, axi):
    def _AllFields(field, data):
        v = []
        for ptype in data.pf.particle_types:
            data.pf._last_freq = (ptype, None)
            if ptype == "all" or \
                ptype in data.pf.known_filters:
                  continue
            v.append(data[ptype, fname][:,axi])
        rv = uconcatenate(v, axis=0)
        return rv
    return _AllFields

def particle_deposition_functions(ptype, coord_name, mass_name, registry):
    orig = set(registry.keys())
    def particle_count(field, data):
        pos = data[ptype, coord_name]
        d = data.deposit(pos, method = "count")
        d = data.pf.arr(d, input_units = "cm**-3")
        return data.apply_units(d, field.units)

    registry.add_field(("deposit", "%s_count" % ptype),
             function = particle_count,
             validators = [ValidateSpatial()],
             display_name = "\\mathrm{%s Count}" % ptype,
             projection_conversion = '1')

    def particle_mass(field, data):
        pos = data[ptype, coord_name]
        d = data.deposit(pos, [data[ptype, mass_name]], method = "sum")
        return data.apply_units(d, field.units)

    registry.add_field(("deposit", "%s_mass" % ptype),
             function = particle_mass,
             validators = [ValidateSpatial()],
             display_name = "\\mathrm{%s Mass}" % ptype,
             units = "g")
             
    def particle_density(field, data):
        pos = data[ptype, coord_name]
        mass = data[ptype, mass_name]
        pos.convert_to_units("code_length")
        mass.convert_to_units("code_mass")
        d = data.deposit(pos, [data[ptype, mass_name]], method = "sum")
        d = data.pf.arr(d, "code_mass")
        d /= data["index", "cell_volume"]
        return d

    registry.add_field(("deposit", "%s_density" % ptype),
             function = particle_density,
             validators = [ValidateSpatial()],
             display_name = "\\mathrm{%s Density}" % ptype,
             units = "g/cm**3")

    def particle_cic(field, data):
        pos = data[ptype, coord_name]
        d = data.deposit(pos, [data[ptype, mass_name]], method = "cic")
        d = data.apply_units(d, data[ptype, mass_name].units)
        d /= data["index", "cell_volume"]
        return d

    registry.add_field(("deposit", "%s_cic" % ptype),
             function = particle_cic,
             validators = [ValidateSpatial()],
             display_name = "\\mathrm{%s CIC Density}" % ptype,
             units = "g/cm**3")

    # Now some translation functions.

    def particle_ones(field, data):
        v = np.ones(data[ptype, mass_name].shape, dtype="float64")
        return data.apply_units(v, field.units)

    registry.add_field((ptype, "particle_ones"),
                       function = particle_ones,
                       particle_type = True,
                       units = "")

    registry.alias((ptype, "ParticleMass"), (ptype, mass_name),
                    units = "g")

    registry.alias((ptype, "ParticleMassMsun"), (ptype, mass_name),
                    units = "Msun")

    def particle_mesh_ids(field, data):
        pos = data[ptype, coord_name]
        ids = np.zeros(pos.shape[0], dtype="float64") - 1
        # This is float64 in name only.  It will be properly cast inside the
        # deposit operation.
        #_ids = ids.view("float64")
        data.deposit(pos, [ids], method = "mesh_id")
        return data.apply_units(ids, "")
    registry.add_field((ptype, "mesh_id"),
            function = particle_mesh_ids,
            validators = [ValidateSpatial()],
            particle_type = True)

    return list(set(registry.keys()).difference(orig))

def particle_scalar_functions(ptype, coord_name, vel_name, registry):

    # Now we have to set up the various velocity and coordinate things.  In the
    # future, we'll actually invert this and use the 3-component items
    # elsewhere, and stop using these.
    
    # Note that we pass in _ptype here so that it's defined inside the closure.
    orig = set(registry.keys())

    def _get_coord_funcs(axi, _ptype):
        def _particle_velocity(field, data):
            return data[_ptype, vel_name][:,axi]
        def _particle_position(field, data):
            return data[_ptype, coord_name][:,axi]
        return _particle_velocity, _particle_position
    for axi, ax in enumerate("xyz"):
        v, p = _get_coord_funcs(axi, ptype)
        registry.add_field((ptype, "particle_velocity_%s" % ax),
            particle_type = True, function = v,
            units = "code_length")
        registry.add_field((ptype, "particle_position_%s" % ax),
            particle_type = True, function = p,
            units = "code_length")

    return list(set(registry.keys()).difference(orig))

def particle_vector_functions(ptype, coord_names, vel_names, registry):

    # This will column_stack a set of scalars to create vector fields.
    orig = set(registry.keys())

    def _get_vec_func(_ptype, names):
        def particle_vectors(field, data):
            v = [data[_ptype, name].in_units(field.units)
                  for name in names]
            c = np.column_stack(v)
            return data.apply_units(c, field.units)
        return particle_vectors
    registry.add_field((ptype, "Coordinates"),
                       function=_get_vec_func(ptype, coord_names),
                       units = "code_length",
                       particle_type=True)
    registry.add_field((ptype, "Velocities"),
                       function=_get_vec_func(ptype, vel_names),
                       units = "cm / s",
                       particle_type=True)
    return list(set(registry.keys()).difference(orig))

def standard_particle_fields(registry, ptype,
                             spos = "particle_position_%s",
                             svel = "particle_velocity_%s"):
    # This function will set things up based on the scalar fields and standard
    # yt field names.

    def _particle_velocity_magnitude(field, data):
        """ M{|v|} """
        bulk_velocity = data.get_field_parameter("bulk_velocity")
        if bulk_velocity is None:
            bulk_velocity = np.zeros(3)
        return np.sqrt((data[ptype, svel % 'x'] - bulk_velocity[0])**2
                     + (data[ptype, svel % 'y'] - bulk_velocity[1])**2
                     + (data[ptype, svel % 'z'] - bulk_velocity[2])**2 )
    
        registry.add_field((ptype, "particle_velocity_magnitude"),
                  function=_particle_velocity_magnitude,
                  particle_type=True,
                  take_log=False,
                  units="cm/s")

    def _particle_specific_angular_momentum(field, data):
        """
        Calculate the angular of a particle velocity.  Returns a vector for each
        particle.
        """
        if data.has_field_parameter("bulk_velocity"):
            bv = data.get_field_parameter("bulk_velocity")
        else: bv = np.zeros(3, dtype=np.float64)
        xv = data[ptype, svel % 'x'] - bv[0]
        yv = data[ptype, svel % 'y'] - bv[1]
        zv = data[ptype, svel % 'z'] - bv[2]
        center = data.get_field_parameter('center')
        coords = np.array([data[ptype, spos % 'x'],
                           data[ptype, spos % 'y'],
                           data[ptype, spos % 'z']], dtype=np.float64)
        new_shape = tuple([3] + [1]*(len(coords.shape)-1))
        r_vec = coords - np.reshape(center,new_shape)
        v_vec = np.array([xv,yv,zv], dtype=np.float64)
        return np.cross(r_vec, v_vec, axis=0)

    registry.add_field((ptype, "particle_specific_angular_momentum"),
              function=_particle_specific_angular_momentum,
              particle_type=True,
              units="cm**2/s",
              validators=[ValidateParameter("center")])

    def _particle_specific_angular_momentum_x(field, data):
        if data.has_field_parameter("bulk_velocity"):
            bv = data.get_field_parameter("bulk_velocity")
        else: bv = np.zeros(3, dtype=np.float64)
        center = data.get_field_parameter('center')
        y = data[ptype, spos % "y"] - center[1]
        z = data[ptype, spos % "z"] - center[2]
        yv = data[ptype, svel % "y"] - bv[1]
        zv = data[ptype, svel % "z"] - bv[2]
        return yv*z - zv*y

    registry.add_field((ptype, "particle_specific_angular_momentum_x"),
              function=_particle_specific_angular_momentum_x,
              particle_type=True,
              units="cm**2/s",
              validators=[ValidateParameter("center")])

    def _particle_specific_angular_momentum_y(field, data):
        if data.has_field_parameter("bulk_velocity"):
            bv = data.get_field_parameter("bulk_velocity")
        else: bv = np.zeros(3, dtype=np.float64)
        center = data.get_field_parameter('center')
        x = data[ptype, spos % "x"] - center[0]
        z = data[ptype, spos % "z"] - center[2]
        xv = data[ptype, svel % "x"] - bv[0]
        zv = data[ptype, svel % "z"] - bv[2]
        return -(xv*z - zv*x)

    registry.add_field((ptype, "particle_specific_angular_momentum_y"),
              function=_particle_specific_angular_momentum_y,
              particle_type=True,
              units="cm**2/s",
              validators=[ValidateParameter("center")])

    def _particle_specific_angular_momentum_z(field, data):
        if data.has_field_parameter("bulk_velocity"):
            bv = data.get_field_parameter("bulk_velocity")
        else: bv = np.zeros(3, dtype=np.float64)
        center = data.get_field_parameter('center')
        x = data[ptype, spos % "x"] - center[0]
        y = data[ptype, spos % "y"] - center[1]
        xv = data[ptype, svel % "x"] - bv[0]
        yv = data[ptype, svel % "y"] - bv[1]
        return xv*y - yv*x

    registry.add_field((ptype, "particle_specific_angular_momentum_z"),
              function=_particle_specific_angular_momentum_z,
              particle_type=True,
              units="cm**2/s",
              validators=[ValidateParameter("center")])

    def _particle_angular_momentum(field, data):
        return data[ptype, "particle_mass"] \
             * data[ptype, "particle_specific_angular_momentum"]

    def _particle_angular_momentum_x(field, data):
        return data[ptype, "particle_mass"] * \
               data[ptype, "particle_specific_angular_momentum_x"]
    registry.add_field((ptype, "particle_angular_momentum_x"),
             function=_particle_angular_momentum_x,
             units="g*cm**2/s", particle_type=True,
             validators=[ValidateParameter('center')])

    def _particle_angular_momentum_y(field, data):
        return data[ptype, "particle_mass"] * \
               data[ptype, "particle_specific_angular_momentum_y"]
    registry.add_field((ptype, "particle_angular_momentum_y"),
             function=_particle_angular_momentum_y,
             units="g*cm**2/s", particle_type=True,
             validators=[ValidateParameter('center')])

    def _particle_angular_momentum_z(field, data):
        return data[ptype, "particle_mass"] * \
               data[ptype, "particle_specific_angular_momentum_z"]
    registry.add_field((ptype, "particle_angular_momentum_z"),
             function=_particle_angular_momentum_z,
             units="g*cm**2/s", particle_type=True,
             validators=[ValidateParameter('center')])

    from .field_functions import \
        get_radius

    def _particle_radius(field, data):
        return get_radius(data, "particle_position_")
    registry.add_field((ptype, "particle_radius"),
              function=_particle_radius,
              validators=[ValidateParameter("center")],
              units="cm", particle_type = True,
              display_name = "Particle Radius")

    def _particle_radius_spherical(field, data):
        normal = data.get_field_parameter('normal')
        center = data.get_field_parameter('center')
        bv = data.get_field_parameter("bulk_velocity")
        pos = spos
        pos = np.array([data[ptype, pos % ax] for ax in "xyz"])
        theta = get_sph_theta(pos, center)
        phi = get_sph_phi(pos, center)
        pos = pos - np.reshape(center, (3, 1))
        sphr = get_sph_r_component(pos, theta, phi, normal)
        return sphr

    registry.add_field((ptype, "particle_radius_spherical"),
              function=_particle_radius_spherical,
              particle_type=True, units="cm/s",
              validators=[ValidateParameter("normal"), 
                          ValidateParameter("center")])

    def _particle_theta_spherical(field, data):
        normal = data.get_field_parameter('normal')
        center = data.get_field_parameter('center')
        bv = data.get_field_parameter("bulk_velocity")
        pos = spos
        pos = np.array([data[ptype, pos % ax] for ax in "xyz"])
        theta = get_sph_theta(pos, center)
        phi = get_sph_phi(pos, center)
        pos = pos - np.reshape(center, (3, 1))
        spht = get_sph_theta_component(pos, theta, phi, normal)
        return spht

    registry.add_field((ptype, "particle_theta_spherical"),
              function=_particle_theta_spherical,
              particle_type=True, units="cm/s",
              validators=[ValidateParameter("normal"), 
                          ValidateParameter("center")])

    def _particle_phi_spherical(field, data):
        normal = data.get_field_parameter('normal')
        center = data.get_field_parameter('center')
        bv = data.get_field_parameter("bulk_velocity")
        pos = spos
        pos = np.array([data[ptype, pos % ax] for ax in "xyz"])
        theta = get_sph_theta(pos, center)
        phi = get_sph_phi(pos, center)
        pos = pos - np.reshape(center, (3, 1))
        vel = vel - np.reshape(bv, (3, 1))
        sphp = get_sph_phi_component(pos, theta, phi, normal)
        return sphp

    registry.add_field((ptype, "particle_phi_spherical"),
              function=_particle_phi_spherical,
              particle_type=True, units="cm/s",
              validators=[ValidateParameter("normal"), 
                          ValidateParameter("center")])

    def _particle_radial_velocity(field, data):
        normal = data.get_field_parameter('normal')
        center = data.get_field_parameter('center')
        bv = data.get_field_parameter("bulk_velocity")
        pos = spos
        pos = np.array([data[ptype, pos % ax] for ax in "xyz"])
        vel = svel
        vel = np.array([data[ptype, vel % ax] for ax in "xyz"])
        theta = get_sph_theta(pos, center)
        phi = get_sph_phi(pos, center)
        pos = pos - np.reshape(center, (3, 1))
        vel = vel - np.reshape(bv, (3, 1))
        sphr = get_sph_r_component(vel, theta, phi, normal)
        return sphr

    registry.add_field((ptype, "particle_radial_velocity"),
              function=_particle_radial_velocity,
              particle_type=True, units="cm/s",
              validators=[ValidateParameter("normal"), 
                          ValidateParameter("center")])

    def _particle_theta_velocity(field, data):
        normal = data.get_field_parameter('normal')
        center = data.get_field_parameter('center')
        bv = data.get_field_parameter("bulk_velocity")
        pos = spos
        pos = np.array([data[ptype, pos % ax] for ax in "xyz"])
        vel = svel
        vel = np.array([data[ptype, vel % ax] for ax in "xyz"])
        theta = get_sph_theta(pos, center)
        phi = get_sph_phi(pos, center)
        pos = pos - np.reshape(center, (3, 1))
        vel = vel - np.reshape(bv, (3, 1))
        spht = get_sph_theta_component(vel, theta, phi, normal)
        return spht

    registry.add_field((ptype, "particle_theta_velocity"),
              function=_particle_theta_velocity,
              particle_type=True, units="cm/s",
              validators=[ValidateParameter("normal"), 
                          ValidateParameter("center")])

    def _particle_phi_velocity(field, data):
        normal = data.get_field_parameter('normal')
        center = data.get_field_parameter('center')
        bv = data.get_field_parameter("bulk_velocity")
        pos = np.array([data[ptype, spos % ax] for ax in "xyz"])
        vel = np.array([data[ptype, svel % ax] for ax in "xyz"])
        theta = get_sph_theta(pos, center)
        phi = get_sph_phi(pos, center)
        pos = pos - np.reshape(center, (3, 1))
        vel = vel - np.reshape(bv, (3, 1))
        sphp = get_sph_phi_component(vel, phi, normal)
        return sphp

    registry.add_field((ptype, "particle_phi_velocity"),
              function=_particle_phi_velocity,
              particle_type=True, units="cm/s",
              validators=[ValidateParameter("normal"), 
                          ValidateParameter("center")])

    def _get_cic_field(fname, units):
        def _cic_particle_field(field, data):
            """
            Create a grid field for particle quantities weighted by particle
            mass, using cloud-in-cell deposit.
            """
            pos = data[ptype, 'Coordinates']
            # Get back into density
            pden = data[ptype, 'particle_mass'] / data["index", "cell_volume"] 
            top = data.deposit(pos, [data[('all', particle_field)]*pden],
                               method = 'cic')
            bottom = data.deposit(pos, [pden], method = 'cic')
            top[bottom == 0] = 0.0
            bnz = bottom.nonzero()
            top[bnz] /= bottom[bnz]
            d = data.pf.arr(top, input_units = units)
            return top

    for ax in 'xyz':
        registry.add_field(
            ("deposit", "%s_cic_velocity_%s" % (ptype, ax)),
            function=_get_cic_field(svel % ax, "cm/s"),
            units = "cm/s", take_log=False,
            validators=[ValidateSpatial(0)])

def add_particle_average(registry, ptype, field_name, 
                         weight = "particle_mass",
                         density = True):
    field_units = registry[ptype, field_name].units
    def _pfunc_avg(field, data):
        pos = data[ptype, "Coordinates"]
        f = data[ptype, field_name]
        wf = data[ptype, weight]
        f *= wf
        v = data.deposit(pos, [f], method = "sum")
        w = data.deposit(pos, [wf], method = "sum")
        v /= w
        if density: v /= data["index", "cell_volume"]
        v[np.isnan(v)] = 0.0
        return v
    fn = ("deposit", "%s_avg_%s" % (ptype, field_name))
    registry.add_field(fn, function=_pfunc_avg,
                       validators = [ValidateSpatial(0)],
                       particle_type = False,
                       units = field_units)
    return fn

def add_smoothed_field(ptype, coord_name, mass_name, smoothed_field, registry,
                       smoothing_type = "neighbor_smoothing", nneighbors = 64):
    field_name = ("deposit", "%s_smoothed_%s" % (ptype, smoothed_field))
    field_units = registry[ptype, smoothed_field].units
    def _smoothed_quantity(field, data):
        pos = data[ptype, coord_name]
        mass = data[ptype, mass_name]
        dep_mass = np.zeros_like(mass)
        quan = data[ptype, smoothed_field]
        data.smooth(pos, [mass, dep_mass], method="neighbor_mass_dep",
                          index_fields = [data["cell_volume"]],
                          nneighbors = nneighbors, create_octree = True)
        # Now, what dep_mass is is a total mass deposited, but in units of the
        # code -- not in the units of either the quantity or the mass.  So we
        # need to convert it back to cm^3 eventually.
        rv = data.smooth(pos, [mass, dep_mass, quan], method="neighbor_smoothing",
                          nneighbors = nneighbors, create_octree = True)[0]

        return rv
    registry.add_field(field_name, function = _smoothed_quantity,
                       validators = [ValidateSpatial(0)],
                       units = field_units)
    return [field_name]

def add_mass_conserved_smoothed_field(ptype, coord_name, mass_name,
        smoothing_length_name, smoothed_field, registry):
    field_name = ("deposit", "%s_smoothed_%s" % (ptype, smoothed_field))
    field_units = "(%s) / cm**3" % (registry[ptype, smoothed_field].units,)
    def _mass_cons(field, data):
        pos = data[ptype, coord_name].in_units("code_length")
        hsml = data[ptype, smoothing_length_name].in_units("code_length")
        mass = data[ptype, mass_name].in_cgs()
        vol = data["cell_volume"].in_cgs()
        dep_mass = mass.copy()
        dep_mass[:] = 0.0
        quan = data[ptype, smoothed_field]
        data.smooth(pos, [mass, hsml, dep_mass],
                method="mass_deposition_coeff",
                index_fields = [vol],
                create_octree = True)
        rv = data.smooth(pos, [mass, hsml, dep_mass, quan],
                         method="conserved_mass",
                         create_octree = True)[0]
        return data.apply_units(rv, field.units)
    registry.add_field(field_name, function = _mass_cons,
                       validators = [ValidateSpatial(0)],
                       units = field_units)
    return [field_name]

