"""
Quantities that can be derived from Enzo data that may also required additional
arguments.  (Standard arguments -- such as the center of a distribution of
points -- are excluded here, and left to the EnzoDerivedFields.)



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

from yt.config import ytcfg
from yt.units.yt_array import YTArray, uconcatenate, array_like_field
from yt.fields.field_info_container import \
    FieldDetector
from yt.utilities.data_point_utilities import FindBindingEnergy
from yt.utilities.exceptions import YTFieldNotFound
from yt.utilities.parallel_tools.parallel_analysis_interface import \
    ParallelAnalysisInterface, parallel_objects
from yt.utilities.lib.Octree import Octree
from yt.utilities.physical_constants import \
    gravitational_constant_cgs, \
    mass_sun_cgs, \
    HUGE
from yt.utilities.math_utils import prec_accum

derived_quantity_registry = {}

class DerivedQuantity(ParallelAnalysisInterface):
    num_vals = -1

    class __metaclass__(type):
        def __init__(cls, name, b, d):
            type.__init__(cls, name, b, d)
            if name != "DerivedQuantity":
                derived_quantity_registry[name] = cls

    def __init__(self, data_source):
        self.data_source = data_source

    def count_values(self, *args, **kwargs):
        return

    def __call__(self, *args, **kwargs):
        self.count_values(*args, **kwargs)
        chunks = self.data_source.chunks([], chunking_style="io")
        storage = {}
        for sto, ds in parallel_objects(chunks, -1, storage = storage):
            sto.result = self.process_chunk(ds, *args, **kwargs)
        # Now storage will have everything, and will be done via pickling, so
        # the units will be preserved.  (Credit to Nathan for this
        # idea/implementation.)
        values = [ [] for i in range(self.num_vals) ]
        for key in sorted(storage):
            for i in range(self.num_vals):
                values[i].append(storage[key][i])
        # These will be YTArrays
        values = [self.data_source.pf.arr(values[i]) for i in range(self.num_vals)]
        values = self.reduce_intermediate(values)
        return values

    def process_chunk(self, data, *args, **kwargs):
        raise NotImplementedError

    def reduce_intermediate(self, values):
        raise NotImplementedError

class DerivedQuantityCollection(object):
    def __new__(cls, data_source, *args, **kwargs):
        inst = object.__new__(cls)
        inst.data_source = data_source
        for f in inst.keys():
            setattr(inst, camelcase_to_underscore(f), inst[f])
        return inst

    def __getitem__(self, key):
        dq = derived_quantity_registry[key]
        # Instantiate here, so we can pass it the data object
        # Note that this means we instantiate every time we run help, etc
        # I have made my peace with this.
        return dq(self.data_source)

    def keys(self):
        return derived_quantity_registry.keys()

class WeightedAverage(DerivedQuantity):

    def count_values(self, fields, weight):
        # This is a list now
        self.num_vals = len(fields) + 1

    def __call__(self, fields, weight):
        fields = ensure_list(fields)
        rv = super(WeightedAverage, self).__call__(fields, weight)
        if len(rv) == 1: rv = rv[0]
        return rv

    def process_chunk(self, data, fields, weight):
        vals = [(data[field] * data[weight]).sum(dtype=np.float64)
                for field in fields]
        wv = data[weight].sum(dtype=np.float64)
        return vals + [wv]

    def reduce_intermediate(self, values):
        w = values.pop(-1).sum(dtype=np.float64)
        return [v.sum(dtype=np.float64)/w for v in values]

class TotalValue(DerivedQuantity):

    def count_values(self, fields):
        # This is a list now
        self.num_vals = len(fields)

    def __call__(self, fields):
        fields = ensure_list(fields)
        rv = super(TotalValue, self).__call__(fields)
        if len(rv) == 1: rv = rv[0]
        return rv

    def process_chunk(self, data, fields):
        vals = [data[field].sum(dtype=np.float64)
                for field in fields]
        return vals

    def reduce_intermediate(self, values):
        return [v.sum(dtype=np.float64) for v in values]

class TotalMass(TotalValue):
    def __call__(self):
        fi = self.data_source.pf.field_info
        fields = []
        if ("gas", "cell_mass") in fi:
            fields.append(("gas", "cell_mass"))
        if ("all", "particle_mass") in fi:
            fields.append(("all", "particle_mass"))
        rv = super(TotalMass, self).__call__(fields)
        return rv

class CenterOfMass(DerivedQuantity):
    def count_values(self, use_cells = True, use_particles = False):
        # This is a list now
        self.num_vals = 0
        if use_cells:
            self.num_vals += 4
        if use_particles:
            self.num_vals += 4

    def process_chunk(self, data, use_cells = True, use_particles = False):
        vals = []
        if use_cells:
            vals += [(data[ax] * data["cell_mass"]).sum(dtype=np.float64)
                     for ax in 'xyz']
            vals.append(data["cell_mass"].sum(dtype=np.float64))
        if use_particles:
            vals += [(data["particle_position_%s" % ax] *
                      data["particle_mass"]).sum(dtype=np.float64)
                     for ax in 'xyz']
            vals.append(data["particle_mass"].sum(dtype=np.float64))
        return vals

    def reduce_intermediate(self, values):
        if len(values) not in (4, 8):
            raise RuntimeError
        x = values.pop(0).sum(dtype=np.float64)
        y = values.pop(0).sum(dtype=np.float64)
        z = values.pop(0).sum(dtype=np.float64)
        w = values.pop(0).sum(dtype=np.float64)
        if len(values) > 0:
            # Note that this could be shorter if we pre-initialized our x,y,z,w
            # values as YTQuantity objects.
            x += values.pop(0).sum(dtype=np.float64)
            y += values.pop(0).sum(dtype=np.float64)
            z += values.pop(0).sum(dtype=np.float64)
            w += values.pop(0).sum(dtype=np.float64)
        return [v/w for v in [x, y, z]]

def _WeightedVariance(data, field, weight):
    """
    This function returns the variance of a field.

    :param field: The target field
    :param weight: The field to weight by

    Returns the weighted variance and the weighted mean.
    """
    my_weight = data[weight].sum(dtype=np.float64)
    if my_weight == 0:
        return 0.0, 0.0, 0.0
    my_mean = (data[field] * data[weight]).sum(dtype=np.float64) / my_weight
    my_var2 = (data[weight] * (data[field] - my_mean)**2).sum(dtype=np.float64) / my_weight
    return my_weight, my_mean, my_var2
def _combWeightedVariance(data, my_weight, my_mean, my_var2):
    all_weight = my_weight.sum()
    all_mean = (my_weight * my_mean).sum() / all_weight
    return [np.sqrt((my_weight * (my_var2 + (my_mean - all_mean)**2)).sum() / 
                    all_weight), all_mean]

class BulkVelocity(WeightedAverage):
    def __call__(self, ftype = "gas"):
        fields = [(ftype, "velocity_%s" % ax) for ax in 'xyz']
        weight = (ftype, "cell_mass")
        return super(BulkVelocity, self).__call__(fields, weight)

class AngularMomentumVector(WeightedAverage):
    def __call__(self, ftype = "gas"):
        fields = [(ftype, "specific_angular_momentum_%s" % ax)
                  for ax in 'xyz']
        weight = (ftype, "cell_mass")
        return super(AngularMomentumVector, self).__call__(fields, weight)

class ParticleAngularMomentumVector(WeightedAverage):
    def __call__(self, ptype = "all"):
        fields = [(ptype, "particle_specific_angular_momentum_%s" % ax)
                  for ax in 'xyz']
        weight = (ptype, "particle_mass")
        return super(ParticleAngularMomentumVector, self).__call__(fields, weight)

class Extrema(DerivedQuantity):
    def count_values(self, fields, non_zero):
        self.num_vals = len(fields) * 2

    def __call__(self, fields, non_zero = False):
        fields = ensure_list(fields)
        rv = super(Extrema, self).__call__(fields, non_zero)
        if len(rv) == 1: rv = rv[0]
        return rv

    def process_chunk(self, data, fields, non_zero):
        vals = []
        for field in fields:
            fd = data[field]
            if non_zero: fd = fd[fd > 0.0]
            if fd.size > 0:
                vals += [fd.min(), fd.max()]
            else:
                vals += [array_like_field(data.pf, HUGE, field),
                         array_like_field(data.pf, -HUGE, field)]
        return vals

    def reduce_intermediate(self, values):
        # The values get turned into arrays here.
        return [(mis.min(), mas.max() )
                for mis, mas in zip(values[::2], values[1::2])]

class MaxLocation(DerivedQuantity):
    def count_values(self, *args, **kwargs):
        self.num_vals = 5

    def __call__(self, field):
        rv = super(MaxLocation, self).__call__(field)
        if len(rv) == 1: rv = rv[0]
        return rv

    def process_chunk(self, data, field):
        field = data._determine_fields(field)[0]
        ma = array_like_field(data.pf, -HUGE, field)
        mx = array_like_field(data.pf, -1, "x")
        my = array_like_field(data.pf, -1, "y")
        mz = array_like_field(data.pf, -1, "z")
        maxi = -1
        if data[field].size > 0:
            maxi = np.argmax(data[field])
            ma = data[field][maxi]
            mx, my, mz = [data[ax][maxi] for ax in 'xyz']
        return (ma, maxi, mx, my, mz)

    def reduce_intermediate(self, values):
        i = np.argmax(values[0]) # ma is values[0]
        return [val[i] for val in values]

class MinLocation(DerivedQuantity):
    def count_values(self, *args, **kwargs):
        self.num_vals = 5

    def __call__(self, field):
        rv = super(MinLocation, self).__call__(field)
        if len(rv) == 1: rv = rv[0]
        return rv

    def process_chunk(self, data, field):
        field = data._determine_fields(field)[0]
        ma = array_like_field(data.pf, HUGE, field)
        mx = array_like_field(data.pf, -1, "x")
        my = array_like_field(data.pf, -1, "y")
        mz = array_like_field(data.pf, -1, "z")
        mini = -1
        if data[field].size > 0:
            mini = np.argmin(data[field])
            ma = data[field][mini]
            mx, my, mz = [data[ax][mini] for ax in 'xyz']
        return (ma, mini, mx, my, mz)

    def reduce_intermediate(self, values):
        i = np.argmin(values[0]) # ma is values[0]
        return [val[i] for val in values]

def _BaryonSpinParameter(data):
    """
    This function returns the spin parameter for the baryons, but it uses
    the particles in calculating enclosed mass.
    """
    m_enc = _TotalMass(data)
    amx = data["specific_angular_momentum_x"]*data["cell_mass"]
    amy = data["specific_angular_momentum_y"]*data["cell_mass"]
    amz = data["specific_angular_momentum_z"]*data["cell_mass"]
    j_mag = np.array([amx.sum(dtype=np.float64), amy.sum(dtype=np.float64), amz.sum(dtype=np.float64)])
    e_term_pre = np.sum(data["cell_mass"]*data["velocity_magnitude"]**2.0,dtype=np.float64)
    weight=data["cell_mass"].sum(dtype=np.float64)
    return j_mag, m_enc, e_term_pre, weight
def _combBaryonSpinParameter(data, j_mag, m_enc, e_term_pre, weight):
    # Because it's a vector field, we have to ensure we have enough dimensions
    if len(j_mag.shape) < 2: j_mag = np.expand_dims(j_mag, 0)
    W = weight.sum()
    M = m_enc.sum()
    J = np.sqrt(((j_mag.sum(axis=0))**2.0).sum())/W
    E = np.sqrt(e_term_pre.sum()/W)
    spin = J * E / (M * mass_sun_cgs * gravitational_constant_cgs)
    return spin

def _ParticleSpinParameter(data):
    """
    This function returns the spin parameter for the baryons, but it uses
    the particles in calculating enclosed mass.
    """
    m_enc = _TotalMass(data)
    amx = data["particle_specific_angular_momentum_x"]*data["particle_mass"]
    if amx.size == 0: return (np.zeros((3,), dtype=np.float64), m_enc, 0, 0)
    amy = data["particle_specific_angular_momentum_y"]*data["particle_mass"]
    amz = data["particle_specific_angular_momentum_z"]*data["particle_mass"]
    j_mag = np.array([amx.sum(dtype=np.float64), amy.sum(dtype=np.float64), amz.sum(dtype=np.float64)])
    e_term_pre = np.sum(data["particle_mass"]
                       *data["particle_velocity_magnitude"]**2.0,dtype=np.float64)
    weight=data["particle_mass"].sum(dtype=np.float64)
    return j_mag, m_enc, e_term_pre, weight
