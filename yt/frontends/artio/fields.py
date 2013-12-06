"""
ARTIO-specific fields




"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

from yt.fields.field_info_container import \
    FieldInfoContainer, \
    FieldInfo, \
    NullFunc, \
    TranslationFunc, \
    ValidateParameter, \
    ValidateDataField, \
    ValidateProperty, \
    ValidateSpatial, \
    ValidateGridType
import yt.fields.universal_fields
from yt.fields.universal_fields import \
    _get_conv
from yt.fields.particle_fields import \
    particle_deposition_functions, \
    particle_vector_functions, \
    particle_scalar_functions
import numpy as np

KnownARTIOFields = FieldInfoContainer()
add_artio_field = KnownARTIOFields.add_field

ARTIOFieldInfo = FieldInfoContainer.create_with_fallback(FieldInfo)
add_field = ARTIOFieldInfo.add_field

known_artio_fields = [
    'Density', 'TotalEnergy',
    'XMomentumDensity', 'YMomentumDensity', 'ZMomentumDensity',
    'Pressure', 'GasEnergy',
    'MetalDensitySNII', 'MetalDensitySNIa',
    'Potential', 'PotentialHydro']

#Add the fields, then later we'll individually defined units and names
for f in known_artio_fields:
    add_artio_field(f, function=NullFunc, take_log=True,
                    validators=[ValidateDataField(f)])

add_artio_field("Gamma", function=NullFunc, take_log=False,
                validators=[ValidateDataField("Gamma")])


def _convertDensity(data):
    return data.convert("Density")
KnownARTIOFields["Density"].units = r"\rm{g}/\rm{cm}^3"
KnownARTIOFields["Density"]._convert_function = _convertDensity


def _convertTotalEnergy(data):
    return data.convert("GasEnergy")
KnownARTIOFields["TotalEnergy"].units = r"\rm{g}/\rm{cm}^3"
KnownARTIOFields["TotalEnergy"]._convert_function = _convertTotalEnergy


def _convertXMomentumDensity(data):
    tr = data.convert("Mass")*data.convert("Velocity")
    tr *= (data.convert("Density")/data.convert("Mass"))
    return tr
KnownARTIOFields["XMomentumDensity"].units = r"\rm{mg}/\rm{s}/\rm{cm}^3"
KnownARTIOFields["XMomentumDensity"]._convert_function = \
    _convertXMomentumDensity


def _convertYMomentumDensity(data):
    tr = data.convert("Mass")*data.convert("Velocity")
    tr *= (data.convert("Density")/data.convert("Mass"))
    return tr
KnownARTIOFields["YMomentumDensity"].units = r"\rm{mg}/\rm{s}/\rm{cm}^3"
KnownARTIOFields["YMomentumDensity"]._convert_function =\
    _convertYMomentumDensity


def _convertZMomentumDensity(data):
    tr = data.convert("Mass")*data.convert("Velocity")
    tr *= (data.convert("Density")/data.convert("Mass"))
    return tr
KnownARTIOFields["ZMomentumDensity"].units = r"\rm{mg}/\rm{s}/\rm{cm}^3"
KnownARTIOFields["ZMomentumDensity"]._convert_function = \
    _convertZMomentumDensity


def _convertPressure(data):
    return data.convert("Pressure")
KnownARTIOFields["Pressure"].units = r"\rm{g}/\rm{cm}/\rm{s}^2"
KnownARTIOFields["Pressure"]._convert_function = _convertPressure


def _convertGamma(data):
    return 1.0
KnownARTIOFields["Gamma"].units = r""
KnownARTIOFields["Gamma"]._convert_function = _convertGamma


def _convertGasEnergy(data):
    return data.convert("GasEnergy")
KnownARTIOFields["GasEnergy"].units = r"\rm{ergs}/\rm{g}"
KnownARTIOFields["GasEnergy"]._convert_function = _convertGasEnergy


def _convertMetalDensitySNII(data):
    return data.convert('Density')
KnownARTIOFields["MetalDensitySNII"].units = r"\rm{g}/\rm{cm}^3"
KnownARTIOFields["MetalDensitySNII"]._convert_function = \
    _convertMetalDensitySNII


def _convertMetalDensitySNIa(data):
    return data.convert('Density')
KnownARTIOFields["MetalDensitySNIa"].units = r"\rm{g}/\rm{cm}^3"
KnownARTIOFields["MetalDensitySNIa"]._convert_function = \
    _convertMetalDensitySNIa


def _convertPotential(data):
    return data.convert("Potential")
KnownARTIOFields["Potential"].units = r"\rm{g}/\rm{cm}^3"
KnownARTIOFields["Potential"]._convert_function = \
    _convertPotential


def _convertPotentialHydro(data):
    return data.convert("Potential")
KnownARTIOFields["PotentialHydro"].units = r"\rm{g}/\rm{cm}^3"
KnownARTIOFields["PotentialHydro"]._convert_function = \
    _convertPotentialHydro


####### Derived fields
def _temperature(field, data):
    tr = data["GasEnergy"]/data["Density"]
    #Gamma fixed not field *(data["Gamma"]-1)*wmu
    tr[np.isnan(tr)] = 0.0
    #dhr - this is bad, don't just mask nan's,
    #make the user think about what they're doing
    return tr


def _converttemperature(data):
    x = data.pf.conversion_factors["Temperature"] * \
        data.pf.conversion_factors["Density"] / \
        data.pf.conversion_factors["GasEnergy"]
    return x
add_field("Temperature", function=_temperature, units="K", take_log=True)


def _metallicity_snII(field, data):
    tr = data["MetalDensitySNII"] / data["Density"]
    return tr
add_field("Metallicity_SNII", function=_metallicity_snII,
          units="Zsun", take_log=True)


def _metallicity_snIa(field, data):
    tr = data["MetalDensitySNIa"] / data["Density"]
    return tr
add_field("Metallicity_SNIa", function=_metallicity_snIa,
          units="Zsun", take_log=True)


def _metallicity(field, data):
    tr = data["Metal_Density"] / data["Density"]
    return tr
add_field("Metallicity", function=_metallicity,
          units="Zsun", take_log=True)


def _x_velocity(field, data):
    tr = data["XMomentumDensity"]/data["Density"]
    return tr
add_field("x-velocity", function=_x_velocity,
          units="cm/s", take_log=False)


def _y_velocity(field, data):
    tr = data["YMomentumDensity"]/data["Density"]
    return tr
add_field("y-velocity", function=_y_velocity,
          units="cm/s", take_log=False)


def _z_velocity(field, data):
    tr = data["ZMomentumDensity"]/data["Density"]
    return tr
add_field("z-velocity", function=_z_velocity,
          units="cm/s", take_log=False)


def _metal_density(field, data):
    tr = data["MetalDensitySNIa"]
    tr += data["MetalDensitySNII"]
    return tr
add_field("Metal_Density", function=_metal_density,
          units="Zsun", take_log=True)

##################################################
#Particle fields

def _setup_particle_fields(registry, ptype):
    for ax in 'xyz':
        registry.add_field(
                        (ptype, "particle_velocity_%s" % ax),
                        function=NullFunc,
                        convert_function=_get_conv("particle_velocity_%s" % ax),
                        particle_type=True)
        registry.add_field(
                        (ptype, "particle_position_%s" % ax),
                        function=NullFunc,
                        particle_type=True)

    def _convertParticleMass(data):
        return np.float64(data.convert('particle_mass'))
    registry.add_field((ptype, "particle_mass"),
              function=NullFunc,
              convert_function=_convertParticleMass,
              units=r"\rm{g}",
              particle_type=True)

    registry.add_field((ptype, "particle_mass_initial"),
              function=NullFunc,
              convert_function=_get_conv("particle_mass"),
              units=r"\rm{g}",
              particle_type=True)

    registry.add_field((ptype, "creation_time"),
            function=NullFunc,
            convert_function=_get_conv("particle_creation_time"),
            units=r"\rm{s}",
            particle_type=True)

    for field in ["particle_index",
                  "particle_species",
                  "particle_metallicity1",
                  "particle_metallicity2"]:
        registry.add_field((ptype, field),
            function=NullFunc, particle_type=True)

    particle_vector_functions(ptype,
        ["particle_position_%s" % ax for ax in 'xyz'],
        ["particle_velocity_%s" % ax for ax in 'xyz'],
        registry)
    particle_deposition_functions(ptype, "Coordinates", "particle_mass",
        registry)

#add_artio_field("creation_time", function=NullFunc, particle_type=True)
def _particle_age(field, data):
    pa = b2t(data['stars','creation_time'])
#    tr = np.zeros(pa.shape,dtype='float')-1.0
#    tr[pa>0] = pa[pa>0]
    tr = pa
    return tr
add_field(("stars","particle_age"), function=_particle_age, units=r"\rm{s}",
          particle_type=True)

def mass_dm(field, data):
    tr = np.ones(data.ActiveDimensions, dtype='float32')
    idx = data["particle_type"] < 5
    # dhr - particle types are strings...
    #make a dumb assumption that the mass is evenly spread out in the grid
    #must return an array the shape of the grid cells
    if np.sum(idx) > 0:
        #divide by the volume
        tr /= np.prod(data['CellVolumeCode']*data.pf['mpchcm']**3.0)
        #Multiply by total contaiend mass
        tr *= np.sum(data['particle_mass'][idx])*data.pf['Msun']
        #print tr.shape
        return tr
    else:
        return tr*1e-9
        #dhr - where does this 1e-9 come from?

add_field("particle_cell_mass_dm", function=mass_dm, units=r"\mathrm{M_{sun}}",
          validators=[ValidateSpatial(0)],
          take_log=False,
          projection_conversion="1")


def _spdensity(field, data):
    grid_mass = np.zeros(data.ActiveDimensions, dtype='float32')
    if data.star_mass.shape[0] == 0:
        return grid_mass
    amr_utils.CICDeposit_3(data.star_position_x,
                           data.star_position_y,
                           data.star_position_z,
                           data.star_mass.astype('float32'),
                           data.star_mass.shape[0],
                           grid_mass,
                           np.array(data.LeftEdge).astype(np.float64),
                           np.array(data.ActiveDimensions).astype(np.int32),
                           np.float64(data['dx']))
    return grid_mass

#add_field("star_density", function=_spdensity,
#          validators=[ValidateSpatial(0)], convert_function=_convertDensity)


def _simple_density(field, data):
    mass = np.sum(data.star_mass)
    volume = data['dx']*data.ActiveDimensions.prod().astype('float64')
    return mass/volume

add_field("star_density", function=_simple_density,
          validators=[ValidateSpatial(0)], convert_function=_convertDensity)


#stolen from frontends/art/
#All of these functions are to convert from hydro time var to
#proper time
sqrt = np.sqrt
sign = np.sign


def find_root(f, a, b, tol=1e-6):
    c = (a+b)/2.0
    last = -np.inf
    assert(sign(f(a)) != sign(f(b)))
    while np.abs(f(c)-last) > tol:
        last = f(c)
        if sign(last) == sign(f(b)):
            b = c
        else:
            a = c
        c = (a+b)/2.0
    return c


def quad(fintegrand, xmin, xmax, n=1e4):
    spacings = np.logspace(np.log10(xmin), np.log10(xmax), n)
    integrand_arr = fintegrand(spacings)
    val = np.trapz(integrand_arr, dx=np.diff(spacings))
    return val


def a2b(at, Om0=0.27, Oml0=0.73, h=0.700):
    def f_a2b(x):
        val = 0.5*sqrt(Om0) / x**3.0
        val /= sqrt(Om0/x**3.0 + Oml0 + (1.0-Om0-Oml0)/x**2.0)
        return val
    #val, err = si.quad(f_a2b,1,at)
    val = quad(f_a2b, 1, at)
    return val


def b2a(bt, **kwargs):
    #converts code time into expansion factor
    #if Om0 ==1and OmL == 0 then b2a is (1 / (1-td))**2
    #if bt < -190.0 or bt > -.10:  raise 'bt outside of range'
    f_b2a = lambda at: a2b(at, **kwargs)-bt
    return find_root(f_b2a, 1e-4, 1.1)
    #return so.brenth(f_b2a,1e-4,1.1)
    #return brent.brent(f_b2a)


def a2t(at, Om0=0.27, Oml0=0.73, h=0.700):
    integrand = lambda x: 1./(x*sqrt(Oml0+Om0*x**-3.0))
    #current_time,err = si.quad(integrand,0.0,at,epsabs=1e-6,epsrel=1e-6)
    current_time = quad(integrand, 1e-4, at)
    #spacings = np.logspace(-5,np.log10(at),1e5)
    #integrand_arr = integrand(spacings)
    #current_time = np.trapz(integrand_arr,dx=np.diff(spacings))
    current_time *= 9.779/h
    return current_time


def b2t(tb, n=1e2, logger=None, **kwargs):
    tb = np.array(tb)
    if len(np.atleast_1d(tb)) == 1: 
        return a2t(b2a(tb))
    if tb.shape == ():
        return None 
    if len(tb) < n:
        n = len(tb)
    age_min = a2t(b2a(tb.max(), **kwargs), **kwargs)
    age_max = a2t(b2a(tb.min(), **kwargs), **kwargs)
    tbs = -1.*np.logspace(np.log10(-tb.min()),
                          np.log10(-tb.max()), n)
    ages = []
    for i, tbi in enumerate(tbs):
        ages += a2t(b2a(tbi)),
        if logger:
            logger(i)
    ages = np.array(ages)
    fb2t = np.interp(tb, tbs, ages)
    #fb2t = interp1d(tbs,ages)
    return fb2t*1e9*31556926


def spread_ages(ages, logger=None, spread=.0e7*365*24*3600):
    #stars are formed in lumps; spread out the ages linearly
    da = np.diff(ages)
    assert np.all(da <= 0)
    #ages should always be decreasing, and ordered so
    agesd = np.zeros(ages.shape)
    idx, = np.where(da < 0)
    idx += 1
    #mark the right edges
    #spread this age evenly out to the next age
    lidx = 0
    lage = 0
    for i in idx:
        n = i-lidx
        #n stars affected
        rage = ages[i]
        lage = max(rage-spread, 0.0)
        agesd[lidx:i] = np.linspace(lage, rage, n)
        lidx = i
        #lage=rage
        if logger:
            logger(i)
    #we didn't get the last iter
    i = ages.shape[0]-1
    n = i-lidx
    #n stars affected
    rage = ages[i]
    lage = max(rage-spread, 0.0)
    agesd[lidx:i] = np.linspace(lage, rage, n)
    return agesd
