"""
The basic field info container resides here.  These classes, code specific and
universal, are the means by which we access fields across YT, both derived and
native.



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import types
import numpy as np
import inspect
import copy

from yt.funcs import *

from yt.utilities.lib import CICDeposit_3, obtain_rvec, obtain_rv_vec
from yt.utilities.cosmology import Cosmology
from field_info_container import \
    add_field, \
    ValidateDataField, \
    ValidateGridType, \
    ValidateParameter, \
    ValidateSpatial, \
    NeedsGridType, \
    NeedsOriginalGrid, \
    NeedsDataField, \
    NeedsProperty, \
    NeedsParameter, \
    NullFunc

from yt.utilities.physical_constants import \
     mh, \
     me, \
     sigma_thompson, \
     clight, \
     kboltz, \
     G, \
     rho_crit_now, \
     speed_of_light_cgs, \
     km_per_cm, keV_per_K

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
     
# Note that, despite my newfound efforts to comply with PEP-8,
# I violate it here in order to keep the name/func_name relationship

def _dx(field, data):
    return data.dds[0]
    return np.ones(data.ActiveDimensions, dtype='float64') * data.dds[0]
add_field('dx', function=_dx, display_field=False,
          validators=[ValidateSpatial(0)])

def _dy(field, data):
    return data.dds[1]
    return np.ones(data.ActiveDimensions, dtype='float64') * data.dds[1]
add_field('dy', function=_dy, display_field=False,
          validators=[ValidateSpatial(0)])

def _dz(field, data):
    return data.dds[2]
    return np.ones(data.ActiveDimensions, dtype='float64') * data.dds[2]
add_field('dz', function=_dz,
          display_field=False, validators=[ValidateSpatial(0)])

def _coordX(field, data):
    dim = data.ActiveDimensions[0]
    return (np.ones(data.ActiveDimensions, dtype='float64')
                   * np.arange(data.ActiveDimensions[0])[:,None,None]
            +0.5) * data['dx'] + data.LeftEdge[0]
add_field('x', function=_coordX, display_field=False,
          validators=[ValidateSpatial(0)])

def _coordY(field, data):
    dim = data.ActiveDimensions[1]
    return (np.ones(data.ActiveDimensions, dtype='float64')
                   * np.arange(data.ActiveDimensions[1])[None,:,None]
            +0.5) * data['dy'] + data.LeftEdge[1]
add_field('y', function=_coordY, display_field=False,
          validators=[ValidateSpatial(0)])

def _coordZ(field, data):
    dim = data.ActiveDimensions[2]
    return (np.ones(data.ActiveDimensions, dtype='float64')
                   * np.arange(data.ActiveDimensions[2])[None,None,:]
            +0.5) * data['dz'] + data.LeftEdge[2]
add_field('z', function=_coordZ, display_field=False,
          validators=[ValidateSpatial(0)])

def _GridLevel(field, data):
    return np.ones(data.ActiveDimensions)*(data.Level)
add_field("GridLevel", function=_GridLevel,
          validators=[ValidateGridType(),
                      ValidateSpatial(0)])

def _GridIndices(field, data):
    return np.ones(data["Ones"].shape)*(data.id-data._id_offset)
add_field("GridIndices", function=_GridIndices,
          validators=[ValidateGridType(),
                      ValidateSpatial(0)], take_log=False)

def _OnesOverDx(field, data):
    return np.ones(data["Ones"].shape,
                   dtype=data["Density"].dtype)/data['dx']
add_field("OnesOverDx", function=_OnesOverDx,
          display_field=False)

def _Zeros(field, data):
    return np.zeros(data.ActiveDimensions, dtype='float64')
add_field("Zeros", function=_Zeros,
          validators=[ValidateSpatial(0)],
          projection_conversion="unitary",
          display_field = False)

def _Ones(field, data):
    return np.ones(data.ActiveDimensions, dtype='float64')
add_field("Ones", function=_Ones,
          validators=[ValidateSpatial(0)],
          projection_conversion="unitary",
          display_field = False)
add_field("CellsPerBin", function=_Ones, validators=[ValidateSpatial(0)],
          display_field = False)

def _SoundSpeed(field, data):
    if data.pf["EOSType"] == 1:
        return np.ones(data["Density"].shape, dtype='float64') * \
                data.pf["EOSSoundSpeed"]
    return ( data.pf["Gamma"]*data["Pressure"] / \
             data["Density"] )**(1.0/2.0)
add_field("SoundSpeed", function=_SoundSpeed,
          units=r"\rm{cm}/\rm{s}")

def _RadialMachNumber(field, data):
    """M{|v|/t_sound}"""
    return np.abs(data["RadialVelocity"]) / data["SoundSpeed"]
add_field("RadialMachNumber", function=_RadialMachNumber)

def _MachNumber(field, data):
    """M{|v|/t_sound}"""
    return data["VelocityMagnitude"] / data["SoundSpeed"]
add_field("MachNumber", function=_MachNumber)

def _CourantTimeStep(field, data):
    t1 = data['dx'] / (
        data["SoundSpeed"] + \
        abs(data["x-velocity"]))
    t2 = data['dy'] / (
        data["SoundSpeed"] + \
        abs(data["y-velocity"]))
    t3 = data['dz'] / (
        data["SoundSpeed"] + \
        abs(data["z-velocity"]))
    return np.minimum(np.minimum(t1,t2),t3)
def _convertCourantTimeStep(data):
    # SoundSpeed and z-velocity are in cm/s, dx is in code
    return data.convert("cm")
add_field("CourantTimeStep", function=_CourantTimeStep,
          convert_function=_convertCourantTimeStep,
          units=r"$\rm{s}$")

def _ParticleVelocityMagnitude(field, data):
    """M{|v|}"""
    bulk_velocity = data.get_field_parameter("bulk_velocity")
    if bulk_velocity == None:
        bulk_velocity = np.zeros(3)
    return ( (data["particle_velocity_x"]-bulk_velocity[0])**2.0 + \
             (data["particle_velocity_y"]-bulk_velocity[1])**2.0 + \
             (data["particle_velocity_z"]-bulk_velocity[2])**2.0 )**(1.0/2.0)
add_field("ParticleVelocityMagnitude", function=_ParticleVelocityMagnitude,
          particle_type=True, 
          take_log=False, units=r"\rm{cm}/\rm{s}")

def _VelocityMagnitude(field, data):
    """M{|v|}"""
    velocities = obtain_rv_vec(data)
    return np.sqrt(np.sum(velocities**2,axis=0))
add_field("VelocityMagnitude", function=_VelocityMagnitude,
          take_log=False, units=r"\rm{cm}/\rm{s}")

def _TangentialOverVelocityMagnitude(field, data):
    return np.abs(data["TangentialVelocity"])/np.abs(data["VelocityMagnitude"])
add_field("TangentialOverVelocityMagnitude",
          function=_TangentialOverVelocityMagnitude,
          take_log=False)

def _Pressure(field, data):
    """M{(Gamma-1.0)*rho*E}"""
    return (data.pf["Gamma"] - 1.0) * \
           data["Density"] * data["ThermalEnergy"]
add_field("Pressure", function=_Pressure, units=r"\rm{dyne}/\rm{cm}^{2}")

def _TempkeV(field, data):
    return data["Temperature"] * keV_per_K
add_field("TempkeV", function=_TempkeV, units=r"\rm{keV}",
          display_name="Temperature")

def _Entropy(field, data):
    if data.has_field_parameter("mu"):
        mw = mh*data.get_field_parameter("mu")
    else :
        mw = mh
    try:
        gammam1 = data.pf["Gamma"] - 1.0
    except:
        gammam1 = 5./3. - 1.0
    return kboltz * data["Temperature"] / \
           ((data["Density"]/mw)**gammam1)
add_field("Entropy", units=r"\rm{ergs}\ \rm{cm}^{3\gamma-3}",
          function=_Entropy)

### spherical coordinates: r (radius)
def _sph_r(field, data):
    center = data.get_field_parameter("center")
      
    coords = obtain_rvec(data)

    return get_sph_r(coords)

def _Convert_sph_r_CGS(data):
   return data.convert("cm")

add_field("sph_r", function=_sph_r,
         validators=[ValidateParameter("center")],
         convert_function = _Convert_sph_r_CGS, units=r"\rm{cm}")


### spherical coordinates: theta (angle with respect to normal)
def _sph_theta(field, data):
    center = data.get_field_parameter("center")
    normal = data.get_field_parameter("normal")
    
    coords = obtain_rvec(data)

    return get_sph_theta(coords, normal)

add_field("sph_theta", function=_sph_theta,
         validators=[ValidateParameter("center"),ValidateParameter("normal")])


### spherical coordinates: phi (angle in the plane perpendicular to the normal)
def _sph_phi(field, data):
    center = data.get_field_parameter("center")
    normal = data.get_field_parameter("normal")
    
    coords = obtain_rvec(data)

    return get_sph_phi(coords, normal)

add_field("sph_phi", function=_sph_phi,
         validators=[ValidateParameter("center"),ValidateParameter("normal")])

### cylindrical coordinates: R (radius in the cylinder's plane)
def _cyl_R(field, data):
    center = data.get_field_parameter("center")
    normal = data.get_field_parameter("normal")
      
    coords = obtain_rvec(data)

    return get_cyl_r(coords, normal)

def _Convert_cyl_R_CGS(data):
   return data.convert("cm")

add_field("cyl_R", function=_cyl_R,
         validators=[ValidateParameter("center"),ValidateParameter("normal")],
         convert_function = _Convert_cyl_R_CGS, units=r"\rm{cm}")
add_field("cyl_RCode", function=_cyl_R,
          validators=[ValidateParameter("center"),ValidateParameter("normal")],
          units=r"Radius (code)")


### cylindrical coordinates: z (height above the cylinder's plane)
def _cyl_z(field, data):
    center = data.get_field_parameter("center")
    normal = data.get_field_parameter("normal")
    
    coords = obtain_rvec(data)

    return get_cyl_z(coords, normal)

def _Convert_cyl_z_CGS(data):
   return data.convert("cm")

add_field("cyl_z", function=_cyl_z,
         validators=[ValidateParameter("center"),ValidateParameter("normal")],
         convert_function = _Convert_cyl_z_CGS, units=r"\rm{cm}")


### cylindrical coordinates: theta (angle in the cylinder's plane)
def _cyl_theta(field, data):
    center = data.get_field_parameter("center")
    normal = data.get_field_parameter("normal")
    
    coords = obtain_rvec(data)

    return get_cyl_theta(coords, normal)

add_field("cyl_theta", function=_cyl_theta,
         validators=[ValidateParameter("center"),ValidateParameter("normal")])

### The old field DiskAngle is the same as the spherical coordinates'
### 'theta' angle. I'm keeping DiskAngle for backwards compatibility.
def _DiskAngle(field, data):
    return data['sph_theta']

add_field("DiskAngle", function=_DiskAngle,
          take_log=False,
          validators=[ValidateParameter("center"),
                      ValidateParameter("normal")],
          display_field=False)


### The old field Height is the same as the cylindrical coordinates' z
### field. I'm keeping Height for backwards compatibility.
def _Height(field, data):
    return data['cyl_z']

def _convertHeight(data):
    return data.convert("cm")
def _convertHeightAU(data):
    return data.convert("au")
add_field("Height", function=_Height,
          convert_function=_convertHeight,
          validators=[ValidateParameter("center"),
                      ValidateParameter("normal")],
          units=r"cm", display_field=False)
add_field("HeightAU", function=_Height,
          convert_function=_convertHeightAU,
          validators=[ValidateParameter("center"),
                      ValidateParameter("normal")],
          units=r"AU", display_field=False)

def _cyl_RadialVelocity(field, data):
    normal = data.get_field_parameter("normal")
    velocities = obtain_rv_vec(data)

    theta = data['cyl_theta']

    return get_cyl_r_component(velocities, theta, normal)

def _cyl_RadialVelocityABS(field, data):
    return np.abs(_cyl_RadialVelocity(field, data))
def _Convert_cyl_RadialVelocityKMS(data):
    return km_per_cm
add_field("cyl_RadialVelocity", function=_cyl_RadialVelocity,
          units=r"\rm{cm}/\rm{s}",
          validators=[ValidateParameter("normal")])
add_field("cyl_RadialVelocityABS", function=_cyl_RadialVelocityABS,
          units=r"\rm{cm}/\rm{s}",
          validators=[ValidateParameter("normal")])
add_field("cyl_RadialVelocityKMS", function=_cyl_RadialVelocity,
          convert_function=_Convert_cyl_RadialVelocityKMS, units=r"\rm{km}/\rm{s}",
          validators=[ValidateParameter("normal")])
add_field("cyl_RadialVelocityKMSABS", function=_cyl_RadialVelocityABS,
          convert_function=_Convert_cyl_RadialVelocityKMS, units=r"\rm{km}/\rm{s}",
          validators=[ValidateParameter("normal")])

def _cyl_TangentialVelocity(field, data):
    normal = data.get_field_parameter("normal")
    velocities = obtain_rv_vec(data)
    theta = data['cyl_theta']

    return get_cyl_theta_component(velocities, theta, normal)

def _cyl_TangentialVelocityABS(field, data):
    return np.abs(_cyl_TangentialVelocity(field, data))
def _Convert_cyl_TangentialVelocityKMS(data):
    return km_per_cm
add_field("cyl_TangentialVelocity", function=_cyl_TangentialVelocity,
          units=r"\rm{cm}/\rm{s}",
          validators=[ValidateParameter("normal")])
add_field("cyl_TangentialVelocityABS", function=_cyl_TangentialVelocityABS,
          units=r"\rm{cm}/\rm{s}",
          validators=[ValidateParameter("normal")])
add_field("cyl_TangentialVelocityKMS", function=_cyl_TangentialVelocity,
          convert_function=_Convert_cyl_TangentialVelocityKMS, units=r"\rm{km}/\rm{s}",
          validators=[ValidateParameter("normal")])
add_field("cyl_TangentialVelocityKMSABS", function=_cyl_TangentialVelocityABS,
          convert_function=_Convert_cyl_TangentialVelocityKMS, units=r"\rm{km}/\rm{s}",
          validators=[ValidateParameter("normal")])

def _DynamicalTime(field, data):
    """
    The formulation for the dynamical time is:
    M{sqrt(3pi/(16*G*rho))} or M{sqrt(3pi/(16G))*rho^-(1/2)}
    Note that we return in our natural units already
    """
    return (3.0*np.pi/(16*G*data["Density"]))**(1./2.)
add_field("DynamicalTime", function=_DynamicalTime,
           units=r"\rm{s}")

def JeansMassMsun(field,data):
    return (MJ_constant * 
            ((data["Temperature"]/data["MeanMolecularWeight"])**(1.5)) *
            (data["Density"]**(-0.5)))
add_field("JeansMassMsun",function=JeansMassMsun,units=r"\rm{Msun}")

def _CellMass(field, data):
    return data["Density"] * data["CellVolume"]
def _convertCellMassMsun(data):
    return 5.027854e-34 # g^-1
add_field("CellMass", function=_CellMass, units=r"\rm{g}")
add_field("CellMassMsun", units=r"M_{\odot}",
          function=_CellMass,
          convert_function=_convertCellMassMsun)

def _CellMassCode(field, data):
    return data["Density"] * data["CellVolumeCode"]
def _convertCellMassCode(data):
    return 1.0/data.convert("Density")
add_field("CellMassCode", 
          function=_CellMassCode,
          convert_function=_convertCellMassCode)

def _TotalMass(field,data):
    return (data["Density"]+data["particle_density"]) * data["CellVolume"]
add_field("TotalMass", function=_TotalMass, units=r"\rm{g}")
add_field("TotalMassMsun", units=r"M_{\odot}",
          function=_TotalMass,
          convert_function=_convertCellMassMsun)

def _StarMass(field,data):
    return data["star_density"] * data["CellVolume"]
add_field("StarMassMsun", units=r"M_{\odot}",
          function=_StarMass,
          convert_function=_convertCellMassMsun)

def _Matter_Density(field,data):
    return (data['Density'] + data['particle_density'])
add_field("Matter_Density",function=_Matter_Density,units=r"\rm{g}/\rm{cm^3}")

def _ComovingDensity(field, data):
    ef = (1.0 + data.pf.current_redshift)**3.0
    return data["Density"]/ef
add_field("ComovingDensity", function=_ComovingDensity, units=r"\rm{g}/\rm{cm}^3")

# This is rho_total / rho_cr(z).
def _Convert_Overdensity(data):
    return 1.0 / (rho_crit_now * data.pf.hubble_constant**2 * 
                (1+data.pf.current_redshift)**3)
add_field("Overdensity",function=_Matter_Density,
          convert_function=_Convert_Overdensity, units=r"")

# This is (rho_total - <rho_total>) / <rho_total>.
def _DensityPerturbation(field, data):
    rho_bar = rho_crit_now * data.pf.omega_matter * \
        data.pf.hubble_constant**2 * \
        (1.0 + data.pf.current_redshift)**3
    return ((data['Matter_Density'] - rho_bar) / rho_bar)
add_field("DensityPerturbation",function=_DensityPerturbation,units=r"")

# This is rho_b / <rho_b>.
def _Baryon_Overdensity(field, data):
    if data.pf.has_key('omega_baryon_now'):
        omega_baryon_now = data.pf['omega_baryon_now']
    else:
        omega_baryon_now = 0.0441
    return data['Density'] / (omega_baryon_now * rho_crit_now * 
                              (data.pf.hubble_constant**2) * 
                              ((1+data.pf.current_redshift)**3))
add_field("Baryon_Overdensity", function=_Baryon_Overdensity, 
          units=r"")

# Weak lensing convergence.
# Eqn 4 of Metzler, White, & Loken (2001, ApJ, 547, 560).
def _convertConvergence(data):
    if not data.pf.parameters.has_key('cosmology_calculator'):
        data.pf.parameters['cosmology_calculator'] = Cosmology(
            HubbleConstantNow=(100.*data.pf.hubble_constant),
            OmegaMatterNow=data.pf.omega_matter, OmegaLambdaNow=data.pf.omega_lambda)
    # observer to lens
    DL = data.pf.parameters['cosmology_calculator'].AngularDiameterDistance(
        data.pf.parameters['observer_redshift'], data.pf.current_redshift)
    # observer to source
    DS = data.pf.parameters['cosmology_calculator'].AngularDiameterDistance(
        data.pf.parameters['observer_redshift'], data.pf.parameters['lensing_source_redshift'])
    # lens to source
    DLS = data.pf.parameters['cosmology_calculator'].AngularDiameterDistance(
        data.pf.current_redshift, data.pf.parameters['lensing_source_redshift'])
    return (((DL * DLS) / DS) * (1.5e14 * data.pf.omega_matter * 
                                (data.pf.hubble_constant / speed_of_light_cgs)**2 *
                                (1 + data.pf.current_redshift)))
add_field("WeakLensingConvergence", function=_DensityPerturbation, 
          convert_function=_convertConvergence, 
          projection_conversion='mpccm')

def _CellVolume(field, data):
    if data['dx'].size == 1:
        try:
            return data['dx'] * data['dy'] * data['dz'] * \
                np.ones(data.ActiveDimensions, dtype='float64')
        except AttributeError:
            return data['dx'] * data['dy'] * data['dz']
    return data["dx"] * data["dy"] * data["dz"]
def _ConvertCellVolumeMpc(data):
    return data.convert("mpc")**3.0
def _ConvertCellVolumeCGS(data):
    return data.convert("cm")**3.0
add_field("CellVolumeCode", units=r"\rm{BoxVolume}^3",
          function=_CellVolume)
add_field("CellVolumeMpc", units=r"\rm{Mpc}^3",
          function=_CellVolume,
          convert_function=_ConvertCellVolumeMpc)
add_field("CellVolume", units=r"\rm{cm}^3",
          function=_CellVolume,
          convert_function=_ConvertCellVolumeCGS)

def _ChandraEmissivity(field, data):
    logT0 = np.log10(data["Temperature"]) - 7
    return ((data["NumberDensity"].astype('float64')**2.0) \
            *(10**(-0.0103*logT0**8 \
                   +0.0417*logT0**7 \
                   -0.0636*logT0**6 \
                   +0.1149*logT0**5 \
                   -0.3151*logT0**4 \
                   +0.6655*logT0**3 \
                   -1.1256*logT0**2 \
                   +1.0026*logT0**1 \
                   -0.6984*logT0) \
              +data["Metallicity"]*10**(0.0305*logT0**11 \
                                        -0.0045*logT0**10 \
                                        -0.3620*logT0**9 \
                                        +0.0513*logT0**8 \
                                        +1.6669*logT0**7 \
                                        -0.3854*logT0**6 \
                                        -3.3604*logT0**5 \
                                        +0.4728*logT0**4 \
                                        +4.5774*logT0**3 \
                                        -2.3661*logT0**2 \
                                        -1.6667*logT0**1 \
                                        -0.2193*logT0)))
def _convertChandraEmissivity(data):
    return 1.0 #1.0e-23*0.76**2
add_field("ChandraEmissivity", function=_ChandraEmissivity,
          convert_function=_convertChandraEmissivity,
          projection_conversion="1")

def _XRayEmissivity(field, data):
    return ((data["Density"].astype('float64')**2.0) \
            *data["Temperature"]**0.5)
def _convertXRayEmissivity(data):
    return 2.168e60
add_field("XRayEmissivity", function=_XRayEmissivity,
          convert_function=_convertXRayEmissivity,
          projection_conversion="1")

def _SZKinetic(field, data):
    vel_axis = data.get_field_parameter('axis')
    if vel_axis > 2:
        raise NeedsParameter(['axis'])
    vel = data["%s-velocity" % ({0:'x',1:'y',2:'z'}[vel_axis])]
    return (vel*data["Density"])
def _convertSZKinetic(data):
    return 0.88*((sigma_thompson/mh)/clight)
add_field("SZKinetic", function=_SZKinetic,
          convert_function=_convertSZKinetic,
          validators=[ValidateParameter('axis')])

def _SZY(field, data):
    return (data["Density"]*data["Temperature"])
def _convertSZY(data):
    conv = (0.88/mh) * (kboltz)/(me * clight*clight) * sigma_thompson
    return conv
add_field("SZY", function=_SZY, convert_function=_convertSZY)

def _AveragedDensity(field, data):
    nx, ny, nz = data["Density"].shape
    new_field = np.zeros((nx-2,ny-2,nz-2), dtype='float64')
    weight_field = np.zeros((nx-2,ny-2,nz-2), dtype='float64')
    i_i, j_i, k_i = np.mgrid[0:3,0:3,0:3]
    for i,j,k in zip(i_i.ravel(),j_i.ravel(),k_i.ravel()):
        sl = [slice(i,nx-(2-i)),slice(j,ny-(2-j)),slice(k,nz-(2-k))]
        new_field += data["Density"][sl] * data["CellMass"][sl]
        weight_field += data["CellMass"][sl]
    # Now some fancy footwork
    new_field2 = np.zeros((nx,ny,nz))
    new_field2[1:-1,1:-1,1:-1] = new_field/weight_field
    return new_field2
add_field("AveragedDensity",
          function=_AveragedDensity,
          validators=[ValidateSpatial(1, ["Density"])])

def _DivV(field, data):
    # We need to set up stencils
    if data.pf["HydroMethod"] == 2:
        sl_left = slice(None,-2,None)
        sl_right = slice(1,-1,None)
        div_fac = 1.0
    else:
        sl_left = slice(None,-2,None)
        sl_right = slice(2,None,None)
        div_fac = 2.0
    ds = div_fac * data['dx'].flat[0]
    f  = data["x-velocity"][sl_right,1:-1,1:-1]/ds
    f -= data["x-velocity"][sl_left ,1:-1,1:-1]/ds
    if data.pf.dimensionality > 1:
        ds = div_fac * data['dy'].flat[0]
        f += data["y-velocity"][1:-1,sl_right,1:-1]/ds
        f -= data["y-velocity"][1:-1,sl_left ,1:-1]/ds
    if data.pf.dimensionality > 2:
        ds = div_fac * data['dz'].flat[0]
        f += data["z-velocity"][1:-1,1:-1,sl_right]/ds
        f -= data["z-velocity"][1:-1,1:-1,sl_left ]/ds
    new_field = np.zeros(data["x-velocity"].shape, dtype='float64')
    new_field[1:-1,1:-1,1:-1] = f
    return new_field
def _convertDivV(data):
    return data.convert("cm")**-1.0
add_field("DivV", function=_DivV,
            validators=[ValidateSpatial(1,
            ["x-velocity","y-velocity","z-velocity"])],
          units=r"\rm{s}^{-1}", take_log=False,
          convert_function=_convertDivV)

def _AbsDivV(field, data):
    return np.abs(data['DivV'])
add_field("AbsDivV", function=_AbsDivV,
          units=r"\rm{s}^{-1}")

def _Contours(field, data):
    return -np.ones_like(data["Ones"])
add_field("Contours", validators=[ValidateSpatial(0)], take_log=False,
          display_field=False, function=_Contours)
add_field("tempContours", function=_Contours,
          validators=[ValidateSpatial(0), ValidateGridType()],
          take_log=False, display_field=False)

def obtain_velocities(data):
    return obtain_rv_vec(data)

def _convertSpecificAngularMomentum(data):
    return data.convert("cm")
def _convertSpecificAngularMomentumKMSMPC(data):
    return data.convert("mpc")/1e5

def _SpecificAngularMomentumX(field, data):
    xv, yv, zv = obtain_velocities(data)
    rv = obtain_rvec(data)
    return yv*rv[2,:] - zv*rv[1,:]
def _SpecificAngularMomentumY(field, data):
    xv, yv, zv = obtain_velocities(data)
    rv = obtain_rvec(data)
    return -(xv*rv[2,:] - zv*rv[0,:])
def _SpecificAngularMomentumZ(field, data):
    xv, yv, zv = obtain_velocities(data)
    rv = obtain_rvec(data)
    return xv*rv[1,:] - yv*rv[0,:]
for ax in 'XYZ':
    n = "SpecificAngularMomentum%s" % ax
    add_field(n, function=eval("_%s" % n),
              convert_function=_convertSpecificAngularMomentum,
              units=r"\rm{cm}^2/\rm{s}", validators=[ValidateParameter("center")])

def _AngularMomentumX(field, data):
    return data["CellMass"] * data["SpecificAngularMomentumX"]
add_field("AngularMomentumX", function=_AngularMomentumX,
         units=r"\rm{g}\/\rm{cm}^2/\rm{s}", vector_field=False,
         validators=[ValidateParameter('center')])
def _AngularMomentumY(field, data):
    return data["CellMass"] * data["SpecificAngularMomentumY"]
add_field("AngularMomentumY", function=_AngularMomentumY,
         units=r"\rm{g}\/\rm{cm}^2/\rm{s}", vector_field=False,
         validators=[ValidateParameter('center')])
def _AngularMomentumZ(field, data):
    return data["CellMass"] * data["SpecificAngularMomentumZ"]
add_field("AngularMomentumZ", function=_AngularMomentumZ,
         units=r"\rm{g}\/\rm{cm}^2/\rm{s}", vector_field=False,
         validators=[ValidateParameter('center')])

def _ParticleSpecificAngularMomentum(field, data):
    """
    Calculate the angular of a particle velocity.  Returns a vector for each
    particle.
    """
    if data.has_field_parameter("bulk_velocity"):
        bv = data.get_field_parameter("bulk_velocity")
    else: bv = np.zeros(3, dtype='float64')
    xv = data["particle_velocity_x"] - bv[0]
    yv = data["particle_velocity_y"] - bv[1]
    zv = data["particle_velocity_z"] - bv[2]
    center = data.get_field_parameter('center')
    coords = np.array([data['particle_position_x'],
                       data['particle_position_y'],
                       data['particle_position_z']], dtype='float64')
    new_shape = tuple([3] + [1]*(len(coords.shape)-1))
    r_vec = coords - np.reshape(center,new_shape)
    v_vec = np.array([xv,yv,zv], dtype='float64')
    return np.cross(r_vec, v_vec, axis=0)
#add_field("ParticleSpecificAngularMomentum",
#          function=_ParticleSpecificAngularMomentum, particle_type=True,
#          convert_function=_convertSpecificAngularMomentum, vector_field=True,
#          units=r"\rm{cm}^2/\rm{s}", validators=[ValidateParameter('center')])
def _convertSpecificAngularMomentumKMSMPC(data):
    return km_per_cm*data.convert("mpc")
#add_field("ParticleSpecificAngularMomentumKMSMPC",
#          function=_ParticleSpecificAngularMomentum, particle_type=True,
#          convert_function=_convertSpecificAngularMomentumKMSMPC, vector_field=True,
#          units=r"\rm{km}\rm{Mpc}/\rm{s}", validators=[ValidateParameter('center')])

def _ParticleSpecificAngularMomentumX(field, data):
    if data.has_field_parameter("bulk_velocity"):
        bv = data.get_field_parameter("bulk_velocity")
    else: bv = np.zeros(3, dtype='float64')
    center = data.get_field_parameter('center')
    y = data["particle_position_y"] - center[1]
    z = data["particle_position_z"] - center[2]
    yv = data["particle_velocity_y"] - bv[1]
    zv = data["particle_velocity_z"] - bv[2]
    return yv*z - zv*y
def _ParticleSpecificAngularMomentumY(field, data):
    if data.has_field_parameter("bulk_velocity"):
        bv = data.get_field_parameter("bulk_velocity")
    else: bv = np.zeros(3, dtype='float64')
    center = data.get_field_parameter('center')
    x = data["particle_position_x"] - center[0]
    z = data["particle_position_z"] - center[2]
    xv = data["particle_velocity_x"] - bv[0]
    zv = data["particle_velocity_z"] - bv[2]
    return -(xv*z - zv*x)
def _ParticleSpecificAngularMomentumZ(field, data):
    if data.has_field_parameter("bulk_velocity"):
        bv = data.get_field_parameter("bulk_velocity")
    else: bv = np.zeros(3, dtype='float64')
    center = data.get_field_parameter('center')
    x = data["particle_position_x"] - center[0]
    y = data["particle_position_y"] - center[1]
    xv = data["particle_velocity_x"] - bv[0]
    yv = data["particle_velocity_y"] - bv[1]
    return xv*y - yv*x
for ax in 'XYZ':
    n = "ParticleSpecificAngularMomentum%s" % ax
    add_field(n, function=eval("_%s" % n), particle_type=True,
              convert_function=_convertSpecificAngularMomentum,
              units=r"\rm{cm}^2/\rm{s}", validators=[ValidateParameter("center")])
    add_field(n + "KMSMPC", function=eval("_%s" % n), particle_type=True,
              convert_function=_convertSpecificAngularMomentumKMSMPC,
              units=r"\rm{cm}^2/\rm{s}", validators=[ValidateParameter("center")])

def _ParticleAngularMomentum(field, data):
    return data["ParticleMass"] * data["ParticleSpecificAngularMomentum"]
#add_field("ParticleAngularMomentum",
#          function=_ParticleAngularMomentum, units=r"\rm{g}\/\rm{cm}^2/\rm{s}",
#          particle_type=True, validators=[ValidateParameter('center')])
def _ParticleAngularMomentumMSUNKMSMPC(field, data):
    return data["ParticleMass"] * data["ParticleSpecificAngularMomentumKMSMPC"]
#add_field("ParticleAngularMomentumMSUNKMSMPC",
#          function=_ParticleAngularMomentumMSUNKMSMPC,
#          units=r"M_{\odot}\rm{km}\rm{Mpc}/\rm{s}",
#          particle_type=True, validators=[ValidateParameter('center')])

def _ParticleAngularMomentumX(field, data):
    return data["CellMass"] * data["ParticleSpecificAngularMomentumX"]
add_field("ParticleAngularMomentumX", function=_ParticleAngularMomentumX,
         units=r"\rm{g}\/\rm{cm}^2/\rm{s}", particle_type=True,
         validators=[ValidateParameter('center')])
def _ParticleAngularMomentumY(field, data):
    return data["CellMass"] * data["ParticleSpecificAngularMomentumY"]
add_field("ParticleAngularMomentumY", function=_ParticleAngularMomentumY,
         units=r"\rm{g}\/\rm{cm}^2/\rm{s}", particle_type=True,
         validators=[ValidateParameter('center')])
def _ParticleAngularMomentumZ(field, data):
    return data["CellMass"] * data["ParticleSpecificAngularMomentumZ"]
add_field("ParticleAngularMomentumZ", function=_ParticleAngularMomentumZ,
         units=r"\rm{g}\/\rm{cm}^2/\rm{s}", particle_type=True,
         validators=[ValidateParameter('center')])

def get_radius(data, field_prefix):
    center = data.get_field_parameter("center")
    DW = data.pf.domain_right_edge - data.pf.domain_left_edge
    radius = np.zeros(data[field_prefix+"x"].shape, dtype='float64')
    r = radius.copy()
    if any(data.pf.periodicity):
        rdw = radius.copy()
    for i, ax in enumerate('xyz'):
        np.subtract(data["%s%s" % (field_prefix, ax)], center[i], r)
        if data.pf.dimensionality < i+1:
            break
        if data.pf.periodicity[i] == True:
            np.abs(r, r)
            np.subtract(r, DW[i], rdw)
            np.abs(rdw, rdw)
            np.minimum(r, rdw, r)
        np.power(r, 2.0, r)
        np.add(radius, r, radius)
    np.sqrt(radius, radius)
    return radius

def _ParticleRadius(field, data):
    return get_radius(data, "particle_position_")
def _Radius(field, data):
    return get_radius(data, "")

def _ConvertRadiusCGS(data):
    return data.convert("cm")
add_field("ParticleRadius", function=_ParticleRadius,
          validators=[ValidateParameter("center")],
          convert_function = _ConvertRadiusCGS, units=r"\rm{cm}",
          particle_type = True,
          display_name = "Particle Radius")
add_field("Radius", function=_Radius,
          validators=[ValidateParameter("center")],
          convert_function = _ConvertRadiusCGS, units=r"\rm{cm}")

def _ConvertRadiusMpc(data):
    return data.convert("mpc")
add_field("RadiusMpc", function=_Radius,
          validators=[ValidateParameter("center")],
          convert_function = _ConvertRadiusMpc, units=r"\rm{Mpc}",
          display_name = "Radius")
add_field("ParticleRadiusMpc", function=_ParticleRadius,
          validators=[ValidateParameter("center")],
          convert_function = _ConvertRadiusMpc, units=r"\rm{Mpc}",
          particle_type=True,
          display_name = "Particle Radius")

def _ConvertRadiuskpc(data):
    return data.convert("kpc")
add_field("ParticleRadiuskpc", function=_ParticleRadius,
          validators=[ValidateParameter("center")],
          convert_function = _ConvertRadiuskpc, units=r"\rm{kpc}",
          particle_type=True,
          display_name = "Particle Radius")
add_field("Radiuskpc", function=_Radius,
          validators=[ValidateParameter("center")],
          convert_function = _ConvertRadiuskpc, units=r"\rm{kpc}",
          display_name = "Radius")

def _ConvertRadiuskpch(data):
    return data.convert("kpch")
add_field("ParticleRadiuskpch", function=_ParticleRadius,
          validators=[ValidateParameter("center")],
          convert_function = _ConvertRadiuskpch, units=r"\rm{kpc}/\rm{h}",
          particle_type=True,
          display_name = "Particle Radius")
add_field("Radiuskpch", function=_Radius,
          validators=[ValidateParameter("center")],
          convert_function = _ConvertRadiuskpc, units=r"\rm{kpc}/\rm{h}",
          display_name = "Radius")

def _ConvertRadiuspc(data):
    return data.convert("pc")
add_field("ParticleRadiuspc", function=_ParticleRadius,
          validators=[ValidateParameter("center")],
          convert_function = _ConvertRadiuspc, units=r"\rm{pc}",
          particle_type=True,
          display_name = "Particle Radius")
add_field("Radiuspc", function=_Radius,
          validators=[ValidateParameter("center")],
          convert_function = _ConvertRadiuspc, units=r"\rm{pc}",
          display_name="Radius")

def _ConvertRadiusAU(data):
    return data.convert("au")
add_field("ParticleRadiusAU", function=_ParticleRadius,
          validators=[ValidateParameter("center")],
          convert_function = _ConvertRadiusAU, units=r"\rm{AU}",
          particle_type=True,
          display_name = "Particle Radius")
add_field("RadiusAU", function=_Radius,
          validators=[ValidateParameter("center")],
          convert_function = _ConvertRadiusAU, units=r"\rm{AU}",
          display_name = "Radius")

add_field("ParticleRadiusCode", function=_ParticleRadius,
          validators=[ValidateParameter("center")],
          particle_type=True,
          display_name = "Particle Radius (code)")
add_field("RadiusCode", function=_Radius,
          validators=[ValidateParameter("center")],
          display_name = "Radius (code)")

def _RadialVelocity(field, data):
    normal = data.get_field_parameter("normal")
    velocities = obtain_rv_vec(data)    
    theta = data['sph_theta']
    phi   = data['sph_phi']

    return get_sph_r_component(velocities, theta, phi, normal)

def _RadialVelocityABS(field, data):
    return np.abs(_RadialVelocity(field, data))
def _ConvertRadialVelocityKMS(data):
    return km_per_cm
add_field("RadialVelocity", function=_RadialVelocity,
          units=r"\rm{cm}/\rm{s}")
add_field("RadialVelocityABS", function=_RadialVelocityABS,
          units=r"\rm{cm}/\rm{s}")
add_field("RadialVelocityKMS", function=_RadialVelocity,
          convert_function=_ConvertRadialVelocityKMS, units=r"\rm{km}/\rm{s}")
add_field("RadialVelocityKMSABS", function=_RadialVelocityABS,
          convert_function=_ConvertRadialVelocityKMS, units=r"\rm{km}/\rm{s}")

def _TangentialVelocity(field, data):
    return np.sqrt(data["VelocityMagnitude"]**2.0
                 - data["RadialVelocity"]**2.0)
add_field("TangentialVelocity", 
          function=_TangentialVelocity,
          take_log=False, units=r"\rm{cm}/\rm{s}")

def _CuttingPlaneVelocityX(field, data):
    x_vec, y_vec, z_vec = [data.get_field_parameter("cp_%s_vec" % (ax))
                           for ax in 'xyz']
    bulk_velocity = data.get_field_parameter("bulk_velocity")
    if bulk_velocity == None:
        bulk_velocity = np.zeros(3)
    v_vec = np.array([data["%s-velocity" % ax] for ax in 'xyz']) \
                - bulk_velocity[...,np.newaxis]
    return np.dot(x_vec, v_vec)
add_field("CuttingPlaneVelocityX", 
          function=_CuttingPlaneVelocityX,
          validators=[ValidateParameter("cp_%s_vec" % ax)
                      for ax in 'xyz'], units=r"\rm{km}/\rm{s}")
def _CuttingPlaneVelocityY(field, data):
    x_vec, y_vec, z_vec = [data.get_field_parameter("cp_%s_vec" % (ax))
                           for ax in 'xyz']
    bulk_velocity = data.get_field_parameter("bulk_velocity")
    if bulk_velocity == None:
        bulk_velocity = np.zeros(3)
    v_vec = np.array([data["%s-velocity" % ax] for ax in 'xyz']) \
                - bulk_velocity[...,np.newaxis]
    return np.dot(y_vec, v_vec)
add_field("CuttingPlaneVelocityY", 
          function=_CuttingPlaneVelocityY,
          validators=[ValidateParameter("cp_%s_vec" % ax)
                      for ax in 'xyz'], units=r"\rm{km}/\rm{s}")

def _CuttingPlaneBx(field, data):
    x_vec, y_vec, z_vec = [data.get_field_parameter("cp_%s_vec" % (ax))
                           for ax in 'xyz']
    b_vec = np.array([data["B%s" % ax] for ax in 'xyz'])
    return np.dot(x_vec, b_vec)
add_field("CuttingPlaneBx", 
          function=_CuttingPlaneBx,
          validators=[ValidateParameter("cp_%s_vec" % ax)
                      for ax in 'xyz'], units=r"\rm{Gauss}")
def _CuttingPlaneBy(field, data):
    x_vec, y_vec, z_vec = [data.get_field_parameter("cp_%s_vec" % (ax))
                           for ax in 'xyz']
    b_vec = np.array([data["B%s" % ax] for ax in 'xyz'])
    return np.dot(y_vec, b_vec)
add_field("CuttingPlaneBy", 
          function=_CuttingPlaneBy,
          validators=[ValidateParameter("cp_%s_vec" % ax)
                      for ax in 'xyz'], units=r"\rm{Gauss}")

def _MeanMolecularWeight(field,data):
    return (data["Density"] / (mh *data["NumberDensity"]))
add_field("MeanMolecularWeight",function=_MeanMolecularWeight,units=r"")

def _JeansMassMsun(field,data):
    MJ_constant = (((5*kboltz)/(G*mh))**(1.5)) * \
    (3/(4*3.1415926535897931))**(0.5) / 1.989e33

    return (MJ_constant *
            ((data["Temperature"]/data["MeanMolecularWeight"])**(1.5)) *
            (data["Density"]**(-0.5)))
add_field("JeansMassMsun",function=_JeansMassMsun,
          units=r"\rm{M_{\odot}}")

# We add these fields so that the field detector can use them
for field in ["particle_position_%s" % ax for ax in "xyz"] + \
             ["ParticleMass"]:
    # This marker should let everyone know not to use the fields, but NullFunc
    # should do that, too.
    add_field(field, function=NullFunc, particle_type = True,
        units=r"UNDEFINED")

def _pdensity(field, data):
    blank = np.zeros(data.ActiveDimensions, dtype='float64')
    if data["particle_position_x"].size == 0: return blank
    CICDeposit_3(data["particle_position_x"].astype(np.float64),
                 data["particle_position_y"].astype(np.float64),
                 data["particle_position_z"].astype(np.float64),
                 data["ParticleMass"],
                 data["particle_position_x"].size,
                 blank, np.array(data.LeftEdge).astype(np.float64),
                 np.array(data.ActiveDimensions).astype(np.int32),
                 np.float64(data['dx']))
    np.divide(blank, data["CellVolume"], blank)
    return blank
add_field("particle_density", function=_pdensity,
          validators=[ValidateGridType()],
          display_name=r"\mathrm{Particle}\/\mathrm{Density}")

def _MagneticEnergy(field,data):
    """This assumes that your front end has provided Bx, By, Bz in
    units of Gauss. If you use MKS, make sure to write your own
    MagneticEnergy field to deal with non-unitary \mu_0.
    """
    return (data["Bx"]**2 + data["By"]**2 + data["Bz"]**2)/(8*np.pi)
add_field("MagneticEnergy",function=_MagneticEnergy,
          units=r"\rm{ergs}\/\rm{cm}^{-3}",
          display_name=r"\rm{Magnetic}\/\rm{Energy}")

def _BMagnitude(field,data):
    """This assumes that your front end has provided Bx, By, Bz in
    units of Gauss. If you use MKS, make sure to write your own
    BMagnitude field to deal with non-unitary \mu_0.
    """
    return np.sqrt((data["Bx"]**2 + data["By"]**2 + data["Bz"]**2))
add_field("BMagnitude",
          function=_BMagnitude,
          display_name=r"|B|", units=r"\rm{Gauss}")

def _PlasmaBeta(field,data):
    """This assumes that your front end has provided Bx, By, Bz in
    units of Gauss. If you use MKS, make sure to write your own
    PlasmaBeta field to deal with non-unitary \mu_0.
    """
    return data['Pressure']/data['MagneticEnergy']
add_field("PlasmaBeta",
          function=_PlasmaBeta,
          display_name=r"\rm{Plasma}\/\beta", units="")

def _MagneticPressure(field,data):
    return data['MagneticEnergy']
add_field("MagneticPressure",
          function=_MagneticPressure,
          display_name=r"\rm{Magnetic}\/\rm{Pressure}",
          units=r"\rm{ergs}\/\rm{cm}^{-3}")

def _BPoloidal(field,data):
    normal = data.get_field_parameter("normal")

    Bfields = np.array([data['Bx'], data['By'], data['Bz']])

    theta = data['sph_theta']
    phi   = data['sph_phi']

    return get_sph_theta_component(Bfields, theta, phi, normal)

add_field("BPoloidal", function=_BPoloidal,
          units=r"\rm{Gauss}",
          validators=[ValidateParameter("normal")])

def _BToroidal(field,data):
    normal = data.get_field_parameter("normal")

    Bfields = np.array([data['Bx'], data['By'], data['Bz']])

    phi   = data['sph_phi']

    return get_sph_phi_component(Bfields, phi, normal)

add_field("BToroidal", function=_BToroidal,
          units=r"\rm{Gauss}",
          validators=[ValidateParameter("normal")])

def _BRadial(field,data):
    normal = data.get_field_parameter("normal")

    Bfields = np.array([data['Bx'], data['By'], data['Bz']])

    theta = data['sph_theta']
    phi   = data['sph_phi']

    return get_sph_r_component(Bfields, theta, phi, normal)

add_field("BRadial", function=_BRadial,
          units=r"\rm{Gauss}",
          validators=[ValidateParameter("normal")])

def _VorticitySquared(field, data):
    mylog.debug("Generating vorticity on %s", data)
    # We need to set up stencils
    if data.pf["HydroMethod"] == 2:
        sl_left = slice(None,-2,None)
        sl_right = slice(1,-1,None)
        div_fac = 1.0
    else:
        sl_left = slice(None,-2,None)
        sl_right = slice(2,None,None)
        div_fac = 2.0
    new_field = np.zeros(data["x-velocity"].shape)
    dvzdy = (data["z-velocity"][1:-1,sl_right,1:-1] -
             data["z-velocity"][1:-1,sl_left,1:-1]) \
             / (div_fac*data["dy"].flat[0])
    dvydz = (data["y-velocity"][1:-1,1:-1,sl_right] -
             data["y-velocity"][1:-1,1:-1,sl_left]) \
             / (div_fac*data["dz"].flat[0])
    new_field[1:-1,1:-1,1:-1] += (dvzdy - dvydz)**2.0
    del dvzdy, dvydz
    dvxdz = (data["x-velocity"][1:-1,1:-1,sl_right] -
             data["x-velocity"][1:-1,1:-1,sl_left]) \
             / (div_fac*data["dz"].flat[0])
    dvzdx = (data["z-velocity"][sl_right,1:-1,1:-1] -
             data["z-velocity"][sl_left,1:-1,1:-1]) \
             / (div_fac*data["dx"].flat[0])
    new_field[1:-1,1:-1,1:-1] += (dvxdz - dvzdx)**2.0
    del dvxdz, dvzdx
    dvydx = (data["y-velocity"][sl_right,1:-1,1:-1] -
             data["y-velocity"][sl_left,1:-1,1:-1]) \
             / (div_fac*data["dx"].flat[0])
    dvxdy = (data["x-velocity"][1:-1,sl_right,1:-1] -
             data["x-velocity"][1:-1,sl_left,1:-1]) \
             / (div_fac*data["dy"].flat[0])
    new_field[1:-1,1:-1,1:-1] += (dvydx - dvxdy)**2.0
    del dvydx, dvxdy
    new_field = np.abs(new_field)
    return new_field
def _convertVorticitySquared(data):
    return data.convert("cm")**-2.0
add_field("VorticitySquared", function=_VorticitySquared,
          validators=[ValidateSpatial(1,
              ["x-velocity","y-velocity","z-velocity"])],
          units=r"\rm{s}^{-2}",
          convert_function=_convertVorticitySquared)

def _Shear(field, data):
    """
    Shear is defined as [(dvx/dy + dvy/dx)^2 + (dvz/dy + dvy/dz)^2 +
                         (dvx/dz + dvz/dx)^2 ]^(0.5)
    where dvx/dy = [vx(j-1) - vx(j+1)]/[2dy]
    and is in units of s^(-1)
    (it's just like vorticity except add the derivative pairs instead
     of subtracting them)
    """
    # We need to set up stencils
    if data.pf["HydroMethod"] == 2:
        sl_left = slice(None,-2,None)
        sl_right = slice(1,-1,None)
        div_fac = 1.0
    else:
        sl_left = slice(None,-2,None)
        sl_right = slice(2,None,None)
        div_fac = 2.0
    new_field = np.zeros(data["x-velocity"].shape)
    if data.pf.dimensionality > 1:
        dvydx = (data["y-velocity"][sl_right,1:-1,1:-1] -
                data["y-velocity"][sl_left,1:-1,1:-1]) \
                / (div_fac*data["dx"].flat[0])
        dvxdy = (data["x-velocity"][1:-1,sl_right,1:-1] -
                data["x-velocity"][1:-1,sl_left,1:-1]) \
                / (div_fac*data["dy"].flat[0])
        new_field[1:-1,1:-1,1:-1] += (dvydx + dvxdy)**2.0
        del dvydx, dvxdy
    if data.pf.dimensionality > 2:
        dvzdy = (data["z-velocity"][1:-1,sl_right,1:-1] -
                data["z-velocity"][1:-1,sl_left,1:-1]) \
                / (div_fac*data["dy"].flat[0])
        dvydz = (data["y-velocity"][1:-1,1:-1,sl_right] -
                data["y-velocity"][1:-1,1:-1,sl_left]) \
                / (div_fac*data["dz"].flat[0])
        new_field[1:-1,1:-1,1:-1] += (dvzdy + dvydz)**2.0
        del dvzdy, dvydz
        dvxdz = (data["x-velocity"][1:-1,1:-1,sl_right] -
                data["x-velocity"][1:-1,1:-1,sl_left]) \
                / (div_fac*data["dz"].flat[0])
        dvzdx = (data["z-velocity"][sl_right,1:-1,1:-1] -
                data["z-velocity"][sl_left,1:-1,1:-1]) \
                / (div_fac*data["dx"].flat[0])
        new_field[1:-1,1:-1,1:-1] += (dvxdz + dvzdx)**2.0
        del dvxdz, dvzdx
    new_field = new_field**0.5
    new_field = np.abs(new_field)
    return new_field
def _convertShear(data):
    return data.convert("cm")**-1.0
add_field("Shear", function=_Shear,
          validators=[ValidateSpatial(1,
              ["x-velocity","y-velocity","z-velocity"])],
          units=r"\rm{s}^{-1}",
          convert_function=_convertShear, take_log=False)

def _ShearCriterion(field, data):
    """
    Shear is defined as [(dvx/dy + dvy/dx)^2 + (dvz/dy + dvy/dz)^2 +
                         (dvx/dz + dvz/dx)^2 ]^(0.5)
    where dvx/dy = [vx(j-1) - vx(j+1)]/[2dy]
    and is in units of s^(-1)
    (it's just like vorticity except add the derivative pairs instead
     of subtracting them)

    Divide by c_s to leave Shear in units of cm**-1, which 
    can be compared against the inverse of the local cell size (1/dx) 
    to determine if refinement should occur.
    """
    # We need to set up stencils
    if data.pf["HydroMethod"] == 2:
        sl_left = slice(None,-2,None)
        sl_right = slice(1,-1,None)
        div_fac = 1.0
    else:
        sl_left = slice(None,-2,None)
        sl_right = slice(2,None,None)
        div_fac = 2.0
    new_field = np.zeros(data["x-velocity"].shape)
    if data.pf.dimensionality > 1:
        dvydx = (data["y-velocity"][sl_right,1:-1,1:-1] -
                data["y-velocity"][sl_left,1:-1,1:-1]) \
                / (div_fac*data["dx"].flat[0])
        dvxdy = (data["x-velocity"][1:-1,sl_right,1:-1] -
                data["x-velocity"][1:-1,sl_left,1:-1]) \
                / (div_fac*data["dy"].flat[0])
        new_field[1:-1,1:-1,1:-1] += (dvydx + dvxdy)**2.0
        del dvydx, dvxdy
    if data.pf.dimensionality > 2:
        dvzdy = (data["z-velocity"][1:-1,sl_right,1:-1] -
                data["z-velocity"][1:-1,sl_left,1:-1]) \
                / (div_fac*data["dy"].flat[0])
        dvydz = (data["y-velocity"][1:-1,1:-1,sl_right] -
                data["y-velocity"][1:-1,1:-1,sl_left]) \
                / (div_fac*data["dz"].flat[0])
        new_field[1:-1,1:-1,1:-1] += (dvzdy + dvydz)**2.0
        del dvzdy, dvydz
        dvxdz = (data["x-velocity"][1:-1,1:-1,sl_right] -
                data["x-velocity"][1:-1,1:-1,sl_left]) \
                / (div_fac*data["dz"].flat[0])
        dvzdx = (data["z-velocity"][sl_right,1:-1,1:-1] -
                data["z-velocity"][sl_left,1:-1,1:-1]) \
                / (div_fac*data["dx"].flat[0])
        new_field[1:-1,1:-1,1:-1] += (dvxdz + dvzdx)**2.0
        del dvxdz, dvzdx
    new_field /= data["SoundSpeed"]**2.0
    new_field = new_field**(0.5)
    new_field = np.abs(new_field)
    return new_field

def _convertShearCriterion(data):
    return data.convert("cm")**-1.0
add_field("ShearCriterion", function=_ShearCriterion,
          validators=[ValidateSpatial(1,
              ["x-velocity","y-velocity","z-velocity", "SoundSpeed"])],
          units=r"\rm{cm}^{-1}",
          convert_function=_convertShearCriterion, take_log=False)

def _ShearMach(field, data):
    """
    Dimensionless Shear (ShearMach) is defined nearly the same as shear, 
    except that it is scaled by the local dx/dy/dz and the local sound speed.
    So it results in a unitless quantity that is effectively measuring 
    shear in mach number.  

    In order to avoid discontinuities created by multiplying by dx/dy/dz at
    grid refinement boundaries, we also multiply by 2**GridLevel.

    Shear (Mach) = [(dvx + dvy)^2 + (dvz + dvy)^2 +
                    (dvx + dvz)^2  ]^(0.5) / c_sound
    """
    # We need to set up stencils
    if data.pf["HydroMethod"] == 2:
        sl_left = slice(None,-2,None)
        sl_right = slice(1,-1,None)
        div_fac = 1.0
    else:
        sl_left = slice(None,-2,None)
        sl_right = slice(2,None,None)
        div_fac = 2.0
    new_field = np.zeros(data["x-velocity"].shape)
    if data.pf.dimensionality > 1:
        dvydx = (data["y-velocity"][sl_right,1:-1,1:-1] -
                data["y-velocity"][sl_left,1:-1,1:-1]) \
                / (div_fac)
        dvxdy = (data["x-velocity"][1:-1,sl_right,1:-1] -
                data["x-velocity"][1:-1,sl_left,1:-1]) \
                / (div_fac)
        new_field[1:-1,1:-1,1:-1] += (dvydx + dvxdy)**2.0
        del dvydx, dvxdy
    if data.pf.dimensionality > 2:
        dvzdy = (data["z-velocity"][1:-1,sl_right,1:-1] -
                data["z-velocity"][1:-1,sl_left,1:-1]) \
                / (div_fac)
        dvydz = (data["y-velocity"][1:-1,1:-1,sl_right] -
                data["y-velocity"][1:-1,1:-1,sl_left]) \
                / (div_fac)
        new_field[1:-1,1:-1,1:-1] += (dvzdy + dvydz)**2.0
        del dvzdy, dvydz
        dvxdz = (data["x-velocity"][1:-1,1:-1,sl_right] -
                data["x-velocity"][1:-1,1:-1,sl_left]) \
                / (div_fac)
        dvzdx = (data["z-velocity"][sl_right,1:-1,1:-1] -
                data["z-velocity"][sl_left,1:-1,1:-1]) \
                / (div_fac)
        new_field[1:-1,1:-1,1:-1] += (dvxdz + dvzdx)**2.0
        del dvxdz, dvzdx
    new_field *= ((2.0**data.level)/data["SoundSpeed"])**2.0
    new_field = new_field**0.5
    new_field = np.abs(new_field)
    return new_field
add_field("ShearMach", function=_ShearMach,
          validators=[ValidateSpatial(1,
              ["x-velocity","y-velocity","z-velocity","SoundSpeed"])],
          units=r"\rm{Mach}",take_log=False)

def _gradPressureX(field, data):
    # We need to set up stencils
    if data.pf["HydroMethod"] == 2:
        sl_left = slice(None,-2,None)
        sl_right = slice(1,-1,None)
        div_fac = 1.0
    else:
        sl_left = slice(None,-2,None)
        sl_right = slice(2,None,None)
        div_fac = 2.0
    new_field = np.zeros(data["Pressure"].shape, dtype='float64')
    ds = div_fac * data['dx'].flat[0]
    new_field[1:-1,1:-1,1:-1]  = data["Pressure"][sl_right,1:-1,1:-1]/ds
    new_field[1:-1,1:-1,1:-1] -= data["Pressure"][sl_left ,1:-1,1:-1]/ds
    return new_field
def _gradPressureY(field, data):
    # We need to set up stencils
    if data.pf["HydroMethod"] == 2:
        sl_left = slice(None,-2,None)
        sl_right = slice(1,-1,None)
        div_fac = 1.0
    else:
        sl_left = slice(None,-2,None)
        sl_right = slice(2,None,None)
        div_fac = 2.0
    new_field = np.zeros(data["Pressure"].shape, dtype='float64')
    ds = div_fac * data['dy'].flat[0]
    new_field[1:-1,1:-1,1:-1]  = data["Pressure"][1:-1,sl_right,1:-1]/ds
    new_field[1:-1,1:-1,1:-1] -= data["Pressure"][1:-1,sl_left ,1:-1]/ds
    return new_field
def _gradPressureZ(field, data):
    # We need to set up stencils
    if data.pf["HydroMethod"] == 2:
        sl_left = slice(None,-2,None)
        sl_right = slice(1,-1,None)
        div_fac = 1.0
    else:
        sl_left = slice(None,-2,None)
        sl_right = slice(2,None,None)
        div_fac = 2.0
    new_field = np.zeros(data["Pressure"].shape, dtype='float64')
    ds = div_fac * data['dz'].flat[0]
    new_field[1:-1,1:-1,1:-1]  = data["Pressure"][1:-1,1:-1,sl_right]/ds
    new_field[1:-1,1:-1,1:-1] -= data["Pressure"][1:-1,1:-1,sl_left ]/ds
    return new_field
def _convertgradPressure(data):
    return 1.0/data.convert("cm")
for ax in 'XYZ':
    n = "gradPressure%s" % ax
    add_field(n, function=eval("_%s" % n),
              convert_function=_convertgradPressure,
              validators=[ValidateSpatial(1, ["Pressure"])],
              units=r"\rm{dyne}/\rm{cm}^{3}")

def _gradPressureMagnitude(field, data):
    return np.sqrt(data["gradPressureX"]**2 +
                   data["gradPressureY"]**2 +
                   data["gradPressureZ"]**2)
add_field("gradPressureMagnitude", function=_gradPressureMagnitude,
          validators=[ValidateSpatial(1, ["Pressure"])],
          units=r"\rm{dyne}/\rm{cm}^{3}")

def _gradDensityX(field, data):
    # We need to set up stencils
    if data.pf["HydroMethod"] == 2:
        sl_left = slice(None,-2,None)
        sl_right = slice(1,-1,None)
        div_fac = 1.0
    else:
        sl_left = slice(None,-2,None)
        sl_right = slice(2,None,None)
        div_fac = 2.0
    new_field = np.zeros(data["Density"].shape, dtype='float64')
    ds = div_fac * data['dx'].flat[0]
    new_field[1:-1,1:-1,1:-1]  = data["Density"][sl_right,1:-1,1:-1]/ds
    new_field[1:-1,1:-1,1:-1] -= data["Density"][sl_left ,1:-1,1:-1]/ds
    return new_field
def _gradDensityY(field, data):
    # We need to set up stencils
    if data.pf["HydroMethod"] == 2:
        sl_left = slice(None,-2,None)
        sl_right = slice(1,-1,None)
        div_fac = 1.0
    else:
        sl_left = slice(None,-2,None)
        sl_right = slice(2,None,None)
        div_fac = 2.0
    new_field = np.zeros(data["Density"].shape, dtype='float64')
    ds = div_fac * data['dy'].flat[0]
    new_field[1:-1,1:-1,1:-1]  = data["Density"][1:-1,sl_right,1:-1]/ds
    new_field[1:-1,1:-1,1:-1] -= data["Density"][1:-1,sl_left ,1:-1]/ds
    return new_field
def _gradDensityZ(field, data):
    # We need to set up stencils
    if data.pf["HydroMethod"] == 2:
        sl_left = slice(None,-2,None)
        sl_right = slice(1,-1,None)
        div_fac = 1.0
    else:
        sl_left = slice(None,-2,None)
        sl_right = slice(2,None,None)
        div_fac = 2.0
    new_field = np.zeros(data["Density"].shape, dtype='float64')
    ds = div_fac * data['dz'].flat[0]
    new_field[1:-1,1:-1,1:-1]  = data["Density"][1:-1,1:-1,sl_right]/ds
    new_field[1:-1,1:-1,1:-1] -= data["Density"][1:-1,1:-1,sl_left ]/ds
    return new_field
def _convertgradDensity(data):
    return 1.0/data.convert("cm")
for ax in 'XYZ':
    n = "gradDensity%s" % ax
    add_field(n, function=eval("_%s" % n),
              convert_function=_convertgradDensity,
              validators=[ValidateSpatial(1, ["Density"])],
              units=r"\rm{g}/\rm{cm}^{4}")

def _gradDensityMagnitude(field, data):
    return np.sqrt(data["gradDensityX"]**2 +
                   data["gradDensityY"]**2 +
                   data["gradDensityZ"]**2)
add_field("gradDensityMagnitude", function=_gradDensityMagnitude,
          validators=[ValidateSpatial(1, ["Density"])],
          units=r"\rm{g}/\rm{cm}^{4}")

def _BaroclinicVorticityX(field, data):
    rho2 = data["Density"].astype('float64')**2
    return (data["gradPressureY"] * data["gradDensityZ"] -
            data["gradPressureZ"] * data["gradDensityY"]) / rho2
def _BaroclinicVorticityY(field, data):
    rho2 = data["Density"].astype('float64')**2
    return (data["gradPressureZ"] * data["gradDensityX"] -
            data["gradPressureX"] * data["gradDensityZ"]) / rho2
def _BaroclinicVorticityZ(field, data):
    rho2 = data["Density"].astype('float64')**2
    return (data["gradPressureX"] * data["gradDensityY"] -
            data["gradPressureY"] * data["gradDensityX"]) / rho2
for ax in 'XYZ':
    n = "BaroclinicVorticity%s" % ax
    add_field(n, function=eval("_%s" % n),
          validators=[ValidateSpatial(1, ["Density", "Pressure"])],
          units=r"\rm{s}^{-1}")

def _BaroclinicVorticityMagnitude(field, data):
    return np.sqrt(data["BaroclinicVorticityX"]**2 +
                   data["BaroclinicVorticityY"]**2 +
                   data["BaroclinicVorticityZ"]**2)
add_field("BaroclinicVorticityMagnitude",
          function=_BaroclinicVorticityMagnitude,
          validators=[ValidateSpatial(1, ["Density", "Pressure"])],
          units=r"\rm{s}^{-1}")

def _VorticityX(field, data):
    # We need to set up stencils
    if data.pf["HydroMethod"] == 2:
        sl_left = slice(None,-2,None)
        sl_right = slice(1,-1,None)
        div_fac = 1.0
    else:
        sl_left = slice(None,-2,None)
        sl_right = slice(2,None,None)
        div_fac = 2.0
    new_field = np.zeros(data["z-velocity"].shape, dtype='float64')
    new_field[1:-1,1:-1,1:-1] = (data["z-velocity"][1:-1,sl_right,1:-1] -
                                 data["z-velocity"][1:-1,sl_left,1:-1]) \
                                 / (div_fac*data["dy"].flat[0])
    new_field[1:-1,1:-1,1:-1] -= (data["y-velocity"][1:-1,1:-1,sl_right] -
                                  data["y-velocity"][1:-1,1:-1,sl_left]) \
                                  / (div_fac*data["dz"].flat[0])
    return new_field
def _VorticityY(field, data):
    # We need to set up stencils
    if data.pf["HydroMethod"] == 2:
        sl_left = slice(None,-2,None)
        sl_right = slice(1,-1,None)
        div_fac = 1.0
    else:
        sl_left = slice(None,-2,None)
        sl_right = slice(2,None,None)
        div_fac = 2.0
    new_field = np.zeros(data["z-velocity"].shape, dtype='float64')
    new_field[1:-1,1:-1,1:-1] = (data["x-velocity"][1:-1,1:-1,sl_right] -
                                 data["x-velocity"][1:-1,1:-1,sl_left]) \
                                 / (div_fac*data["dz"].flat[0])
    new_field[1:-1,1:-1,1:-1] -= (data["z-velocity"][sl_right,1:-1,1:-1] -
                                  data["z-velocity"][sl_left,1:-1,1:-1]) \
                                  / (div_fac*data["dx"].flat[0])
    return new_field
def _VorticityZ(field, data):
    # We need to set up stencils
    if data.pf["HydroMethod"] == 2:
        sl_left = slice(None,-2,None)
        sl_right = slice(1,-1,None)
        div_fac = 1.0
    else:
        sl_left = slice(None,-2,None)
        sl_right = slice(2,None,None)
        div_fac = 2.0
    new_field = np.zeros(data["x-velocity"].shape, dtype='float64')
    new_field[1:-1,1:-1,1:-1] = (data["y-velocity"][sl_right,1:-1,1:-1] -
                                 data["y-velocity"][sl_left,1:-1,1:-1]) \
                                 / (div_fac*data["dx"].flat[0])
    new_field[1:-1,1:-1,1:-1] -= (data["x-velocity"][1:-1,sl_right,1:-1] -
                                  data["x-velocity"][1:-1,sl_left,1:-1]) \
                                  / (div_fac*data["dy"].flat[0])
    return new_field
def _convertVorticity(data):
    return 1.0/data.convert("cm")
for ax in 'XYZ':
    n = "Vorticity%s" % ax
    add_field(n, function=eval("_%s" % n),
              convert_function=_convertVorticity,
              validators=[ValidateSpatial(1, 
                          ["x-velocity", "y-velocity", "z-velocity"])],
              units=r"\rm{s}^{-1}")

def _VorticityMagnitude(field, data):
    return np.sqrt(data["VorticityX"]**2 +
                   data["VorticityY"]**2 +
                   data["VorticityZ"]**2)
add_field("VorticityMagnitude", function=_VorticityMagnitude,
          validators=[ValidateSpatial(1, 
                      ["x-velocity", "y-velocity", "z-velocity"])],
          units=r"\rm{s}^{-1}")

def _VorticityStretchingX(field, data):
    return data["DivV"] * data["VorticityX"]
def _VorticityStretchingY(field, data):
    return data["DivV"] * data["VorticityY"]
def _VorticityStretchingZ(field, data):
    return data["DivV"] * data["VorticityZ"]
for ax in 'XYZ':
    n = "VorticityStretching%s" % ax
    add_field(n, function=eval("_%s" % n),
              validators=[ValidateSpatial(0)])
def _VorticityStretchingMagnitude(field, data):
    return np.sqrt(data["VorticityStretchingX"]**2 +
                   data["VorticityStretchingY"]**2 +
                   data["VorticityStretchingZ"]**2)
add_field("VorticityStretchingMagnitude", 
          function=_VorticityStretchingMagnitude,
          validators=[ValidateSpatial(1, 
                      ["x-velocity", "y-velocity", "z-velocity"])],
          units=r"\rm{s}^{-1}")

def _VorticityGrowthX(field, data):
    return -data["VorticityStretchingX"] - data["BaroclinicVorticityX"]
def _VorticityGrowthY(field, data):
    return -data["VorticityStretchingY"] - data["BaroclinicVorticityY"]
def _VorticityGrowthZ(field, data):
    return -data["VorticityStretchingZ"] - data["BaroclinicVorticityZ"]
for ax in 'XYZ':
    n = "VorticityGrowth%s" % ax
    add_field(n, function=eval("_%s" % n),
              validators=[ValidateSpatial(1, 
                          ["x-velocity", "y-velocity", "z-velocity"])],
              units=r"\rm{s}^{-2}")
def _VorticityGrowthMagnitude(field, data):
    result = np.sqrt(data["VorticityGrowthX"]**2 +
                     data["VorticityGrowthY"]**2 +
                     data["VorticityGrowthZ"]**2)
    dot = np.zeros(result.shape)
    for ax in "XYZ":
        dot += data["Vorticity%s" % ax] * data["VorticityGrowth%s" % ax]
    result = np.sign(dot) * result
    return result
add_field("VorticityGrowthMagnitude", function=_VorticityGrowthMagnitude,
          validators=[ValidateSpatial(1, 
                      ["x-velocity", "y-velocity", "z-velocity"])],
          units=r"\rm{s}^{-1}",
          take_log=False)
def _VorticityGrowthMagnitudeABS(field, data):
    return np.sqrt(data["VorticityGrowthX"]**2 +
                   data["VorticityGrowthY"]**2 +
                   data["VorticityGrowthZ"]**2)
add_field("VorticityGrowthMagnitudeABS", function=_VorticityGrowthMagnitudeABS,
          validators=[ValidateSpatial(1, 
                      ["x-velocity", "y-velocity", "z-velocity"])],
          units=r"\rm{s}^{-1}")

def _VorticityGrowthTimescale(field, data):
    domegax_dt = data["VorticityX"] / data["VorticityGrowthX"]
    domegay_dt = data["VorticityY"] / data["VorticityGrowthY"]
    domegaz_dt = data["VorticityZ"] / data["VorticityGrowthZ"]
    return np.sqrt(domegax_dt**2 + domegay_dt**2 + domegaz_dt**2)
add_field("VorticityGrowthTimescale", function=_VorticityGrowthTimescale,
          validators=[ValidateSpatial(1, 
                      ["x-velocity", "y-velocity", "z-velocity"])],
          units=r"\rm{s}")

########################################################################
# With radiation pressure
########################################################################

def _VorticityRadPressureX(field, data):
    rho = data["Density"].astype('float64')
    return (data["RadAccel2"] * data["gradDensityZ"] -
            data["RadAccel3"] * data["gradDensityY"]) / rho
def _VorticityRadPressureY(field, data):
    rho = data["Density"].astype('float64')
    return (data["RadAccel3"] * data["gradDensityX"] -
            data["RadAccel1"] * data["gradDensityZ"]) / rho
def _VorticityRadPressureZ(field, data):
    rho = data["Density"].astype('float64')
    return (data["RadAccel1"] * data["gradDensityY"] -
            data["RadAccel2"] * data["gradDensityX"]) / rho
def _convertRadAccel(data):
    return data.convert("x-velocity")/data.convert("Time")
for ax in 'XYZ':
    n = "VorticityRadPressure%s" % ax
    add_field(n, function=eval("_%s" % n),
              convert_function=_convertRadAccel,
              validators=[ValidateSpatial(1, 
                   ["Density", "RadAccel1", "RadAccel2", "RadAccel3"])],
              units=r"\rm{s}^{-1}")

def _VorticityRadPressureMagnitude(field, data):
    return np.sqrt(data["VorticityRadPressureX"]**2 +
                   data["VorticityRadPressureY"]**2 +
                   data["VorticityRadPressureZ"]**2)
add_field("VorticityRadPressureMagnitude",
          function=_VorticityRadPressureMagnitude,
          validators=[ValidateSpatial(1, 
                      ["Density", "RadAccel1", "RadAccel2", "RadAccel3"])],
          units=r"\rm{s}^{-1}")

def _VorticityRPGrowthX(field, data):
    return -data["VorticityStretchingX"] - data["BaroclinicVorticityX"] \
           -data["VorticityRadPressureX"]
def _VorticityRPGrowthY(field, data):
    return -data["VorticityStretchingY"] - data["BaroclinicVorticityY"] \
           -data["VorticityRadPressureY"]
def _VorticityRPGrowthZ(field, data):
    return -data["VorticityStretchingZ"] - data["BaroclinicVorticityZ"] \
           -data["VorticityRadPressureZ"]
for ax in 'XYZ':
    n = "VorticityRPGrowth%s" % ax
    add_field(n, function=eval("_%s" % n),
              validators=[ValidateSpatial(1, 
                       ["Density", "RadAccel1", "RadAccel2", "RadAccel3"])],
              units=r"\rm{s}^{-1}")
def _VorticityRPGrowthMagnitude(field, data):
    result = np.sqrt(data["VorticityRPGrowthX"]**2 +
                     data["VorticityRPGrowthY"]**2 +
                     data["VorticityRPGrowthZ"]**2)
    dot = np.zeros(result.shape)
    for ax in "XYZ":
        dot += data["Vorticity%s" % ax] * data["VorticityGrowth%s" % ax]
    result = np.sign(dot) * result
    return result
add_field("VorticityRPGrowthMagnitude", function=_VorticityGrowthMagnitude,
          validators=[ValidateSpatial(1, 
                      ["Density", "RadAccel1", "RadAccel2", "RadAccel3"])],
          units=r"\rm{s}^{-1}",
          take_log=False)
def _VorticityRPGrowthMagnitudeABS(field, data):
    return np.sqrt(data["VorticityRPGrowthX"]**2 +
                   data["VorticityRPGrowthY"]**2 +
                   data["VorticityRPGrowthZ"]**2)
add_field("VorticityRPGrowthMagnitudeABS", 
          function=_VorticityRPGrowthMagnitudeABS,
          validators=[ValidateSpatial(1, 
                      ["Density", "RadAccel1", "RadAccel2", "RadAccel3"])],
          units=r"\rm{s}^{-1}")

def _VorticityRPGrowthTimescale(field, data):
    domegax_dt = data["VorticityX"] / data["VorticityRPGrowthX"]
    domegay_dt = data["VorticityY"] / data["VorticityRPGrowthY"]
    domegaz_dt = data["VorticityZ"] / data["VorticityRPGrowthZ"]
    return np.sqrt(domegax_dt**2 + domegay_dt**2 + domegaz_dt**2)
add_field("VorticityRPGrowthTimescale", function=_VorticityRPGrowthTimescale,
          validators=[ValidateSpatial(1, 
                      ["Density", "RadAccel1", "RadAccel2", "RadAccel3"])],
          units=r"\rm{s}^{-1}")
