#
# Physical Constants and Units Conversion Factors
#

# Masses
mass_hydrogen_cgs = 1.67e-24 # g
mass_electron_cgs = 9.11e-28 # g
# Velocities
speed_of_light_cgs = 2.99792458e10 # cm/s, exact

# Cross Sections
cross_section_thompson_cgs = 6.65e-25 # cm^2

# Physical Constants
boltzmann_constant_cgs = 1.3806504e-16 # erg K^-1
gravitational_constant_cgs  = 6.67428e-8 # cm^3 g^-1 s^-2

rho_crit_now = 1.8788e-29 # g times h^2 (critical mass for closure, Cosmology)

# Misc. Approximations
mass_mean_atomic_cosmology = 1.22
mass_mean_atomic_galactic = 2.3

# Conversion Factors:  X au * mpc_per_au = Y au
# length
mpc_per_mpc   = 1
mpc_per_kpc   = 1e-3
mpc_per_pc    = 1e-6
mpc_per_au    = 4.847e-12
mpc_per_rsun  = 2.253e-14
mpc_per_miles = 5.216e-20
mpc_per_cm    = 3.24e-25
km_per_pc     = 1.3806504e13 
km_per_m      = 1e-3
km_per_cm     = 1e-5

m_per_fpc     = 0.0324077649

au_per_mpc    = 2.063e11
rsun_per_mpc  = 4.43664e13
miles_per_mpc = 1.917e19
cm_per_mpc    = 3.0857e24
cm_per_km     = 1e5
pc_per_km     = 3.24e-14
pc_per_cm     = 3.24e-19
#Short cuts
G = gravitational_constant_cgs
me = mass_electron_cgs
mp = mass_hydrogen_cgs
clight = speed_of_light_cgs
kboltz = boltzmann_constant_cgs
sigma_thompson = cross_section_thompson_cgs
