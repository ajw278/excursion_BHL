
#unit definitions
day2s = 60.*60.*24.0
year2s = day2s*365.
Myr2s = 1e6*year2s
pc2cm = 3.086e18
Gcgs = 6.6743e-8
Msol2g = 1.988e33
km2cm = 1e5
km2pc = 3.241e-14
pc2cm = km2cm/km2pc
au2cm = 1.496e13
Mearth2Msol = 3.0027e-6

Lsolergs = 3.82e33

#Sound speed in km s-1
cs_ = 0.2

#Frequency of solar neighbourhood in Myr-1
Omega_ = 2.6e-2 

#Velocity dispersion at the scale height 
sigvh_ = 6.0

#Scale height galactic disc in pc
h_ = (sigvh_*1e5*Myr2s/Omega_)/pc2cm

#Local average (midplane) density in units of Msol/pc2
rho0_ = 6./h_

#Surface density of solar neighbourhood in Msol/pc2
Sigma0_ = 2.*rho0_*h_

#Lower disc mass limit (dispersal criterion)
mllim = 3e-5
