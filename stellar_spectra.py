import os



#os.environ["PYSYN_CDBS"] = "/Users/andrewwinter/Documents/pysyn_data/grp/redcat/trds/"
#os.environ["PYSYN_CDBS"] = "/home/awinter/Documents/pysyn_data/grp/redcat/trds/"
os.environ["PYSYN_CDBS"] = "/Users/andrew/Documents/Stellar_models/"

import stellar_evolution as se
import pysynphot as S
import matplotlib.pyplot as plt
import numpy as np
import planck
import warnings

import scipy.interpolate as interpolate


G = 6.67e-8
Msol  =1.989e33
au = 1.496e13
sig_SB = 5.67e-5 #K^-4
mp = 1.6726e-24
k_B = 1.81e-16 #cgs K^-1
Lsol = 3.828e33 #erg/s
year = 365.*24.*60.0*60.0
#Rsol in cm
Rsol = 6.957e10
h = 6.626e-27  # Planck's constant in erg*s
c = 2.998e10   # Speed of light in cm/s
eV = 1.602e-12


h = 6.626e-27  # Planck's constant in erg*s
c = 2.998e10   # Speed of light in cm/s
k_B = 1.381e-16 # Boltzmann constant in erg/K

def blackbody_spectrum(wavelength, temperature):
    exponent = h * c / (wavelength * k_B * temperature)
    # To prevent overflow, we limit the exponent value
    exponent = np.clip(exponent, 1e-3, 30)
    return (2 * h * c**2 / wavelength**5) / (np.exp(exponent) - 1)


# Free-fall velocity
def free_fall_velocity(M_star, R_star, r_M):
	return np.sqrt(2 * G * M_star * (1/R_star - 1/r_M))

# Shock temperature
def shock_temperature(v_ff):
	return (3./16.0) * (mu * mH * v_ff**2) / k_B


def column_energy_flux(f, R_star, M_star, Mdot_acc, r_M):
	vff = free_fall_velocity(M_star, R_star, r_M)
	F = 0.5*Mdot_acc*vff**2
	F /= f*4.*np.pi*R_star*R_star
	return F

def coll_temp_approx(Tstar, F):
	Tcol = ((sig_SB*Tstar**4 + F)/sig_SB)**(1./4.)
	return Tcol

# Accretion shock luminosity
def accretion_luminosity(M_star, R_star, M_dot, r_M):
	return (G * M_star * M_dot) / R_star * (1 - R_star / r_M)

# Planck function for blackbody radiation
def planck_wavelength(wavelength_angstrom, T):
	wavelength_cm = np.asarray(wavelength_angstrom) * 1e-8  # Convert Angstrom to cm
	x = h * c / (wavelength_cm * k_B * T)  # Dimensionless argument h nu / k T

	# Safe exponentiation to prevent overflow
	x_safe = np.clip(x, None, 700)  # Clip values above 700 to avoid np.exp overflow


	# Use NumPy's where to apply Rayleigh-Jeans for small x, full Planck otherwise
	B_lambda = (2.0 * k_B * T) / (wavelength_cm**4) * (c**2)
	B_lambda[x>1e-10] = (2.0 * h * c**2) / (wavelength_cm[x>1e-10]**5) * (1.0 / (np.exp(x_safe[x>1e-10]) - 1.0)) 
	
	return B_lambda/1e8   # Convert from per cm to per Angstrom

# Blackbody emission from the shock region
def shock_flux(wavelength_angstrom, T, area):
	B_lambda = planck_wavelength(wavelength_angstrom, T)
	return B_lambda * area

def magnetospheric_radius(M_star, M_dot, R_star, B=1000.0, xi=0.7):
	mdm = B*R_star**3
	return xi*((mdm**4.)/(4.*G*M_star*M_dot**2))**(1./7.)

# Total accretion shock spectrum approximated as by Mendigutia et al. 2011 (actually reference within MDCH04..)
def accretion_shock_spectrum_approx(M_star_sol, R_star_sol, M_dot_Msolyr, Tstar, wavelength_A, f=0.01):

	M_star = M_star_sol*Msol
	R_star = R_star_sol*Rsol
	M_dot = M_dot_Msolyr*Msol/year

	r_M = magnetospheric_radius(M_star, M_dot, R_star, B=1000.0, xi=0.7)

	print('r_m/r_star', r_M/R_star)
	if r_M<=2.*R_star:
		print('Warning, truncating R_M below')
		r_M = 2.*R_star
	elif r_M>=100.0*R_star:
		print('Warning, truncating R_M above')
		r_M = 100.0*R_star


	# Free-fall velocity
	v_ff = free_fall_velocity(M_star, R_star, r_M)

	#print('v_ff (km/s)', v_ff/1e5)

	# Shock temperature
	#T_shock = shock_temperature(v_ff)
	F = column_energy_flux(f, R_star,  M_star, M_dot, r_M)

	T_coll = coll_temp_approx(Tstar, F)

	# Luminosity from the shock
	L_acc = accretion_luminosity(M_star, R_star, M_dot, r_M)

	# Filling factor (fraction of the surface covered by accretion)
	A_coll = f * 4 * np.pi * R_star**2

	# Spectrum from the shock as a function of wavelength
	spectrum = shock_flux(wavelength_A, T_coll, A_coll)

	#print('Normalisation:', L_acc / np.trapz(spectrum, wavelength_A))
	spectrum *= L_acc / np.trapz(spectrum, wavelength_A)

	return spectrum
    
    
def get_spectra(mstar, age, metallicity=0.0, Mdot_acc=0.0):
	
	# Get stellar properties
	Teff, log_g, log_L, R, star_mass = se.fetch_stellar_properties(mstar, age)
	
	# Compute the stellar spectrum using Castelli & Kurucz atmosphere models
	#sp = S.Icat('ck04models', Teff, metallicity, log_g)
	try:
		sp = S.Icat('phoenix', Teff, metallicity, log_g)
	except:
		print('Warning: using blackbody spectrum because stellar parameters outside of atmosphere model range')
		sp = S.BlackBody(Teff)
	
	"""if mstar>0.9:
		
		sp_alt = S.BlackBody(Teff)
		Ltot = np.trapz(sp.flux*np.pi*4.0*R*R*Rsol*Rsol, sp.wave)
		LBB = np.trapz(sp_alt.flux*np.pi*4.0*R*R*Rsol*Rsol, sp_alt.wave)
		
		Lnorm = Lsol*10.**log_L / Ltot
		LnormBB = Lsol*10.**log_L / LBB
		plt.plot(sp.wave, Lnorm*sp.flux)
		plt.plot(sp_alt.wave, LnormBB*sp_alt.flux)
		plt.xscale('log')
		plt.yscale('log')
		plt.show()"""
		
	#sp = S.Icat('k93models', Teff, metallicity, log_g)
	
	#Renormalize given the stellar luminosity 
	Ltot = np.trapz(sp.flux*np.pi*4.0*R*R*Rsol*Rsol, sp.wave)
	
	Lnorm = Lsol*10.**log_L / Ltot
	
	#if (Lnorm-1.)/Lnorm>0.5:
	#	raise warnings.warn('Luminosity of the spectra very different to the evoluton model')
	acc_cont=0.
	if Mdot_acc>0.0:

		flux_acc = accretion_shock_spectrum_approx(star_mass, R, Mdot_acc, Teff, sp.wave, f=0.01)/(4.*np.pi*R*R*Rsol*Rsol)
		Lacc = G*star_mass*Mdot_acc*Msol*Msol/year / (R*Rsol)
		#Teff_acc = (Lacc/4/np.pi/(R*R*Rsol*Rsol)/sig_SB)**(0.25)
		#Teff_acc_th = G*star_mass*mp*Msol/(3.*k_B*R*Rsol)
		
		#sp_acc = S.BlackBody(Teff_acc)
		#flux_acc = Lacc*sp_acc.flux/Ltot_acc
		#wave_acc = sp_acc.wave
		
		wave_acc = sp.wave
		
		Ltot_acc = np.trapz(flux_acc*np.pi*4.0*R*R*Rsol*Rsol, wave_acc)
		
		
		flux_acc *= Lacc/Ltot_acc
		
		f_acc = interpolate.interp1d(wave_acc, flux_acc, fill_value=0.0, bounds_error=False)
		
		acc_cont = f_acc(sp.wave)
		"""print(Mdot_acc, mstar)
	
		plt.plot(sp.wave, sp.flux*Lnorm)
		plt.plot(sp.wave,acc_cont)
		plt.yscale('log')
		plt.xscale('log')
		plt.show()"""
	return sp.wave, sp.flux*Lnorm+acc_cont, R*Rsol


def compute_luminosity(wave, flux, Rstar, wavelength_start=0.0, wavelength_end = np.inf):
	"""
	Compute the luminosity of the star between given wavelengths.

	Parameters:
	wave (array): Wavelength array in Angstroms.
	flux (array): Flux array in erg/cm^2/s/Å.
	wavelength_start (float): Starting wavelength in Angstroms.
	wavelength_end (float): Ending wavelength in Angstroms.

	Returns:
	float: Luminosity in erg/s.
	"""
	# Mask to select the wavelength range
	mask = (wave >= wavelength_start) & (wave <= wavelength_end)

	# Integrate the flux over the selected wavelength range
	integrated_flux = np.trapz(flux[mask]*4.*np.pi*Rstar*Rstar, wave[mask])

	mean_energy = np.trapz(flux[mask]*4.*np.pi*Rstar*Rstar*(12398.0/wave[mask]), wave[mask])/integrated_flux
	
	# Convert to luminosity (erg/s)
	# Note: The factor of 4πR^2 is already included in the flux normalization, 
	# so we don't need to include it again here.
	luminosity = integrated_flux

	return luminosity, mean_energy*eV

def compute_fuv_luminosity_over_time(mstar, metallicity=0.0, age_start=1e5, age_end=1e7, age_res=40):
	ages = np.logspace(np.log10(age_start), np.log10(age_end), age_res)
	fuv_luminosities = np.zeros(ages.shape)

	for iage, age in enumerate(ages):
		wave, flux, Rstar = get_spectra(mstar, age, metallicity)
		#plt.plot(wave, flux)
		fuv_luminosity, mean_e = compute_luminosity(wave, flux, Rstar, 910., 2070.0)  # FUV range in Angstroms
		fuv_luminosities[iage] =  fuv_luminosity
	
	#plt.xscale('log')
	return ages, fuv_luminosities


def compute_fractional_uv_luminosity_over_time(mstar, metallicity=0.0, age_start=1e5, age_end=1e7, age_res=40, ages = None, Mdot_accs = 0.0):
	if ages is None:
		ages = np.logspace(np.log10(age_start), np.log10(age_end), age_res)
	uv_luminosities = np.zeros(ages.shape)
	frac_uv = np.zeros(ages.shape)
	
	Mdot_accs = np.ones(len(ages))*Mdot_accs

	for iage, age in enumerate(ages):
		wave, flux, Rstar = get_spectra(mstar, age, metallicity, Mdot_acc=Mdot_accs[iage])
		#plt.plot(wave, flux)
		uv_luminosity, mean_e = compute_luminosity(wave, flux, Rstar, 0., 2070.0) 
		tot_luminosity, mean_e = compute_luminosity(wave, flux, Rstar, 0., np.inf)  # FUV range in Angstroms
		uv_luminosities[iage] =  uv_luminosity
		frac_uv[iage] =  uv_luminosity/tot_luminosity
	
	#plt.xscale('log')
	return ages, frac_uv, uv_luminosities
	


def compute_counts_over_time(mstar, metallicity=0.0, age_start=1e5, age_end=1e7, age_res=40, ages = None, Mdot_accs = 0.0, wavelim=910.0):
	if ages is None:
		ages = np.logspace(np.log10(age_start), np.log10(age_end), age_res)
	counts = np.zeros(ages.shape)
	
	Mdot_accs = np.ones(len(ages))*Mdot_accs

	for iage, age in enumerate(ages):
		wave, flux, Rstar = get_spectra(mstar, age, metallicity, Mdot_acc=Mdot_accs[iage])
		#plt.plot(wave, flux)
		uv_luminosity, mean_e = compute_luminosity(wave, flux, Rstar, 0., wavelim) 
		counts[iage] =  uv_luminosity/mean_e
	
	#plt.xscale('log')
	return ages, counts


def compute_fractional_uv_luminosity_over_time(mstar, metallicity=0.0, age_start=1e5, age_end=1e7, age_res=40, ages = None, Mdot_accs = 0.0, wavelim=2070.0):
	if ages is None:
		ages = np.logspace(np.log10(age_start), np.log10(age_end), age_res)
	uv_luminosities = np.zeros(ages.shape)
	frac_uv = np.zeros(ages.shape)
	
	Mdot_accs = np.ones(len(ages))*Mdot_accs

	for iage, age in enumerate(ages):
		wave, flux, Rstar = get_spectra(mstar, age, metallicity, Mdot_acc=Mdot_accs[iage])
		#plt.plot(wave, flux)
		uv_luminosity, mean_e = compute_luminosity(wave, flux, Rstar, 0., wavelim) 
		tot_luminosity, mean_e = compute_luminosity(wave, flux, Rstar, 0., np.inf)  # FUV range in Angstroms
		uv_luminosities[iage] =  uv_luminosity
		frac_uv[iage] =  uv_luminosity/tot_luminosity
	
	#plt.xscale('log')
	return ages, frac_uv, uv_luminosities

# Define a function to plot the evolution of FUV luminosity
def plot_fuv_luminosity_evolution(stellar_masses, metallicity=0.0):
	plt.figure(figsize=(12, 8))

	for mstar in stellar_masses:
		ages, fuv_luminosities = compute_fuv_luminosity_over_time(mstar, metallicity)
		plt.plot(ages, fuv_luminosities, label=f'{mstar} $M_\\odot$')

	plt.xlabel('Age (years)')
	plt.ylabel('FUV Luminosity (erg/s)')
	plt.yscale('log')
	plt.xscale('log')
	plt.ylim([1e33, 1e40])
	plt.legend()
	plt.grid(True)
	plt.show()
	
# Define a function to plot the evolution of FUV luminosity
def plot_fractional_uv_luminosity_evolution(stellar_masses, metallicity=0.0, Mdot_accs=0.0):
	plt.figure(figsize=(12, 8))

	for mstar in stellar_masses:
		ages, frac_uv, fuv_luminosities = compute_fractional_uv_luminosity_over_time(mstar, metallicity, Mdot_accs=Mdot_accs)
		plt.plot(ages, frac_uv, label=f'{mstar} $M_\\odot$')

	plt.xlabel('Age (years)')
	plt.ylabel('Fractional Lyman luminosity (erg/s)')
	plt.yscale('log')
	plt.xscale('log')
	plt.ylim([1e-10, 1.0])
	plt.legend()
	plt.grid(True)
	plt.show()
	
def plot_spectrum(mass, age, metallicity=0.0):
	
	wave, flux, R = get_spectra(mass, age, metallicity=metallicity)
	# Plot the spectrum
	plt.figure(figsize=(10, 6))
	plt.plot(wave, flux)
	plt.xlabel('Wavelength [Å]')
	plt.ylabel('Flux [erg cm$^2$ s$^{-1}$ Å$^{-1}$]')
	plt.xscale('log')
	plt.yscale('log')
	plt.grid(True)
	plt.show()

if __name__=='__main__':

	stellar_masses = [0.5, 1, 2]
	plot_fractional_uv_luminosity_evolution(stellar_masses, metallicity=0.0, Mdot_accs=0.0)
	exit()
	plot_fuv_luminosity_evolution(stellar_masses, metallicity=0.0)

	

