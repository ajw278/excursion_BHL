import numpy as np
import matplotlib.pyplot as plt
import stellar_evolution as se
import stellar_spectra as ss
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from scipy.interpolate import RegularGridInterpolator

plt.rc('text', usetex=True)

"""pplt.rc.update({
    'linewidth': 1, 'ticklabelweight': 'bold', 'axeslabelweight': 'bold'
})"""
G = 6.67e-8
Msol  =1.989e33
au = 1.496e13
sig_SB = 5.67e-5 #K^-4
mp = 1.6726e-24
k_B = 1.81e-16 #cgs K^-1
Lsol = 3.828e33 #erg/s
year = 365.*24.*60.0*60.0

ref_density = 1e-20


def compute_Rcrit(Mstar, Lstar, flang=np.pi/2, mu=2.3):
	const1 = np.sqrt(8.*np.pi*sig_SB)
	const2 = (2.*G*mu*mp/k_B)**2
	
	return const1*const2*(Mstar**2 /np.sqrt(Lstar*flang))



def compute_Rwind(Mstar, Lstar_UV, rho0, eps_geo=0.1):
	const1 = (G*Mstar*2.)**(1.5)
	const2 = 2./np.pi
	return ((const2/const1)*Lstar_UV*eps_geo/rho0)**2
	

def compute_Rstrom(Phi_Ly, rho0, alphaB=2.6e-13):
	return ((3./4./np.pi)*Phi_Ly*(1./alphaB)*(mp/rho0)**2 )**(1./3.)
	
def compute_Mdotwind(Mstar, Lstar_UV, Racc, eps_geo=0.1):
	return Lstar_UV*eps_geo*(Racc/G/Mstar)

def get_Rwind_interpolator(mstar_input):	
	try:
		# Load precomputed grids
		mstar_space = np.load('Rwind_mstar.npy')
		age_space = np.load('Rwind_age.npy')
		mdot_space = np.load('Rwind_mdotacc.npy')
		Rwinds = np.load('Rwind_grid.npy')
	except:
		construct_grid()
		mstar_space = np.load('Rwind_mstar.npy')
		age_space = np.load('Rwind_age.npy')
		mdot_space = np.load('Rwind_mdotacc.npy')
		Rwinds = np.load('Rwind_grid.npy')
		
	
	# Find the closest stellar mass in the grid
	mstar_idx = np.abs(mstar_space - mstar_input).argmin()
	# Interpolation in age and Mdot space for the closest stellar mass
	Rwind_interpolator = RegularGridInterpolator((np.log10(age_space), np.log10(mdot_space)), np.log10(Rwinds[mstar_idx, :, :].T), bounds_error=False, fill_value=np.nan)
	
	return Rwind_interpolator


def interpolate_Rwind(mstar_input, age_input, mdot_input, rho0=ref_density, Rwind_interpolator=None, debug=False):
	if Rwind_interpolator is None:
		Rwind_interpolator = get_Rwind_interpolator(mstar_input)

	# Normalize the input accretion rate by the square of the stellar mass
	mdot_norm_input = np.asarray(mdot_input)
	if isinstance(mdot_norm_input, float) or isinstance(mdot_norm_input, np.float64):
		mdot_norm_input = np.array([mdot_norm_input])
	
	
	mdot_norm_input[mdot_norm_input >1e-6] =1e-6
	mdot_norm_input[mdot_norm_input <1e-13] =1e-13
	
	
	if isinstance(age_input, float) or isinstance(age_input, np.float64):
		age_input = np.array([age_input])
	age_input[age_input <0.1] =0.1
	age_input[age_input >30.0] =30.0
	
	if debug:
		interpolated_Rwind = 10.**Rwind_interpolator((np.log10(age_input), np.log10(mdot_norm_input)))
		print(np.log10(age_input), np.log10(mdot_norm_input))
		print(interpolated_Rwind*(ref_density/rho0)**2/au)
		
		Mdv =  mdot_input
		_, ion_frac_acc, LUV_acc = ss.compute_fractional_uv_luminosity_over_time(mstar_input, 
						                                             metallicity=0.0, 
						                                             ages=age_input*1e6, 
						                                             Mdot_accs=mdot_input, 
						                                             wavelim=2070.0)
		Rwind_dbg = compute_Rwind(mstar_input*Msol, LUV_acc, rho0)
		print('Rwind direct:',Rwind_dbg/au)
		print('Rwind interp:', interpolated_Rwind*(ref_density/rho0)**2/au)
		
		
	else:
		try:
			interpolated_Rwind = 10.**Rwind_interpolator((np.log10(age_input), np.log10(mdot_norm_input)))
		except:
			print(age_input, mdot_norm_input)
			exit()
	"""
	_, ion_frac_acc, LUV_acc = ss.compute_fractional_uv_luminosity_over_time(mstar_input, 
					                                             metallicity=0.0, 
					                                             ages=age_input*1e6, 
					                                             Mdot_accs=mdot_input, 
					                                             wavelim=2070.0)
					                                             	
	compute_Rwind(mstar_input*Msol, LUV_acc, rho0) #
	"""
	#print("Warning: you were checking the interpolation before!")		                                             
	return interpolated_Rwind*(ref_density/rho0)**2
	

def create_contour_plot_for_star(mstar_input, ax, norm, cmap,levels=np.arange(1., 7.5, 0.5)):
	try:
		# Load precomputed grids
		mstar_space = np.load('Rwind_mstar.npy')
		age_space = np.load('Rwind_age.npy')
		mdot_space = np.load('Rwind_mdotacc.npy')
		Rwinds = np.load('Rwind_grid.npy')
	except:
		construct_grid()
		mstar_space = np.load('Rwind_mstar.npy')
		age_space = np.load('Rwind_age.npy')
		mdot_space = np.load('Rwind_mdotacc.npy')
		Rwinds = np.load('Rwind_grid.npy')
		
		


	# Find the closest stellar mass in the grid
	mstar_idx = np.abs(mstar_space - mstar_input).argmin()
	closest_mstar = mstar_space[mstar_idx]
	
	print(Rwinds.shape)

	age_space_int = np.logspace(-1.0, 1.0, 40)
	mdot_space_int = np.logspace(-10., -6., 45)

	# Create meshgrid for age and mdot space
	age_grid, mdot_grid = np.meshgrid(age_space_int, mdot_space_int)
	age_grid_in, mdot_grid_in = np.meshgrid(age_space, mdot_space)

	# Interpolate Rwind values
	Rwind_values = np.array([interpolate_Rwind(mstar_input, age, mdot) for age, mdot in zip(np.ravel(age_grid), np.ravel(mdot_grid))])
	Rwind_values = Rwind_values.reshape(age_grid.shape)
	
	print(np.log10(Rwinds[mstar_idx,:,:]/au))
	# Create contour plot
	contour = ax.contourf(age_space_int, mdot_space_int, np.log10(Rwind_values/au), cmap=cmap, norm=norm, levels=levels) # Create contour plot
	print(age_grid_in.shape, mdot_grid_in.shape, Rwinds[mstar_idx,:,:].shape)
	sc = ax.scatter(age_grid_in, mdot_grid_in*(mstar_input)**2, c=np.log10(Rwinds[mstar_idx,:,:]/au), cmap=cmap, norm=norm, edgecolors='black')
	
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_title(f'Stellar Mass: {mstar_input} $M_\\odot$')
	ax.set_xlabel('Age (years)')
	ax.set_ylabel('Accretion Rate (normalized)')
	return contour

def construct_grid(mstar_space=np.logspace(-1., 1., 15), 
                   age_space=np.logspace(-1., 1.5, 40), 
                   mdot_space=np.logspace(-13., -6., 50)):
    
	Nmst = len(mstar_space)
	Nage = len(age_space)
	Nmdot = len(mdot_space)
	Rwinds = np.zeros((Nmst, Nmdot, Nage))

	np.save('Rwind_mstar', mstar_space)
	np.save('Rwind_age', age_space)
	np.save('Rwind_mdotacc', mdot_space)

	for imstar in range(Nmst):
		print('Mstar:', mstar_space[imstar])
		for imdot in range(Nmdot):
			Mdv = np.ones(len(age_space)) * mdot_space[imdot] 
			_, ion_frac_acc, LUV_acc = ss.compute_fractional_uv_luminosity_over_time(mstar_space[imstar], 
						                                             metallicity=0.0, 
						                                             ages=age_space*1e6, 
						                                             Mdot_accs=Mdv, 
						                                             wavelim=2070.0)
			Rwinds[imstar, imdot, :] = compute_Rwind(mstar_space[imstar]*Msol, LUV_acc, ref_density)

	np.save('Rwind_grid', Rwinds)
    
	return Rwinds

def plot_Rstrom(rho0=1e-24):

	
	# Create a grid of stellar mass and luminosity
	Mstar = np.logspace(-0.5, 1., 50) * Msol # From 0.1 to 30 solar masses
	Lstar = np.logspace(10, 50, 60) # From 0.1 to 1000 solar luminosities

	# Create a meshgrid
	Mstar_grid, Lstar_grid = np.meshgrid(Mstar, Lstar, indexing='ij')

	# Compute Rcrit for each point in the grid
	Rcrit_grid = compute_Rstrom(Lstar_grid, rho0)
	
	
	# Plotting
	#fig, ax = plt.subplots(figsize=(10, 8))
	
	fig = pplt.figure( share=False, refwidth=3.9)
	ax = fig.subplot(111)
	cp = ax.contourf(Mstar_grid / Msol, Lstar_grid , np.log10(Rcrit_grid/au), transpose=True, cmap='viridis',  levels=np.arange(1., 7.5, 0.5))
	# Shorter colorbar for critical radius
	cbar = ax.colorbar(cp, label='log $R_\mathrm{S}$ [au]', loc='r')
	#cbar.ax.get_yaxis().labelpad = 15
	ax.set_xscale('log')
	ax.set_yscale('log')
	#ax.set_xlim([Mstar[0]/Msol, Mstar[-1]/Msol])
	ax.set_xlabel('Stellar mass: $m_*$ [$M_\odot$]')
	ax.set_ylabel('High energy stellar Luminosity: $L_{*,\mathrm{<2070}}$ [$L_\odot$]')
	# Define initial masses
	initial_masses = [0.5,  1.0,1.3, 2.0, 3.0, 4.0]  # in solar masses

	# Time evolution
	ages = np.logspace(np.log10(0.1), np.log10(10.0), 6)*1e6  # From 0 to 10 Myr, every 1 Myr
	ages_evol = np.logspace(np.log10(0.1), np.log10(10.0), 100)*1e6  # From 0 to 10 Myr, every 1 Myr
	cmap = plt.get_cmap('plasma', len(ages))  # Use a colormap

	# Plot the time evolution of each initial mass
	for iminit, minit in enumerate(initial_masses):
		_, _, log_Ls, _, masses = se.fetch_stellar_properties(minit , ages)
		
		
		mda = np.ones(len(ages))*1e-7 *minit**2
		
		_,  counts_acc_h =  ss.compute_counts_over_time(minit, metallicity=0.0,ages = ages, Mdot_accs=mda, wavelim=910.0)
		_,  counts_acc_l =  ss.compute_counts_over_time(minit, metallicity=0.0,ages = ages, Mdot_accs=mda*0.1, wavelim=910.0)
		_,  counts_na =  ss.compute_counts_over_time(minit, metallicity=0.0,ages = ages, Mdot_accs=0.0, wavelim=910.0)
		print(counts_acc_h)
		
		if iminit ==0:
			
			ax.scatter(masses, counts_acc_h, c=np.log10(ages/1e6), marker='s',s=20, cmap=cmap, edgecolor='black', label='$\dot{M}_\mathrm{acc} = 10^{-7} (m_*/1\,M_\odot)^2 \, M_\odot$ yr$^{-1}$')
			ax.scatter(masses, counts_acc_l, c=np.log10(ages/1e6), marker='^',  s=80, cmap=cmap, edgecolor='black', label='$\dot{M}_\mathrm{acc} = 10^{-8} (m_*/1\,M_\odot)^2 \, M_\odot$ yr$^{-1}$')
			
			scatter = ax.scatter(masses, counts_na, c=np.log10(ages/1e6),s=20, cmap=cmap, edgecolor='black', label='No accretion')
		else:
			ax.scatter(masses, counts_acc_h, c=np.log10(ages/1e6), marker='s',s=20, cmap=cmap, edgecolor='black')
			
			ax.scatter(masses, counts_acc_l, c=np.log10(ages/1e6), s=80, marker='^', cmap=cmap, edgecolor='black')
			scatter = ax.scatter(masses, counts_na, c=np.log10(ages/1e6),s=20, cmap=cmap, edgecolor='black')
		#ax.plot(masses, 10.**log_Ls, color='r', linewidth=0.2)

	# Add a colorbar for ages
	# Add a colorbar for ages at the top
	cbar_top = ax.colorbar(scatter, orientation='horizontal',label='log Age [Myr]', loc='t')
	#cbar_top.set_label('log Age [Myr]', labelpad=10)"""
	ax.legend(loc='best', fontsize=8, ncol=1)
	ax.format(yscale='log', ylim=(Lstar[0], Lstar[-1]),  yformatter=('sci', 0), xformatter=('sigfig', 7))
	ax.tick_params(which='both', left=True, right=True, top=True, bottom=True)
	fig.savefig('Rstrom.pdf', bbox_inches='tight', format='pdf')
	plt.show()
	


def plot_Rwind(rho0=1e-24):

	import proplot as pplt
	# Create a grid of stellar mass and luminosity
	Mstar = np.logspace(-0.5, 1., 50) * Msol # From 0.1 to 30 solar masses
	Lstar = np.logspace(-10, 3, 60) * Lsol # From 0.1 to 1000 solar luminosities

	# Create a meshgrid
	Mstar_grid, Lstar_grid = np.meshgrid(Mstar, Lstar, indexing='ij')

	# Compute Rcrit for each point in the grid
	Rcrit_grid = compute_Rwind(Mstar_grid, Lstar_grid, rho0)

	# Plotting
	#fig, ax = plt.subplots(figsize=(10, 8))
	
	fig = pplt.figure( share=False, refwidth=3.9)
	ax = fig.subplot(111)
	cp = ax.contourf(Mstar_grid / Msol, Lstar_grid / Lsol, np.log10(Rcrit_grid/au), transpose=True, cmap='viridis',  levels=np.arange(1., 7.5, 0.5))
	# Shorter colorbar for critical radius
	cbar = ax.colorbar(cp, label='log $R_\mathrm{wind}$ [au]', loc='r')
	#cbar.ax.get_yaxis().labelpad = 15
	ax.set_xscale('log')
	ax.set_yscale('log')
	#ax.set_xlim([Mstar[0]/Msol, Mstar[-1]/Msol])
	ax.set_xlabel('Stellar mass: $m_*$ [$M_\odot$]')
	ax.set_ylabel('High energy stellar Luminosity: $L_{*,\mathrm{<2070}}$ [$L_\odot$]')
	# Define initial masses
	initial_masses = [0.5,  1.0,1.3, 2.0, 3.0, 4.0]  # in solar masses

	# Time evolution
	ages = np.logspace(np.log10(0.1), np.log10(10.0), 6)*1e6  # From 0 to 10 Myr, every 1 Myr
	ages_evol = np.logspace(np.log10(0.1), np.log10(10.0), 100)*1e6  # From 0 to 10 Myr, every 1 Myr
	cmap = plt.get_cmap('plasma', len(ages))  # Use a colormap

	# Plot the time evolution of each initial mass
	for iminit, minit in enumerate(initial_masses):
		_, _, log_Ls, _, masses = se.fetch_stellar_properties(minit , ages)
		
		
		mda = np.ones(len(ages))*1e-7 *minit**2
		
		_, ion_frac_acc, LUV_acc =  ss.compute_fractional_uv_luminosity_over_time(minit, metallicity=0.0,ages = ages, Mdot_accs=mda, wavelim=2070.0)
		_, ion_frac_acc_low, LUV_acc_low =  ss.compute_fractional_uv_luminosity_over_time(minit, metallicity=0.0,ages = ages, Mdot_accs=mda*0.1, wavelim=2070.0)
		
		_, ion_frac, LUV =  ss.compute_fractional_uv_luminosity_over_time(minit, metallicity=0.0,ages = ages, Mdot_accs=0.0, wavelim=2070.0)
		
		if iminit ==0:
			
			ax.scatter(masses, LUV_acc/Lsol, c=np.log10(ages/1e6), marker='s',s=20, cmap=cmap, edgecolor='black', label='$\dot{M}_\mathrm{acc} = 10^{-7} (m_*/1\,M_\odot)^2 \, M_\odot$ yr$^{-1}$')
			ax.scatter(masses, LUV_acc_low/Lsol, c=np.log10(ages/1e6), marker='^',  s=80, cmap=cmap, edgecolor='black', label='$\dot{M}_\mathrm{acc} = 10^{-8} (m_*/1\,M_\odot)^2 \, M_\odot$ yr$^{-1}$')
			
			scatter = ax.scatter(masses, LUV/Lsol, c=np.log10(ages/1e6),s=20, cmap=cmap, edgecolor='black', label='No accretion')
		else:
			ax.scatter(masses, LUV_acc/Lsol, c=np.log10(ages/1e6), marker='s',s=20, cmap=cmap, edgecolor='black')
			
			ax.scatter(masses, LUV_acc_low/Lsol, c=np.log10(ages/1e6), s=80, marker='^', cmap=cmap, edgecolor='black')
			scatter = ax.scatter(masses, LUV/Lsol, c=np.log10(ages/1e6),s=20, cmap=cmap, edgecolor='black')
		#ax.plot(masses, 10.**log_Ls, color='r', linewidth=0.2)

	# Add a colorbar for ages
	# Add a colorbar for ages at the top
	cbar_top = ax.colorbar(scatter, orientation='horizontal',label='log Age [Myr]', loc='t')
	#cbar_top.set_label('log Age [Myr]', labelpad=10)"""
	ax.legend(loc='best', fontsize=8, ncol=1)
	ax.format(yscale='log', ylim=(Lstar[0]/Lsol, Lstar[-1]/Lsol),  yformatter=('sci', 0), xformatter=('sigfig', 7))
	ax.tick_params(which='both', left=True, right=True, top=True, bottom=True)
	fig.savefig('Rwind.pdf', bbox_inches='tight', format='pdf')
	plt.show()
	
	

def plot_Rcrit():
	import proplot as pplt
	# Create a grid of stellar mass and luminosity
	Mstar = np.logspace(-1.5, 1.0, 50) * Msol # From 0.1 to 30 solar masses
	Lstar = np.logspace(-3, 3, 60) * Lsol # From 0.1 to 1000 solar luminosities

	# Create a meshgrid
	Mstar_grid, Lstar_grid = np.meshgrid(Mstar, Lstar, indexing='ij')

	# Compute Rcrit for each point in the grid
	Rcrit_grid = compute_Rcrit(Mstar_grid, Lstar_grid)

	# Plotting
	#fig, ax = plt.subplots(figsize=(10, 8))
	
	fig = pplt.figure(share=False, refwidth=3.9)
	ax = fig.subplot(111)
	cp = ax.contourf(Mstar_grid / Msol, Lstar_grid / Lsol, np.log10(Rcrit_grid/au), transpose=True, cmap='viridis',  levels=np.arange(1., 7.5, 0.5))
	# Shorter colorbar for critical radius
	cbar = ax.colorbar(cp, label='log $R_\mathrm{therm}$ [au]', loc='r')
	#cbar.ax.get_yaxis().labelpad = 15
	ax.set_xscale('log')
	ax.set_yscale('log')
	#ax.set_xlim([Mstar[0]/Msol, Mstar[-1]/Msol])
	ax.set_xlabel('Stellar mass: $m_*$ [$M_\odot$]')
	ax.set_ylabel('Stellar Luminosity: $L_*$ [$L_\odot$]')
	# Define initial masses
	initial_masses = [0.1, 0.5,  1.0, 1.5, 2.0, 5]  # in solar masses

	# Time evolution
	ages = np.logspace(np.log10(0.1), np.log10(10.0), 21)*1e6  # From 0 to 10 Myr, every 1 Myr
	ages_evol = np.logspace(np.log10(0.1), np.log10(10.0), 100)*1e6  # From 0 to 10 Myr, every 1 Myr
	cmap = plt.get_cmap('plasma', len(ages))  # Use a colormap

	# Plot the time evolution of each initial mass
	for minit in initial_masses:
		_, _, log_Ls, _, masses = se.fetch_stellar_properties(minit , ages)
		_, _, log_Ls_hr, _, masses_hr = se.fetch_stellar_properties(minit, ages_evol)
		
		   
		scatter = ax.scatter(masses, 10.**log_Ls, c=np.log10(ages/1e6), cmap=cmap, edgecolor='black')
		#ax.plot(masses, 10.**log_Ls, color='r', linewidth=0.2)

	# Add a colorbar for ages
	# Add a colorbar for ages at the top
	cbar_top = ax.colorbar(scatter, orientation='horizontal',label='log Age [Myr]', loc='t')
	#cbar_top.set_label('log Age [Myr]', labelpad=10)"""
	
	ax.format(yscale='log', ylim=(Lstar[0]/Lsol, Lstar[-1]/Lsol),  yformatter=('sigfig', 7), xformatter=('sigfig', 7))
	ax.tick_params(which='both', left=True, right=True, top=True, bottom=True)
	fig.savefig('Rtherm.pdf', bbox_inches='tight', format='pdf')
	plt.show()
		
if __name__=='__main__':
	#construct_grid()
	
	#plot_Rwind(rho0=1e-22)
	#exit()
	# Load data to determine the normalization range
	mstar_space = np.load('Rwind_mstar.npy')
	age_space = np.load('Rwind_age.npy')
	mdot_space = np.load('Rwind_mdotacc.npy')
	Rwinds = np.load('Rwind_grid.npy')

	# Calculate min and max for normalization
	vmin = 1.0
	vmax = 7.0
	norm = Normalize(vmin=vmin, vmax=vmax)
	cmap = 'viridis'

	# Set up figure and axes for 2x2 plot
	fig, axes = plt.subplots(2, 2, figsize=(12, 10))

	# List of stellar masses to plot
	stellar_masses = [0.6, 0.7, 0.8, 1.0]

	# Plot each stellar mass
	for mstar, ax in zip(stellar_masses, axes.flatten()):
		contour = create_contour_plot_for_star(mstar, ax, norm, cmap)

	# Adjust layout and add colorbar
	plt.tight_layout()
	cbar = fig.colorbar(contour, ax=axes.ravel().tolist(), orientation='vertical')
	cbar.set_label('Wind Radius (Rwind)')

	plt.show()
	#plot_Rcrit()
	#plot_Rstrom(rho0=1e-24)
