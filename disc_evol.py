import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from scipy.integrate import solve_ivp
from consts_defaults import *
import wind_limits as wl
import copy

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#e41a1c','#984ea3', '#dede00', '#999999','cyan', 'teal']

"""
mdot_tacc
Given a BHL accretion rate and star-disc accretion time-scale, self-consistently solve the disc evolution
"""
def mdot_tacc(Mdot_BHL, R_BHL, teval_, tarr_, dts, mstars, rho_BHL, dv_BHL, plot=False, mu=-.5, sigma=1.0 , fM=0.0, fM_disp = 0.0, G=Gcgs, wind=False):

	# Apply convolution
	mdot_star = np.zeros((len(dts), len(Mdot_BHL), len(teval_)))
	disc_mass = np.zeros(mdot_star.shape)
	frac_tend = np.zeros(mdot_star.shape)
	disc_radius = np.zeros(mdot_star.shape)
	disc_vturb = np.zeros(mdot_star.shape)
	rho_amb = np.zeros(mdot_star.shape)
	v_amb = np.zeros(mdot_star.shape)
	mdot_BHL = np.zeros(mdot_star.shape)
	

	if plot:
		fig, ax = plt.subplots(figsize=(5.,4.))
	for ikern, dt_ in enumerate(dts):
		dt= dt_
		# Calculate the Gaussian kernel for each element in data
		for istar in range(len(Mdot_BHL)):
			Rwinterp = wl.get_Rwind_interpolator(mstars[istar]/Msol2g)
			
			if dt_=='ln':
				#dt = 10.**np.random.normal(loc=mu, scale=sigma)
				dt = Myr2s*10.**np.random.uniform(mu-sigma, mu+sigma)
			
			tmp_data =copy.copy(Mdot_BHL[istar])
			
			rho_func = interpolate.interp1d(tarr_[istar], rho_BHL[istar])
			BHL_func = interpolate.interp1d(tarr_[istar], tmp_data)
			
			
			RBHL_func = interpolate.interp1d(tarr_[istar], R_BHL[istar])
			

			Rdisc = 250.0*(mstars[istar]/Msol2g)*au2cm
			
			if wind:
				def mmdotEt_func(t, y):

					#print(t/Myr2s)
					MdBHL = BHL_func(t)
					
					Mdot_Msoly = y[0]*year2s/dt/Msol2g
					Rwind = wl.interpolate_Rwind(mstars[istar]/Msol2g, t/Myr2s, Mdot_Msoly, rho0=rho_func(t), Rwind_interpolator=Rwinterp)
					RBHL = RBHL_func(t)
					MdBHL *= float(np.exp(-(float(Rwind)/float(RBHL))**2))

					mdot = -y[0]/dt + MdBHL
					vin2 = 2*G*mstars[istar]/Rdisc

					vt_ = np.sqrt(2.*max(float(y[1]),0.0)/max(float(y[0]), 1e-30))
					dt_e = 0.1*Rdisc/max(vt_, 1e-60)

					Ein  =  0.5*MdBHL*vin2
					Eout =  y[1]/dt_e
					Eacc = 0.5*y[0]*vt_*vt_/dt
					Edot = Ein-Eout-Eacc
					#if mdot>0.0 and y[0]>minit:
					#	mdot = mdot*np.exp(1.0)*np.exp(-(y[0]/ulim)**2)
					return mdot, Edot
			else:
				def mmdotEt_func(t, y):

					#print(t/Myr2s)
					MdBHL = BHL_func(t)
						
					mdot = -y[0]/dt + MdBHL
					vin2 = 2*G*mstars[istar]/Rdisc

					vt_ = np.sqrt(2.*max(float(y[1]),0.0)/max(float(y[0]), 1e-30))
					dt_e = 0.1*Rdisc/max(vt_, 1e-60)

					Ein  =  0.5*MdBHL*vin2
					Eout =  y[1]/dt_e
					Eacc = 0.5*y[0]*vt_*vt_/dt
					Edot = Ein-Eout-Eacc
					#if mdot>0.0 and y[0]>minit:
					#	mdot = mdot*np.exp(1.0)*np.exp(-(y[0]/ulim)**2)
					return mdot, Edot
			
			m0 = fM*Msol2g*((mstars[istar]/Msol2g)**2)
			if m0>0.0:
				m0 = 10.**(np.log10(m0)+ np.random.normal(loc=0.0, scale=fM_disp))
			sol = solve_ivp(mmdotEt_func, (teval_[0], teval_[-1]), [m0, 0.0], method='RK45', t_eval=teval_, rtol=1e-10, atol=1e-10)
			mdisc = sol.y[0].flatten()
			Edisc = sol.y[1].flatten()

			vt = np.sqrt(2.*Edisc/(mdisc+1e-60))

			if istar==0 and plot and dt_!='ln':
				#plt.plot(tarr_, kernel[npad:-npad], c=CB_color_cycle[ikern], linestyle='dotted', linewidth=1, label='Kernel')
				plt.plot(sol.t.flatten()/Myr2s, mdisc/Msol2g, c=CB_color_cycle[ikern], linestyle='solid', linewidth=1, label='$\\tau_\mathrm{acc}=%.1lf$ Myr'%(dt/Myr2s))
			
			
			
			#Assuming constant time intervals between steps! 
			ihalf = np.arange(len(teval_))//2
			sd = mdisc/dt
			
			
			Rwind_arr = wl.interpolate_Rwind(mstars[istar]/Msol2g, teval_/Myr2s, sd*year2s/Msol2g, rho0=rho_func(teval_), Rwind_interpolator=Rwinterp)
			RBHL_eval = RBHL_func(teval_[0])
			BHL_eval = BHL_func(teval_)
			BHL_eval[Rwind_arr>RBHL_eval] = 0.0
			mdot_BHL[ikern][istar] = BHL_eval
			
			print('Stellar mass:', mstars[istar]/Msol2g)
			"""plt.plot(teval_/Myr2s, mdot_BHL[ikern][istar]*year2s/Msol2g)
			plt.plot(teval_/Myr2s, BHL_func(teval_)*year2s/Msol2g)
			plt.plot(teval_/Myr2s, Rwind_arr/RBHL_eval, label='Rrat')
			plt.yscale('log')
			plt.legend(loc='best')
			plt.show()"""
			
			disc_mass[ikern][istar] = mdisc
			mdot_star[ikern][istar] = sd
			frac_tend[ikern][istar] = 1.-mdisc[ihalf]*np.exp(-(teval_-teval_[ihalf])/dt)/(mdisc+1e-30)

			disc_radius[ikern][istar] = RBHL_func(teval_[0])
			disc_vturb[ikern][istar] = vt
			
			
			
			
			rho_amb[ikern][istar] = np.interp(teval_, tarr_[istar], rho_BHL[istar])
			v_amb[ikern][istar] = np.interp(teval_, tarr_[istar], dv_BHL[istar])

			iacc = np.where(BHL_func(teval_)>mdot_star[ikern][istar])[0]
			for ir in iacc:
				disc_radius[ikern][istar][ir:] = RBHL_func(teval_[ir])

			
			if ((istar+1)%5)==0:
				print('Disc evolution computation complete for star %d/%d '%(istar+1, len(Mdot_BHL))) 
		
		print('Disc mass calculation complete for %d/%d accretion times'%(ikern+1, len(dts)))
	if plot:
		ax.tick_params(which='both', top=True, bottom=True, left=True, right=True)
		plt.ylabel('Disc mass: ${M}_\mathrm{disc}$ [$M_\odot$]')
		plt.xlabel('Time: $t$ [Myr]')
		plt.xlim([0., int(teval_[-1]/Myr2s+0.5)])
		plt.yscale('log')
		
		plt.legend(loc='best', ncols=2, fontsize=8)
		plt.savefig('disc_mass_evol.pdf', format='pdf', bbox_inches='tight')
		plt.show()

		fig, ax = plt.subplots(figsize=(5.,4.))
		plt.plot(teval_/Myr2s, Mdot_BHL[0]*year2s/Msol2g, c='k', linestyle='solid', linewidth=1, label='BHL')
		for ikern, dt in enumerate(dts):
			if dt!='ln':
				plt.plot(teval_/Myr2s, mdot_star[ikern][0]*year2s/Msol2g, c=CB_color_cycle[ikern], linestyle='solid', linewidth=1, label='$\\tau_\mathrm{acc}=%.1lf$ Myr'%(dt/Myr2s))
		
		plt.yscale('log')
		plt.xlim([0., int(teval_[-1]/Myr2s+0.5)])
		plt.ylabel('Accretion rate: $\dot{M}_\mathrm{acc}$ [$M_\odot$ yr$^{-1}$]')
		plt.xlabel('Time: $t$ [Myr]')
		ax.tick_params(which='both', top=True, bottom=True, left=True, right=True)
		plt.legend(loc='best', ncols=2, fontsize=8)
		plt.savefig('accretion_rate_evol.pdf', bbox_inches='tight', format='pdf')
		plt.show()


		fig, ax = plt.subplots(figsize=(5.,4.))
		for ikern, dt in enumerate(dts):
			if dt!='ln':
				plt.plot(teval_/Myr2s, frac_tend[ikern][0], c=CB_color_cycle[ikern], linestyle='solid', linewidth=1, label='$\\tau_\mathrm{acc}=%.1lf$ Myr'%(dt/Myr2s))
		
		#plt.yscale('log')
		#plt.ylim([1e-5, 1.0])
		plt.ylim([0.0, 1.0])
		plt.xlim([0., int(teval_[-1]/Myr2s+0.5)])
		plt.ylabel('Half-life replenishment mass fraction: $f_{M, 1/2}$')
		plt.xlabel('Time: $t$ [Myr]')
		ax.tick_params(which='both', top=True, bottom=True, left=True, right=True)
		plt.legend(loc='best', ncols=2, fontsize=8)
		plt.savefig('material_fraction.pdf', bbox_inches='tight', format='pdf')
		plt.show()


		fig, ax = plt.subplots(figsize=(5.,4.))
		for ikern, dt in enumerate(dts):
			if dt!='ln':
				plt.plot(teval_/Myr2s, disc_radius[ikern][0]/au2cm, c=CB_color_cycle[ikern], linestyle='solid', linewidth=1, label='$\\tau_\mathrm{acc}=%.1lf$ Myr'%dt)
		
		plt.yscale('log')
		#plt.ylim([1e-5, 1.0])
		
		plt.xlim([0., int(teval_[-1]/Myr2s+0.5)])
		plt.ylabel('Disc radius: $R_\mathrm{disc}$')
		plt.xlabel('Time: $t$ [Myr]')
		ax.tick_params(which='both', top=True, bottom=True, left=True, right=True)
		plt.legend(loc='best', ncols=2, fontsize=8)
		plt.savefig('disc_radius.pdf', bbox_inches='tight', format='pdf')
		plt.show()




	return disc_mass, mdot_star, frac_tend, disc_radius, disc_vturb, mdot_BHL, rho_amb, v_amb 
