import numpy as np

from turbfuncs import *
from consts_defaults import *

import scipy.special as ss
import matplotlib.pyplot as plt

import star_forming_region as sfr_c
import sfr_database as sfrdb

import cloud as cl

plt.rc('text', usetex=True)

			
"""
class trajectory_grid
Generate a grid structure for the random drawing of trajectories. The grid structure comes with all of the physical parameters inbuilt
"""	
class trajectory_grid():
	def __init__(self, rmax=100.0*pc2cm, rmin=0.01*pc2cm, drfact=0.95, Ps_v=Ev_k,kappa = Omega_*np.sqrt(2.)/1e6/year2s, Omega=Omega_/1e6/year2s, h=h_*pc2cm, G=Gcgs, rho0=rho0_*Msol2g/(pc2cm)**3, Espect=Ev_k, eta=1.0, cs=cs_*1e5, **kwargs):
		self.rmax = rmax
		self.rmin = rmin
		
		self.drfact = drfact
		
		self.setup_grid()
		
		self.rmin = rmin
		self.rmax = rmax
		self.ksp =  np.logspace(np.log10(1./rmax), np.log10(1./rmin), 2048)
		self.drfact = drfact
		
		
		self.setup_grid()
		
		self.rho0 = rho0
		self.Espect=Ps_v
		self.G = G
		self.h = h
		self.Omega = Omega
		self.kappa = kappa
		self.cs = cs
		self.eta = eta
		self.Espect=Espect
		
		self.setup_qphys()
		
	
	
	def setup_grid(self):
		nlevel=1
		
		self.rlevels = [self.rmax]
		self.dr = [0.0]
		while self.rlevels[-1]>self.rmin:
			self.rlevels.append(self.rlevels[-1]*self.drfact)
			dr = self.rlevels[-1] - self.rlevels[-2]
			self.dr.append(dr)
			nlevel+=1
		
		self.nlevels = nlevel
		
		self.rlevels = np.array(self.rlevels)
		self.dr = np.array(self.dr)
	
	
	def setup_qphys(self):
		self.rhocs = []
		self.sig2_R = []
		self.sig2_k = []
		self.sig2v_R = []
		self.mu_lnrho = []
		self.delta_c = []
		self.Delta_S =[]
		self.Delta_Sv =[]
		self.tau_R = []
		
		for ir, r in enumerate(self.rlevels):
			k = 1./r
			rc = rho_crit(k, kappa = self.kappa, Omega=self.Omega, h=self.h, G=self.G, rho0=self.rho0, Espect=self.Espect, cs=self.cs)
			s2_v = variancev_R(r, self.ksp,  cs=self.cs, Espect = self.Espect, kappa=self.kappa, Omega=self.Omega, h=self.h)
			
			s2_k = variance_k(k, cs=self.cs, kappa=self.kappa, Espect = self.Espect)
			s2_R = variance_R(r, self.ksp, kappa=self.kappa, cs=self.cs, Espect = self.Espect)
			mu_lnrho = -s2_k / 2.0

			delta_c = np.log(rc/self.rho0) - mu_lnrho
			tau = self.eta*r/np.sqrt(vturb2_k(k, cs=self.cs, Espect = self.Espect))
			
			if ir==0:
				self.Delta_S.append(0.0)
				self.Delta_Sv.append(0.0)
			else:
				self.Delta_S.append(s2_R -self.sig2_R[-1] )
				self.Delta_Sv.append(s2_v - self.sig2v_R[-1]) # - self.sig2v_R[-1])
			
			self.sig2_R.append(s2_R)
			self.sig2v_R.append(s2_v)
			self.mu_lnrho.append(mu_lnrho)
			self.delta_c.append(delta_c)
			self.tau_R.append(tau)
			self.rhocs.append(rc)
			self.sig2_k.append(s2_k)
		
		self.rhocs = np.array(self.rhocs)
		self.sig2_R = np.array(self.sig2_R)
		self.sig2_k = np.array(self.sig2_k)
		self.sig2v_R = np.array(self.sig2v_R)
		self.mu_lnrho = np.array(self.mu_lnrho)
		self.delta_c = np.array(self.delta_c)
		self.Delta_S = np.array(self.Delta_S)
		self.Delta_Sv = np.array(self.Delta_Sv)
		self.tau_R = np.array(self.tau_R)

		return None


			
"""
class trajectory
Randomly draw a trajectory of Gaussian perturbations, given a grid structure (above)
"""	
class trajectory():
	def __init__(self, grid=None, delta=None, v=None, vdir=None, iscale=0, iresample=None, dt_factor=0.1, **kwargs):
	
		if grid is None:
			grid = trajectory_grid(**kwargs)
		
		self.grid = grid
		
		self.t = 0.0
		self.ts = np.array([])
		self.dt_factor = dt_factor
		self.dt = np.amin(self.grid.tau_R)*self.dt_factor
		

		self.kwargs = kwargs
		self.cloud = cl.bound_clump(**kwargs)

		self.icols = []
		self.icol = np.nan
		self.tcollapse = np.inf
		self.tdisperse = np.inf

		self.iscale = iscale

		self.Ddelta = np.zeros(self.grid.rlevels.shape)
		self.Dv = np.zeros((len(self.grid.rlevels),3))
		
		self.iscale = iscale
		if delta is None and v is None:
			self.resample_all(0)
		else:

			if v is None:
				self.resample_v_trajectory(0)
			else:
				self.Dv = np.diff(v, prepend=v[0], axis=0)
				if not iresample is None:
					self.resample_v_trajectory(iresample)


			if delta is None:
				self.resample_d_trajectory(0)
			else:
				self.Ddelta = np.diff(delta, prepend=delta[0])
				if not iresample is None:
					self.resample_d_trajectory(iresample)
		

		self.vs =[]
		self.deltas = []


		
	def resample_d_trajectory(self, imax):
		self.cloud = cl.bound_clump(**self.kwargs)
		DS = self.grid.Delta_S[imax:]
		u_delta = np.random.uniform(size=DS.shape)
		Delta_delta =  np.sqrt(2.*DS)*ss.erfinv(2.*u_delta-1.)
		self.Ddelta[imax:] = Delta_delta
		self.delta = np.cumsum(self.Ddelta)
		self.cloud.find_collapse(self)
		return self.Ddelta
		
	def resample_v_trajectory(self, imax):
		
		DSv = self.grid.Delta_Sv[imax:, np.newaxis]
		u_v = np.random.uniform(size=(len(DSv), 3))
		Delta_v = np.sqrt(2.*DSv)*ss.erfinv(2.*u_v-1.)
		self.Dv[imax:] = Delta_v
		self.v = np.cumsum(self.Dv, axis=0)
		return self.Dv
	
	def resample_all(self, imax):
		self.resample_v_trajectory(imax)
		self.resample_d_trajectory(imax)

	def add_trajectory(self, Ddelta=None, Dv=None, tstep=True):
		
		if Ddelta is None:
			Ddelta = self.Ddelta
		if Dv is None:
			Dv = self.Dv
		self.delta = np.cumsum(Ddelta)
		self.v = np.cumsum(Dv, axis=0)
		self.deltas.append(self.delta)
		self.vs.append(self.v)
		self.ts = np.append(self.ts, self.t)
		self.icols.append(self.icol)
		return None

	
	def time_step(self, dt=None, ilevel=None, cloud_evolve=True):
		
		if ilevel is None:
			ilevel = self.iscale
		if dt is None:
			dt = self.dt
		
		self.t = self.t+dt
		Ddelta_new = self.Ddelta
		Dv_new = self.Dv
		nrand = len(self.Ddelta[ilevel:])
		
		Ddelta_new[ilevel:] = self.Ddelta[ilevel:]*np.exp(-dt/self.grid.tau_R[ilevel:])
		Ddelta_new[ilevel:] += np.random.normal(loc=0.0, scale=1.0, size=nrand)*np.sqrt(self.grid.Delta_S[ilevel:]*(1.-np.exp(-2.*dt/self.grid.tau_R[ilevel:])))
		
		
		Dv_new[ilevel:] = self.Dv[ilevel:]*np.exp(-dt/self.grid.tau_R[ilevel:])[:, np.newaxis]
		Dv_new[ilevel:] += np.random.normal(loc=0.0, scale=1.0, size=(nrand, 3))*np.sqrt(self.grid.Delta_Sv[ilevel:]*(1.-np.exp(-2.*dt/self.grid.tau_R[ilevel:])))[:, np.newaxis]/np.sqrt(3.0)
		
		self.Ddelta = Ddelta_new
		self.Dv = Dv_new
		
		self.add_trajectory()
		if cloud_evolve:
			self.icol = self.cloud.evolve(self)
			if self.cloud.formed and self.t<self.tcollapse:
				self.tcollapse = self.t
			if self.cloud.dispersed and self.t<self.tdisperse:
				self.tdisperse = self.t
	
	def evolve(self, tend=None, ilevel=None, dt=None, cloud_evolve=True, tlim=100.*Myr2s, terminate='form'):
		
		if len(self.ts)==0:
			self.add_trajectory()
		if terminate=='form':
			while not self.cloud.formed:
				self.time_step(dt=dt, ilevel=ilevel)
		elif terminate=='disperse':
			while not self.cloud.dispersed:
				self.time_step(dt=dt, ilevel=ilevel)
		elif terminate=='disperse_min':
			while not self.cloud.dispersed and self.ts[-1]<tlim:
				self.time_step(dt=dt, ilevel=ilevel)
			while self.ts[-1]<self.tdisperse+self.tcollapse+tend and self.ts[-1]<tlim:
				self.time_step(dt=dt, ilevel=ilevel)
			if self.ts[-1]>=tlim:
				print('Warning: time limit reached for trajectory, cloud still bound after %.2lf Myr'%(tlim/Myr2s))
				self.tdisperse = tlim
		else:
			while self.ts[-1]<tend and (not self.cloud.dispersed or not cloud_evolve):
				self.time_step(dt=dt, ilevel=ilevel, cloud_evolve=cloud_evolve)
		
		self.deltas = np.array(self.deltas)
		self.vs = np.array(self.vs)
		return self.ts, self.deltas, self.vs
		
	def plot_trajectories(self):
		ds = np.array(self.deltas)
		vs = np.array(self.vs)
		ts = np.array(self.ts)
		
		
		plt.plot(self.cloud.tclouds/Myr2s, self.cloud.Mclouds/Msol2g)
		plt.plot(self.cloud.tclouds/Myr2s, self.cloud.mstars/Msol2g)
		plt.yscale('log')
		plt.show()
		
		istep=1
		plt.plot(ts[::istep], vs[::istep,:, 0]/1e5, color='b')
		plt.plot(ts[::istep], vs[::istep,:, 1]/1e5, color='r', linestyle='dashed')
		plt.plot(ts[::istep], vs[::istep,:, 2]/1e5, color='g', linestyle='dotted')
		plt.show()
		
		plt.plot(self.grid.rlevels/pc2cm, self.grid.rho0*np.exp(self.grid.delta_c+self.mu_lnrho), color='k', linewidth=1)
		plt.plot(self.grid.rlevels/pc2cm, self.grid.rho0*np.exp(ds[::istep].T), color='r', linewidth=1)
		plt.axvline(self.cloud.R/pc2cm, color='k', linestyle='dashed')
		plt.yscale('log')
		plt.xscale('log')
		plt.show()


"""
GMC_MF
Generate the mass function for giant molecular clouds, following the approach of Hopkins 2012
In order to normalise, we estimate the turbulent time-scale on the scale height of the disc
"""	

def GMC_MF(Nsample=1000,  rmax=10.*h_*pc2cm):
	grid = trajectory_grid(rmin=0.01*pc2cm,rmax=rmax, drfact=0.95)

	eps_SF= 0.5
	tcross = (h_*pc2cm)/(sigvh_*1e5)/Myr2s


	qtraj = trajectory(grid=grid,  dt_factor=0.1)


	m = np.zeros(Nsample)
	ms = np.zeros(Nsample)
	r = np.zeros(Nsample)
	rho = np.zeros(Nsample)
	t = np.zeros(Nsample)
	tcross_arr = np.zeros(Nsample)
	iformed = np.zeros(Nsample, dtype=bool)
	ntry = 0
	for iN in range(Nsample):
		qtraj = trajectory(grid=grid,  dt_factor=0.1)
		ntry+=1
		while not qtraj.cloud.formed:
			qtraj.resample_all(0)
			ntry+=1
		ms[iN] = eps_SF*qtraj.cloud.M/Msol2g
		#qtraj.evolve(terminate='form')
		m[iN] = qtraj.cloud.M/Msol2g
		r[iN] = qtraj.cloud.R/pc2cm
		t[iN] = qtraj.cloud.tform/Myr2s
		tcross_arr[iN] = qtraj.grid.tau_R[qtraj.cloud.icol]/Myr2s
		rho[iN] = qtraj.cloud.rho_med*(pc2cm)**3/Msol2g
		#if t[iN]>0.0:
		iformed[iN]=True
	
	
	trange = 10.0
	rrange = 200.0
	rate_norm = (1./8.)*(0.333*4.*np.pi*rrange**3)*trange*1.0/ntry
	
	mbins = np.logspace(-0.5, 8., 80)
	bcents = (mbins[1:]+mbins[:-1])/2.
	
	npuv =  np.zeros(mbins.shape)
	npuv_st = np.zeros(mbins.shape)
	
	weight = (rho/m)*(1./tcross)
	for ith, threshold in enumerate(mbins):
		itmp = m>threshold
		
		npuv[ith] = np.sum(itmp*weight)
		itmp_st = ms>threshold
		
		npuv_st[ith] = np.sum(itmp_st*weight)
	
	npuv *= rate_norm
	npuv_st *= rate_norm
	
	
	fig, ax = plt.subplots(figsize=(5, 5))

	# Scatter plots
	plt.scatter(mbins, npuv, label='GMCs', s=10, alpha=0.7)  # smaller points
	plt.scatter(mbins, npuv_st, label='Stars w/ $\epsilon = %.1lf$'%eps_SF, s=10, alpha=0.7)  # smaller points

	# Line plot
	#plt.plot(mbins, 20.0 * (mbins / 100.)**-1., label=r'$\beta = -1$', linewidth=1, color='k')  
	#plt.plot(bcents, 100.0 * (bcents / 100.)**-1.8, label=r'$\beta = -1.9$', linewidth=1, color='k', linestyle='dashed')  

	plt.xscale('log')
	plt.yscale('log')

	# Highlight region between 0.1 and 10 on the y-axis
	ax.axhspan(10.0,2000.0, color='yellow', alpha=0.3)
	# Highlight region between 0.1 and 10 on the y-axis
	ax.axvspan(10.0, 500.0, color='cyan', alpha=0.2)

	# Add label to the shaded region
	# Add label to the shaded region
	label_x = 1e4  # x-coordinate of the label
	label_y = 30.0 # y-coordinate of the label
	ax.text(label_x, label_y, 'Locally well-sampled', fontsize=12, ha='left', va='center')

	label_x = 15.0  
	label_y = 1e-2
	ax.text(label_x, label_y, 'Selected GMCs', fontsize=12, ha='left', va='center')

	plt.xlim([1e1, 1e6])
	plt.ylim([1e-3, 1e3])

	plt.ylabel('\# Regions, mass $>M$ , age $<%d$ Myr within $%d$ pc' % (trange, rrange))
	plt.xlabel('Mass: $M$ [$M_\odot$]')

	# Legend
	plt.legend(loc='best')

	# Tick parameters
	ax.tick_params(which='both', top=True, right=True, left=True, bottom=True, direction='in')

	plt.savefig('region_selection.pdf', bbox_inches='tight', format='pdf')
	plt.show()

"""
get_density_history
Get the BHL accretion rate histories for stars for a representative sample of GMCs
"""
def get_density_history(GMCmin=10.0, GMCmax=1000.0, Nregions=200, Nstars=2000, probnorm=1e22, dt_fact=0.02, rmax=4.*h_*pc2cm):

	sfdb = sfrdb.sfr_database()
	grid = trajectory_grid(rmax=rmax, rmin=0.01*pc2cm, drfact=0.95)
	ntry= 0
	nstars = sfdb.nstars
	nregions = sfdb.nregions
	while (nstars<Nstars) or nregions<Nregions:
		qtraj = trajectory(grid=grid,  dt_factor=dt_fact)
		icol = qtraj.icol 
		mcloud = qtraj.cloud.M/Msol2g
		ntry+=1
		while (mcloud<GMCmin or mcloud>GMCmax):
			qtraj = trajectory(grid=grid,  dt_factor=dt_fact)
			qtraj.resample_all(0)
			mcloud = qtraj.cloud.M/Msol2g
			ntry+=1
		
		prob = probnorm*(qtraj.cloud.rho_med_crit/mcloud)
		urand = np.random.uniform()
		if urand<prob:
			#Form a star forming region
			sfr= sfr_c.star_forming_region(qtraj)
			out = sfr.get_BHL_history()

			if not out is None and len(sfr.msts)>0:
				sfdb.add_region(sfr)
			nstars = sfdb.nstars
			nregions = sfdb.nregions

			print('Total number of stars', nstars)
			print('Total number of regions', nregions)


	return sfdb



	
	
