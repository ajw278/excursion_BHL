import numpy as np

from consts_defaults import *

import orbit as orb
from turbfuncs import *

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#e41a1c','#984ea3', '#dede00', '#999999','cyan', 'teal']

"""
class: star_forming_region
Given a trajectory in density perturbation space, evolve the star forming region including the local densities and velocities of stars
"""
class star_forming_region():
	def __init__(self, trajectory, mrand_func=None, tevolve=20.0*Myr2s, RGC = 8e3*pc2cm, Mgal=1e12*Msol2g, cs=cs_*1e5, G=Gcgs, tform=None):
		self.traj = trajectory

		self.dt = self.traj.dt
		self.tevolve = tevolve
		self.run_bound = False
		self.run_texpulsion = False
		self.RGC = RGC
		self.Mgal = Mgal
		self.G = G
		self.cs = cs
		self.tmax= tevolve

	def draw_formation_time(self):

		tvals = self.traj.ts
		ivals = tvals<=self.traj.tdisperse
		mstars = self.traj.cloud.msts[ivals]
		tvals = tvals[ivals]
		dmdt = np.diff(mstars)
		dmdt[dmdt<0.0] = 0.0
		p = dmdt/np.sum(dmdt)
		iform = np.random.choice(np.arange(len(tvals[:-1])), p=p)
		self.tform = tvals[iform]
		#self.Rform = (np.random.uniform()**(1./3.))*self.traj.cloud.Rclouds[iform]

		return self.tform #, self.Rform
	

	def mrand_func(self, min=-2., max=0.5):
		return Msol2g*10.**(-2.+(max-min)*np.random.uniform())

	#From the 
	def texpulsion(self):
		if not self.run_bound:
			self.bound_evol()
	
	def calc_rtide_galaxy(self, M):
			return self.RGC*(M/(3.*self.Mgal))**(1./3.)
	
	def rtide_galaxy(self):
		RT_MW = self.RGC*(self.mstar/(3.*self.Mgal))**(1./3.)
		return RT_MW
		
	def get_BHL_history(self, Mlim=1e3*Msol2g, nstarlim=1000):
		if len(self.traj.ts)<2:
			self.traj.evolve(tend =self.tmax, cloud_evolve=True, terminate='disperse_min')


		Mcl = self.traj.cloud.Mclouds 
		Rcl = self.traj.cloud.Rclouds
		vcl  = self.traj.cloud.vclouds
		mst = self.traj.cloud.msts
		tcl = self.traj.cloud.tclouds


		delts = self.traj.deltas
		rho_amb_all = self.traj.grid.rho0*np.exp(delts)


		tall_cl = self.traj.ts

		#Recentre all time coordinates on cloud formation time
		tall_cl -= self.traj.cloud.tform
		tcl -= self.traj.cloud.tform
		tdisp = self.traj.tdisperse-self.traj.cloud.tform
		idisp  = np.argmin(np.absolute(tall_cl-tdisp))

		self.Mcl = Mcl
		self.vcl = vcl
		self.Mstcl = mst
		self.tcl = tcl

		Mgas_func = interpolate.interp1d(tcl, Mcl, fill_value=0.0, bounds_error=False)
		Rgas_func = interpolate.interp1d(tcl, Rcl/2./1.3, fill_value=0.0, bounds_error=False)
	

		t_evols = []
		t_forms = []
		msts = []
		Mdot_accs =[]
		dv_local = []
		rho_local = []
		R_accs = []
		nstars = int((mst[-1]/Msol2g)+0.5)


		if mst[-1]>Mlim:
			print('Mass above requested mass-limit.')
			return None


		print('Number of stars in region:', nstars)
		if nstars>nstarlim:
			print('Limiting to %d stars'%nstarlim)
			nstars = nstarlim
		for ist in range(nstars):
			self.mstar = self.mrand_func()
			t0 = self.draw_formation_time()
			while self.tform>self.traj.ts[-1]-10.*Myr2s:
				t0 = self.draw_formation_time()



			tf = t0 - self.traj.cloud.tform
		
			itsf = len(tall_cl)
			it0 = np.argmin(np.absolute(tall_cl-tf))

			t_eval = tall_cl[it0:idisp]
			tall = tall_cl[it0:itsf]


			self.t_star = tall- tall[0]
			self.rho_amb = rho_amb_all[it0:]
			v_gas = self.traj.vs[it0:]

			v_star  = np.zeros((itsf-it0, 3))

			if len(t_eval)>0:
				#Get the evolution over the bound phase
				t_tmp, r, vvect, rho, Mcl, MdBHL = orb.orbit_sampler_Plummer(t_eval, self.mstar, Mgas_func, Rgas_func, G=self.G, cs=self.cs)
				
				
				self.v_bound =  np.linalg.norm(vvect, axis=-1)
				
				self.t_bound = t_tmp
				self.rho_bound = rho
				self.r_bound  = r
				self.Mcl_bound = Mcl[it0:idisp]
				self.vcl_bound = vcl[it0:idisp]

				self.MBHL_bound = MdBHL


				v_star[:idisp-it0] = vvect + vcl[it0:idisp]
				v_star[idisp-it0:] = v_star[idisp-it0-1]
				iout = self.r_bound>2.*1.3*Rgas_func(t_tmp)
				iunb = self.r_bound>self.calc_rtide_galaxy(Mgas_func(t_tmp))

				ibound = np.where(~iout&~iunb)[0]
			else:
				v_star[:] = vcl[it0]

			
			self.v_amb = np.linalg.norm(v_gas-v_star[:, np.newaxis, :], axis=-1)

			RBHL =  BHL_radius(self.mstar, self.v_amb, cs=self.cs, G=self.G)
			Rscale = self.traj.grid.rlevels[np.newaxis, :]*np.ones(RBHL.shape)
			RtideG = self.rtide_galaxy()*np.ones(RBHL.shape)
			
			Racc = np.amin(np.array([RBHL, Rscale, RtideG]), axis=0)

			Mdotacc  = Mdot_BHL(Racc, self.rho_amb, self.v_amb, cs=self.cs)

			iMdmax = np.argmax(Mdotacc, axis=1)
			itimes = np.arange(len(Mdotacc))
			Mdotacc = Mdotacc[itimes, iMdmax]
			dv_amb = self.v_amb[itimes, iMdmax]
			rho_amb = self.rho_amb[itimes, iMdmax]

			if len(t_eval)>0:
				Mdotacc[ibound] = self.MBHL_bound[ibound]
				dv_amb[ibound] = self.v_bound[ibound]
				rho_amb[ibound] = self.rho_bound[ibound]

			t_evols.append(self.t_star)
			t_forms.append(self.tform)
			msts.append(self.mstar)
			Mdot_accs.append(Mdotacc)
			R_accs.append(Racc[itimes, iMdmax])
			dv_local.append(dv_amb)
			rho_local.append(rho_amb)

		self.t_forms = t_forms
		self.msts = msts
		self.t_evols = t_evols
		self.Mdot_accs = Mdot_accs
		self.dv_local = dv_local
		self.rho_local = rho_local
		self.R_accs = R_accs
		
		return t_forms, msts, t_evols, Mdot_accs, R_accs, dv_local, rho_local


