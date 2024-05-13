import numpy as np

from consts_defaults import *
from turbfuncs import *
import scipy.special as ss


"""
Class: bound_clump
Deals with the collapse and star formation of a gravitationally unstable clump in the ISM
Follows the approach of Girichidis et al. 2014
"""
class bound_clump():
	#Set initial sound speed, core SF efficiency, wind velocity and SF efficiency per free-fall time
	def __init__(self, cs0=cs_*1e5,  eps_core=0.5,  h = h_*pc2cm,G=Gcgs, Rmin=0.001*pc2cm,  **kwargs):
		self.tform = np.inf
		self.bound=False
		self.t=0.0
		
		self.formed=False
		self.dispersed=False

		self.cs0 = cs0
		self.cs = cs0
		
		self.eps_core = eps_core 
		
		self.h = h
		
		self.G = G
		self.tff_rho = np.sqrt(3.*np.pi/32./self.G) 
		self.M = 0.0
		self.R = 0.0
		self.t = 0.0
		self.v = np.array([0.0,0.0,0.0])
		self.sv = 0.0
		self.mst = 0.0
		self.vr = 0.0
		self.Rmin = Rmin
		self.a = 2.0
		self.Q = 32.*self.G/(3.*np.pi)
		self.rho_accr = (self.tff_rho/0.1/Myr2s)**2
		
		self.Mdot_acc = 0.0
		self.dvdt = 0.0


		self.icol=np.nan
		
		self.rho_mean = 0.0
		self.rho_med = 0.0
		self.dRdt = 0.0
		self.initialise_arrays()
		self.update_arrays()
	
	def initialise_arrays(self):
		self.Mclouds = np.array([])
		self.Rclouds =np.array([])
		self.tclouds = np.array([])
		self.vclouds = np.array([])
		self.rho_meds = np.array([])
		self.tau_sfs = np.array([])
		self.sfrs = np.array([])
		self.icols = np.array([])
		self.msts = np.array([])
		self.Mdot_accs = np.array([])
		
	def update_arrays(self):
		self.Mclouds = np.append(self.Mclouds, self.M)
		self.Rclouds= np.append(self.Rclouds, self.R)
		self.rho_meds = np.append(self.rho_meds, self.rho_med)
		if len(self.vclouds)>0:
			self.vclouds = np.append(self.vclouds, np.array([self.v]), axis=0)
		else:
			self.vclouds = np.array([self.v])
		self.tclouds= np.append(self.tclouds, self.t)
		self.sfrs = np.append(self.sfrs, self.calc_SFR(self.t))
		self.Mdot_accs = np.append(self.Mdot_accs, self.Mdot_acc)
		self.msts = np.append(self.msts, self.mst)
		self.icols = np.append(self.icols, self.icol)

	
	def form(self, icol, traj):

		median_density = traj.grid.rho0*np.exp(traj.grid.delta_c[icol] + traj.grid.mu_lnrho[icol])
		sigmav = np.sqrt(np.sum(traj.grid.Delta_Sv[icol:]))
		radius = traj.grid.rlevels[icol]
		velocity = traj.v[icol]
		
		self.icol = icol
		self.tform = traj.t
		self.rho_med = median_density
		self.Mdot_BHL = 0.0
		self.R = radius
		self.R0 = radius
		self.tau_ff0 = self.tau_ff()
		self.v = velocity
		self.sv = sigmav
		self.t = traj.t
		self.formed=True
		self.dispersed=False
		self.Espect = traj.grid.Espect
		self.cs = traj.grid.cs
		self.kappa = traj.grid.kappa
		self.lnrho_var0  =variance_k(1./self.R0,  cs=self.cs, kappa=self.kappa, Espect = self.Espect)
		self.lnrho_med0  = np.log(median_density)

		self.rho_med_crit = median_density
		self.calc_Mgas()
		self.vol = self.M/self.rho_med
	
	def lognormal(self, x, mu, var):
		return (1./x/np.sqrt(2.*np.pi*var))*np.exp(-(np.log(x)-mu)**2/2./var)
		
	def lognormal_CDF(self, x, mu, var):
		arg = (np.log(x)-mu)/np.sqrt(2.*var)
		cdf = 0.5*(1.+ss.erf(arg))
		return cdf
	
	def Prho0_M(self, rho0):
		mu = self.lnrho_med0
		var=  self.lnrho_var0
		factor = np.exp(mu + var/2.)
		return self.lognormal(rho0, mu, var)
	
	def CDFrho0_M(self, rho0):
		mu = self.lnrho_med0
		var=  self.lnrho_var0
		factor = np.exp(mu + var/2.)
		arg = (mu+ var - np.log(rho0))/(np.sqrt(2.*var))
		return 0.5*(1.+ss.erf(arg))
	
	def calc_drhodt(self, t, rho0):
		td = t-self.tform
		tff0 =self.tff_rho/np.sqrt(rho0)
		factor1 = rho0*td*12.*self.Q/self.a
		Qprod = self.Q*rho0*td*td
		tau = td/tff0
		return np.absolute(factor1*(1.- Qprod)**(-self.a-1.))
		
		
	def calc_q(self, t, rho0):
		tff0 = self.tff_rho/np.sqrt(rho0)
		tau = (t-self.tform)/tff0
		
		num = (1.-tau**2)**(self.a+1.)
		denom = (1.+(self.a-1.)*tau**2)
		return num/denom
	
	def calc_rho_acc(self, t):
		td = t - self.tform
		sq = (2.*self.Q*self.rho_accr*td*td +1.)/(2.*self.Q*self.Q*self.rho_accr*td**4)
		rho_i = sq + np.sqrt(sq**2 - 1./(self.Q**2 * t**4))
		return rho_i
	
	
    
	def calc_SFR(self,t, M=None, rho=None, update=True, var_lrho=None):
		if M is None:
			M=self.M
		
		if M>0.0:
			rho0 = np.exp(self.lnrho_med0)
			
			rhoacc =self.calc_rho_acc(t)
			rhosp = np.logspace(rho0-100.0, rho0+10.0,1000)
			PM = self.Prho0_M(rhosp)
			drhodt = self.calc_drhodt(t, rhoacc)
			q= self.calc_q(t, rhoacc)
			dMdrho = self.Prho0_M(rhoacc)*q
			
			#vol = self.M/self.rho_med
			SFR = dMdrho*q*drhodt
			if SFR<0.0:
				raise Warning('Star formation rate <0 in cloud calculation')
			if update:
				self.SFR=SFR
			return SFR
		else:
			return 0.0
		
	
	def find_collapse(self, traj):
		delta = traj.delta
		col = delta>traj.grid.delta_c
		if np.sum(col>0):
			icol =np.argmax(col)
			if np.isnan(self.icol):
				self.form(icol, traj)
				self.icol = icol
		else:
			self.icol = self.icols[-1]
		
		return self.icol

	def calc_Mgas(self, rho=None, R=None, update=True):
		if rho is None:
			rho = self.rho_med
		if R is None:
			R = self.R
		h = self.h
		M = 4.*np.pi*rho*(h**3)*(R*R/h/h/2. + (1.+R/h)*np.exp(-R/h) -1.)
		if update:
			self.M =M
		return M
	
	
	def calc_rhogasM(self, M=None, rc=None, update=True):
		if M is None:
			M = self.M
		if rc is None:
			rc = self.R
		h = self.h
		rho = M/(4.*np.pi*(h**3)*(rc*rc/h/h/2. + (1.+rc/h)*np.exp(-rc/h) -1.))
		if update:
			self.rho_med = rho
		return rho


	def tau_ff(self, rho=None):
		if rho is None:
			rho = self.rho_med
		return self.tff_rho/np.sqrt(rho)
	
	def epsilon(self):
		return self.eps_core

	
	def accretion_step(self, dt,  traj):
		delts = traj.delta
		s2_R = traj.grid.sig2_R
		rho_amb = traj.grid.rho0*np.exp(delts - s2_R/2. )
		dv_gas = traj.v - self.v
		dv_amb  = np.linalg.norm(dv_gas, axis=-1)
		RBHL =  BHL_radius(self.M, dv_amb, cs=self.cs, G=self.G)

		Rscale = traj.grid.rlevels*np.ones(RBHL.shape)
		RtideG = self.rtide_galaxy()*np.ones(RBHL.shape)

		Racc = np.amin(np.array([RBHL, Rscale, RtideG]), axis=0)

		Mdotacc  = Mdot_BHL(Racc, rho_amb, dv_amb, cs=self.cs)
		tcross = Rscale/dv_amb
		Mdotacc_lim = (rho_amb*(1./3.)*4.*np.pi*Rscale**3)/tcross
		
		imax = np.argmax(Mdotacc)
		vmax = traj.v[imax]

		self.Mdot_acc = np.amax(Mdotacc)
		self.dvdt = self.Mdot_acc * vmax / self.M

		self.v += self.dvdt*dt
		self.M += self.Mdot_acc*dt

		return self.Mdot_acc


	def calc_rhocrit(self, traj):
		self.rho_med_crit = rho_crit(1./self.R, kappa = traj.grid.kappa, Omega=traj.grid.Omega, h=traj.grid.h, G=traj.grid.G, rho0=traj.grid.rho0, Espect=traj.grid.Espect, cs=traj.grid.cs)
		return self.rho_med_crit
	
	
	def collapse_step(self, dt,  traj):
		tnew = traj.t
		if tnew<self.tau_ff0:
			self.R = self.R0*(1.-((tnew-self.tform)/self.tau_ff0)**2)**(self.a/3.)
			self.calc_rhogasM()
		else:
			self.R = self.Rmin
			self.M = 0.0
		if self.R<self.Rmin:
			self.R = self.Rmin
	

	def SF_step(self, dt, traj):
		
		rhoacc =self.calc_rho_acc(traj.t)
		cdf = self.CDFrho0_M(rhoacc)
		sf_frac = cdf
		self.mst = self.M*sf_frac
		if self.mst/self.M>=self.eps_core:
			self.mst = self.eps_core*self.M
			self.M = 0.0
			self.dispersed=True
		
		self.calc_rhogasM()
		return self.mst
		
	def check_dispersal(self):

		if self.mst>=self.M:
			self.dispersed = True
			self.M = 0.0
			self.icol = np.nan
			return True
		
		return False
		

	def evolve(self, traj, rejuvinate=False):
		dt = traj.t - self.t
		if (rejuvinate or not self.dispersed) and dt>0.0:
			if self.formed and not self.dispersed:
				
				#self.accretion_step(dt, traj)
				self.SF_step(dt, traj)
				self.collapse_step(dt,  traj)
				if np.isnan(self.M):
					print(self.M, self.R, self.rho_med)
					raise Warning('Cloud mass is NaN in evolve timestep')
				self.check_dispersal()
		
		if (not self.formed) or rejuvinate:
			self.find_collapse(traj)
		

		self.t = self.t+dt

		self.update_arrays()

		return self.icol
