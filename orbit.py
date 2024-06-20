import numpy as np
import rotating_cluster as rc
from consts_defaults import *
from scipy.integrate import solve_ivp

from turbfuncs import *

"""
make_cluster
Function to draw position and velocities of star from a (rotating) Plummer profile
"""
def make_cluster(Nstars, M, a, b, k, G=Gcgs):
    zs = rc.sample_z(M, a, b, k, Nz=Nstars, zres=1000, zmaxfact=100.0, zminfact=1e-8, rmaxfact=100.0)
    rs = rc.sample_r_all(zs, M, a, b, k, rres=100, rmaxfact=100.0, rminfact=1e-8)
    vr, vz, vphi = rc.sample_vrzphi(rs, zs, M, a, b, k, G=G)
  
    return rs, vr, vphi, zs, vz
    
"""
orbit_sample_Plummer
Integrate an orbit inside a plummer potential, where both the mass and scale radius may be a function of time
"""
def orbit_sampler_Plummer(t_eval,mst, Mass_func, a_func, G=Gcgs, cs=cs_/1e5):
	
	def density_func(t, r):
		M_ = Mass_func(t)
		a = a_func(t)
		rho = 3.*M_/(4.*np.pi*a*a*a) 
		rho *= (1.+r*r/a/a)**(-5./2.)
		return rho
	
	def Menc_func(t, r):
		M_ = Mass_func(t)
		a = a_func(t)
		return M_*(r**3)/(r**2 + a**2)**(3./2.)
		

	# Define the derivative function for the Runge-Kutta method
	def derivative(t, rv):
		x, vx, y, vy, z, vz = rv
		R = np.sqrt(x**2 + y**2 +z**2)
		a = a_func(t)
		ax = -G * Mass_func(t) * x / ((R**2 + a**2)**1.5)
		ay = -G * Mass_func(t) * y / ( (R**2 + a**2)**1.5)
		az = -G * Mass_func(t) * z / ( (R**2 + a**2)**1.5)
		return np.array([vx, ax, vy, ay, vz, az])
	r= np.inf
	a0 = a_func(t_eval[0])
	while r>a0:
		r, vr, vphi, z, vz = make_cluster(1, max(Mass_func(t_eval[0]), 1e-10), a0,a0, 0.0, G=G)

	theta = 2.*np.pi*np.random.uniform()
	x0 = float(r*np.cos(theta)) #float(np.sqrt(r*r+z*z))
	vx0 = float(vr*np.cos(theta) - vphi*np.sin(theta)) #float(np.sqrt(vr*vr +vz*vz))
	y0 = float(r*np.sin(theta))
	vy0 = float(vr*np.sin(theta) + vphi * np.cos(theta)) #float(vphi)
	z0 = float(z)
	vz0 = float(vz)
	
	initial_conditions = np.array([x0, vx0, y0, vy0, z0, vz0])
	sol = solve_ivp(derivative, [t_eval[0], t_eval[-1]], initial_conditions, t_eval=t_eval, method='RK45', dense_output=True, atol=1e-4, rtol=1e-4)

	# Evaluate the solution
	t_ = t_eval
	rv_ = sol.sol(t_)
	r_ = np.sqrt(rv_[0]**2 + rv_[2]**2 + rv_[4]**2)

	v_ = np.array([rv_[1],rv_[3], rv_[5]]).T

	rho_ = density_func(t_, r_)

	RBHL = BHL_radius(mst, np.linalg.norm(v_, axis=-1), cs=cs, G=G)

	Rtide = r*(mst/3./Menc_func(t_, r_))**(1./3.)

	Racc = np.amin(np.array([RBHL, Rtide]), axis=0)

	Mdotacc  = Mdot_BHL(Racc, rho_, np.linalg.norm(v_, axis=-1), cs=cs)

	return t_, r_, v_, rho_, Mass_func(t_eval), Mdotacc



