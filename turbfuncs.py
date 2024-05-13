import numpy as np
from consts_defaults import *
import scipy.interpolate as interpolate
"""def Fourier_tophat(k, R):
	Ftp = np.ones((len(k),len(R)))
	kgr = np.ones(Ftp.shape)*kgr[:, np.newaxis]
	Rgr = np.ones(Ftp.shape)*Rgr[np.newaxis,:]
	Ftp[kgr>(1./Rgr)] = 0.0
	return Ftp"""
	

def Fourier_tophat(k, R):
	Ftp = np.ones(k.shape)
	#Ftp *= np.exp(-k*k*R*R)
	Ftp[k>(1./R)] = 0.0
	return Ftp


def runge_kutta_step(f,y, dt):
	k1 = dt * f(y, 0.0)
	k2 = dt * f(y + 0.5 * k1, 0.5 * dt)
	k3 = dt * f(y + 0.5 * k2, 0.5 * dt)
	k4 = dt * f(y + k3, dt)

	y_next = y + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

	return y_next

def BHL_radius(mst, v, cs=cs_*1e5, G=Gcgs):
	return 2 * G * mst /(cs*cs +v *v) 

def Mdot_BHL(Racc, rho, v, cs=cs_*1e5, lamb=np.exp(1.5)/4.):
    accrate = np.pi*Racc*Racc*rho*np.sqrt(lamb*cs*cs +v*v)
    return accrate

#Vt^2(k) = E(k)/k
#k = 1/R 
#v(1pc) = 1 km/s = 10^5 cm/s 
#1.5*0.63e5
def Ev_k(k, v0 = sigvh_*1e5, p =2., k0 = 1./h_/pc2cm, h=h_*pc2cm, shear=True):
	if shear:
		return ((1 + 1./(k*h)**2 )**((1.-p)/2.)) *((v0**2)/k0)*(k/k0)**-p
	else:
		return ((v0**2)/k0)*(k/k0)**-p


def rtide_clump(rstar, mstar, Mclump):
	RT_cl = rstar*(mstar/(3.*Mclump))**(1./3.)
	return RT_cl


def vturb2_k(k, cs=cs_*1e5, Espect = Ev_k):
	return Espect(k)*k
	

def variance_k(k, cs=cs_*1e5, kappa=Omega_*np.sqrt(2.)/1e6/year2s, Espect = Ev_k):
	vt2 = vturb2_k(k, Espect=Espect)
	return np.log(1.+ (0.75*vt2)/(cs**2+(kappa/k)**2 ))


def variance_R(R, ksp, kappa=Omega_*np.sqrt(2.)/1e6/year2s, cs=cs_*1e5, Espect = Ev_k):
	Ftp = Fourier_tophat(ksp, R)
	var_k = variance_k(ksp, cs=cs, kappa=kappa, Espect = Espect)
	return np.trapz(var_k*Ftp**2, np.log(ksp))


def variancev_k(k, cs=cs_*1e5, Espect = Ev_k, kappa=Omega_*np.sqrt(2.)/1e6/year2s, Omega=Omega_/1e6/year2s, h = h_*pc2cm):
	return vturb2_k(k, Espect=Espect) + cs*cs  #sigg_k(k, Omega=Omega, cs=cs, Espect = Espect)**2 /(1.+sigg_k(k)**2/sigg_k(1./h)**2)

def variancev_R(R, ksp, cs=cs_*1e5, Espect = Ev_k, kappa=Omega_*np.sqrt(2.)/1e6/year2s, Omega=Omega_/1e6/year2s, h=h_*pc2cm):
	Ftp = Fourier_tophat(ksp, R)
	var_k = variancev_k(ksp, cs=cs, kappa=kappa, Espect = Espect, h=h)
	return np.trapz(var_k*Ftp**2, np.log(ksp))

def sigg_k(k, Omega=Omega_/1e6/year2s, cs=cs_*1e5, Espect = Ev_k):
	return np.sqrt(cs**2 + vturb2_k(k, cs=cs, Espect = Espect)+cs*cs)

#def sigg_h(h=h_*pc2cm, Omega=Omega_/1e6/year2s):
#	return Omega*h

def Q0_h(h=h_*pc2cm, kappa=Omega_*np.sqrt(2.)/1e6/year2s, Sigma0=Sigma0_*Msol2g/(pc2cm**2),  G=Gcgs):
	return kappa*sigg_k(1./h)/np.pi/G/Sigma0
	
def k_tilde(k, h=h_*pc2cm):
	return np.absolute(k)*h

def kappa_tilde(kappa = Omega_*np.sqrt(2.)/1e6/year2s, Omega=Omega_/1e6/year2s):
	return kappa/Omega

def rho_crit(k, kappa = Omega_*np.sqrt(2.)/1e6/year2s, Omega=Omega_/1e6/year2s, h=h_*pc2cm, G=Gcgs, rho0=rho0_*Msol2g/(pc2cm)**3, Espect=Ev_k, cs=cs_*1e5):
	kapt = kappa_tilde(kappa=kappa, Omega=Omega)
	kt = k_tilde(k, h=h)
	Sigma0 = 2.*rho0*h
	Q0 = Q0_h(h=h, kappa=kappa, Sigma0=Sigma0, G=G)
	sgk = sigg_k(k, Omega=Omega, cs=cs, Espect=Espect)
	sgh = sigg_k(1./h, Omega=Omega, cs=cs, Espect=Espect) #sigg_h(h=h, Omega=Omega)
	if Q0>6.0 or Q0<0.5:
		print('Warning: Q0 outside of usual range...')
		print('Q0 : ', Q0)
	return rho0*(Q0/2./kapt)*(1.+kt)*(kt*(sgk/sgh)**2 + (kapt**2)/kt)

	
	
def get_kroupa_imf(m1=0.08, p1=0.3, m2=0.5, p2=1.3, m3=1.0, p3=2.3, p4=2.7,  mmin=0.01):
    
    msp = np.logspace(-3.0, 2., 10000)
    
    xi  = msp**-p1
    f1 = (m1**-p1)/(m1**-p2)
    f2 = f1*(m2**-p2)/(m2**-p3)
    f3 = f2*(m3**-p3)/(m3**-p4)
    xi[msp>m1] = f1*(msp[msp>m1]**-p2)
    xi[msp>m2] = f2*(msp[msp>m2]**-p3)
    xi[msp>m3] = f3*(msp[msp>m3]**-p4)
    xi[msp<mmin] = 0.0
    
    xi /= np.trapz(xi, msp)

    return interpolate.interp1d(msp, xi)

def get_imf_cdf():
    
    msp = np.logspace(-3.0, 2, 10000)
    
    imf_func= get_kroupa_imf()
    
    imf = imf_func(msp)

    cdf = np.cumsum(imf*np.gradient(msp))
    cdf -= cdf[0]
    cdf /=cdf[-1]

    
    return  interpolate.interp1d(msp, cdf)



def get_imf_icdf():
    
    msp = np.logspace(-3.0, 2, 10000)
    
    imf_func= get_kroupa_imf()
    
    imf = imf_func(msp)

    cdf = np.cumsum(imf*np.gradient(msp))
    cdf -= cdf[0]
    cdf /=cdf[-1]

    
    return  interpolate.interp1d(cdf, msp)

