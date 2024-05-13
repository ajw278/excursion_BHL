import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import scipy.integrate as integrate
from consts_defaults import *

#Functions used to draw stars from a (rotating) Plummer potential
#see Vergara et al. 2021

def zeta(z, b):
    return np.sqrt(z*z+b*b)

def density(r, z, M, a, b):
    z_ = zeta(z, b)
    rho = (M*b*b/np.pi/4.)*(a*r*r + (a+z_*3.)*(a+z_)*(a+z_) )
    rho /= z_*z_*z_*((r*r + (a+ z_)*(a+ z_))**(5./2.))
    
    return rho

def sigma1D(r, z, M, a, b, G=Gcgs):
    z_ = zeta(z, b)
    rhosigsq = (G*b*b*M*M/8./np.pi)*(a+z_)*(a+z_)
    rhosigsq /= z_*z_*((r*r+(a+z_)*(a+z_))**3)
    
    rho = density(r, z, M, a, b)
    return np.sqrt(rhosigsq/rho)

def vphisq_mean(r, z, M, a, b, G=Gcgs):
    z_ = zeta(z, b)
    rhobvp = (G*M*M*b*b/4./np.pi/z_/z_)*(a*r*r/z_ + (a+z_)*(a+z_)/2.)
    rhobvp /= (r*r+(a+z_)*(a+z_))**3
    rho = density(r, z, M, a, b)
    return rhobvp/rho

def vphi_mean(r, z, M, a, b, k, G=Gcgs):
    sigphi = sigma1D_phi(r, z, M, a, b, k, G=G)
    vphisq_m = vphisq_mean(r, z, M, a, b, G=G)
    vpm = np.sqrt(np.absolute(vphisq_m - sigphi**2))
    return vpm

def sigma1D_phi(r, z, M, a, b, k, G=Gcgs):
    vpsm = vphisq_mean(r, z, M, a, b, G=G)
    sig1d = sigma1D(r, z, M, a, b)
    return np.sqrt(k*k*sig1d*sig1d + (1.-k*k)*vpsm)

def sigma3D(r, z, M, a, b, k, G=Gcgs):
    s12 =  sigma1D(r, z, M, a, b, G=G)
    s3 = sigma1D_phi(r, z, M, a, b, k, G=G)
    return np.sqrt(2.*s12*s12 + s3*s3)

def cummassdz_fixz(r, z, M, a, b):
    z_ = zeta(z, b)
    cmz = (M*b*b/2.)*((a*r*r + (a + z_)**3)/(z_*z_*z_*(r*r +(a+z_)**2 )**1.5))
    return cmz

def sample_vrzphi(r, z, M, a, b, k, G=Gcgs):
    sigp = sigma1D_phi(r, z, M, a, b, k, G=G)
    sigrz = sigma1D(r, z, M, a, b, G=G)
    vphim = vphi_mean(r, z, M, a, b, k, G=G)
    vr = np.random.normal(loc=0.0, scale=sigrz)
    vz = np.random.normal(loc=0.0, scale=sigrz)
    vphi = np.random.normal(loc=vphim, scale=sigp)
    return vr, vz, vphi
    

def sample_z(M, a, b, k, Nz=1000, zres=1000, zmaxfact=100.0, zminfact=1e-8, rmaxfact=100.0):
    L = a+b
    rmax = rmaxfact*L
    zsp = np.logspace(np.log10(zminfact*L), np.log10(zmaxfact*L), zres)
    cmz = cummassdz_fixz(rmax, zsp, M, a, b)
    cmz_int = integrate.cumtrapz(cmz, zsp, initial=0.0)
    cmzn_inv = interpolate.interp1d(cmz_int/cmz_int[-1], zsp)
    xrand = np.random.uniform(size=Nz)
    sign_rand  = np.random.uniform(size=Nz)
    sign_rand[sign_rand<0.5] = -1.
    sign_rand[sign_rand>=0.5] = 1.
    return sign_rand*cmzn_inv(xrand)

def sample_r(zfix, M, a, b, k, rres=100, rmaxfact=100.0, rminfact=1e-8): 
    L = a+b
    rsp = np.logspace(np.log10(rminfact*L), np.log10(rmaxfact*L), rres)
    cmr = density(rsp, zfix, M, a, b)*2*np.pi*rsp
    cmr_int = integrate.cumtrapz(cmr, rsp, initial=0.0)
    cmrn_inv = interpolate.interp1d(cmr_int/cmr_int[-1], rsp)
    xrand  = np.random.uniform()
    return cmrn_inv(xrand)

def sample_r_all(zs, M, a, b, k, rres=100, rmaxfact=100.0, rminfact=1e-8):
    rs = np.zeros(zs.shape)
    ir = 0
    for zfix in zs:
        rs[ir] =  sample_r(zfix, M, a, b, k, rres=rres, rmaxfact=rmaxfact, rminfact=rminfact)
        ir+=1
    return rs 

def sample_phi(Nphi=1000):
    return 2.*np.pi*np.random.uniform(size=Nphi)
    

