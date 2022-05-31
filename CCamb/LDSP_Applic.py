import sys
import os
import astropy
from pathlib import Path
import camb
from camb import model
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from astropy import units as u
from astropy import constants as c

#print(sys.version_info)
os.environ['PATH'] += os.pathsep + '/Library/TeX/texbin'

rcParams.update({'font.size':12})
rcParams.update({'text.usetex':True})

WD = 'D:/SNU/Cosm'

#%% (a)
#Now get matter power spectra and sigma8 at redshift 0 and 0.8
npoints = 1000

As_init = 2.1073e-9
h = 0.677
omb = 0.048
om = 0.31

# h= 0.6777
# om = 0.307115
# omb = 0.048206

ombh2 = omb * h**2
omch2 = (om-omb) * h**2
pars = camb.CAMBparams()
pars.set_cosmology(H0=100*h, ombh2=ombh2, omch2=omch2)
pars.InitPower.set_params(ns=0.96, As=As_init)
#Note non-linear corrections couples to smaller scales than you want
pars.set_matter_power(redshifts=[0.], kmax=100.0)

#Linear spectra
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)
kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=10,
                                              npoints=npoints)
s8 = np.array(results.get_sigma8())

# normalization with sigma8 = 0.8228
s8_ratio = 0.8228/s8[0]
As = s8_ratio**2 * As_init

# pars = camb.CAMBparams()
# pars.set_cosmology(H0=100*h, ombh2=ombh2, omch2=omch2)
# pars.InitPower.set_params(ns=0.96, As=As)
# pars.set_matter_power(redshifts=[0.], kmax=2.0)

# #Non-Linear spectra (Halofit)
# pars.NonLinear = model.NonLinear_both
# results.calc_power_spectra(pars)
# kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=10, npoints=npoints)

kh8 = 1 / 8 #* h

plt.figure()
plt.loglog(kh, pk[0,:], color='k')
#plt.loglog(kh_nonlin, pk_nonlin[0,:], color='r')
plt.ylabel(r'$P(k)$ [$ (h^{-1} {\rm Mpc})^3 $]')
plt.xlabel(r'$k$ [$h$ Mpc$^{-1}$]')
plt.legend(['linear'], loc='lower left')
plt.title(f'Matter power at z={z[0]}')
plt.axvline(kh8, c='k', lw=0.8, ls='--')
plt.savefig(WD+'/N1a.png', dpi=300)


#%% (b)
# filter definitions
def tophat(k, r_f):
    num = np.sin(k*r_f) - k*r_f*np.cos(k*r_f)
    den = (k*r_f)**3
    return 3 * num / den

def gaussian(k, r_f):
    return np.exp(-(k*r_f)**2 * 0.5)

def sharp_k(k, r_f):
    _thres = 1 / r_f #np.pi / r_f
    # _thres = np.pi / r_f #TODO
    _filter = np.ones_like(k) # * 3 / (4*np.pi*r_f**3)
    _filter[k > _thres] = 0
    return _filter

def sig_squared(filt, k, pk, r_f):
    integrand = k**2 * pk * filt(k, r_f)**2 / np.pi**2 / 2.
    return np.trapz(integrand, k)
    
def mass_to_r(mass, h, filt='tophat'):
    _rho = om*2.7754e11 * u.Msun / u.Mpc**3 #* h**2
    
    gam_f = {'tophat': 4*np.pi/3,
             'gaussian': (2*np.pi)**(3/2),
             'sharp_k': 6*np.pi**2}
    
    r3 = mass / _rho / gam_f[filt]
        
    return r3**(1/3)

mass = 10**np.linspace(0, 3, npoints) * 1e12*u.Msun


fig = plt.figure()
ax = fig.add_subplot(111)
filters = [ gaussian,tophat, sharp_k]
colors = ['k', 'b', 'r']
for filt, color in zip(filters, colors):
    func = lambda x : sig_squared(filt, kh, pk, x)
    r_f = 10**np.linspace(-1., 2.5, npoints)
    vfunc = np.vectorize(func)
    sig2 = vfunc(r_f)
    
    ax.loglog(r_f, np.sqrt(sig2), c=color)
ax.set_xlim([0.2, 300])
ax.set_ylim([1e-2, 8])
ax.set_xlabel('$R$ [$h^{-1}$ Mpc]')
ax.set_ylabel('$\sigma_\delta$')
plt.legend(['Gaussian', 'Top-Hat', 'Sharp-k'])
plt.savefig(WD+'/N2_un1.png', dpi=300)

fig = plt.figure()
ax = fig.add_subplot(111)
for filt, color in zip(filters, colors):
    func = lambda x : sig_squared(filt, kh, pk, x)
    r_f = mass_to_r(mass, 1, filt=filt.__name__)
    vfunc = np.vectorize(func)
    sig2 = vfunc(r_f.value)
    
    ax.loglog(mass/1e12, sig2, c=color)
ax.set_xlim([1, 1e3])
# ax.set_ylim([1e-2, 8])
ax.set_xlabel('$M$ [$10^{12} h^{-1} M_{\odot}$]')
ax.set_ylabel('$\sigma^2_{\delta}(M)$')
plt.legend(['Gaussian','Top-Hat',  'Sharp-k'])
plt.tight_layout()
plt.savefig(WD+'/N1_b.png', dpi=300)

fig = plt.figure()
ax = fig.add_subplot(111)
for filt, color in zip(filters, colors):
    func = lambda x : sig_squared(filt, kh, pk, x)
    r_f = mass_to_r(mass, 1, filt=filt.__name__)
    vfunc = np.vectorize(func)
    sig2 = vfunc(r_f.value)
    
    ax.plot(r_f, sig2, c=color)
# ax.set_xlim([1, 1e3])
# ax.set_ylim([1e-2, 8])
ax.set_xlabel('$R_f$ [Mpc]')
ax.set_ylabel('$\sigma^2_{\delta}(M)$')
plt.legend(['Top-Hat', 'Gaussian', 'Sharp-k'])
plt.tight_layout()
plt.savefig(WD+'/N2_un2', dpi=300)
#%% (c)

'''
The Press-Schechter mass function with normalization factor of 2,
$$\frac{dN}{d\ln M} = \sqrt{\frac{2}{\pi}}\frac{\bar{\rho}}{M}\frac{\delta_c}{\sigma_M} \exp \left(-\frac{\delta_c^2}{2\sigma^2_M}\right)\left|\frac{d\ln \sigma_M}{d \ln M}\right|$$
'''

# rho_mean = 2.7754e11 * u.Msun / u.Mpc**3# * h**2
rho_mean = om*(3*(100*u.km/u.s/u.Mpc)**2/(8*np.pi*c.G)).to(u.Msun*u.Mpc**(-3))
delt_c = 3/5*(3/4)**(2/3)*(2*np.pi)**(2/3)

func = lambda x : sig_squared(tophat, kh, pk, x)
r_f = mass_to_r(mass, 1, filt='tophat')
vfunc = np.vectorize(func)
sig2 = vfunc(r_f.value)
sig = np.sqrt(sig2)

dlnsig = np.gradient(np.log(sig))
dlnM = np.gradient(np.log(mass.value))
multi_func_PS = np.sqrt(2/np.pi) * delt_c/sig * np.exp(-0.5*delt_c**2/sig2)
dNdlnM_PS = multi_func_PS * rho_mean/mass * np.abs(dlnsig/dlnM)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.log10(mass/u.Msun), dNdlnM_PS, c='k')
# ax.set_xlim([1, 1e3])
# ax.set_ylim([1e-2, 8])
ax.set_yscale('log')
ax.set_xlabel('$\log M (h^{-1}M_{\odot})$')
ax.set_ylabel('$dN/d\ln M$ [$(h^{-1}$Mpc$)^{-3}$]')
# plt.legend(['Top-Hat', 'Gaussian', 'Sharp-k'])
plt.tight_layout()
plt.savefig(WD+'/N1c.png', dpi=300)

#%% (d)
'''
Sheth-Tormen mass function
$$f(\sigma) = A\sqrt{\frac{2a}{\pi}}\left[1+\left(\frac{\sigma^2}{a\delta_c^2}\right)^P\right]\exp\left(-\frac{a}{2}\frac{\delta_c^2}{\sigma^2}\right)$$
'''
A = 0.3222
a = 0.707
P = 0.3
nu = delt_c/sig
multi_func_ST = A*np.sqrt(2*a/np.pi)*(1+(1/a/nu**2)**P)*nu*np.exp(-a/2*nu**2)

dNdlnM_ST = multi_func_ST * rho_mean/mass * np.abs(dlnsig/dlnM)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.log10(mass/u.Msun), dNdlnM_ST, c='k')
# ax.set_xlim([1, 1e3])
# ax.set_ylim([1e-2, 8])
ax.set_yscale('log')
ax.set_xlabel('$\log M (h^{-1}M_{\odot})$')
ax.set_ylabel('$dN/d\ln M$ [$(h^{-1}$Mpc$)^{-3}$]')
# plt.legend(['Top-Hat', 'Gaussian', 'Sharp-k'])
plt.tight_layout()
plt.savefig(WD+'/N1d.png', dpi=300)

#%% (e)
'''
Jenkins mass function
$$f(\sigma) = 0.315 \exp[-|\ln \sigma^{-1} +0.61|^{3.8}]$$
'''

multi_func_Jenkins = 0.315*np.exp(-np.abs(np.log(1/sig)+0.61)**3.8)
dNdlnM_Jenkins = multi_func_Jenkins * rho_mean/mass * np.abs(dlnsig/dlnM)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.log10(mass/u.Msun), dNdlnM_Jenkins, c='k')
# ax.set_xlim([1, 1e3])
# ax.set_ylim([1e-2, 8])
ax.set_yscale('log')
ax.set_xlabel('$\log M (h^{-1}M_{\odot})$')
ax.set_ylabel('$dN/d\ln M$ [$(h^{-1}$Mpc$)^{-3}$]')
# plt.legend(['Top-Hat', 'Gaussian', 'Sharp-k'])
plt.tight_layout()
plt.savefig(WD+'/N1_e.png', dpi=300)

#%% (c)+(d)+(e)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.log10(mass/u.Msun), dNdlnM_PS, c='k', label='PS')
ax.plot(np.log10(mass/u.Msun), dNdlnM_ST, c='r', label='ST')
ax.plot(np.log10(mass/u.Msun), dNdlnM_Jenkins, c='b', label='Jenkins')
ax.set_xlim([12, 15])
# ax.set_ylim([1e-2, 8])
ax.set_yscale('log')
ax.set_xlabel('$\log M (h^{-1}M_{\odot})$')
ax.set_ylabel('$dN/d\ln M$ [$(h^{-1}$Mpc$)^{-3}$]')
plt.legend()
plt.tight_layout()
plt.savefig(WD+'/prob1cde.png', dpi=300)