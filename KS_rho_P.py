# Implementation of the Komatsu & Seljak 2001 model for the gas density and pressure profiles
import numpy as np
import pickle as pk

from astropy import units as u
from astropy import constants as const
from astropy.io import fits

import sys, os

import matplotlib
import matplotlib.pyplot as pl

font = {'size': 18}
matplotlib.rc('font', **font)
pl.rc('text', usetex=False)
pl.rc('font', family='serif')

from colossus.cosmology import cosmology

cosmo = cosmology.setCosmology('planck18')
from colossus.halo import mass_so

mdef = '200c'
# mdef = '200c'
# mdef = '500c'
import astropy.units as units
import astropy.constants as const

import scipy as sp

from tqdm import tqdm
from colossus.halo import mass_so


#Eq. 8
def m(x):
    return np.log(1 + x) - x / (1 + x)


# Eq. 9
def int_mu_un2(x):
    return 1 - (np.log(1 + x)) / x


# Eq. 21
def s_star_func(xstar):
    return -1 * (1 + 2 * xstar / (1 + xstar))


# Eq. 23
def eta0_func(xstar, c, gamma):
    s_star = s_star_func(xstar)
    m_xstar = m(xstar)
    mc = m(c)
    int_mu_un2_xstar = int_mu_un2(xstar)
    val = (1 / gamma) * ((-3 / s_star) * (c * m_xstar / (xstar * mc)) + 3 * (gamma - 1) * (c / mc) * int_mu_un2_xstar)
    return val


# Fig.2
def get_gamma(c):
    xstar_vals = np.linspace(c / 2, 2 * c, 6)
    gamma_vals = np.linspace(1.0, 1.7, 80)
    eta0_all = np.zeros((len(xstar_vals), len(gamma_vals)))
    for jx in range(len(xstar_vals)):
        for jg in range(len(gamma_vals)):
            eta0_all[jx, jg] = eta0_func(xstar_vals[jx], c, gamma_vals[jg])

    deta0_dxstar_all = np.zeros(len(gamma_vals))
    for jg in range(len(gamma_vals)):
        deta0_dxstar_all[jg] = np.mean(np.gradient(eta0_all[:, jg], xstar_vals))
    indmin = np.argmin(np.abs(deta0_dxstar_all))
    return gamma_vals[indmin]


# Eq. 19
def ygas(x, gamma, c):
    eta0 = eta0_func(c, c, gamma)
    int_mu_un2_x = int_mu_un2(x)
    mc = m(c)
    val = 1 - (3 / eta0) * ((gamma - 1) / gamma) * (c / mc) * int_mu_un2_x
    return val**(1 / (gamma - 1))


# Eq. 20 (coeff here is supplied when calculating the final pressure profile)
def Tgas(x, gamma, c, Mvir):
    ygasx = ygas(x, gamma, c)
    return (ygasx**(gamma - 1.))


# Eq. 15 (coeff here is free and setting it to 1.)
def rhogas(x, gamma, c, coeff=1.):
    ygasx = ygas(x, gamma, c)
    return coeff * ygasx


def coeffgas_full(gamma, c, Mvir=1e15, fb_cosmo=0.048 / 0.3):
    r = np.linspace(0.01, 5, 2000)
    rvir = mass_so.M_to_R(Mvir, 0.0, mdef) / 1000.
    rs = rvir / c
    x = r / rs
    ygasx = ygas(x, gamma, c)
    indsel = np.where(r < rvir)[0]
    ygasx_int = sp.integrate.simps((ygasx * 4 * np.pi * r**2)[indsel], r[indsel])
    coeff = (Mvir / ygasx_int) * fb_cosmo
    return coeff


def rhogas_r(r, gamma, c, Mvir=1e15, fb_cosmo=0.048 / 0.3):
    rvir = mass_so.M_to_R(Mvir, 0.0, mdef) / 1000.
    rs = rvir / c
    x = r / rs
    ygasx = ygas(x, gamma, c)
    coeff = coeffgas_full(gamma, c, Mvir=Mvir, fb_cosmo=fb_cosmo)
    return coeff * ygasx


def P0gas(x, gamma, c, Mvir, coeff_rho=1., coeff_T=1.):
    kB_Tgas = Tgas(x, gamma, c, Mvir)
    rho_gas = rhogas(x, gamma, c, coeff=coeff_rho)
    P_gas = rho_gas * (kB_Tgas)
    return P_gas


def P0gas_r(r, gamma, c, Mvir, rmax_rvir=5.0):
    rvir = mass_so.M_to_R(Mvir, 0.0, mdef) / 1000.
    rs = rvir / c
    x = r / rs
    mu = 0.59
    eta0 = eta0_func(c, c, gamma)
    kB_T0 = (1. / 3.) * eta0 * ((Mvir * units.Msun / (rvir * units.Mpc)) * mu * (const.G * const.m_p)).to(units.keV)
    kB_Tgas = kB_T0 * Tgas(x, gamma, c, Mvir)
    rho_gas = rhogas_r(r, gamma, c, Mvir=Mvir, fb_cosmo=cosmo.Ob0 / cosmo.Om0)
    P_gas = 55 * rho_gas * ((cosmo.h**2) / (1e14)) * (kB_Tgas / 8.)
    indsel = np.where(r > rmax_rvir * rvir)[0]
    P_gas[indsel] = 0.0
    return P_gas


def P0gas_2Dr(rp_array, gamma, c, Mvir, rmax_rvir=5.0):
    Px_2D = np.zeros_like(rp_array)
    for jr in range(len(rp_array)):
        r = np.linspace(1.03 * rp_array[jr], 5.0, 500)
        rvir = mass_so.M_to_R(Mvir, 0.0, mdef) / 1000.
        rs = rvir / c
        x = r / rs
        mu = 0.59
        eta0 = eta0_func(c, c, gamma)
        kB_T0 = (1. / 3.) * eta0 * ((Mvir * units.Msun / (rvir * units.Mpc)) * mu * (const.G * const.m_p)).to(units.keV)
        kB_Tgas = kB_T0 * Tgas(x, gamma, c, Mvir)
        rho_gas = rhogas_r(r, gamma, c, Mvir=Mvir, fb_cosmo=cosmo.Ob0 / cosmo.Om0)
        Px_3D = 55 * rho_gas * ((cosmo.h**2) / (1e14)) * (kB_Tgas / 8.)
        indsel = np.where(r > rmax_rvir * rvir)[0]
        Px_3D[indsel] = 0.0
        Px_2D[jr] = 2 * sp.integrate.simps(r * Px_3D / (np.sqrt(r**2 - (rp_array[jr])**2)), r)
    return Px_2D


def get_cSZ(rp_array, gamma, c, Mvir, rmax_rvir=2.0):
    Px_2D = np.zeros_like(rp_array)
    for jr in range(len(rp_array)):
        r = np.linspace(1.01 * rp_array[jr], 5.0, 500)
        rvir = mass_so.M_to_R(Mvir, 0.0, mdef) / 1000.
        rs = rvir / c
        x = r / rs
        mu = 0.59
        eta0 = eta0_func(c, c, gamma)
        kB_T0 = (1. / 3.) * eta0 * ((Mvir * units.Msun / (rvir * units.Mpc)) * mu * (const.G * const.m_p)).to(units.keV)
        kB_Tgas = kB_T0 * Tgas(x, gamma, c, Mvir)
        rho_gas = rhogas_r(r, gamma, c, Mvir=Mvir, fb_cosmo=cosmo.Ob0 / cosmo.Om0)
        Px_3D = 55 * rho_gas * ((cosmo.h**2) / (1e14)) * (kB_Tgas / 8.)
        indsel = np.where(r > rmax_rvir * rvir)[0]
        Px_3D[indsel] = 0.0
        Px_2D[jr] = 2 * sp.integrate.simps(r * Px_3D / (np.sqrt(r**2 - (rp_array[jr])**2)), r)

    dlog_rp = np.log(rp_array)[1] - np.log(rp_array)[0]
    dlogPc1_dlogrp = np.gradient(np.log(Px_2D), dlog_rp)
    indsel = np.where(dlogPc1_dlogrp < -1)[0][0]
    rp_val_n1 = rp_array[indsel]
    r200m = mass_so.M_to_R(Mvir, 0.0, mdef) / 1000.
    c_SZ = r200m / rp_val_n1

    return c_SZ


def rhoDM(x, c, Mvir):
    mc = m(c)
    rvir = mass_so.M_to_R(Mvir, 0.0, mdef) / 1000.
    rhos = (c**3) * (Mvir / (4 * np.pi * (rvir**3) * mc))
    ydm = rhos * (1 / (x * ((1 + x)**2)))
    # if x>c:
    try:
        ydm[x > c] = 0.0
    except:
        if x > c:
            ydm = 0.0
    return ydm


def ydm_r(r, c, Mvir):
    rvir = mass_so.M_to_R(Mvir, 0.0, mdef) / 1000.
    rs = rvir / c
    x = r / rs
    return rhoDM(x, c, Mvir)


def rhoDM_2Dr(rp_array, c, Mvir):
    Px_2D = np.zeros_like(rp_array)
    for jr in range(len(rp_array)):
        r = np.linspace(1.01 * rp_array[jr], 5.0, 100)
        rvir = mass_so.M_to_R(Mvir, 0.0, mdef) / 1000.
        rs = rvir / c
        x = r / rs
        Px_3D = rhoDM(x, c, Mvir)
        Px_2D[jr] = 2 * sp.integrate.simps(r * Px_3D / (np.sqrt(r**2 - (rp_array[jr])**2)), r)
    # print(Px_2D)
    return Px_2D


if __name__ == '__main__':
    c_dm_vals = np.linspace(1.5, 10.0, 10)

    M200c_array = np.logspace(np.log10(1e13), np.log10(1e16), 10)

    r200c_mat = np.zeros((len(M200c_array), len(c_dm_vals)))
    nrp = 300
    Pgas_2D_all = np.zeros((len(M200c_array), len(c_dm_vals), nrp))
    Pgas_3D_all = np.zeros((len(M200c_array), len(c_dm_vals), nrp))
    gamma_all = np.zeros(len(c_dm_vals))
    for jM in tqdm(range(len(M200c_array))):
        M200c = M200c_array[jM]

        for jc in range(len(c_dm_vals)):
            import time
            ti = time.time()
            r200c = mass_so.M_to_R(M200c, 0.0, '200c') / 1000.
            rp_array = np.linspace(0.01, 5.0, nrp)
            gammav = get_gamma(c_dm_vals[jc])
            r200c_mat[jM, jc] = r200c
            Pgas_2D = P0gas_2Dr(rp_array, gammav, c_dm_vals[jc], M200c)
            Pgas_3D = P0gas_r(rp_array, gammav, c_dm_vals[jc], M200c)
            Pgas_2D_all[jM, jc, :] = Pgas_2D
            Pgas_3D_all[jM, jc, :] = Pgas_3D
            indsel = np.where(rp_array > r200c)[0]
            Pgas_2D_cut = np.copy(Pgas_2D)
            Pgas_2D_cut[indsel] = 0.0
            gamma_all[jc] = gammav
            print(time.time() - ti)
