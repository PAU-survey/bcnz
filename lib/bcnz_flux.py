#!/usr/bin/env python
# encoding: UTF8

import pdb
import numpy as np
import time
from numpy.random import normal
from scipy.integrate import quad
from scipy.interpolate import splev

import bpz_flux
import bcnz_exposure

def err_magnitude(conf, zdata, mag):
    """Error in magnitudes from the telescope and sky."""

    in_r = zdata['in_r']
    in_sky = zdata['in_sky']
    t_exp = zdata['t_exp']
    pix_size = conf['scale']**2.
    n_pix = conf['aperture'] / pix_size

    tel_surface = np.pi*(conf['D_tel']/2.)**2.
    pre = (tel_surface/n_pix)*(3631.0*1.51*1e7)

    # Filters are second index..
    yN_sig = pre*10**(-0.4*mag)*(in_r*t_exp)

    # For each filter..
    yN_sky = tel_surface*pix_size*(in_sky*t_exp)

    yStoN =  np.sqrt(n_pix*conf['n_exp'])*yN_sig / \
             np.sqrt(conf['RN']**2 + yN_sig + yN_sky)
    ynoise_ctn = 2.5*np.log10(1 + 0.02)
    yerr_m_obs = 2.5*np.log10(1.+ 1./yStoN)

    yerr_m_obs = np.sqrt(yerr_m_obs**2 + ynoise_ctn**2)

    return yerr_m_obs

def add_noise(conf, zdata, data):
    """Add noise in the magnitudes."""

    # Note that the data is still in magnitudes even if the variable names
    # tell something else.. Yes, its confusing...

    mag = data['f_obs']
    zdata['t_exp'] = bcnz_exposure.texp(conf, zdata)
    err_mag = err_magnitude(conf, zdata, mag)

    ngal, nfilters = err_mag.shape
    add_mag = np.zeros((ngal, nfilters))
    for i in range(ngal):
        for j in range(nfilters):
            add_mag[i,j] += normal(scale=err_mag[i,j])

    mag_filter = np.logical_not(np.logical_or(mag == conf['unobs'], \
                                              mag == conf['undet']))

    mag = data['f_obs']
    mag += mag_filter*add_mag
    data['f_obs'] = mag

    return zdata

def fix_fluxes(conf, zdata, data):
    """Manipulate the flux values."""

    if conf['add_noise']:
        add_noise(conf, zdata, data)

    ids,f_obs,ef_obs,m_0,z_s = bpz_flux.mega_function(conf, zdata, data)

    return ids,f_obs,ef_obs,m_0,z_s

def post_pros(conf, data):
    # FJC seems to have an inconsisten definition..
    data['f_obs'] = np.where(data['f_obs'] == 90, -99, data['f_obs'])

    if 'ID' in data:
        ids = data['ID'].astype(np.int)
    else:
        ids = np.arange(data['f_obs'].shape[0])

    data['ids'] = ids

    if 'm_0' in data:
        data['m_0'] += conf['delta_m_0']

    return data

def get_cols(conf, zdata):

    # Yes, this will be improved later..
    cols_keys = []
    cols = []

    to_iter = [('f_obs', 'flux_cols'), ('ef_obs', 'eflux_cols')]
    for k1,k2 in to_iter:
        cols_keys.append(k1)
        cols.append(zdata[k2])

    col_pars = zdata['col_pars']
    for key, val in col_pars.iteritems():
        if not len(val) == 1:
            continue

        cols_keys.append(key.lower())
        cols.append(int(val)-1)

    return cols_keys, cols
