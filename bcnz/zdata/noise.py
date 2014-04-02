#!/usr/bin/env python
# encoding: UTF8

import copy
import pdb
import numpy as np
import time
from numpy.random import normal
from scipy.interpolate import splev, splrep

import bcnz

def texp(conf, filters):
    """Find exposure time in each of the filters."""

    texpD = {}
    for f in ['up', 'g', 'r', 'i', 'z', 'y', 'Y', 'J', 'H']:
        texpD[f] = conf['exp_{0}'.format(f)]


    for i, ftray in enumerate(conf['trays']):
        exp_time = conf['exp_t{0}'.format(i+1)]
        for f in ftray:
            texpD[f] = exp_time

    res = [texpD[x] for x in filters]

    return res

def err_mag(conf, zdata, mag):
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

    return yerr_m_obs, yStoN

def sn_spls(conf, zdata):
    """Construct splines with magnitude errors and SN."""

    import ipdb
    filters = zdata['filters']
    mag_interp = np.linspace(15., 30)
    mag = np.tile(mag_interp, (len(filters), 1)).T


    merrD, sn_splD = {}, {}
    mag_err, SN = err_mag(conf, zdata, mag)
    for i,f in enumerate(filters):
        merrD[f] = splrep(mag_interp, mag_err[:,i])
        sn_splD[f] = splrep(mag_interp, SN[:,i])

    return {'merrD': merrD, 'sn_splD': sn_splD}


def noise_info(conf, zdata):

    zdata['texp'] = texp(conf, zdata['filters'])
    zdata['sn_spls'] = sn_spls(conf, zdata)
 
    return noise_zdata

def OLDadd_noise(conf, zdata, data):
    """Add noise in the magnitudes."""

    raise NotImplentedError, 'This is moved...'

    zdata['t_exp'] = texp(conf, zdata['filters'])
    mag = data['mag']
    err_mag, SN = err_magnitude(conf, zdata, mag)

    ngal, nfilters = err_mag.shape
    add_mag = np.zeros((ngal, nfilters))
    for i in range(ngal):
        for j in range(nfilters):
            add_mag[i,j] += normal(scale=err_mag[i,j])

    to_use = np.logical_and(err_mag < 0.5, conf['sn_lim'] <= SN)

    mag += to_use*add_mag
    data['mag'] = mag
    data['emag'] = np.where(to_use, err_mag, -99.)


    return data
