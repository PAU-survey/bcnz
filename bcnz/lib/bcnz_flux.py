#!/usr/bin/env python
# encoding: UTF8

import copy
import pdb
import numpy as np
import time
from numpy.random import normal
from scipy.integrate import quad
from scipy.interpolate import splev

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

#    mag_filter = np.logical_not(np.logical_or(mag == conf['unobs'], \
#                                              mag == conf['undet']))

#    mag_filter = err_mag < 0.5
    to_use = err_mag < 0.5
    mag = data['f_obs']
    mag += to_use*add_mag
    data['f_obs'] = mag
    data['ef_obs'] = np.where(to_use, err_mag, -99.)

#    pdb.set_trace()

    return zdata

def mega1(conf, zdata, data):

    assert conf['mag'], 'Only magnitudes are implemented'
    f_obs = data['f_obs']
    ef_obs = data['ef_obs']
    f1 = f_obs.copy()
    ef1 = ef_obs.copy()

    if False:
        seen = np.logical_and(0. < f_obs, f_obs < conf['undet'])
        # Multiplied with one to test if not multiple values are true
        # at once...
        assert (1*seen + 1*not_seen + 1*not_obs == 1).all()

    not_seen = (ef_obs == conf['undet'])
    not_obs = (ef_obs == conf['unobs'])

    seen = np.logical_not(np.logical_or(not_seen, not_obs))
#    pdb.set_trace()
    ef_obs = np.clip(ef_obs, conf['min_magerr'], np.inf)

#    pdb.set_trace()
    # Convert to fluxes
    f_obs = seen*(10**(-0.4*f_obs))
    ef_obs = seen*(10**(0.4*ef_obs)-1)*f_obs

    # One exception...
    ef_obs += not_seen*(10**(-0.4*np.abs(ef_obs)))

    assert (0. <= f_obs).all()
    ef_obs = np.where(not_obs, 1e108, ef_obs)

    return f_obs, ef_obs

def fix_fluxes(conf, zdata, data):
    """Manipulate the flux values."""

    if conf['add_noise']:
        add_noise(conf, zdata, data)

    f_obs, ef_obs = mega1(conf, zdata, data)

    return f_obs,ef_obs

def post_pros(conf, data):
    # FJC seems to have an inconsisten definition..

#    pdb.set_trace()
    data['f_obs'] = np.where(data['f_obs'] == 90, -99, data['f_obs'])
    data['ef_obs'] = np.where(data['ef_obs'] == 99, -99, data['ef_obs'])

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
        if isinstance(val, tuple):
            continue

        cols_keys.append(key.lower())
        cols.append(int(val)-1)

    return cols_keys, cols


