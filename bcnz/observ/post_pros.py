#!/usr/bin/env python
# encoding: UTF8

# All the post processing after reading in the data.
import ipdb
import numpy as np
from scipy.interpolate import splev, splrep
import bcnz

def setid_m0s(conf, data):
    """Set ID and add magnitude zero points."""

    if 'id' in data:
        ids = data['id'].astype(np.int)
    else:
        ids = np.arange(data['mag'].shape[0])

    data['id'] = ids

    if 'm_0' in data:
        data['m_0'] += conf['delta_m_0']

    return data

def add_noise(conf, zdata, data):
    """Add noise in the magnitudes."""

#    zdata['t_exp'] = texp(conf, zdata['filters'])
    mag = data['mag']

    # These lines are added later using a previous data
    # structure.
    snD, merrD = zdata['sn_splD'], zdata['merrD']
    err_mag = np.zeros_like(mag)
    SN = np.zeros_like(mag)
    print('mag range', np.min(mag), np.max(mag))
    for i,fname in enumerate(conf['filters']):
        err_mag[:,i] = splev(mag[:,i], merrD[fname], ext=2)
        SN[:,i] = splev(mag[:,i], snD[fname], ext=2)

#    ipdb.set_trace()
#    err_mag, SN = err_magnitude(conf, zdata, mag)

    ngal, nfilters = err_mag.shape
    add_mag = np.zeros((ngal, nfilters))
    for i in range(ngal):
        for j in range(nfilters):
            add_mag[i,j] += np.random.normal(scale=err_mag[i,j])

    to_use = np.logical_and(err_mag < 0.5, conf['sn_lim'] <= SN)

    mag += to_use*add_mag
    data['mag'] = mag
    data['emag'] = np.where(to_use, err_mag, -99.)


    return data

def post_pros(conf, zdata, data):
    """Post processing of the observations."""

    data = setid_m0s(conf, data)

    if conf['add_noise']:
        data = add_noise(conf, zdata, data)

    data = bcnz.observ.toflux(conf, zdata, data)
    return data
