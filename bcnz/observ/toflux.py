#!/usr/bin/env python
# encoding: UTF8

import pdb
import numpy as np
import scipy

def toflux(conf, zdata, data):
    """Convert input magnitude to fluxes."""

    assert conf['mag'], 'Only magnitudes are implemented'

    mag = data['mag']
    emag = data['emag']

    A1 = mag == conf['unobs']
    A2 = emag == conf['unobs']
    A3 = conf['max_magerr'] < emag
    not_obs = np.logical_or(A1, np.logical_or(A2, A3))

    B1 = (mag == conf['undet'])
    B2 = (emag == conf['undet'])
    not_det = np.logical_or(B1, B2)
#    not_obs = np.logical_or(emag == conf['unobs'], conf['max_magerr'] < emag)
    not_use = np.logical_or(not_det, not_obs)
    touse = np.logical_not(not_use)
    assert (1*touse + 1*not_det + 1*not_obs == 1).all()

#    print(np.min(emag), np.max(emag))
#    pdb.set_trace()

    # Clipping on mag value since some values can be unreasonable high..
    emag = np.clip(emag, conf['min_magerr'], 150.)

    assert np.max(np.abs(touse*mag)) < 30
    assert np.max(touse*emag) < 5.

    # Convert to fluxes
#    X = scipy.seterr(over='ignore')
    f_obs = touse*(10**(-0.4*mag))
    ef_obs = touse*(10**(0.4*emag)-1)*f_obs

    # One exception...
#    ef_obs += not_seen*(10**(-0.4*np.abs(ef_obs)))

    assert (0. <= f_obs).all()
    ef_obs = np.where(not_use, 1e108, ef_obs)

    data['f_obs'] = f_obs
    data['ef_obs'] = ef_obs

    return data