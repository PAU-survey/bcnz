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

    # The problem here is all possible ways people are specifying how
    # to not use certain magnitudes. If *any* of the requirements fail,
    # then do not use the observations. 

    unreal_mag = 50.
    unreal_err = 5.

    np_or = np.logical_or 
    not_use = np.zeros(shape=mag.shape, dtype=np.bool)
    not_use = np_or(not_use, mag == conf['unobs'])
    not_use = np_or(not_use, emag == conf['unobs'])
    not_use = np_or(not_use, mag == conf['undet'])
    not_use = np_or(not_use, emag == conf['undet'])
    not_use = np_or(not_use, conf['max_magerr'] < emag)
    not_use = np_or(not_use, unreal_mag < mag)

    to_use = np.logical_not(not_use)

    # Note: One will get zero from filtering. I want to avoid people entering
    # absolute magnitudes.
    if np.min(to_use*mag) < 0:
        ids = data['id'][1 <= (to_use*mag <= 0).sum(axis=1)]
        raise ValueError('Negative magnitudes for ids: {0}'.format(ids))

    if unreal_mag < np.max(to_use*mag):
        ids = data['id'][1 <= (to_use*mag < unreal_mag).sum(axis=1)]
        raise ValueError('Internal error for ids: {0}'.format(ids))

    if unreal_err < np.max(to_use*emag):
        ids = data['id'][1 <= (unreal_err < to_use*emag).sum(axis=1)]
        raise ValueError('Unrealistic (>{0}) errors for ids: {1}'.format(unreal_err, ids))

    # Too high magnitude errors can cause problems in the chi2
    # expression.
    emag = np.clip(emag, conf['min_magerr'], 150.)

    # Convert from maginitudes to fluxes
#    X = scipy.seterr(over='ignore')
    f_obs = to_use*(10**(-0.4*mag))
    ef_obs = to_use*(10**(0.4*emag)-1)*f_obs

    assert (0. <= f_obs).all()
    ef_obs = np.where(not_use, 1e108, ef_obs)

    data['f_obs'] = f_obs
    data['ef_obs'] = ef_obs

    return data
