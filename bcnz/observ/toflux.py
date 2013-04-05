#!/usr/bin/env python
# encoding: UTF8

import pdb
import numpy as np

def toflux(conf, zdata, data):
    """Convert input magnitude to fluxes."""

    assert conf['mag'], 'Only magnitudes are implemented'

    mag = data['mag']
    emag = data['emag']

    not_seen = (emag == conf['undet'])

    A1 = emag == conf['unobs']
    A2 = conf['max_magerr'] < emag
    not_obs = np.logical_or(A1, A2)
#    not_obs = np.logical_or(emag == conf['unobs'], conf['max_magerr'] < emag)

    seen = np.logical_not(np.logical_or(not_seen, not_obs))
    assert (1*seen + 1*not_seen + 1*not_obs == 1).all()

    print(np.min(emag), np.max(emag))
#    pdb.set_trace()
    emag = np.clip(emag, conf['min_magerr'], np.inf)

    assert np.max(np.abs(seen*mag)) < 30
    assert np.max(seen*emag) < 5.

    # Convert to fluxes
    f_obs = seen*(10**(-0.4*mag))
    ef_obs = seen*(10**(0.4*emag)-1)*f_obs

    # One exception...
#    ef_obs += not_seen*(10**(-0.4*np.abs(ef_obs)))

    assert (0. <= f_obs).all()
    ef_obs = np.where(not_obs, 1e108, ef_obs)

    data['f_obs'] = f_obs
    data['ef_obs'] = ef_obs

    return data
