#!/usr/bin/env python
# encoding: UTF8

# Normalize energy distributions.
import pdb
import numpy as np

# TODO: THIS FILE IS NO LONGER IN USE....

def norm_model(conf, zdata, f_mod):

    if not conf['norm_flux']:
        return f_mod

    n_filter = '5000'
    n_value = 1e-10

    n_ind =  zdata['filters'].index(n_filter)

    n_norm = n_value / f_mod[:,:, n_ind]
    # Currently f_mod is z,t,f. Want f,z,t
    shape_bef = f_mod.shape
    f_mod = f_mod.swapaxes(0,2)
    f_mod = f_mod.swapaxes(1,2)

    f_mod = n_norm*f_mod 

    f_mod = f_mod.swapaxes(0,1)
    f_mod = f_mod.swapaxes(1,2)
    assert shape_bef == f_mod.shape

    print('Used model norm')

    return f_mod

def norm_data(conf, filters, f_obs, ef_obs):

    if not conf['norm_flux']:
        return f_obs, ef_obs

    n_filter = '5000'
    n_value = 1e-10
    n_ind =  filters.index(n_filter)

#    iszero = f_obs[:,n_ind] == 0

    olderr = np.seterr(divide='ignore')
    n_norm = n_value/f_obs[:,n_ind]
    np.seterr(**olderr)

    n_norm[np.isinf(n_norm)] = 1.

#    pdb.set_trace()

    f_obs = (n_norm*f_obs.T).T
    ef_obs = (n_norm*ef_obs.T).T

    print('Used data norm')

    return f_obs, ef_obs
