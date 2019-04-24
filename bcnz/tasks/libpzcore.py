#!/usr/bin/env python
# encoding: UTF8

import time
import numpy as np
import pandas as pd
import xarray as xr

def model_at_z(zs, modelD, fit_bands):
    """Load the models at specific redshifts."""

    t1 = time.time()
    f_modD = {}
    inds = ['band', 'sed', 'ext_law', 'EBV', 'z']
    for key, dep in modelD:
        if not key.startswith('model'):
            continue

        # Later the code depends on this order.
        f_mod = dep.result.reset_index().set_index(inds)
        f_mod = f_mod.to_xarray().flux
        f_mod = f_mod.sel(band=fit_bands)
        f_mod = f_mod.sel(z=zs.values, method='nearest')

        if 'EBV' in f_mod.dims:
            f_mod = f_mod.squeeze('EBV')

        if 'ext_law' in f_mod.dims:
            f_mod = f_mod.squeeze('ext_law')

        f_modD[key] = f_mod.transpose('z', 'band', 'sed')

    print('Time loading model:', time.time() - t1)

    return f_modD


def minimize_at_z(f_mod, flux, flux_err, NBlist, BBlist, **config):
    """Minimize at a known redshift."""

    var_inv = 1. / flux_err**2

    # Problematic entries.        
    mask = np.isnan(flux)
    var_inv.values[mask] = 0.
    flux = flux.fillna(0.)

    A_NB = np.einsum('gf,gfs,gft->gst', var_inv.sel(band=NBlist), \
           f_mod.sel(band=NBlist), f_mod.sel(band=NBlist))
    b_NB = np.einsum('gf,gf,gfs->gs', var_inv.sel(band=NBlist), flux.sel(band=NBlist), \
           f_mod.sel(band=NBlist))
    A_BB = np.einsum('gf,gfs,gft->gst', var_inv.sel(band=BBlist), \
           f_mod.sel(band=BBlist), f_mod.sel(band=BBlist))
    b_BB = np.einsum('gf,gf,gfs->gs', var_inv.sel(band=BBlist), flux.sel(band=BBlist), \
           f_mod.sel(band=BBlist))

    # Testing to scale to the narrow bands. In that case the code above is not needed.
    scale_to = NBlist
    var_inv_NB = var_inv.sel(band=scale_to)
    flux_NB = flux.sel(band=scale_to)
    f_mod_NB = f_mod.sel(band=scale_to)
    S1 = (var_inv_NB*flux_NB).sum(dim='band')

    # Since we need these entries in the beginning...
    k = np.ones(len(flux))
    b = b_BB + k[:,np.newaxis]*b_NB
    A = A_BB + k[:,np.newaxis,np.newaxis]**2*A_NB

    v = 100*np.ones_like(b)

    ref_id = flux.ref_id.values
    coords = {'ref_id': ref_id, 'band': f_mod.band}
    coords_norm = {'ref_id': ref_id, 'model': f_mod.sed}


    t1 = time.time()
    for i in range(config['Niter']):
        a = np.einsum('gst,gt->gs', A, v)

        m0 = b / a
        vn = m0*v

        v = vn
        # Extra step for the amplitude
        if 0 < i and i % config['Nskip'] == 0:
            # Testing a new form for scaling the amplitude...
            flux_mod = np.einsum('gfs,gs->gf', f_mod_NB, v)
            S1 = np.einsum('gf,gf,gf->g', var_inv_NB, flux_NB, flux_mod)
            S2 = np.einsum('gf,gf->g', var_inv_NB, flux_mod**2)

            k = S1 / S2

            # Just to avoid crazy values ...
            k = np.clip(k, 0.1, 10)

            b = b_BB + k[:,None]*b_NB
            A = A_BB + k[:,None,None]**2*A_NB

    # I was comparing with the standard algorithm above...
    v_scaled = v
    k_scaled = k

    L = []
    L.append(np.einsum('g,gfs,gs->gf', k_scaled, f_mod.sel(band=NBlist), v_scaled))
    L.append(np.einsum('gfs,gs->gf', f_mod.sel(band=BBlist), v_scaled))

    Fx = np.hstack(L)

    coords['band'] = NBlist + BBlist
    Fx = xr.DataArray(Fx, coords=coords, dims=('ref_id', 'band'))

    chi2x = var_inv*(flux - Fx)**2

    return chi2x, Fx


def galcat_to_arrays(data_df, filters, scale_input=True):
    """Convert the galcat dataframe to arrays."""

    # Seperating this book keeping also makes it simpler to write
    # up different algorithms.

    dims = ('gal', 'band')
    flux = xr.DataArray(data_df['flux'][filters], dims=dims)
    flux_err = xr.DataArray(data_df['flux_err'][filters], dims=dims)

    # Previously I found that using on flux system or another made a
    # difference.
    if scale_input:
        print('Convert away from PAU fluxes...')
        ab_factor = 10**(0.4*26)
        cosmos_scale = ab_factor * 10**(0.4*48.6)

        flux /= cosmos_scale
        flux_err /= cosmos_scale

    # Not exacly the best, but Numpy had problems with the fancy
    # indexing.
    to_use = ~np.isnan(flux_err)

    # This gave problems in cases where all measurements was removed..
    flux.values = np.where(to_use, flux.values, 1e-100) #0.) 

    var_inv = 1./(flux_err + 1e-100)**2
    var_inv.values = np.where(to_use, var_inv, 1e-100)
    flux_err.values = np.where(to_use, flux_err, 1e-100)

    return flux, flux_err, var_inv


def minimize_all_z(config, ref_id, f_mod, flux, var_inv):
    """Minimize the chi2 expression."""

    NBlist = list(filter(lambda x: x.startswith('NB'), flux.band.values))
    BBlist = list(filter(lambda x: not x.startswith('NB'), flux.band.values))

    A_NB = np.einsum('gf,zfs,zft->gzst', var_inv.sel(band=NBlist), \
           f_mod.sel(band=NBlist), f_mod.sel(band=NBlist))
    b_NB = np.einsum('gf,gf,zfs->gzs', var_inv.sel(band=NBlist), flux.sel(band=NBlist), \
           f_mod.sel(band=NBlist))
    A_BB = np.einsum('gf,zfs,zft->gzst', var_inv.sel(band=BBlist), \
           f_mod.sel(band=BBlist), f_mod.sel(band=BBlist))
    b_BB = np.einsum('gf,gf,zfs->gzs', var_inv.sel(band=BBlist), flux.sel(band=BBlist), \
           f_mod.sel(band=BBlist))

    # Testing to scale to the narrow bands. In that case the code above is not needed.
    scale_to = NBlist
    var_inv_NB = var_inv.sel(band=scale_to)
    flux_NB = flux.sel(band=scale_to)
    f_mod_NB = f_mod.sel(band=scale_to)
    S1 = (var_inv_NB*flux_NB).sum(dim='band')

    # Since we need these entries in the beginning...
    k = np.ones((len(flux), len(f_mod.z)))
    b = b_BB + k[:,:,np.newaxis]*b_NB
    A = A_BB + k[:,:,np.newaxis,np.newaxis]**2*A_NB

    v = 100*np.ones_like(b)

    ref_id = np.array(ref_id)
    coords = {'ref_id': ref_id, 'band': f_mod.band, 'z': f_mod.z}
    coords_norm = {'ref_id': ref_id, 'z': f_mod.z, 'model': f_mod.model}

    t1 = time.time()
    for i in range(config['Niter']):
        a = np.einsum('gzst,gzt->gzs', A, v)

        m0 = b / a
        vn = m0*v

        v = vn
        # Extra step for the amplitude
        if 0 < i and i % config['Nskip'] == 0:
            # Testing a new form for scaling the amplitude...
            S2 = np.einsum('gf,zfs,gzs->gz', var_inv_NB, f_mod_NB, v)
            k = (S1.values/S2.T).T

            # Just to avoid crazy values ...
            k = np.clip(k, 0.1, 10)

            b = b_BB + k[:,:,np.newaxis]*b_NB
            A = A_BB + k[:,:,np.newaxis,np.newaxis]**2*A_NB

    # I was comparing with the standard algorithm above...
    v_scaled = v
    k_scaled = k

    L = []
    L.append(np.einsum('gz,zfs,gzs->gzf', k_scaled, f_mod.sel(band=NBlist), v_scaled))
    L.append(np.einsum('zfs,gzs->gzf', f_mod.sel(band=BBlist), v_scaled))

    Fx = np.dstack(L)
    coords['band'] = NBlist + BBlist
    Fx = xr.DataArray(Fx, coords=coords, dims=('ref_id', 'z', 'band'))

    # Now with another scaling...
    chi2x = var_inv*(flux - Fx)**2
    Fx = xr.DataArray(Fx, coords=coords, dims=('ref_id', 'z', 'band'))
    chi2x = var_inv*(flux - Fx)**2
    chi2x = chi2x.sum(dim='band')

    norm = xr.DataArray(v_scaled, coords=coords_norm, dims=\
                        ('ref_id','z','model'))

    return chi2x, norm

def bestfit_all_z(config, modelD, data_df):
    """Combines the chi2 estimate for all models into a single structure."""

    flux, _, var_inv = galcat_to_arrays(data_df, config['filters'])

    keys = list(modelD.keys())
    L = []
    for key in keys:
        print('key', key)
        L.append(minimize_all_z(config, data_df.index, modelD[key], flux, var_inv))

    dim = pd.Index(keys, name='run')
    chi2L, normL = zip(*L)

    chi2 = xr.concat(chi2L, dim=dim)
    norm = xr.concat(normL, dim=dim)

    print(chi2)
    print(norm)

    return chi2, norm
