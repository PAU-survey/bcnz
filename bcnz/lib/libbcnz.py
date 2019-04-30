#!/usr/bin/env python
# encoding: UTF8

import time
import numpy as np
import pandas as pd
import xarray as xr

import libpzqual

def galcat_to_arrays(data_df, filters, scale_input=True):
    """Convert the galcat dataframe to arrays."""

    # Seperating this book keeping also makes it simpler to write
    # up different algorithms.

    hirarchical = True
    if hirarchical:
        # Bit of a hack..
        bands = [x.replace('flux_err_','') for x in data_df.columns \
                 if x.startswith('flux_err_')]
        cols_flux = [f'flux_{x}' for x in bands]
        cols_error = [f'flux_err_{x}' for x in bands]

        D = {'dims': ('ref_id', 'band'), 'coords': 
             {'band': bands, 'ref_id': data_df.index}}

        flux = xr.DataArray(data_df[cols_flux].values, **D)
        flux_err = xr.DataArray(data_df[cols_error].values, **D)
    else:
        dims = ('ref_id', 'band')
        flux = xr.DataArray(data_df['flux'][filters], dims=dims)
        flux_err = xr.DataArray(data_df['flux_err'][filters], dims=dims)

    # Previously I found that using on flux system or another made a
    # difference.
    if scale_input:
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


def _core_allz(config, ref_id, f_mod, flux, var_inv):
    """Minimize the chi2 expression."""

    NBlist = list(filter(lambda x: x.startswith('NB'), flux.band.values))
    BBlist = list(filter(lambda x: not x.startswith('NB'), flux.band.values))

    #print('var_inv',  var_inv.shape, var_inv.coords)
    #print('f_mod',  f_mod.shape, f_mod.coords)

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

def minimize_all_z(data_df, config, modelD):
    """Combines the chi2 estimate for all models into a single structure."""

    flux, _, var_inv = galcat_to_arrays(data_df, config['filters'])
    ref_id = data_df.index

    keys = list(modelD.keys())
    L = []
    for key in keys:
        f_mod = modelD[key]
        L.append(_core_allz(config, ref_id, f_mod, flux, var_inv))

    dim = pd.Index([int(x) for x in keys], name='run')
    chi2L, normL = zip(*L)

    chi2 = xr.concat(chi2L, dim=dim)
    norm = xr.concat(normL, dim=dim)

    return chi2, norm

def get_model(name, model, norm, pzcat, z):
    """Get the model magnitudes at a given redshift."""
    
    tmp_model = model.sel_points(z=z, run=pzcat.best_run.values, dim='ref_id')
    tmp_norm = norm.sel_points(ref_id=pzcat.index, z=z, run=pzcat.best_run.values)
    tmp_norm = tmp_norm.rename({'points': 'ref_id'})
    best_model = (tmp_model * tmp_norm).sum(dim='model')
    
    return best_model 

def photoz_wrapper(data_df, config, modelD):
    """Estimates the photoz for the models for a given configuration."""

    print('data type', type(data_df))
    chi2, norm = minimize_all_z(data_df, config, modelD)
    pzcat, pz = libpzqual.get_pzcat(chi2, config['odds_lim'], config['width_frac'])

    # Convert the model to an xarray!
    keys = list(modelD.keys())
    vals = [modelD[x] for x in keys]
    model = xr.concat(vals, dim='run')
    model.coords['run'] = keys

    # Model magnitudes at the best fit redshift and z=0.
    best_model = get_model('model', model, norm, pzcat, pzcat.zb.values)
    z0 = np.zeros_like(pzcat.zb)
    flux_z0 = get_model('modelz0', model, norm, pzcat, z0)

    # Combining into one data structure, since Dask does not support
    # hirarchical data structures.
    def prefix(pre, cat):
        return [f'{pre}_{x}' for x in cat.band.values]

    names = list(pzcat.columns) + prefix('model', best_model) + \
            prefix('modelz0', model_z0) +\
            [f'z{x}' for x in range(pz.shape[1])]

    data = np.hstack([pzcat, best_model, model_z0, pz])
    df_out = pd.DataFrame(data, columns=names, index=pzcat.index)

    return df_out
