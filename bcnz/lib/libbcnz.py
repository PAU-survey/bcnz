#!/usr/bin/env python
# encoding: UTF8

import time
import dask
import numpy as np
import pandas as pd
import xarray as xr

from . import libpzqual

def galcat_to_arrays(data_df, bands, scale_input=True):
    """Convert the galcat dataframe to arrays."""

    # Seperating this book keeping also makes it simpler to write
    # up different algorithms.

    hirarchical = True
    if hirarchical:
        cols_flux = [f'flux_{x}' for x in bands]
        cols_error = [f'flux_err_{x}' for x in bands]

        D = {'dims': ('ref_id', 'band'), 'coords': 
             {'band': bands, 'ref_id': data_df.index}}

        flux = xr.DataArray(data_df[cols_flux].values, **D)
        flux_err = xr.DataArray(data_df[cols_error].values, **D)
    else:
        dims = ('ref_id', 'band')
        flux = xr.DataArray(data_df['flux'][bands], dims=dims)
        flux_err = xr.DataArray(data_df['flux_err'][bands], dims=dims)

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

    flux, _, var_inv = galcat_to_arrays(data_df, config['bands'])
    ref_id = data_df.index

    keys = list(modelD.keys())
    L = []
    for key in keys:
        # Supporting both interfaces.
        f_mod = modelD[key]
        if isinstance(f_mod, dask.distributed.client.Future):
            f_mod = f_mod.result()

        L.append(_core_allz(config, ref_id, f_mod, flux, var_inv))

    dim = pd.Index([int(x) for x in keys], name='run')
    chi2L, normL = zip(*L)

    chi2 = xr.concat(chi2L, dim=dim)
    norm = xr.concat(normL, dim=dim)

    return chi2, norm


def flatten_models(modelD):
    """Combines all the models into a flat structure."""

    # Convert the model into a single xarray!
    keys = list(modelD.keys())
    vals = []
    for x in keys:
        part = modelD[x]
        if isinstance(part, dask.distributed.client.Future):
            part = part.result()
        vals.append(part)

    model = xr.concat(vals, dim='run')
    model.coords['run'] = keys

    return model

def get_model(name, model, norm, pzcat, z, scale_input=True):
    """Get the model magnitudes at a given redshift."""
    
    tmp_model = model.sel_points(z=z, run=pzcat.best_run.values, dim='ref_id')
    tmp_norm = norm.sel_points(ref_id=pzcat.index, z=z, run=pzcat.best_run.values)
    tmp_norm = tmp_norm.rename({'points': 'ref_id'})
    best_model = (tmp_model * tmp_norm).sum(dim='model')
    
    columns = [f'{name}_{x}' for x in best_model.band.values]
    best_model = pd.DataFrame(best_model.values, columns=columns, index=pzcat.index)
   
    if scale_input:
        best_model *= 10**(0.4*(26+48.6))

    return best_model 


def get_iband_model(model, norm, pzcat, scale_input=True):
    """The model i-band flux as a function of redshift."""
    
    tmp_model = model.sel(band='I').sel_points(run=pzcat.best_run.values)
    tmp_norm = norm.sel_points(ref_id=pzcat.index, run=pzcat.best_run.values)

    data = (tmp_model * tmp_norm).rename({'points': 'ref_id'}).sum(dim='model')
    columns = [f'iflux_{x}' for x in range(len(tmp_model.z))]

    df = pd.DataFrame(data.values, columns=columns, index=pzcat.index)
    
    if scale_input:
        df *= 10**(0.4*(26+48.6))

    return df

def photoz_wrapper(data_df, config, modelD):
    """Estimates the photoz for the models for a given configuration."""

    chi2, norm = minimize_all_z(data_df, config, modelD)
    pzcat, pz = libpzqual.get_pzcat(chi2, config['odds_lim'], config['width_frac'])
    model = flatten_models(modelD)

    # Model magnitudes at the best fit redshift and z=0.
    best_model = get_model('model', model, norm, pzcat, pzcat.zb.values)
    z0 = 0.01*np.ones_like(pzcat.zb) # yes, a hack.
    model_z0 = get_model('modelz0', model, norm, pzcat, z0)
    iband_model = get_iband_model(model, norm, pzcat)

    
    # Combining into one data structure, since Dask does not support
    # hirarchical data structures.
    
    pz = pd.DataFrame(pz.values, columns = [f'z{x}' for x in range(pz.shape[1])])
    pz.index = pzcat.index

    df_out = pd.concat([pzcat, best_model, model_z0, iband_model, pz], 1)
    
    return df_out
