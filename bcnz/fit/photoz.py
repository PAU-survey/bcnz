# Copyright (C) 2019 Martin B. Eriksen
# This file is part of BCNz <https://github.com/PAU-survey/bcnz>.
#
# BCNz is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BCNz is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BCNz.  If not, see <http://www.gnu.org/licenses/>.
#!/usr/bin/env python
# encoding: UTF8

import time
import dask
import dask.distributed
import numpy as np
import pandas as pd
import xarray as xr
from IPython.core import debugger as ipdb

from . import libpzqual

#np.seterr(over='raise')


def galcat_to_arrays(data_df, bands, scale_input=False):
    """Convert the galcat dataframe to arrays."""

    # Seperating this book keeping also makes it simpler to write
    # up different algorithms.

    hirarchical = True
    if hirarchical:
        cols_flux = [f'flux_{x}' for x in bands]
        cols_error = [f'flux_error_{x}' for x in bands]

        D = {'dims': ('ref_id', 'band'), 'coords': 
             {'band': bands, 'ref_id': data_df.index}}

        flux = xr.DataArray(data_df[cols_flux].values, **D)
        flux_error = xr.DataArray(data_df[cols_error].values, **D)
    else:
        dims = ('ref_id', 'band')
        flux = xr.DataArray(data_df['flux'][bands], dims=dims)
        flux_error = xr.DataArray(data_df['flux_error'][bands], dims=dims)

    # Previously I found that using on flux system or another made a
    # difference.
    if scale_input:
        ab_factor = 10**(0.4*26)
        cosmos_scale = ab_factor * 10**(0.4*48.6)

        flux /= cosmos_scale
        flux_error /= cosmos_scale

    # Not exacly the best, but Numpy had problems with the fancy
    # indexing.
    to_use = ~np.isnan(flux_error)

    # This gave problems in cases where all measurements was removed..
    flux.values = np.where(to_use, flux.values, 1e-100) #0.) 

    var_inv = 1./(flux_error + 1e-100)**2
    var_inv.values = np.where(to_use, var_inv, 1e-100)
    flux_error.values = np.where(to_use, flux_error, 1e-100)


    return flux, flux_error, var_inv


def _core_allz(ref_id, f_mod, flux, var_inv, Niter, Nskip):
    """Minimize the chi2 expression."""

    # Normalizing the models to avoid too large numbers.
    f_mod = f_mod / f_mod.max(dim=('band', 'z'))

    # To avoid very low numbers which would result in a large amplitude...
    f_mod = f_mod.where(f_mod > 1e-3, 0)


    NBlist = list(filter(lambda x: x.startswith('pau_nb'), flux.band.values))
    BBlist = list(filter(lambda x: not x.startswith('pau_nb'), flux.band.values))

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
    for i in range(Niter):
        a = np.einsum('gzst,gzt->gzs', A, v)

        m0 = b / a
        m0 = np.nan_to_num(m0)

        vn = m0*v
        v = vn
            
        # Extra step for the amplitude
        if 0 < i and i % Nskip == 0:
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

def minimize_all_z(data_df, modelD, **config): #fit_bands, Niter, Nskip):
    """Combines the chi2 estimate for all models into a single structure."""

    flux, _, var_inv = galcat_to_arrays(data_df, config['fit_bands'])
    ref_id = data_df.index

    L = []
    args = (config['Niter'], config['Nskip'])
    if hasattr(modelD, 'result'):
        modelD = modelD.result()

    keys = list(modelD.keys())
    for key in keys:
        # Supporting both interfaces.
        f_mod = modelD[key]
        L.append(_core_allz(ref_id, f_mod, flux, var_inv, *args))

    dim = pd.Index([int(x) for x in keys], name='run')
    chi2L, normL = zip(*L)

    chi2 = xr.concat(chi2L, dim=dim)
    norm = xr.concat(normL, dim=dim)

    return chi2, norm


def flatten_models(modelD):
    """Combines all the models into a flat structure."""

    # Convert the model into a single xarray!
    if hasattr(modelD, 'result'):
        modelD = modelD.result()

    keys = list(modelD.keys())
    vals = [modelD[x] for x in keys]
    
    model = xr.concat(vals, dim='run')
    model.coords['run'] = keys

    return model

def get_model(name, model, norm, pzcat, z, scale_input=False):
    """Get the model magnitudes at a given redshift."""
  

    z_xr = xr.DataArray(z, coords={'ref_id': pzcat.index.values})
    bestrun_xr = xr.DataArray(pzcat.best_run.values, coords={'ref_id': pzcat.index.values})

    tmp_model = model.sel(z=z_xr, run=bestrun_xr)
#    tmp_model = model.sel_points(z=z, run=pzcat.best_run.values, dim='ref_id')

    tmp_norm = norm.sel(z=z_xr, run=bestrun_xr)
#    tmp_norm = norm.sel_points(ref_id=pzcat.index, z=z, run=pzcat.best_run.values)
#    tmp_norm = tmp_norm.rename({'points': 'ref_id'})


    best_model = (tmp_model * tmp_norm).sum(dim='model')
    
    columns = [f'{name}_{x}' for x in best_model.band.values]
    best_model = pd.DataFrame(best_model.values, columns=columns, index=pzcat.index)
  
    if scale_input:
        best_model *= 10**(0.4*(26+48.6))


    return best_model 


def get_iband_model(model, norm, pzcat, scale_input=False, i_band=None):
    """The model i-band flux as a function of redshift."""
   
    assert i_band 

#    z_xr = xr.DataArray(z, coords={'z': pzcat.index.values})
    bestrun_xr = xr.DataArray(pzcat.best_run.values, coords={'ref_id': pzcat.index.values})

    tmp_model = model.sel(band=i_band)[bestrun_xr]
    tmp_norm = norm.sel(run=bestrun_xr)

#    ipdb.set_trace() 
#    tmp_model = model.sel(band=i_band).sel_points(run=pzcat.best_run.values)
#    tmp_norm = norm.sel_points(ref_id=pzcat.index, run=pzcat.best_run.values)

    data = (tmp_norm * tmp_model).sum(dim='model')

#    ipdb.set_trace() 
#    data = (tmp_model * tmp_norm).rename({'points': 'ref_id'}).sum(dim='model')
    columns = [f'iflux_{x}' for x in range(len(tmp_model.z))]

    df = pd.DataFrame(data.values, columns=columns, index=pzcat.index)
    
    if scale_input:
        df *= 10**(0.4*(26+48.6))

    return df

def _find_iband(fit_bands):
    """Find the i-band"""

    # This would prevent anyone from running the code without a i-band. If
    # you really want this, please modify the code. For creating randoms
    # we require the modelling in the i-band.
    for iband in ['subaru_i', 'cfht_i','kids_i']:
        if iband in fit_bands:
            return iband
    
    raise ValueError('No i-band is included in the parameters.')

def photoz(galcat, modelD, ebvD, fit_bands, Niter=1000, Nskip=10, odds_lim=0.01,
           width_frac=0.01, i_band='', only_pz=True):
    """Estimates the photoz for the models for a given configuration.

       Args:
           galcat (df): Subset of galaxy catalogue to estimate photo-z.
           modelD (dict): Dictionary containing all the flux models.
           ebvD (dict): Corresponding EBV values.
           fit_bands (list): List of bands to fit.
           Niter (int): Number of iterations in the minimization.
           Nskip (int): How often to normalize between NB and BB.
           odds_lim (float): Limit for estimating the ODDS.
           width_frac (float): Limit when estimating the pz_width.
           i_band (str): The i-band for which to return the model.
           only_pz (bool): Only return the photo-z catalogue. Otherwise return
                          pzcat, best_model, model_z0, iband_model, pz
    """

    config = {'fit_bands': fit_bands, 'Niter': Niter, 'Nskip': Nskip}

    # Propagating this value is a bit tedious when one can infer it
    # from the provided bands..
    i_band = i_band if i_band else _find_iband(fit_bands)

    chi2, norm = minimize_all_z(galcat, modelD, **config)
    pzcat, pz = libpzqual.get_pzcat(chi2, odds_lim, width_frac)

    model = flatten_models(modelD)

    # Set the number of bands. 
    cols_flux = [f'flux_{x}' for x in fit_bands]
    cols_nbflux = [f'flux_{x}' for x in fit_bands if x.startswith('pau_nb')]
    pzcat['n_band'] = (~np.isnan(galcat[cols_flux])).sum(1)
    pzcat['n_narrow'] = (~np.isnan(galcat[cols_nbflux])).sum(1)

    pzcat['ebv'] = pzcat.best_run.replace(ebvD)
    pzcat['ebv'] = pzcat.ebv.astype('float') # Just because of a weird issue.

    # Model magnitudes at the best fit redshift and z=0.
    best_model = get_model('model', model, norm, pzcat, pzcat.zb.values)
    z0 = 0.01*np.ones_like(pzcat.zb) # yes, a hack.
    model_z0 = get_model('modelz0', model, norm, pzcat, z0)
    iband_model = get_iband_model(model, norm, pzcat, i_band=i_band)


    if only_pz:
        return pzcat
    else:
        return pzcat, best_model, model_z0, iband_model, pz

# Interface useful for parallel computation. This could perhaps be elsewhere.
#
def flatten_input(df):
    """Flattens the input dataframe."""

    flux = df.flux.rename(columns=lambda x: f'flux_{x}')
    flux_error = df.flux_error.rename(columns=lambda x: f'flux_error_{x}')

    comb = pd.concat([flux, flux_error], 1)

    return comb

def photoz_flatten(galcat, *args, **kwds):
    """When working with parallel processing frameworks which works simpler
       having a flat hirarchy both in the input and output.
       
       Args:
           galcat (df): Galaxy catalogue in a hirarchical format.
    """

    #galcat = _flatten_input(galcat)
    pzcat, best_model, model_z0, iband_model, pz = photoz(galcat, *args, only_pz=False, **kwds)

    # Combine into a flat data structure. For example Dask does not support a
    # hirarchical data structure.
    pz = pd.DataFrame(pz.values, columns = [f'z{x}' for x in range(pz.shape[1])])
    pz.index = pzcat.index

    df_out = pd.concat([pzcat, best_model, model_z0, iband_model, pz], axis=1)
    
    return df_out

