#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import time
import numpy as np
import pandas as pd
import xarray as xr

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from tqdm import tqdm

def model_at_z(zs, modelD, fit_bands):
    """Load the models at specific redshifts."""

    t1 = time.time()
    f_modD = {}
    inds = ['band', 'sed', 'ext_law', 'EBV', 'z']
    for key, f_mod in modelD.items():
        f_mod = f_mod.sel(band=fit_bands)
        f_mod = f_mod.sel(z=zs.values, method='nearest')

        if 'EBV' in f_mod.dims:
            f_mod = f_mod.squeeze('EBV')

        if 'ext_law' in f_mod.dims:
            f_mod = f_mod.squeeze('ext_law')

        f_modD[key] = f_mod.transpose('z', 'band', 'sed')

    print('Time loading model:', time.time() - t1)

    return f_modD


def minimize_at_z(f_mod, flux, flux_err, NBlist, BBlist, Niter, Nskip):
    """Minimize at a known redshift.
       Args: 
           f_mod (DataArray): Flux model from templates.
           flux (DataArray): The fluxes.
           flux_err (DataArray): The flux errrors
           NBlist (list): List with narrow bands to fit.
           BBlist (list): List with broad bands to fit.
           Niter (int): How many iterations to run.
           Nskip (int): Number of iterations between each BB vs NB adjustment.
    """

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
    for i in range(Niter):
        a = np.einsum('gst,gt->gs', A, v)

        m0 = b / a
        vn = m0*v

        v = vn
        # Extra step for the amplitude
        if 0 < i and i % Nskip == 0:
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
