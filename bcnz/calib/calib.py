# Copyright (C) 2020 Martin B. Eriksen
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

from IPython.core import debugger as ipdb
import sys
import time
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from scipy.optimize import minimize

from . import libcalib


def _prepare_input(modelD, galcat, SNR_min, cosmos_scale, fit_bands):
    """Change the format on some of the input values."""

    # Here we store all observations.
    flux = galcat['flux'].stack().to_xarray()
    flux_error = galcat['flux_error'].stack().to_xarray()

    SN = flux / flux_error
    flux = flux.where(SNR_min < SN)
    flux_error = flux_error.where(SNR_min < SN)

    # this should have been changed in the input..
    if 'level_1' in flux.dims:
        flux = flux.rename({'level_1': 'band'})
        flux_error = flux_error.rename({'level_1': 'band'})

    # ... just to have completely the same.
    if cosmos_scale:
        ab_factor = 10**(0.4*26)
        cosmos_scale = ab_factor * 10**(0.4*48.6)
        flux /= cosmos_scale
        flux_error /= cosmos_scale

    # Empty structure for storint the best flux model.
    dims = ('part', 'ref_id', 'band')
    _model_parts = list(modelD.keys())
    _model_parts.sort()

    coords_flux = {'ref_id': flux.ref_id,
                   'part': _model_parts, 'band': list(fit_bands)}
    flux_model = np.zeros((len(modelD), len(flux), len(fit_bands)))

    flux_model = xr.DataArray(flux_model, dims=dims, coords=coords_flux)

    # Datastructure for storing the results...
    coords_chi2 = {'ref_id': flux.ref_id, 'part': _model_parts}
    chi2 = np.zeros((len(modelD), len(flux)))
    chi2 = xr.DataArray(chi2, dims=('part', 'ref_id'), coords=coords_chi2)

    zp_tot = xr.DataArray(np.ones(len(flux.band)), dims=('band'),
                          coords={'band': flux.band})

    return flux, flux_error, chi2, zp_tot, flux_model


def _zp_min_cost(cost, best_flux, flux, err_inv):
    """Estimate the zero-point through minimizing a cost function."""

    t1 = time.time()
    # And another test for getting the median...
    zp = np.ones(len(flux.band))
    for i, band in enumerate(flux.band):
        A = (best_flux.sel(band=band), flux.sel(band=band),
             err_inv.sel(band=band))

        X = minimize(cost, 1., args=A)
        #assert not isinstance(X.x, np.nan), 'Internal error: found nan'

        zp[i] = X.x

    return zp


def _calc_zp(best_flux, flux, flux_error):
    """Estimate the zero-point."""

    err_inv = 1. / flux_error
    X = (best_flux, flux, err_inv)

    def cost_flux(R, model, flux, err_inv):
        return float(np.abs((err_inv*(flux*R[0] - model)).median()))

    zp = _zp_min_cost(cost_flux, *X)
    zp = xr.DataArray(zp, dims=('band',), coords={'band': flux.band})

    return zp


def _which_filters(fit_bands):
    all_nb = [f'pau_nb{x}' for x in 455+10*np.arange(40)]

    NBlist = [x for x in fit_bands if (x in all_nb)]
    BBlist = [x for x in fit_bands if not (x in all_nb)]

    return NBlist, BBlist


def _find_best_model(modelD, flux_model, flux, flux_error, chi2, fit_bands,
                     Niter, Nskip):
    """Find the best flux model."""

    # Just get a normal list of the models.
    model_parts = [int(x.values) for x in flux_model.part]

    NBlist, BBlist = _which_filters(fit_bands)
    for j, key in enumerate(model_parts):
        K = (modelD[key], flux, flux_error, NBlist, BBlist, Niter, Nskip)
        chi2_part, F = libcalib.minimize_at_z(*K)
        chi2[j, :] = chi2_part.sum(dim='band')

        # Weird ref_id, gal index issue..
        assert (flux_model.ref_id.values == F.ref_id.values).all()
        assert (flux_model.band == F.band).all()
        flux_model.values[j, :] = F.values
        #flux_model[j,:] = F

    # Ok, this is not the only possible assumption!
    best_part = chi2.argmin(dim='part')


    best_flux = flux_model[best_part]

    # Old code for isel_points.
    #best_flux = flux_model.isel_points(ref_id=range(len(flux)), part=best_part)
    #best_flux = xr.DataArray(best_flux, dims=('ref_id', 'band'),
    #                         coords={'ref_id': flux.ref_id, 'band': flux.band})

    return best_flux


def _zero_points(modelD, galcat, fit_bands, SNR_min, cosmos_scale, Nrounds, Niter, learn_rate, Nskip):
    """Estimate the zero-points."""

    # Just simple input transformations.
    flux, flux_error, chi2, zp_tot, flux_model = _prepare_input(modelD, galcat,
                                                                SNR_min, cosmos_scale, fit_bands)

    flux_orig = flux.copy()

    zp_details = {}
    for i in tqdm(range(Nrounds)):
        best_flux = _find_best_model(
            modelD, flux_model, flux, flux_error, chi2, fit_bands, Niter, Nskip)

        zp = _calc_zp(best_flux, flux, flux_error)
        zp = 1 + learn_rate*(zp - 1.)

        flux = flux*zp
        flux_error = flux_error*zp

        zp_tot *= zp
        zp_details[i] = zp_tot.copy()

        # We mostly need this for debug and the paper.
        if i == Nrounds - 1:
            ratio_all = best_flux / flux_orig

    zp_tot = zp_tot.to_series()
    zp_details = pd.DataFrame(zp_details, index=flux.band)

    return zp_tot, zp_details, ratio_all


def sel_subset(galcat, fit_bands):
    """Select which subset to use for finding the zero-points."""

    # Note: I should add the cuts here after making sure that the
    # two pipelines give the same results.
    galcat = galcat[~np.isnan(galcat.zs)]

    # Removing other bands, since it internally gives a problem.
    D = {'flux': galcat.flux[fit_bands],
         'flux_error': galcat.flux_error[fit_bands]}
    cat = pd.concat(D, axis=1)
    cat['zs'] = galcat.zs

    return cat


def calib(galcat, modelD, fit_bands, SNR_min=-5, Nrounds=20, Niter=1001, cosmos_scale=True,
          learn_rate=1.0, Nskip=10, return_details=False):
    """Calibrate zero-points by comparing the result at the spectroscopic redshift.

       Args:
           fit_bands (list): Bands to fit in the comparison.
           SNR_min (float): Cut on minimum SNR value.
           cosmos_scale (bool): Converting fluxes to units used in COSMOS.
           Nrounds(int): How many calibration iterations to run.
           Niter(int): Number of minimization steps.
           learn_rate (float): How fast to update the zero-points.
           Nskip(int): Skipping updating the nb versus bb each iteration.
    """

    config = {'fit_bands': fit_bands, 'SNR_min': SNR_min, 'Nrounds': Nrounds,
              'cosmos_scale': cosmos_scale, 'Niter': Niter,
              'learn_rate': learn_rate, 'Nskip': Nskip}

    # Loads model exactly at the spectroscopic redshift for each galaxy.
    galcat = sel_subset(galcat, fit_bands)
    f_modD = libcalib.model_at_z(galcat.zs, modelD, fit_bands)

    zp, zp_details, ratio_all = _zero_points(f_modD, galcat, **config)
    ratio_all = ratio_all.to_dataframe('ratio')

    if not return_details:
        return zp
    else:
        return zp, zp_details, ratio_all
