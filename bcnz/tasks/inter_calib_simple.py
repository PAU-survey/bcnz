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

from matplotlib import pyplot as plt

descr = {
  'Nskip': 'Number of steps to skip',
  'fit_bands': 'Bands used in the fit',
  'Nrounds': 'How many rounds in the zero-point calibration',
  'Niter': 'Number of iterations in minimization',
  'zp_min': 'How to estimate the zero-points',
  'learn_rate': 'Learning rate',
  'SN_min': 'Minimum median SN',
  'min_ri_ratio': 'Minimum ri flux ratio'
}

import libpzcore

class inter_calib_simple:
    """Calibration per band, keeping an amplitude per galaxy free."""

    version = 1.27
    config = {'free_ampl': True, #False,
              'Nskip': 10,
              'fit_bands': [],
              'Nrounds': 19,
              'Niter': 1000,
              'zp_min': 'flux',
              'learn_rate': 1.0, # Temporarily disabled
              'SN_min': 1.,
              'min_ri_ratio': 0.5, # Temporarily disabled
              'cosmos_scale': False}

    def check_config(self):
        assert self.config['fit_bands'], 'Need to set: fit_bands'


    def _prepare_input(self, modelD, galcat):
        """Change the format on some of the input values."""

        # Here we store all observations.
        flux = galcat['flux'].stack().to_xarray()
        flux_err = galcat['flux_err'].stack().to_xarray()

        SN = flux / flux_err
        flux = flux.where(self.config['SN_min'] < SN)
        flux_err = flux_err.where(self.config['SN_min'] < SN)

        # this should have been changed in the input..        
        if 'level_1' in flux.dims:
            flux = flux.rename({'level_1': 'band'})
            flux_err = flux_err.rename({'level_1': 'band'})

        # ... just to have completely the same.
        if self.config['cosmos_scale']:
            ab_factor = 10**(0.4*26)
            cosmos_scale = ab_factor * 10**(0.4*48.6)
            flux /= cosmos_scale
            flux_err /= cosmos_scale

        # Empty structure for storint the best flux model.
        dims = ('part', 'ref_id', 'band')
        _model_parts = list(modelD.keys())
        _model_parts.sort()

        fit_bands = self.config['fit_bands']
        coords_flux = {'ref_id': flux.ref_id, 'part': _model_parts, 'band': list(fit_bands)}
        flux_model = np.zeros((len(modelD), len(flux), len(fit_bands)))

        flux_model = xr.DataArray(flux_model, dims=dims, coords=coords_flux)

        # Datastructure for storing the results...
        coords_chi2 = {'ref_id': flux.ref_id, 'part': _model_parts}
        chi2 = np.zeros((len(modelD), len(flux)))
        chi2 = xr.DataArray(chi2, dims=('part', 'ref_id'), coords=coords_chi2)

        zp_tot = xr.DataArray(np.ones(len(flux.band)), dims=('band'), \
                 coords={'band': flux.band})

        return flux, flux_err, chi2, zp_tot, flux_model


    def _zp_min_cost(self, cost, best_flux, flux, err_inv):
        """Estimate the zero-point through minimizing a cost function."""

        t1 = time.time()
        # And another test for getting the median...
        zp = np.ones(len(flux.band))
        for i,band in enumerate(flux.band):
            A = (best_flux.sel(band=band), flux.sel(band=band), \
                 err_inv.sel(band=band))

            X = minimize(cost, 1., args=A)
            #assert not isinstance(X.x, np.nan), 'Internal error: found nan'

            zp[i] = X.x

        return zp

    def calc_zp(self, best_flux, flux, flux_err):
        """Estimate the zero-point."""

        err_inv = 1. / flux_err
        X = (best_flux, flux, err_inv)

        zp_min = self.config['zp_min'] 

        def cost_flux(R, model, flux, err_inv):
            return float(np.abs((err_inv*(flux*R[0] - model)).median()))

        zp = self._zp_min_cost(cost_flux, *X)
        zp = xr.DataArray(zp, dims=('band',), coords={'band': flux.band})

        return zp

    def _which_filters(self):
        fit_bands = self.config['fit_bands']
        all_nb = [f'pau_nb{x}' for x in 455+10*np.arange(40)]

        NBlist = [x for x in fit_bands if (x in all_nb)]
        BBlist = [x for x in fit_bands if not (x in all_nb)]

        return NBlist, BBlist

    def find_best_model(self, modelD, flux_model, flux, flux_err, chi2):
        """Find the best flux model."""

        # Just get a normal list of the models.
        model_parts = [str(x.values) for x in flux_model.part]

        NBlist, BBlist = self._which_filters()
        for j,key in enumerate(model_parts):
            print('Part', j)

            K = (modelD[key], flux, flux_err, NBlist, BBlist)
            chi2_part, F = libpzcore.minimize_at_z(*K, **self.config)
            chi2[j,:] = chi2_part.sum(dim='band')

            # Weird ref_id, gal index issue..
            assert (flux_model.ref_id.values == F.ref_id.values).all()
            assert (flux_model.band == F.band).all()
            flux_model.values[j,:] = F.values
            #flux_model[j,:] = F

        # Ok, this is not the only possible assumption!
        best_part = chi2.argmin(dim='part')
        best_flux = flux_model.isel_points(ref_id=range(len(flux)), part=best_part)
        best_flux = xr.DataArray(best_flux, dims=('ref_id', 'band'), \
                    coords={'ref_id': flux.ref_id, 'band': flux.band})

        return best_flux

    def zero_points(self, modelD, galcat): 
        """Estimate the zero-points."""

        # Just simple input transformations.
        flux, flux_err, chi2, zp_tot, flux_model = self._prepare_input(modelD, galcat)
        flux_orig = flux.copy()

        zp_details = {}
        for i in tqdm(range(self.config['Nrounds'])):
            best_flux = self.find_best_model(modelD, flux_model, flux, flux_err, chi2)

            zp = self.calc_zp(best_flux, flux, flux_err)
            zp = 1 + self.config['learn_rate']*(zp - 1.)

            flux = flux*zp
            flux_err = flux_err*zp

            zp_tot *= zp
            zp_details[i] = zp_tot.copy()

            # We mostly need this for debug and the paper.
            if i == self.config['Nrounds'] - 1:
                ratio_all = best_flux / flux_orig

        zp_tot = zp_tot.to_series()
        zp_details = pd.DataFrame(zp_details, index=flux.band)

        return zp_tot, zp_details, ratio_all

    def sel_subset(self, galcat):
        """Select which subset to use for finding the zero-points."""

        # Note: I should add the cuts here after making sure that the
        # two pipelines give the same results.

        galcat = galcat[~np.isnan(galcat.zs)]

        # This cut was needed to avoid negative numbers in the logarithm.
        #SN = galcat.flux / galcat.flux_err
        #ipdb.set_trace()
        #galcat = galcat.loc[self.config['SN_min'] < SN.min(axis=1)]

        # Removing other bands, since it internally gives a problem.
        fit_bands = self.config['fit_bands']
        D = {'flux': galcat.flux[fit_bands], 'flux_err': galcat.flux_err[fit_bands]}
        cat = pd.concat(D, axis=1)
        cat['zs'] = galcat.zs

        return cat


    def entry(self, galcat):
        # Loads model exactly at the spectroscopic redshift for each galaxy.
        galcat = self.sel_subset(galcat)
        D = self.input.depend.items()
        modelD = libpzcore.model_at_z(galcat.zs, D, self.config['fit_bands'])

        zp, zp_details, ratio_all = self.zero_points(modelD, galcat)
        ratio_all = ratio_all.to_dataframe('ratio')

        return zp, zp_details, ratio_all

    def run(self):
        print('starting minimize_free')
        galcat = self.input.galcat.result

        t1 = time.time()
        zp, zp_details, ratio_all = self.entry(galcat)
        print('Time calibrating:', time.time() - t1)

        # Yes, this should not be needed.
        path = self.output.empty_file('default')
        store = pd.HDFStore(path, 'w')
        store['default'] = zp
        store['zp_details'] = zp_details
        store['ratio_all'] = ratio_all
        store.close()
