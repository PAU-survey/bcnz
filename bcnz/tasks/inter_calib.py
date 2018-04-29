#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import sys
import time
import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import minimize

from matplotlib import pyplot as plt

descr = {
  'fit_bands': 'Bands used in the fit',
  'Nrounds': 'How many rounds in the zero-point calibration',
  'Niter': 'Number of iterations in minimization',
  'zp_min': 'How to estimate the zero-points',
  'learn_rate': 'Learning rate',
  'SN_min': 'Minimum median SN',
  'min_ri_ratio': 'Minimum ri flux ratio'
}

class inter_calib:
    """Calibration between the broad and the narrow bands."""

    version = 1.23
    config = {'free_ampl': False,
              'fit_bands': [],
              'Nrounds': 19,
              'Niter': 1000,
              'zp_min': 'mag', # yes, this in not my preferred option...
              'learn_rate': 1.0, # Temporarily disabled
              'SN_min': 1.,
              'min_ri_ratio': 0.5} # Temporarily disabled

    def check_config(self):
        assert self.config['fit_bands'], 'Need to set: fit_bands'

    def minimize(self, f_mod, flux, flux_err):
        """Minimize the model difference at a specific redshift."""

        # Otherwise we will get nans in the result.
        var_inv = 1./flux_err**2
        var_inv = var_inv.fillna(0.)
        xflux = flux.fillna(0.)

        A = np.einsum('gf,gfs,gft->gst', var_inv, f_mod, f_mod)
        b = np.einsum('gf,gf,gfs->gs', var_inv, xflux, f_mod)

        v = np.ones_like(b)
        t1 = time.time()
        for i in range(self.config['Niter']):
            a = np.einsum('gst,gt->gs', A, v)
            m0 = b / a
            vn = m0*v
            v = vn

        gal_id = np.array(flux.ref_id)
        coords = {'ref_id': gal_id, 'band': f_mod.band}
        coords_norm = {'ref_id': gal_id, 'model': np.array(f_mod.sed)}
        F = np.einsum('gfs,gs->gf', f_mod, v)
        flux_model = xr.DataArray(F, coords=coords, dims=('ref_id', 'band'))

        chi2 = var_inv*(flux - F)**2
        chi2.values = np.where(chi2==0., np.nan, chi2)

        return chi2, F

    def minimize_free(self, f_mod, flux, flux_err):
        """Minimize the model difference at a specific redshift."""

        # Otherwise we will get nans in the result.
        var_inv = 1./flux_err**2
        var_inv = var_inv.fillna(0.)
        xflux = flux.fillna(0.)

        NBlist = list(filter(lambda x: x.startswith('NB'), flux.band.values))
        BBlist = list(filter(lambda x: not x.startswith('NB'), flux.band.values))

        A_NB = np.einsum('gf,gfs,gft->gst', var_inv.sel(band=NBlist), \
               f_mod.sel(band=NBlist), f_mod.sel(band=NBlist))
        b_NB = np.einsum('gf,gf,gfs->gs', var_inv.sel(band=NBlist), xflux.sel(band=NBlist), \
               f_mod.sel(band=NBlist))
        A_BB = np.einsum('gf,gfs,gft->gst', var_inv.sel(band=BBlist), \
               f_mod.sel(band=BBlist), f_mod.sel(band=BBlist))
        b_BB = np.einsum('gf,gf,gfs->gs', var_inv.sel(band=BBlist), xflux.sel(band=BBlist), \
               f_mod.sel(band=BBlist))

        k = np.ones((len(flux)))
        b = b_NB + k[:,np.newaxis]*b_BB
        A = A_NB + k[:,np.newaxis,np.newaxis]**2*A_BB

        v = np.ones_like(b)
        t1 = time.time()
        for i in range(self.config['Niter']):
            a = np.einsum('gst,gt->gs', A, v)
            m0 = b / a
            vn = m0*v
            v = vn

            S1 = np.einsum('gt,gt->g', b_BB, v)
            S2 = np.einsum('gs,gst,gt->g', v, A_BB, v)
            k = S1 / S2

            b = b_NB + k[:,np.newaxis]*b_BB
            A = A_NB + k[:,np.newaxis,np.newaxis]**2*A_BB


        gal_id = np.array(flux.ref_id)
        coords = {'ref_id': gal_id, 'band': f_mod.band}
        coords_norm = {'ref_id': gal_id, 'model': np.array(f_mod.sed)}
        F = np.einsum('gfs,gs->gf', f_mod, v)

        # Scaling the model..
        isBB = np.array(list(map(lambda x: not x.startswith('NB'), flux.band.values)))
        F[:,isBB] *= k[:,np.newaxis]
    
        chi2 = var_inv*(flux - F)**2
        chi2.values = np.where(chi2==0., np.nan, chi2)

        flux_model = xr.DataArray(F, coords=coords, dims=('ref_id', 'band'))

        return chi2, F


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

    def zp_mag(self, best_model, flux, flux_err):
        """Zero-points when using an expression minimizing the median
           of a magnitude distribution.
        """

        # The below code is copied from Alex with minimal
        # changes. This code hopefully disappears soon..
        def cost(x, sig,err,bestmodel):
            shift = x
            shift = 10**(-0.4*shift)
            sig = sig * shift
            err = err * shift
            S = -2.5*np.log10(sig)-48.6
            E = 2.5*np.log10(1+err/sig)
            Em = 2.5*np.log10(1-err/sig)
            M = -2.5*np.log10(bestmodel)-48.6
            S[np.isinf(S)] = np.nan
            E[np.isinf(E)] = np.nan
            # Em[np.isinf(Em)] = np.nan
            M[np.isinf(M)] = np.nan
            arr = (S-M) / E
            # arr = np.where(S>M, (S-M) / Em, (S-M) / E)
            arr[(arr<-5)|(arr>5)] = np.nan
            median = np.nanmedian(arr)
            val = np.abs(median)

            return val

        sig = flux.values
        err = flux_err.values
        bestmodel = best_model

        nbands = len(flux.band)
        zp_mag = np.zeros(nbands)
        for i in range(nbands):
            print('Running for band:', i)

            S,E,M = (sig[:,i], err[:,i], bestmodel[:,i])
            E = E[~np.isnan(S)]
            M = M[~np.isnan(S)]
            S = S[~np.isnan(S)]
            res = minimize(cost, x0=[0.,], args=(S,E,M), method='SLSQP', bounds=((-0.1, 0.1),))

            zp_mag[i] = res.x[0]

        zp = 10**(-0.4*zp_mag)

        return zp


    def calc_zp(self, best_flux, flux, flux_err):
        """Estimate the zero-point."""

        err_inv = 1. / flux_err

        X = (best_flux, flux, err_inv)
        zp_min = self.config['zp_min'] 

        if zp_min == 'mag':
            zp = self.zp_mag(*X)
        elif zp_min == 'flux':
            def cost_flux(R, model, flux, err_inv):
                return float(np.abs((err_inv*(flux*R[0] - model)).median()))

            zp = self._zp_min_cost(cost_flux, *X)
        elif zp_min == 'flux2':
            def cost_flux2(R, model, flux, err_inv):
                return float(np.abs((err_inv*(flux*R[0] - model)/R[0]).median()))

            zp = self._zp_min_cost(cost_flux2, *X)
        elif zp_min == 'flux_chi2':
            zp = self._zp_flux_chi2(*X)

        zp = xr.DataArray(zp, dims=('band',), coords={'band': flux.band})

        return zp

    def _prepare_input(self, modelD, galcat):
        """Change the format on some of the input values."""

        # The observations.
        fit_bands = self.config['fit_bands']
        flux = galcat['flux'][fit_bands].stack().to_xarray()
        flux_err = galcat['flux_err'][fit_bands].stack().to_xarray()

        # Empty structure for storint the best flux model.
        dims = ('part', 'ref_id', 'band')
        _model_parts = list(modelD.keys())
        _model_parts.sort()
        coords_flux = {'ref_id': flux.ref_id, 'part': _model_parts, 'band': fit_bands}
        flux_model = np.zeros((len(modelD), len(flux), len(fit_bands)))
        flux_model = xr.DataArray(flux_model, dims=dims, coords=coords_flux)

        # Datastructure for storing the results...
        coords_chi2 = {'ref_id': flux.ref_id, 'part': _model_parts}
        chi2 = np.zeros((len(modelD), len(flux)))
        chi2 = xr.DataArray(chi2, dims=('part', 'ref_id'), coords=coords_chi2)

        zp_tot = xr.DataArray(np.ones(len(flux.band)), dims=('band'), \
                 coords={'band': flux.band})

        return flux, flux_err, chi2, zp_tot, flux_model

    def find_best_model(self, modelD, flux_model, flux, flux_err, chi2):
        """Find the best flux model."""

        # Just get a normal list of the models.
        model_parts = [str(x.values) for x in flux_model.part]

        fmin = self.minimize_free if self.config['free_ampl'] else self.minimize
        for j,key in enumerate(model_parts):
            print('Part', j)

            chi2_part, F = fmin(modelD[key], flux, flux_err)
            chi2[j,:] = chi2_part.sum(dim='band')
            flux_model[j,:] = F

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

        zp_details = {}
        for i in range(self.config['Nrounds']):
            best_flux = self.find_best_model(modelD, flux_model, flux, flux_err, chi2)

            zp = self.calc_zp(best_flux, flux, flux_err)
            zp = 1 + self.config['learn_rate']*(zp - 1.)

            flux = flux*zp
            flux_err = flux_err*zp

            zp_tot *= zp
            zp_details[i] = zp_tot

        zp_tot = zp_tot.to_series()
        zp_details = pd.DataFrame(zp_details, index=flux.band)

        return zp_tot, zp_details

    def get_model(self, zs):
        """Load and store the models as different xarrays."""

        t1 = time.time()
        f_modD = {}
        fit_bands = self.config['fit_bands']

        inds = ['band', 'sed', 'ext_law', 'EBV', 'z']
        for key, dep in self.input.depend.items():
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

    def sel_subset(self, galcat):
        """Select which subset to use for finding the zero-points."""

        # Note: I should add the cuts here after making sure that the
        # two pipelines give the same results.

        galcat = galcat[~np.isnan(galcat.zs)]

        # This cut was needed to avoid negative numbers in the logarithm.
        SN = galcat.flux / galcat.flux_err
        galcat = galcat.loc[self.config['SN_min'] < SN.min(axis=1)]

        return galcat

    def entry(self, galcat):
        # Loads model exactly at the spectroscopic redshift for each galaxy.
        galcat = self.sel_subset(galcat)
        modelD = self.get_model(galcat.zs)

        zp, zp_details = self.zero_points(modelD, galcat)

        return zp, zp_details

    def run(self):
        galcat = self.input.galcat.result

        t1 = time.time()
        zp, zp_details = self.entry(galcat)
        print('Time calibrating:', time.time() - t1)

        # Yes, this should not be needed.
        path = self.output.empty_file('default')
        store = pd.HDFStore(path, 'w')
        store['default'] = zp
        store['zp_details'] = zp_details
        store.close()
