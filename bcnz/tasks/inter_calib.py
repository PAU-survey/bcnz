#!/usr/bin/env python
# encoding: UTF8

import ipdb
import sys
import time
import numpy as np
import pandas as pd
import xarray as xr

from matplotlib import pyplot as plt

descr = {
  'bb_norm': 'Which broad band which is used for normalization',
  'fit_bands': 'Bands used in the fit',
  'Nrounds': 'How many rounds in the zero-point calibration',
  'Niter': 'Number of iterations in minimization',
  'SN_cap': 'Minimum signal-to-noise',
  'min_method': 'How to use the syntetic broad bands',
  'norm_readjust': 'If readjusting the normalizing of each band using the model',
  'zp_type': 'How to estimate the zero-points'
}

class inter_calib:
    """First step of the calibration.."""

    version = 1.14
    config = {'bb_norm': 'cfht_r',
              'fit_bands': [],
              'Nrounds': 5,
              'Niter': 1000,
              'SN_cap': 50,
              'min_method': 'simple',
              'norm_readjust': True,
              'zp_type': 'median'}

    def check_config(self):
        assert self.config['fit_bands'], 'Need to set: fit_bands'

    def find_trans(self, coeff):
        bb_norm = self.config['bb_norm']
        coeff = coeff[coeff.bb == bb_norm]

        trans = coeff.set_index(['bb','nb']).to_xarray().val
        trans = trans.rename({'nb': 'band'})

        return trans

    def find_synbb(self, coeff, galcat):

        trans = self.find_trans(coeff)
        flux = galcat['flux'][trans.band.values].stack().to_xarray()
        synbb = flux.dot(trans)

        return synbb

    def fix_to_synbb(self, coeff, galcat):
        bb_norm = self.config['bb_norm']

        # The synthetic broad band.
        trans = self.find_trans(coeff)
        _flux = galcat['flux'][trans.band.values].stack().to_xarray()
        syn_bb = _flux.dot(trans)
        syn_bb = self.find_synbb(coeff, galcat)

        data_bb = galcat.flux[bb_norm]

        ratio = syn_bb / data_bb.to_xarray()
        ratio = ratio.squeeze('bb')

        flux = galcat['flux'].stack().to_xarray()
        flux_err = galcat['flux_err'].stack().to_xarray()

        # To be absolutely sure the order is the same..
        ratio = ratio.sel(ref_id=flux.ref_id)
        ratio = ratio.fillna(1.)

        # Yeah, this is ugly... Here we need to only scale the broad
        # bands.
        isBB = list(filter(lambda x: not x.startswith('NB'), flux.band.values))
        for i,touse in enumerate(isBB):
            if not touse:
                continue

            flux.values[:,i] *= ratio
            flux_err.values[:,i] *= ratio

        return ratio, flux, flux_err


    def get_model(self, model, zs):
        """Get the model xarray."""

        f_mod = model.to_xarray().f_mod
        f_mod = f_mod.sel(z=zs.values, method='nearest')
        f_mod = f_mod.sel(EBV=0.0)
#        f_mod = f_mod.sel(band=self.config['fit_bands'])

        return f_mod

    def set_min_err(self, flux, flux_err):
        """Set the minimum error before running."""

        SN = flux / flux_err
        SN = np.clip(SN, -np.infty, self.config['SN_cap'])

        flux_err = flux / SN
        var_inv = 1./flux_err**2.
        var_inv = var_inv.fillna(0.)

        return var_inv

    def minimize(self, f_mod, flux, var_inv):

        A = np.einsum('gf,gfs,gft->gst', var_inv, f_mod, f_mod)
        b = np.einsum('gf,gf,gfs->gs', var_inv, flux, f_mod)

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

        return chi2, F #norm

    def calc_zp(self, flux, var_inv, best_flux):
        # This is not expected to be optimal... To be improved!!

        # Note: This should be written properly....
        if self.config['zp_type'] == 'ratio':
            R = best_flux.values / flux.values
            R[np.isinf(R)] = 0.
            zp = np.median(R, axis=0)
        elif self.config['zp_type'] == 'median':
            from scipy.optimize import minimize
            def cost(R, err_inv, flux, model):
                return np.abs(np.median(err_inv*(flux*R[0] - model)))

            t1 = time.time()
            # And another test for getting the median...
            zp = np.ones(len(flux.band))
            for i,band in enumerate(flux.band):
                args = (np.sqrt(var_inv.sel(band=band).values), flux.sel(band=band).values, \
                        best_flux.sel(band=band).values)

                X = minimize(cost, 1., args=args)
                zp[i] = X.x

            print(time.time() - t1)
           
        zp = xr.DataArray(zp, dims=('band'), coords={'band': flux.band})
 
        return zp

    def zero_points(self, modelD, flux, flux_err):
        """Estimate the zero-points."""

        fit_bands = self.config['fit_bands']
        flux = flux.sel(band=fit_bands)
        flux = flux.fillna(0.)
        flux_err = flux_err.sel(band=fit_bands)
        var_inv = self.set_min_err(flux, flux_err)

        model_parts = list(modelD.keys())
        model_parts.sort()

        coords_chi2 = {'ref_id': flux.ref_id, 'part': model_parts}
        chi2 = np.zeros((len(modelD), len(flux)))
        chi2 = xr.DataArray(chi2, dims=('part', 'ref_id'), coords=coords_chi2)

        coords_flux = {'ref_id': flux.ref_id, 'part': model_parts, 'band': fit_bands}
        flux_model = np.zeros((len(modelD), len(flux), len(fit_bands)))
        flux_model = xr.DataArray(flux_model, dims=('part', 'ref_id', 'band'), \
                     coords=coords_flux)

        zp_tot = xr.DataArray(np.ones(len(flux.band)), dims=('band'), \
                 coords={'band': flux.band})

        bb_norm = self.config['bb_norm']
        for i in range(self.config['Nrounds']):
            for j,key in enumerate(model_parts):
                print('Iteration', i, 'Part', j)

                f_mod = modelD[key]
                chi2_part, F = self.minimize(f_mod, flux, var_inv)
                chi2[j,:] = chi2_part.sum(dim='band')
                flux_model[j,:] = F
   
            best_part = chi2.argmin(dim='part')
            best_flux = flux_model.isel_points(ref_id=range(len(flux)), part=best_part)
            best_flux = xr.DataArray(best_flux, dims=('ref_id', 'band'), \
                        coords={'ref_id': flux.ref_id, 'band': flux.band})

            zp = self.calc_zp(flux, var_inv, best_flux)

            zp_tot *= zp

            print('zp round: {}'.format(i))
            print(zp.values)
            flux = flux*zp
            flux_err = flux_err*zp

            if self.config['norm_readjust']:
                # For some galaxies we don't have this band...
                ratio_adjust = best_flux.sel(band=bb_norm).values / flux.sel(band=bb_norm)
                ratio_adjust = ratio_adjust.drop('band')

                ratio_adjust[np.isinf(ratio_adjust)] = 1.
                flux *= ratio_adjust
                flux_err *= ratio_adjust

        return flux, flux_err, zp_tot

    def get_model(self, zs):
        """Load and store the models as different xarrays."""

        t1 = time.time()
        f_modD = {}
        fit_bands = self.config['fit_bands']
        for key, dep in self.input.depend.items():
            if not key.startswith('model'):
                continue

            f_mod = dep.result.to_xarray().f_mod
            f_mod = f_mod.sel(z=zs.values, method='nearest')

            f_mod = f_mod.sel(band=fit_bands)
            if 'EBV' in f_mod.dims:
                f_mod = f_mod.squeeze('EBV')

            f_modD[key] = f_mod

        print('Time loading model:', time.time() - t1)

        return f_modD

    def entry(self, coeff, galcat):
        zs = galcat.zs
        modelD = self.get_model(zs)

        ratio, flux, flux_err = self.fix_to_synbb(coeff, galcat)
        xflux, xflux_err, zp = self.zero_points(modelD, flux, flux_err)

        # Combines the two xarrays into a single sparse dataframe. *not* nice.
        xflux = xflux.to_dataframe('X').reset_index().pivot('ref_id', 'band', 'X')
        xflux_err = xflux_err.to_dataframe('X').reset_index().pivot('ref_id', 'band', 'X')
        cat_out = pd.concat({'flux': xflux, 'flux_err': xflux_err}, axis=1)

        return cat_out, zp

    def run(self):
        coeff = self.input.bbsyn_coeff.result
        galcat = self.input.galcat.result

        t1 = time.time()
        galcat_out, zp = self.entry(coeff, galcat)
        print('Time calibrating:', time.time() - t1)

        path = self.output.empty_file('default')
        store = pd.HDFStore(path, 'w')
        store['default'] = galcat_out
        store['zp'] = zp.to_dataframe('zp')
        store.close()
