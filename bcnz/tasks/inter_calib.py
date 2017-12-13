#!/usr/bin/env python
# encoding: UTF8

import ipdb
import sys
import time
import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import minimize

from matplotlib import pyplot as plt

descr = {
  'fix_to_synbb': 'If fixing to the synthetic band',
  'bb_norm': 'Which broad band which is used for normalization',
  'fit_bands': 'Bands used in the fit',
  'Nrounds': 'How many rounds in the zero-point calibration',
  'Niter': 'Number of iterations in minimization',
  'zp_min': 'How to estimate the zero-points'
}

class inter_calib:
    """Calibration between the broad and the narrow bands."""

    version = 1.21
    config = {'fix_to_synbb': True,
              'free_ampl': False,
              'bb_norm': 'cfht_r',
              'fit_bands': [],
              'Nrounds': 5,
              'Niter': 1000,
              'zp_min': 'flux2'}

    def check_config(self):
        assert self.config['fit_bands'], 'Need to set: fit_bands'

    def find_trans(self, coeff):
        bb_norm = self.config['bb_norm']
        coeff = coeff[coeff.bb == bb_norm]

        trans = coeff.set_index(['bb','nb']).to_xarray().val
        trans = trans.rename({'nb': 'band'})

        return trans

    def find_synbb(self, coeff, galcat):
        """Syntetic flux in one of the bands."""

        # This handles missing flux values.
        # Note: There might be more precise ways of doing this..
        trans = self.find_trans(coeff)
        flux = galcat['flux'][trans.band.values].stack().to_xarray()
        missing_band = np.isnan(flux)
        flux = flux.fillna(0.)

        synbb = flux.dot(trans)

        missing_flux = 1.*missing_band.dot(trans)
        synbb = synbb/(1.-missing_flux)

        return synbb

    def fix_to_synbb(self, coeff, galcat):
        """Fix the data to a synthetic band."""

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

        # TODO: Consider adding the error on the synthetic broad bands.
        if self.config['fix_to_synbb']:
            # Yeah, this is ugly... Here we need to only scale the broad bands.
            isBB = list(filter(lambda x: not x.startswith('NB'), flux.band.values))
            for i,touse in enumerate(isBB):
                if not touse:
                    continue

                flux.values[:,i] *= ratio
                flux_err.values[:,i] *= ratio

        return ratio, flux, flux_err

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

# Note: The commented version is adding the free factor in the fluxes. I experimented
# with this when including k in the model seemingly gave wrong result. One should remove
# t_tot, since this makes little sense. k is the *total* amplitude after multiple
# iterations.
#
#    def minimize_free(self, f_mod, flux, flux_err):
#        """Minimize the model difference at a specific redshift."""
#
#        # Otherwise we will get nans in the result.
#        var_inv = 1./flux_err**2
#        var_inv = var_inv.fillna(0.)
#        xflux = flux.fillna(0.)
#
#        NB = list(filter(lambda x: x.startswith('NB'), flux.band.values))
#        BB = list(filter(lambda x: not x.startswith('BB'), flux.band.values))
#        A = np.einsum('gf,gfs,gft->gst', var_inv, f_mod, f_mod)
#        b_NB = np.einsum('gf,gf,gfs->gs', var_inv.sel(band=NB), xflux.sel(band=NB), f_mod.sel(band=NB))
#        b_BB = np.einsum('gf,gf,gfs->gs', var_inv.sel(band=BB), xflux.sel(band=BB), f_mod.sel(band=BB))
#        C = np.einsum('gf,gf->g', var_inv.sel(band=BB), \
#                      xflux.sel(band=BB)**2)
#
#        k = np.ones((len(xflux)))
#        k_tot = k.copy()
#        b = b_NB + b_BB # np.einsum('g,gf->gf', k, b_BB)
#
#        v = np.ones_like(b)
#        t1 = time.time()
#        for i in range(self.config['Niter']):
#            a = np.einsum('gst,gt->gs', A, v)
#            m0 = b / a
#            vn = m0*v
#
#            # Minimize the chi2 (see notes).
#
#
#            k = np.einsum('g,gs,gs->g',1./C, b_BB, vn)
#            b_BB = np.einsum('g,gf->gf', k, b_BB)
#            b = b_NB + b_BB
#            C = k**2*C
#
##            print('k', k[5], 'ktot', k_tot[5])
##            ipdb.set_trace()
#
#
##            print(k_tot.median())
##            print(np.median(k_tot))
##            ipdb.set_trace()
#            if np.isnan(k_tot).any():
#                ipdb.set_trace()
#
#            k_tot *= k
#
#            print(np.median(k_tot))
#            v = vn
#
##        ipdb.set_trace()
#
#        gal_id = np.array(flux.ref_id)
#        coords = {'ref_id': gal_id, 'band': f_mod.band}
#        coords_norm = {'ref_id': gal_id, 'model': np.array(f_mod.sed)}
#        F = np.einsum('gfs,gs->gf', f_mod, v)
#        flux_model = xr.DataArray(F, coords=coords, dims=('ref_id', 'band'))
#
#        isbb = np.array(list(map(lambda x: not x.startswith('NB'), flux.band.values)))
#
#        flux[:,isbb] *= k_tot[:,np.newaxis]
#
#        chi2 = var_inv*(flux - F)**2
#        chi2.values = np.where(chi2==0., np.nan, chi2)
#
#
#        ipdb.set_trace()
#
#
#        return chi2, F
#

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

    def _zp_flux_chi2(self, best_flux, flux, err_inv):
        """Find the zero-points by minimizing a chi2 expression."""

        # This could be done recursively, but the first round had
        # very few outliers..
        X = (best_flux / flux).median(dim='ref_id')
        F = err_inv*((best_flux / flux) - X)

        flux.values[0.2 < np.abs(F)] = np.nan
        P1 = (flux*best_flux*err_inv**2).sum(dim='ref_id')
        P2 = ((flux*err_inv)**2).sum(dim='ref_id')

        zp = P1 / P2

        return zp

    def calc_zp(self, best_flux, flux, flux_err):
        """Estimate the zero-point."""

        err_inv = 1. / flux_err

        X = (best_flux, flux, err_inv)
        zp_min = self.config['zp_min'] 

#        ipdb.set_trace()

        if zp_min == 'mag':
            # Code to follow quite closely what Alex did.
            def cost_mag(R, bestmodel, sig, err_inv):
                shift = 10**(-0.4*R[0])
                err = 1./err_inv

                sig = sig * shift
                err = err * shift

                S = -2.5*np.log10(sig)-48.6
                E = 2.5*np.log10(1+err/sig)
                M = -2.5*np.log10(bestmodel)-48.6

                median = np.nanmedian((S-M)/E)
                val = np.abs(median)

                return val

            print('starting minimize')
            zp = self._zp_min_cost(cost_mag, *X)
            zp = 10**(-0.4*zp)
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

    def zero_points(self, modelD, flux, flux_err):
        """Estimate the zero-points."""

        fit_bands = self.config['fit_bands']
        flux = flux.sel(band=fit_bands)
#        flux = flux.fillna(0.)
        flux_err = flux_err.sel(band=fit_bands)

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

        fmin = self.minimize_free if self.config['free_ampl'] else self.minimize
        for i in range(self.config['Nrounds']):
            for j,key in enumerate(model_parts):
                print('Iteration', i, 'Part', j)

                f_mod = modelD[key]
                chi2_part, F = fmin(f_mod, flux, flux_err)
                chi2[j,:] = chi2_part.sum(dim='band')
                flux_model[j,:] = F
   
            best_part = chi2.argmin(dim='part')
            best_flux = flux_model.isel_points(ref_id=range(len(flux)), part=best_part)
            best_flux = xr.DataArray(best_flux, dims=('ref_id', 'band'), \
                        coords={'ref_id': flux.ref_id, 'band': flux.band})

            zp = self.calc_zp(best_flux, flux, flux_err)
            zp_tot *= zp

            print('zp round: {}'.format(i))
            print(zp.values)
            flux = flux*zp
            flux_err = flux_err*zp

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
            if 1 < len(f_mod.EBV):
                ipdb.set_trace()

            if 'EBV' in f_mod.dims:
                f_mod = f_mod.squeeze('EBV')

            # Later the code depends on this order.
            f_mod = f_mod.transpose('z', 'band', 'sed')

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

        return cat_out, zp, ratio

    def run(self):
        coeff = self.input.bbsyn_coeff.result
        galcat = self.input.galcat.result

        t1 = time.time()
        galcat_out, zp, ratio = self.entry(coeff, galcat)
        print('Time calibrating:', time.time() - t1)

        path = self.output.empty_file('default')
        store = pd.HDFStore(path, 'w')
        store['default'] = galcat_out
        store['zp'] = zp.to_dataframe('zp')
        store['ratio'] = ratio.to_dataframe('ratio')
        store.close()
