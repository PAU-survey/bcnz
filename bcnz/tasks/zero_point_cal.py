#!/usr/bin/env python
# encoding: UTF8

import ipdb
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.neighbors import KDTree

descr = {
  'cut_frac': 'Fraction removed using a photo-z cut',
  'use_dist': 'If weighting the zero-points based on a distance',
  'Rmin': 'Cut in minimum value of R',
  'Rmax': 'Cut in maximum value of R',
  'median': 'If using the median instead of the mean',
  'weight': 'How to weight the zero-point from different galaxies',
  'norm_first': 'If first doing a relative calibration of NB and BB'
}

class zero_point_cal:
    """Calibrate the zero-points."""

    # One could divide this into two parts, one which is estimating the 
    # zero-points and another applying it.
    version = 1.08
    config = {'cut_frac': 0.211, 'Rmin': 0., 'Rmax': 10,
              'use_dist': False,  
              'weight': False, 'median': False,
              'Nneigh': 10,
              'bands': [],
              'use_col': True,
              'norm_dist': False,
              'norm_first': True}

    def check_config(self):
        assert self.config['bands'], 'Empty list of bands'

    def get_zp_dist(self, data, model, pzcat):
        """Get the zero-points weighting on a distance."""

        flux = data.flux.stack().to_xarray()
        flux_err = data.flux_err.stack().to_xarray()

        # The photo-z outliers might have worse colors.
        odds_cut = pzcat.odds.quantile(self.config['cut_frac'])
        pzcat = pzcat[odds_cut < pzcat.odds]


        flux = flux.sel(ref_id=pzcat.index)
        flux_err = flux_err.sel(ref_id=pzcat.index)

        # Removes the largest outliers..
        R = model / flux
        R.values = np.where((self.config['Rmin'] < R) & \
                            (R < self.config['Rmax']), R, np.nan)


        bands = self.config['bands']
        flux_train = flux
        train = flux_train.sel(band=bands).values
        pred = data.flux[bands].values

        if self.config['use_col']:
            train = np.log10(train[:,1:] / train[:,:-1])
            pred = np.log10(pred[:,1:] / pred[:,:-1])

        # So the different distances have more meaning..
        if self.config['norm_dist']:
            train = (train - train.mean(axis=0)) / np.sqrt(train.var(axis=0))
            pred = (pred - pred.mean(axis=0)) / np.sqrt(pred.var(axis=0))

        N = self.config['Nneigh']
        tree = KDTree(train)

        # Just skipping the first neighbor in case one is using the same
        # catalog for training.

        dist, ind = tree.query(pred, k=N+1)

        ind = flux_train.ref_id[ind].values

        tmp = np.zeros((N,)+data.flux.shape)
        for i in range(N):
            tmp[i] = R.sel(ref_id=ind[:,i+1]).values

        dims = ('i', 'ref_id', 'band')
        coords = {'ref_id': data.index, 'band': R.band, 'i': np.arange(N)}

        zp = xr.DataArray(tmp, dims=dims, coords=coords)
        zp = zp.median(dim='i') if self.config['median'] else \
             zp.mean(dim='i')

        return zp

    def get_zp_uniform(self, data, model, pzcat):
        flux = data['flux'].stack().to_xarray()
        ratio = flux / model.sel(band=data.flux.columns)

        odds_cut = pzcat.odds.quantile(self.config['cut_frac'])
        pzcat = pzcat[odds_cut < pzcat.odds]
        ratio = ratio.loc[pzcat.index]

        zp = ratio.median(dim='ref_id')

        return zp

    def get_zp(self, data, model, pzcat):
        f = self.get_zp_dist if self.config['use_dist'] else \
            self.get_zp_uniform


        return f(data, model, pzcat)

    def first_normalization(self, data, model):

        flux = data['flux'].stack().to_xarray()
        flux_err = data['flux_err'].stack().to_xarray()

        new_bands = ['cfht_u', 'cfht_g', 'cfht_r', 'cfht_i', 'cfht_z']
        new_bands = ['cfht_g', 'cfht_r', 'cfht_i']

        f_mod = model.sel(band=new_bands)
        flux = flux.sel(band=new_bands)
        flux_err = flux_err.sel(band=new_bands)
        var_inv = 1. / flux_err**2.

        S = (var_inv*flux**2).sum(dim='band')
        X = (var_inv*flux*f_mod).sum(dim='band')
        M = (var_inv*f_mod**2).sum(dim='band')

        norm = (X / M).to_series()
        NB = list(map('NB{}'.format, 455+10*np.arange(40)))

        xflux = data['flux'].copy()
        xflux_err = data['flux_err'].copy()

        xflux[NB] = xflux[NB].mul(norm, axis='rows')
        xflux_err[NB] = xflux_err[NB].mul(norm, axis='rows')

        new_data = pd.concat({'flux': xflux, 'flux_err': xflux_err}, axis=1)


        return new_data

    def calibrate_flux(self, data, model, pzcat):
        if self.config['norm_first']:
            data = self.first_normalization(data, model)
  
 
        zp = self.get_zp(data, model, pzcat)
        flux = data.flux.stack().to_xarray()

        flux = flux*zp
        data['flux'] = flux.to_dataframe('flux').unstack()['flux']

        return data

    def run(self):
        with self.input.model.get_store() as store:
            model = store['best_model'].to_xarray().best_model


        pzcat = self.input.model.result
        data = self.input.flux.result

        self.output.result = self.calibrate_flux(data, model, pzcat)
