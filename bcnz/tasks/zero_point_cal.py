#!/usr/bin/env python
# encoding: UTF8

import ipdb
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.neighbors import KDTree

descr = {
  'cut_frac': 'Fraction removed using a photo-z cut',
  'Rmin': 'Cut in minimum value of R',
  'Rmax': 'Cut in maximum value of R',
  'median': 'If using the median instead of the mean',
  'weight': 'How to weight the zero-point from different galaxies',
}

class zero_point_cal:
    """Calibrate the zero-points."""

    # One could divide this into two parts, one which is estimating the 
    # zero-points and another applying it.
    version = 1.04
    config = {'cut_frac': 0., 'Rmin': 0., 'Rmax': 10,
              'weight': False, 'median': False,
              'Nneigh': 10}

    def get_zp(self, data, model, pzcat):
        """Get the zero-points."""

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


        col_train = -2.5*np.log(flux.sel(band='cfht_g') / flux.sel(band='cfht_r'))
        Atrain = np.array([col_train.values]).T

        N = self.config['Nneigh']
        tree = KDTree(Atrain)

        col_fit = -2.5*np.log10(data.flux.cfht_g / data.flux.cfht_r)
        Afit = np.array([col_fit.values]).T

        # Just skipping the first neighbor in case one is using the same
        # catalog for training.
        dist, ind = tree.query(Afit, k=N+1)
        ind = col_train.ref_id[ind].values

        tmp = np.zeros((N,)+data.flux.shape)
        for i in range(N):
            tmp[i] = R.sel(ref_id=ind[:,i+1]).values

        dims = ('i', 'ref_id', 'band')
        coords = {'ref_id': data.index, 'band': R.band, 'i': np.arange(N)}

        zp = xr.DataArray(tmp, dims=dims, coords=coords)
        zp = zp.median(dim='i') if self.config['median'] else \
             zp.mean(dim='i')

        return zp


    def calibrate_flux(self, data, model, pzcat):
        zp = self.get_zp(data, model, pzcat)
        flux = data.flux.stack().to_xarray()

        flux = flux*zp
        data['flux'] = flux.to_dataframe('flux').unstack()['flux']

        return data

    def run(self):
        with self.job.model.get_store() as store:
            model = store['best_model'].to_xarray().best_model
            model = model.rename({'gal': 'ref_id'})

        pzcat = self.job.model.result.unstack() # This is the photo-z job..
        data = self.job.flux.result
        self.job.result = self.calibrate_flux(data, model, pzcat)
