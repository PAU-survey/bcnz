#!/usr/bin/env python
# encoding: UTF8

import ipdb
import numpy as np

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
    version = 1.0
    config = {'cut_frac': 0., 'Rmin': 0., 'Rmax': 10,
              'weight': False, 'median': False}

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


        if self.config['weight']:
            weights = 1. / flux_err**2.
            weights /= weights.sum(dim='ref_id')

            zp = (R*weights).sum(dim='ref_id')
        else:
            zp = R.median(dim='ref_id') if self.config['median'] else \
                 R.mean(dim='ref_id')

        return zp

    def calibrate_flux(self, data, model, pzcat):
        """Calibrate the input fluxes."""

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
