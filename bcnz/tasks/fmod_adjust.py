#!/usr/bin/env python
# encoding: UTF8

import ipdb
import pandas as pd
import xarray as xr

class fmod_adjust:
    """Adjust the model to reflect that the syntetic narrow bands is not
       entirely accurate.
    """

    version = 1.0
    config = {'norm_band': ''}

    def check_config(self):
        assert self.config['norm_band'], \
               'Need to specify the normalization band'

    def ratio(self, coeff, model):
        """Estimate the ratio between the model flux and the syntetic
           broad band.
        """

        norm_band = self.config['norm_band']
        coeff = coeff[coeff.bb == norm_band]
        coeff = coeff.set_index(['bb', 'nb']).to_xarray().val
        coeff = coeff.rename({'bb': 'band'})

        A = model.to_xarray().f_mod
        B = A.sel(band=norm_band)
        A = A.stack(model=['z', 'sed', 'EBV'])
        A = A.sel(band=coeff.nb.values).rename({'band': 'nb'})

        # Estimate the ratio between the t
        synbb = coeff.dot(A).unstack(dim='model')
        F = xr.Dataset({'flux': B, 'flux_syn': synbb})
        F = F.to_dataframe()
        F['ratio'] = F.flux / F.flux_syn

        # To avoid extremely large ratios.
        flux_median = F.groupby('sed').flux.median().rename('flux_median')
        F =  F.reset_index()
        F = F.merge(pd.DataFrame(flux_median), left_on='sed', right_index=True)
        F['fix'] = F.flux < 0.05*F.flux_median
        F.loc[F.fix, 'ratio'] = 1.0

        ratio = F[['band', 'sed', 'z', 'ratio']]

        return ratio

    def apply_ratio(self, ratio, model):
        """Apply the previously estimated ratio to the model."""

        model.index = model.index.droplevel('EBV')
        model = model.to_xarray()
        ratio = ratio.set_index(['z', 'band', 'sed']).to_xarray().ratio

        model = ratio*model

        return model

    def entry(self, coeff, model):
        ratio = self.ratio(coeff, model)
        model = self.apply_ratio(ratio, model)

        model = model.f_mod.to_dataframe()

        return model

    def run(self):
        print('Running fmod_adjust')
        coeff = self.input.bbsyn_coeff.result
        model = self.input.model.result

        self.output.result = self.entry(coeff, model)
