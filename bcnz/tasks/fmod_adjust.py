#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import pandas as pd
import xarray as xr

class fmod_adjust:
    """Adjust the model to reflect that the syntetic narrow bands is not
       entirely accurate.
    """

    version = 1.05
    config = {'norm_band': ''}

    def check_config(self):
        assert self.config['norm_band'], \
               'Need to specify the normalization band'

    def entry(self, coeff, model):
        """Directly scale the model as with the data."""

        # Transform the coefficients.
        norm_band = self.config['norm_band']
        coeff = coeff[coeff.bb == norm_band][['nb', 'val']]
        coeff = coeff.set_index(['nb']).to_xarray().val

        inds = ['band', 'z', 'sed', 'ext_law', 'EBV']
        model = model.set_index(inds)
        model = model.to_xarray().flux
        model_norm = model.sel(band=norm_band)
        model_NB = model.sel(band=coeff.nb.values)

        # Scaling the fluxes..
        synbb = (model_NB.rename({'band': 'nb'})*coeff).sum(dim='nb')
        syn2real = synbb / model_norm

        for i,band in enumerate(model.band):
            if str(band.values).startswith('NB'):
                model[i] *= syn2real

        # Since we can not directly store xarray.
        model = model.to_dataframe()

        return model

    def run(self):
        coeff = self.input.bbsyn_coeff.result
        model = self.input.model.result

        self.output.result = self.entry(coeff, model)
