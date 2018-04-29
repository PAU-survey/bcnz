#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import pandas as pd
import xarray as xr

descr = {'norm_band': 'Band used for the normalization',
         'funky_limit': 'If adding limits to match Alex code.'}

class fmod_adjust:
    """Adjust the model to reflect that the syntetic narrow bands is not
       entirely accurate.
    """

    version = 1.09
    config = {'norm_band': '', 'funky_limit': True,
              'lines_upper': 0.1226}

    def check_config(self):
        assert self.config['norm_band'], \
               'Need to specify the normalization band'

    def funky_hack(self, syn2real, sed, model_norm):
        """Exactly mimic the cuts Alex was making."""

        from matplotlib import pyplot as plt

        print('sed', sed)
        ratio = syn2real.sel(sed=sed).copy()
        if sed == 'OIII':
            ratio[(ratio.z < 0.1) | (0.45 < ratio.z)] = 1.
        elif sed == 'lines': 
            flux = model_norm.sel(sed=sed)
            ratio.values[flux < 0.001*flux.max()] = 1.

            # Yes, this actually happens in an interval.
            upper = self.config['lines_upper']
            ratio[(ratio.z>0.1025) & (ratio.z<upper)] = 1.
        else:
            # The continuum ratios are better behaved.
            pass

        return ratio

    def entry(self, coeff, model):
        """Directly scale the model as with the data."""

        # Transform the coefficients.
        norm_band = self.config['norm_band']
        coeff = coeff[coeff.bb == norm_band][['nb', 'val']]
        coeff = coeff.set_index(['nb']).to_xarray().val

        inds = ['band', 'z', 'sed', 'ext_law', 'EBV']
        model = model.set_index(inds)
        model = model.to_xarray().flux
        model_norm = model.sel(band=norm_band).copy() # A copy is needed for funky_hack..
        model_NB = model.sel(band=coeff.nb.values)

        # Scaling the fluxes..
        synbb = (model_NB.rename({'band': 'nb'})*coeff).sum(dim='nb')
        syn2real = model_norm / synbb

        for j,xsed in enumerate(model.sed):
            sed = str(xsed.values) 
            syn2real_mod = self.funky_hack(syn2real, sed, model_norm)

            for i,xband in enumerate(model.band):
                # Here we scale the narrow bands into the broad band system.
                if str(xband.values).startswith('NB'):
                    model[i,:,j,:,:] *= syn2real_mod

        # Since we can not directly store xarray.
        model = model.to_dataframe()
        model = model.reset_index()

        return model

    def run(self):
        coeff = self.input.bbsyn_coeff.result
        model = self.input.model.result

        self.output.result = self.entry(coeff, model)
