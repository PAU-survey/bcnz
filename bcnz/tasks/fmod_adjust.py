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

    version = 1.08
    config = {'norm_band': '', 'funky_limit': True}

    def check_config(self):
        assert self.config['norm_band'], \
               'Need to specify the normalization band'

    def funky_hack(self, syn2real, sed):
        """Exactly mimic the cuts Alex was making."""

        from matplotlib import pyplot as plt

        ratio = syn2real.sel(sed=sed).copy()
        if sed == 'OIII':
#            ratio[ratio.z < 0.1] = 1.
            ratio[(ratio.z < 0.1) | (0.45 < ratio.z)] = 1.

            plt.plot(ratio.z, ratio[:,0,0])
            plt.show()
        elif sed == 'lines': 
            ipdb.set

            plt.plot(ratio.z, ratio[:,0,0])
            plt.show()

        ipdb.set_trace()

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
        syn2real = model_norm / synbb

        for j,xsed in enumerate(model.sed):
            sed = str(xsed.values) 
            syn2real_mod = self.funky_hack(syn2real, sed)

            for i,xband in enumerate(model.band):
                band = str(xband.values)
                
                if str(band.values).startswith('NB'):
                    model[i] *= syn2real

        # Since we can not directly store xarray.
        model = model.to_dataframe()

        ipdb.set_trace()

        return model

    def run(self):
        coeff = self.input.bbsyn_coeff.result
        model = self.input.model.result

        self.output.result = self.entry(coeff, model)
