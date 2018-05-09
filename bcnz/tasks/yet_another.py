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

descr = {'fit_bands': 'Which bands to fit',
         'Niter': 'Number of iterations in the fit',
         'cosmos_scale': 'If scaling the input'
}

class yet_another:
    """Yet another attempt on calibration the zero-points."""

    version = 1.01
    config = {'fit_bands': [],
              'Niter': 1000,
              'scale_input': True}

    def check_config(self):
        assert self.config['fit_bands'], 'Need to set: fit_bands'


    def get_arrays(self, data_df):
        """Read in the arrays and present them as xarrays."""

        # Seperating this book keeping also makes it simpler to write
        # up different algorithms.

        filters = self.config['fit_bands']
        dims = ('gal', 'band')
        flux = xr.DataArray(data_df['flux'][filters], dims=dims)
        flux_err = xr.DataArray(data_df['flux_err'][filters], dims=dims)

        # Previously I found that using on flux system or another made a
        # difference.
        if self.config['scale_input']:
            print('Convert away from PAU fluxes...')
            ab_factor = 10**(0.4*26)
            cosmos_scale = ab_factor * 10**(0.4*48.6)

            flux /= cosmos_scale
            flux_err /= cosmos_scale

        # Not exacly the best, but Numpy had problems with the fancy
        # indexing.
        to_use = ~np.isnan(flux_err)

        # This gave problems in cases where all measurements was removed..
        flux.values = np.where(to_use, flux.values, 1e-100) #0.) 

        # TODO: I should check if this is actually needed...
        var_inv = 1./(flux_err + 1e-100)**2
        var_inv.values = np.where(to_use, var_inv, 1e-100)
        flux_err.values = np.where(to_use, flux_err, 1e-100)

        return flux, flux_err, var_inv

    def run_photoz(self, models, galcat):
        """Run the photo-z over all the different chunks."""

        flux, flux_err, var_inv = self.get_arrays(galcat)

        #ipdb.set_trace()

        # Setting up the return datastructure.. Yes, this takes too much
        # code.
        chunk = range(len(models))
        gal = galcat.index.values
        z = models[0].z 
        band = flux.band

        dims_chi2 = ('gal',)
        coords_chi2 = {'gal': gal}
        coords_ratio = {'gal': gal, 'band': band}

        chi2D = {}
        ratioD = {}

        t1 = time.time()
        for j, f_mod in enumerate(models):
            # TODO: Here I should consider doing a linear interpolation. It might
            # reduce my karma.

            f_mod = f_mod.sel(z=galcat.zs.values, method='nearest')

            print('Model', j)
            t2 = time.time()
            A = np.einsum('gf,gfs,gft->gst', var_inv, f_mod, f_mod)
            b = np.einsum('gf,gf,gfs->gs', var_inv, flux, f_mod)

            v = 100*np.ones_like(b)

            # And then all the itertions...
            for i in range(self.config['Niter']):
                a = np.einsum('gst,gt->gs', A, v)

                m0 = b / a
                vn = m0*v
                v = vn

            # Estimates the best-fit model... 
            F = np.einsum('gfs,gs->gf', f_mod, v)
            F = xr.DataArray(F, dims=('gal', 'band'), coords=coords_ratio)
            chi2 = (var_inv*(flux - F)**2.).sum(dim='band')

            chi2D[j] = chi2
            ratioD[j] = F / flux


        chi2 = xr.concat([chi2D[x] for x in chunk], dim='chunk')
        chi2.coords['chunk'] = chunk

        # Here we don't need to scale back since we are using the ratio.
        ratio = xr.concat([ratioD[x] for x in chunk], dim='chunk')
        ratio.coords['chunk'] = chunk

        print('time', time.time() - t1)

        return chi2, ratio

    def fix_fmod_format(self, fmod_in):
        """Converts the dataframe to an xarray."""

        inds = ['z', 'band', 'sed', 'ext_law', 'EBV']
        f_mod = fmod_in.reset_index().set_index(inds).to_xarray().flux

        model = ['sed', 'EBV', 'ext_law']
        f_mod = f_mod.stack(model=model)

        f_mod_full = f_mod
        f_mod = f_mod_full.sel(band=self.config['fit_bands'])

        assert not np.isnan(f_mod).any(), 'Missing entries'

        return f_mod

    def load_models(self):
        # To iterate over these in a specific order...
        nparts = len(list(filter(lambda x: x.startswith('model_'), dir(self.input))))

        # One could consider storing these together, but I have not found the use
        # case...
        models = []
        for nr in range(nparts):
            job = self.input.depend['model_{}'.format(nr)]

            print('nr', nr, 'taskid', job.file_path)

            model_in = job.result
            models.append(self.fix_fmod_format(model_in))

        return models

    def entry(self, models, galcat):
        chi2, ratio = self.run_photoz(models, galcat)

        chi2 = chi2.to_dataframe('chi2')
        ratio = ratio.to_dataframe('ratio')

        return chi2, ratio 

    def run(self):
        models = self.load_models()
        galcat = self.input.galcat.result

        chi2, ratio = self.entry(models, galcat)

        # Yes, this should not be needed.
        path = self.output.empty_file('default')
        store = pd.HDFStore(path, 'w')
        store.append('chi2', chi2)
        store.append('ratio', ratio)
        store.close()
