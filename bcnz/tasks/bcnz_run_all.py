#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import time
import numpy as np
import xarray as xr

descr = {'filters': 'Filters used when making the fit',
         'scale_input': 'If scaling the input',
         'Niter': 'Number of iterations in minimization method'}

class bcnz_run_all:
    """Version running the photo-z for all the different chunks at 
       once.
    """

    version = 1.0
    config = {'filters': [],
              'scale_input': True,
              'Niter': 1000}

    chunksize = 10

    def check_config(self):
        assert len(self.config['filters'])

    def get_arrays(self, data_df):
        """Read in the arrays and present them as xarrays."""

        # Seperating this book keeping also makes it simpler to write
        # up different algorithms.

        filters = self.config['filters']
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

        # Setting up the return datastructure.. Yes, this takes too much
        # code.
        dims = ('chunk', 'gal', 'z')
        coords = {'chunk': range(len(models)), 'gal': galcat.index.values, \
                  'z': models[0].z}
        chi2 = np.zeros((len(coords['chunk']), len(coords['gal']), \
                         len(coords['z'])))
        chi2 = xr.DataArray(chi2, dims=dims, coords=coords)

        coords_model = {'gal': galcat.index.values, 'z': models[0].z, 'band': flux.band}

        t1 = time.time()
        for j, f_mod in enumerate(models):
            print('Model', j)
            t2 = time.time()
            A = np.einsum('gf,zfs,zft->gzst', var_inv, f_mod, f_mod)
            b = np.einsum('gf,gf,zfs->gzs', var_inv, flux, f_mod)

            v = 100*np.ones_like(b)

            # And then all the itertions...
            for i in range(self.config['Niter']):
                a = np.einsum('gzst,gzt->gzs', A, v)

                m0 = b / a
                vn = m0*v
                v = vn
           
            # Estimates the best-fit model... 
            F = np.einsum('zfs,gzs->gzf', f_mod, v)
            F = xr.DataArray(F, dims=('gal', 'z', 'band'), coords=coords_model)

            chi2[j] = (var_inv*(flux - F)**2.).sum(dim='band')

        print('time', time.time() - t1)

        return chi2

    def fix_fmod_format(self, fmod_in):
        """Converts the dataframe to an xarray."""

        inds = ['z', 'band', 'sed', 'ext_law', 'EBV']
        f_mod = fmod_in.reset_index().set_index(inds).to_xarray().flux

        model = ['sed', 'EBV', 'ext_law']
        f_mod = f_mod.stack(model=model)

        f_mod_full = f_mod
        f_mod = f_mod_full.sel(band=self.config['filters'])

#        ipdb.set_trace()
#        assert not np.isnan(f_mod_full).any(), 'Missing entries'

        return f_mod #, f_mod_full

    def load_models(self):
        # To iterate over these in a specific order...
        nparts = len(list(filter(lambda x: x.startswith('model_'), dir(self.input))))

        # One could consider storing these together, but I have not found the use
        # case...
        models = []
        for nr in range(nparts):
            model_in = self.input.depend['model_{}'.format(nr)].result
            models.append(self.fix_fmod_format(model_in))

        return models 

    def entry(self):
        models = self.load_models()

        ipdb.set_trace()

    def run(self):
#        self.entry()
        models = self.load_models()

        galcat_store = self.input.galcat.get_store()
        Rin = galcat_store.select('default', iterator=True, chunksize=self.chunksize)

        for i,galcat in enumerate(Rin):
            print('batch', i, 'tot', i*self.chunksize)
            chi2 = self.run_photoz(models, galcat)

            ipdb.set_trace()
