#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger
import time
import numpy as np
import pandas as pd
import xarray as xr

import libpzqual

descr = {'Niter': 'Number of iterations'}

class bcnz_zs_fixed:
    """Find the best amplitude for a fixed redshift."""

    version = 1.02
    config = {'Niter': 500,
              'filters': []}

    def check_config(self):
        assert self.config['filters'], 'Need to specify filters'

    def get_input(self, f_mod, zs):
        """Get the input arrays."""

        f_mod = f_mod.sel(z=zs.values, method='nearest')
        f_mod = f_mod.sel(EBV=0.0)
        f_mod = f_mod.sel(band=self.config['filters'])
        
        return f_mod

    def minimize(self, galcat, f_mod, zspec):
        """Minimize the chi2 expresssion only at the spectroscopic
           redshift.
        """

        # Using the order of the spec catalogue.
        zs = zspec.zs
        galcat = galcat.loc[zs.index]
        f_mod = self.get_input(f_mod, zs)
        flux, flux_err, var_inv = libpzqual.get_arrays(galcat, \
                                  self.config['filters'])

        A = np.einsum('gf,gfs,gft->gst', var_inv, f_mod, f_mod)
        b = np.einsum('gf,gf,gfs->gs', var_inv, flux, f_mod)

        v = np.ones_like(b)

        t1 = time.time()
        for i in range(self.config['Niter']):
            print('i', i)
            a = np.einsum('gst,gt->gs', A, v)
            m0 = b / a
            vn = m0*v

            v = vn

        print('time minimize',  time.time() - t1)

        gal_id = np.array(galcat.index)
        coords = {'gal': gal_id, 'band': f_mod.band}
        coords_norm = {'gal': gal_id, 'model': np.array(f_mod.sed)}
        F = np.einsum('gfs,gs->gf', f_mod, v)

        F = xr.DataArray(F, coords=coords, dims=('gal', 'band'))

        chi2 = var_inv*(flux - F)**2
        norm = xr.DataArray(v, coords=coords_norm, dims=('gal', 'model'))

        return chi2, norm, F

    def entry(self, galcat, model, zspec):
        f_mod = model.to_xarray().f_mod
        chi2, norm, best_model = self.minimize(galcat, f_mod, zspec)

        chi2 = chi2.to_dataframe('chi2')
        norm = norm.to_dataframe('norm')
        best_model = best_model.to_dataframe('best_model')

        return chi2, norm, best_model

    def run(self):
        galcat = self.job.galcat.result
        model = self.job.model.result
        zspec = self.job.zspec.result

        chi2, norm, best_model = self.entry(galcat, model, zspec)

        empty_file = self.job.empty_file('default')
        store = pd.HDFStore(empty_file, 'w')
        store.append('chi2', chi2)
        store.append('norm', norm)
        store.append('best_model', best_model)
        store.close()
