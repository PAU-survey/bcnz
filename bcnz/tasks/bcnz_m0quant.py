#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import time
import numpy as np
import pandas as pd
import xarray as xr

# Here we inherit to avoid copying the code..
import bcnz_run_all

class bcnz_m0quant(bcnz_run_all.bcnz_run_all):
    """Version testing the convergence of the algorithm."""

    # The usage of this version of the code is to output some metrics
    # for how fast the algorithm is converging. This would clutter 
    # the main algorithm and (hopefully) not be very useful output
    # beyond the paper.


    version = 1.0

    def run_photoz(self, models, galcat):
        """Run the photo-z over all the different chunks."""

        flux, flux_err, var_inv = self.get_arrays(galcat)

        m0_quant = np.zeros((len(models), self.config['Niter']))

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

                m0_quant[j,i] = pd.Series(np.abs((m0-1).flatten())).quantile(0.5)


        dims = ('model', 'iter')
        coords = {'model': range(len(models)), 'iter': range(self.config['Niter'])}

        m0_quant = xr.DataArray(m0_quant, dims=dims, coords=coords)

        return m0_quant

    def run(self):
        # The run method was actually not finished in bcnz_run_all
        # when starting this work.
        models = self.load_models()
        galcat_store = self.input.galcat.get_store()
        Rin = galcat_store.select('default', iterator=True, chunksize=self.chunksize)

        path = self.output.empty_file('default')
        store_out = pd.HDFStore(path, 'w')

        for i,galcat in enumerate(Rin):
            print('batch', i, 'tot', i*self.chunksize)
            m0_quant = self.run_photoz(models, galcat)

            m0_quant = m0_quant.to_dataframe('quant')
            m0_quant = m0_quant.reset_index()
            m0_quant['batch'] = i

            m0_quant = m0_quant.set_index(['batch', 'model', 'iter'])
            store_out.append('m0_quant', m0_quant)

        galcat_store.close()
        store_out.close()
