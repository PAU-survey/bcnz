#!/usr/bin/env python

from __future__ import print_function

from IPython.core import debugger as ipdb
import os
import sys
import time
import numpy as np
import pandas as pd
import xarray as xr
import itertools as it

from scipy.interpolate import splrep, splev, splint
from scipy.integrate import trapz, simps

# Yes, this is ugly.
sys.path.append('/home/eriksen/code/bcnz/bcnz/tasks')
sys.path.append('/nfs/pic.es/user/e/eriksen/code/bcnz/bcnz/tasks')

import libpzqual
import libpzcore

descr = {
  'filters': 'Filters to use',
  'seds': 'SEDs to use',
#  'zmin': 'Minimum redshift',
#  'zmax': 'Maximum redshift',
#  'dz': 'Grid width in redshift',
  'chi2_algo': 'The chi2 algorithm',
#  'use_lines': 'If including emission lines',
#  'use_ext': 'If including extinction'
  'width_frac': 'Percentage on each side',
  'towrite': 'The fields to write',
  'scale_input': 'If scaling the input for numerical reasons.',
  'Nskip': 'Number of entries to skip when scaling the broad bands',
#  'scale_to': 'Which bands to scale it to'
}

class bcnz_photoz_simple:
    """Fitting a linear combination to the observations."""

    # This code is an experiment with adding the free amplitude between
    # the two systems. Previously this was kind of working!

    version = 1.16
    config = {
      'filters': [],
      'seds': [],
      'zmin': 0.01,
      'zmax': 2.0,
      'dz': 0.01,
      'odds_lim': 0.01,
      'Niter': 300,
      'line_weight': 2.,
      'chi2_algo': 'min',
      'use_ext': False,
      'width_frac': 0.01,
      'towrite': ['best_model'],
      'scale_input': True,
      'Nskip': 10,
#      'scale_to': []
    }

    def check_conf(self):
        assert self.config['filters'], 'Need to set filters'
        assert not self.config['seds'], 'This option is not used...'

    def best_model(self, norm, f_mod, peaks):
        """Estimate the best fit model."""

        # Moving this to a separate task is not as easy as it seems. The amplitude 
        # data is huge and creates problems.
        L = [] 

        fluxA = np.zeros((len(peaks), len(f_mod.band)))
        for i,gal in enumerate(peaks.index):
            zgal = peaks.loc[gal].zb
            norm_gal = norm.sel(gal=gal, z=zgal).values
            fmod_gal = f_mod.sel(z=zgal).values

            fluxA[i] = np.dot(fmod_gal, norm_gal)

        coords = {'gal': norm.gal, 'band': f_mod.band}
        flux = xr.DataArray(fluxA, dims=('gal', 'band'), coords=coords)

        return flux

    def get_models(self):
        """Dictionary with all the current models."""
 
        modelD = {}
        for key, dep in self.input.depend.items():
            print(key)
            if not key.startswith('model_'):
                continue

            inds = ['z', 'band', 'sed', 'ext_law', 'EBV']
            fmod_in = dep.result
            f_mod = fmod_in.reset_index().set_index(inds).to_xarray().flux

            model = ['sed', 'EBV', 'ext_law']
            f_mod = f_mod.stack(model=model)

            bands = bands = [x for x in f_mod.band.values if not x.startswith('pau_')]
            f_mod = f_mod.sel(band=bands)

            #assert not np.isnan(f_mod).any(), 'Missing entries'
            nr = str(key.replace('model_', ''))
            modelD[nr] = f_mod

        return modelD

    def run(self):
        modelD = self.get_models()

        galcat_store = self.input.galcat.get_store()
        chunksize = 10
        Rin = galcat_store.select('default', iterator=True, chunksize=chunksize)

        towrite = self.config['towrite']
        path = self.output.empty_file('default')
        store = pd.HDFStore(path)
        for i,galcat in enumerate(Rin):
            print('batch', i, 'tot', i*chunksize)

            index_name = galcat.index.name

            chi2, norm = libpzcore.minimize_all_z(modelD, galcat, self.config['filters'])
            pzcat = libpzqual.get_pzcat(chi2, self.config['odds_lim'], self.config['width_frac'])
            store.append('default', pzcat)

        store.close()
