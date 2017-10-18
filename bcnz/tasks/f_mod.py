#!/usr/bin/env python
# encoding: UTF8

import ipdb
import time
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import splrep, splev, splint

descr = {
#  'filters': 'Filters to use',
  'seds': 'SEDs to use',
  'zmin': 'Minimum redshift',
  'zmax': 'Maximum redshift',
  'dz': 'Grid width in redshift',
  'use_lines': 'If including emission lines',
  'use_ext': 'If including extinction',
  'sep_lines': 'Lines with separate model'
}

class f_mod:
    """Combined model for all the fluxes."""

    version = 1.03
    config = {
#      'filters': [],
      'seds': [],
      'zmin': 0.01,
      'zmax': 2.0,
      'dz': 0.01,
      'use_lines': True,
      'use_ext': False,
      'sep_lines': []
    }

    def check_config(self):
#        assert self.config['filters'], 'Need to check filters'
        assert self.config['seds'], 'Need to check seds'

    def rebin_redshift(self, f_mod, zgrid):
        """Rebin the model to the grid used in the calculations."""

        t1 = time.time()
        z = f_mod.z.values

        # This order is chosen for hopefully better memory access 
        # times.
        nz = len(zgrid)
        nband = len(f_mod.band)
        nmodel = len(f_mod.model)
        A = np.ones((nz, nmodel, nband))

        print('Starting to regrid')
        for i,model in enumerate(f_mod.model.values):
            for j,band in enumerate(f_mod.band.values):
                y = f_mod.sel(band=band, model=model).values
                spl = splrep(z, y)
                A[:,i,j] = splev(zgrid, spl)

        print('Time regridding', time.time() - t1)

        coords = {'z': zgrid, 'band': f_mod.band, 'model': f_mod.model}
        fmod_new = xr.DataArray(A, coords=coords, dims=('z', 'model', 'band'))

        return fmod_new

    def _model_array(self, ab, zgrid, seds):
        """Construct the model array."""

        ab = ab.set_index(['band','sed', 'z', 'EBV'])
        f_mod = ab.to_xarray().flux

        f_mod = f_mod.sel(sed=seds) #, band=fL)
#        f_mod = f_mod.sel(band=self.config['filters'])
        f_mod = f_mod.stack(model=['sed', 'EBV'])

        f_mod = self.rebin_redshift(f_mod, zgrid)

        return f_mod

    def select_lines(self, ab_lines):

        # This method could be extended if wanting different templates
        # for the different emission lines.
        ab_lines = ab_lines[ab_lines.sed == 'lines']
        seds = ['lines']

        return ab_lines, seds

# THIS WAS THE MOST GENERAL FORM....
#
#    def select_lines(self, ab_lines):
#
#        sep_lines = self.config['sep_lines'] 
#        if sep_lines:
#            ab_sep = ab_lines[ab_lines.sed.isin(sep_lines+['lines'])]
#
#            ab_sep = ab_sep.set_index(['z', 'sed', 'ext', 'EBV', 'band'])
#            x_sep = ab_sep.to_xarray().flux
#
#            x_other = x_sep.sel(sed=sep_lines).sum(dim='sed')
#            x_rest = x_sep.sel(sed='lines') - x_other
#
#            ab_lines = ab_sep.reset_index()
#            ab_lines = ab_lines[ab_lines.sed != 'lines']
#
#            ab_rest = x_rest.to_dataframe().reset_index()
#            ab_lines = pd.concat([ab_lines, ab_rest])
#
#            seds = sep_lines + ['lines']
#        else:
#            ab_lines = ab_lines[ab_lines.sed == 'lines']
#            seds = ['lines']
#
#        return ab_lines, seds


    def select_lines(self, ab_lines):

        has_X = self.config['sep_lines'] 
        if has_X:
            lines_O = ['OII', 'OIII_1', 'OIII_2']

            ab_O = ab_lines[ab_lines.sed.isin(lines_O)]

            ab_O = ab_O.set_index(['z', 'sed', 'ext', 'EBV', 'band'])
            x_O = ab_O.to_xarray().flux
            x_O = x_O.sum(dim='sed')

            ab_main = ab_lines[ab_lines.sed == 'lines']
            ab_main = ab_main.set_index(['z', 'sed', 'ext', 'EBV', 'band'])
            x_main = ab_main.to_xarray().flux
            x_main = x_main.sum(dim='sed')

            x_main = x_main - x_O
            
            ab_main = x_main.to_dataframe().reset_index()
            ab_O = x_O.to_dataframe().reset_index()

            ab_main['sed'] = 'lines'
            ab_O['sed'] = 'O'

            ab_lines = pd.concat([ab_main, ab_O])
            seds = ['O', 'lines']
        else:
            ab_lines = ab_lines[ab_lines.sed == 'lines']
            seds = ['lines']

        return ab_lines, seds



    def model(self, ab_cont, ab_lines):

        if not self.config['use_ext']:
            ab_cont = ab_cont[ab_cont.EBV == 0.]
            ab_lines = ab_lines[ab_lines.EBV == 0.]

        C = self.config
#        fL = C['filters']

        zgrid = np.arange(C['zmin'], C['zmax']+C['dz'], C['dz'])
        fmod_cont = self._model_array(ab_cont, zgrid, C['seds'])

        if self.config['use_lines']:
            ab_lines, seds_lines = self.select_lines(ab_lines)
            fmod_lines = self._model_array(ab_lines, zgrid, seds_lines)
            fmod = xr.concat([fmod_cont, fmod_lines], dim='model')
        else:
            fmod = fmod_cont

        return fmod

    def run(self):
        self.check_config()

        ab = self.job.ab.result
        ab_el = self.job.ab_lines.result

        f_mod = self.model(ab, ab_el)
        f_mod = f_mod.unstack(dim='model')
        f_mod = f_mod.to_dataframe(name='f_mod')

        self.job.result = f_mod
