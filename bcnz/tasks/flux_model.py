#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger
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
  'sep_lines': 'Lines with separate model'
}

class flux_model:
    """Combined model for all the fluxes."""

    version = 1.04
    config = {
#      'filters': [],
      'seds': [],
      'zmin': 0.01,
      'zmax': 2.0,
      'dz': 0.01,
      'use_lines': True,
      'sep_lines': []
    }

    def check_config(self):
        assert self.config['seds'], 'Config missing: seds'

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

#        ipdb.set_trace()

        f_mod = f_mod.sel(sed=seds)
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

        sep_lines = self.config['sep_lines'] 
        if sep_lines == 'all':
            # ab_lines is a view from selecting based on the EBV parameter.
            ab_lines = ab_lines.copy()

            seds_in = list(ab_lines.sed.unique())
            mapping = dict([(x,'line_'+x) for x in seds_in if not x=='lines'])
            ab_lines['sed'] = ab_lines.sed.replace(mapping)

            seds = list(mapping.values())
            seds.sort()

        elif sep_lines == 'O':
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

    def workaround_concat(self, fmod_cont, fmod_lines):
        # For some reason the xarray concat, 
        #    xr.concat([fmod_cont, fmod_lines], dim='model')
        # stopped working. I don't why. This workaround is lengthy 
        # and ugly. Please check if it workse later on newer
        # versions of xarray.
        # 
        fmod_cont = fmod_cont.unstack(dim='model')
        fmod_lines = fmod_lines.unstack(dim='model')

        fmod_cont = fmod_cont.sel(band=fmod_lines.band)
        fmod_cont = fmod_cont.sel(z=fmod_lines.z)

        EBV = round(float(fmod_cont.EBV[0]), 4)
        fmod_cont = fmod_cont.squeeze(['EBV'])
        fmod_lines = fmod_lines.squeeze(['EBV'])
       
        fmod = xr.concat([fmod_cont, fmod_lines], dim='sed')
        fmod = fmod.rename({'EBV': 'tmp'})
        fmod = fmod.expand_dims('EBV', -1)
        fmod.coords['EBV'] = np.array([EBV])
        del fmod['tmp']

        fmod = fmod.stack(model=('sed', 'EBV'))

        return fmod

    def model(self, ab_cont, ab_lines):

        C = self.config
        zgrid = np.arange(C['zmin'], C['zmax']+C['dz'], C['dz'])
        fmod_cont = self._model_array(ab_cont, zgrid, C['seds'])

        if self.config['use_lines']:
            ab_lines, seds_lines = self.select_lines(ab_lines)
            fmod_lines = self._model_array(ab_lines, zgrid, seds_lines)
            fmod = self.workaround_concat(fmod_cont, fmod_lines)
        else:
            fmod = fmod_cont

        return fmod

    def run(self):
        ab = self.input.ab.result
        ab_el = self.input.ab_lines.result

        f_mod = self.model(ab, ab_el)
        f_mod = f_mod.unstack(dim='model')
        f_mod = f_mod.to_dataframe(name='f_mod')

        if 1 < len(f_mod.to_xarray().EBV):
            ipdb.set_trace()

        self.output.result = f_mod
