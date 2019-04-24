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

sys.path.append('/home/eriksen/code/bcnz/bcnz/tasks')
sys.path.append('/nfs/pic.es/user/e/eriksen/code/bcnz/bcnz/tasks')
sys.path.append(os.path.expanduser('~/Dropbox/pauphotoz/bcnz/bcnz/tasks'))
import libpzqual

#import .libpzqual

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

class bcnz_try10:
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

        # Moving this to a separate task  is not as easy as it seems. The amplitude 
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

    def best_norm(self, norm, peaks):
        """Select the best norm."""

        # Not pretty... I often have a problem with selecing pairs..
        # Ok, this is not pretty..
        L = []
        for gal,zb in zip(peaks.index, peaks.zb):
            L.append(norm.sel(gal=gal, z=zb).values)

        A = np.array(L)
        coords = {'gal': norm.gal, 'model': norm.model}
        best_model = xr.DataArray(A, dims=('gal', 'model'), coords=coords)

        return best_model

    def sed_iband(self, norm, f_mod, peaks):
        """Find the sed contribut most to the cfht_i band."""

        # Here the choice of cfht_i remains to be tested using
        # data...

        # We are not interested in which extinction the galaxy has.
        norm = norm.unstack(dim='model').sum(dim='EBV')
        f_mod = f_mod.unstack(dim='model').sum(dim='EBV')

        coords = {'gal': norm.gal, 'sed': f_mod.sed}
        flux_band = np.zeros((len(coords['gal']), len(coords['sed'])))
        flux_band = xr.DataArray(flux_band, dims=('gal', 'sed'), coords=coords)
        for i,(gal,zb) in enumerate(zip(peaks.index, peaks.zb)):
            zgal = peaks.loc[gal].zb
            norm_gal = norm.sel(gal=gal, z=zgal)
            fmod_gal = f_mod.sel(z=zgal, band='cfht_i')

            flux_band[i] = fmod_gal*norm_gal

        flux_band = flux_band / flux_band.sum(dim='sed')
        iband_sed = flux_band.sed[flux_band.argmax(dim='sed')]

        # Ugly hack since appending with different lengths give problems.
        iband_sed = [str(x).ljust(5) for x in iband_sed.values]

        return iband_sed


    def photoz(self, chi2, norm):
        """Defines many of the quantities entering in the calogues."""

        pzcat = pd.DataFrame(index=chi2.gal)

        has_sed = 'sed' in chi2.dims
        dims = ['sed', 'z'] if has_sed else ['z']
        delta_chi2 = chi2 - chi2.min(dim=dims)
        pzt = np.exp(-0.5*delta_chi2)

        pz = pzt.sum('sed') if has_sed else pzt
        pz = pz / pz.sum(dim='z')

        zb = libpzqual.zb(pz)
        pzcat['zb'] = zb
        pzcat['odds'] = libpzqual.odds(pz, zb, self.config['odds_lim'])
        pzcat['pz_width'] = libpzqual.pz_width(pz, zb, self.config['width_frac'])
        pzcat['zb_bpz2'] = libpzqual.zb_bpz2(pz)

        pzcat['chi2'] = np.array(chi2.min(dim=dims))

        return pzcat, pz

    def fix_fmod_format(self, fmod_in):
        """Converts the dataframe to an xarray."""


        inds = ['z', 'band', 'sed', 'ext_law', 'EBV']
        f_mod = fmod_in.reset_index().set_index(inds).to_xarray().flux

        model = ['sed', 'EBV', 'ext_law']
        f_mod = f_mod.stack(model=model)

        f_mod_full = f_mod
        f_mod = f_mod_full.sel(band=self.config['filters'])

        assert not np.isnan(f_mod).any(), 'Missing entries'

        return f_mod, f_mod_full

    def run(self):
        galcat = self.input.galcat.result


        f_mod, f_mod_full = self.fix_fmod_format(self.input.model.result)

        galcat_store = self.input.galcat.get_store()
        chunksize = 10
        Rin = galcat_store.select('default', iterator=True, chunksize=chunksize)

        towrite = self.config['towrite']
        path = self.output.empty_file('default')
        store = pd.HDFStore(path)
        for i,galcat in enumerate(Rin):
            print('batch', i, 'tot', i*chunksize)

            index_name = galcat.index.name

            chi2, norm = libpzcore.minimize_all_z(f_mod, galcat, self.config['filters'])
            peaks, pz = self.photoz(chi2, norm)
            best_model = self.best_model(norm, f_mod_full, peaks)

#            peaks['sed_iband'] = self.sed_iband(norm, f_mod_full, peaks)
            peaks.index.name = index_name

            if 'best_model' in towrite:
                best_model = best_model.rename({'gal': index_name})
                store.append('best_model', best_model.to_dataframe('best_model'))

            if 'chi2' in towrite:
                chi2 = chi2.rename({'gal': index_name})
                store.append('chi2', chi2.to_dataframe('chi2'))

            if 'norm' in towrite:
                best_norm = best_norm.rename({'best_norm': index_name})
                best_norm = self.best_norm(norm, peaks)
                best_norm = best_norm.unstack(dim='model')
                store.append('best_norm', best_norm.to_dataframe('best_model'))

            pz = pz.rename({'gal': index_name})
            store.append('default', peaks)
            store.append('pz', pz.to_dataframe('pz'))
 

        store.close()
