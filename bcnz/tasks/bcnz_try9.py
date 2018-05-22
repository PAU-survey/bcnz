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

from matplotlib import pyplot as plt
import xdolphin as xd

sys.path.append('/home/eriksen/code/bcnz/bcnz/tasks')
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
  'scale_to': 'Which bands to scale it to'
}

class bcnz_try9:
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
      'scale_to': []
    }

    def check_conf(self):
        assert self.config['filters'], 'Need to set filters'
        assert not self.config['seds'], 'This option is not used...'

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

        var_inv = 1./(flux_err + 1e-100)**2
        var_inv.values = np.where(to_use, var_inv, 1e-100)
        flux_err.values = np.where(to_use, flux_err, 1e-100)


        return flux, flux_err, var_inv

    def normal_chi2(self, ab, data_df):
        """chi2 estimation."""

        flux, flux_err, var_inv = self.get_arrays(data_df)

        f_mod = self.model(ab)
        S = (var_inv*flux**2).sum(dim='f')
        X = (var_inv*flux).dot(f_mod)
        M = var_inv.dot(f_mod**2)

        chi2 = S - X**2 / M
        norm = X / M

        return chi2, norm

    def chi2_min(self, f_mod, data_df):
        """Minimize the chi2 expression."""

        flux, flux_err, var_inv = self.get_arrays(data_df)

        t1 = time.time()
        A = np.einsum('gf,zfs,zft->gzst', var_inv, f_mod, f_mod)
        print('time A',  time.time() - t1)

        t1 = time.time()
        b = np.einsum('gf,gf,zfs->gzs', var_inv, flux, f_mod)
        print('time b',  time.time() - t1)

        Ap = np.where(A > 0, A, 0)
        An = np.where(A < 0, -A, 0)

        v = 100*np.ones_like(b)

        gal_id = np.array(data_df.index)
        coords = {'gal': gal_id, 'band': f_mod.band, 'z': f_mod.z}
        coords_norm = {'gal': gal_id, 'z': f_mod.z, 'model': f_mod.model}

        if (Ap < 0).any():
            Ap = np.clip(Ap, 1e-50, np.infty)

        t1 = time.time()
        for i in range(self.config['Niter']):
            a = np.einsum('gzst,gzt->gzs', Ap, v)

            m0 = b / a
            vn = m0*v

            v = vn

        print('time minimize',  time.time() - t1)

        F = np.einsum('zfs,gzs->gzf', f_mod, v)
        F = xr.DataArray(F, coords=coords, dims=('gal', 'z', 'band'))

        chi2 = var_inv*(flux - F)**2

        pb = np.exp(-0.5*chi2.sum(dim='band'))
        pb = pb / (1e-100 + pb.sum(dim='z'))
        chi2 = chi2.sum(dim='band')

        norm = xr.DataArray(v, coords=coords_norm, dims=\
                            ('gal','z','model'))

        return chi2, norm


    def chi2_min_free(self, f_mod, data_df):
        """Minimize the chi2 expression."""

        flux, flux_err, var_inv = self.get_arrays(data_df)

        NBlist = list(filter(lambda x: x.startswith('NB'), flux.band.values))
        BBlist = list(filter(lambda x: not x.startswith('NB'), flux.band.values))

        A = np.einsum('gf,zfs,zft->gzst', var_inv, f_mod, f_mod)
        b = np.einsum('gf,gf,zfs->gzs', var_inv, flux, f_mod)

        A_NB = np.einsum('gf,zfs,zft->gzst', var_inv.sel(band=NBlist), \
               f_mod.sel(band=NBlist), f_mod.sel(band=NBlist))
        b_NB = np.einsum('gf,gf,zfs->gzs', var_inv.sel(band=NBlist), flux.sel(band=NBlist), \
               f_mod.sel(band=NBlist))
        A_BB = np.einsum('gf,zfs,zft->gzst', var_inv.sel(band=BBlist), \
               f_mod.sel(band=BBlist), f_mod.sel(band=BBlist))
        b_BB = np.einsum('gf,gf,zfs->gzs', var_inv.sel(band=BBlist), flux.sel(band=BBlist), \
               f_mod.sel(band=BBlist))

        # We might need this...
        scale_to = self.config['scale_to']
        if not len(scale_to):
            scale_to = BBlist

        var_inv_BB = var_inv.sel(band=scale_to)
        flux_BB = flux.sel(band=scale_to)
        f_mod_BB = f_mod.sel(band=scale_to)
        S1 = (var_inv_BB*flux_BB).sum(dim='band')

        # Since we need these entries in the beginning...
        k = np.ones((len(flux), len(f_mod.z)))
        b = b_NB + k[:,:,np.newaxis]*b_BB
        A = A_NB + k[:,:,np.newaxis,np.newaxis]**2*A_BB

        v = 100*np.ones_like(b)

        gal_id = np.array(data_df.index)
        coords = {'gal': gal_id, 'band': f_mod.band, 'z': f_mod.z}
        coords_norm = {'gal': gal_id, 'z': f_mod.z, 'model': f_mod.model}

        t1 = time.time()
        for i in range(self.config['Niter']):
            a = np.einsum('gzst,gzt->gzs', A, v)

            m0 = b / a
            vn = m0*v

            v = vn
            # Extra step for the amplitude
            if 0 < i and i % self.config['Nskip'] == 0:
                # Testing a new form for scaling the amplitude...
                S2 = np.einsum('gf,zfs,gzs->gz', var_inv_BB, f_mod_BB, v)
                k = (S1.values/S2.T).T

                b = b_NB + k[:,:,np.newaxis]*b_BB
                A = A_NB + k[:,:,np.newaxis,np.newaxis]**2*A_BB

# TODO: Delete this when knowing the algorithm below seems to work.
#        # Testing with the old algorithm..
#        v_scaled = v.copy()
#        k_scaled = k.copy()
#        k = np.ones((len(flux), len(f_mod.z)))
#        b = b_NB + k[:,:,np.newaxis]*b_BB
#        A = A_NB + k[:,:,np.newaxis,np.newaxis]**2*A_BB
#
#        v = 100*np.ones_like(b)
#        for i in range(self.config['Niter']):
#            a = np.einsum('gzst,gzt->gzs', A, v)
#
#            m0 = b / a
#            vn = m0*v
#            v = vn
#
#        # Some more tests...
#        F = np.einsum('zfs,gzs->gzf', f_mod, v_scaled)
#        F = xr.DataArray(F, coords=coords, dims=('gal', 'z', 'band'))
#        chi2 = var_inv*(flux - F)**2

        # I was comparing with the standard algorithm above...
        v_scaled = v
        k_scaled = k

        L = []
        L.append(np.einsum('zfs,gzs->gzf', f_mod.sel(band=NBlist), v_scaled))
        L.append(np.einsum('gz,zfs,gzs->gzf', k_scaled, f_mod.sel(band=BBlist), v_scaled))

        Fx = np.dstack(L)
        coords['band'] = NBlist + BBlist
        Fx = xr.DataArray(Fx, coords=coords, dims=('gal', 'z', 'band'))

        # Now with another scaling...
        chi2x = var_inv*(flux - Fx)**2
        Fx = xr.DataArray(Fx, coords=coords, dims=('gal', 'z', 'band'))
        chi2x = var_inv*(flux - Fx)**2

#        S = pd.Series((chi2x / chi2).values.flatten())
#        S = S[S < 10]
#        S.hist(bins=100)
#        plt.show()
        chi2x = chi2x.sum(dim='band')

#        ipdb.set_trace()

        norm = xr.DataArray(v_scaled, coords=coords_norm, dims=\
                            ('gal','z','model'))

        return chi2x, norm

#        ipdb.set_trace()
#
#        Fx = np.einsum('zfs,gzs->gzf', f_mod, v_scaled)
#        ipdb.set_trace()
#
#        print('time minimize',  time.time() - t1)
#
#        F = np.einsum('zfs,gzs->gzf', f_mod, v)
#        F = xr.DataArray(F, coords=coords, dims=('gal', 'z', 'band'))
#        chi2 = var_inv*(flux - F)**2
#
#        pb = np.exp(-0.5*chi2.sum(dim='band'))
#        pb = pb / (1e-100 + pb.sum(dim='z'))
#        chi2 = chi2.sum(dim='band')
#
#        norm = xr.DataArray(v, coords=coords_norm, dims=\
#                            ('gal','z','model'))
#
#        return chi2, norm


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
        self.check_conf()

        algo = self.config['chi2_algo']
        key = 'chi2_{}'.format(algo)
        assert hasattr(self, key), 'No such key: {}'.format(key)
        f_algo = getattr(self, key)

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

            chi2, norm = f_algo(f_mod, galcat)
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
