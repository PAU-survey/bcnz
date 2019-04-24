#!/usr/bin/env python

from __future__ import print_function

from IPython.core import debugger
import os
import time
import numpy as np
import pandas as pd
import xarray as xr
import itertools as it

from scipy.interpolate import splrep, splev, splint
from scipy.integrate import trapz, simps

from matplotlib import pyplot as plt
import xdolphin as xd

import sys
sys.path.append('/home/eriksen/source/bcnz/bcnz/tasks')
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
  'Nrand': 'Number of random realizations',
  'Niter_rand': 'Number of iterations when minimizing on the randoms'
}

class bcnz_try5:
    """Fitting a linear combination to the observations."""

    # Some of these configuration options are no longer valid and 
    # moved into the flux_model code...
    version = 1.11
    config = {
      'filters': [],
      'seds': [],
      'zmin': 0.01,
      'zmax': 2.0,
      'dz': 0.01,
      'odds_lim': 0.01,
      'Niter': 200,
      'line_weight': 2.,
      'chi2_algo': 'min',
      'use_ext': False,
      'width_frac': 0.01,
      'towrite': ['best_model'],
      'Nrand': 3,
      'Niter_rand': 10
    }

    def check_conf(self):
        assert self.config['filters'], 'Need to set filters'
        assert self.config['seds'], 'Need to set: seds'


    def get_arrays(self, data_df):
        """Read in the arrays and present them as xarrays."""

        # Seperating this book keeping also makes it simpler to write
        # up different algorithms.

        filters = self.config['filters']
        dims = ('gal', 'band')
        flux = xr.DataArray(data_df['flux'][filters], dims=dims)
        flux_err = xr.DataArray(data_df['flux_err'][filters], dims=dims)

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

    def _add_estimates(self, cat):
        """Adds some estimates to the catalog."""

        cat['var_inv'] = 1./cat.flux_err**2.
        cat['w_flux'] = cat.var_inv * cat.flux
        cat['w_flux_err'] = cat.var_inv * cat.flux_err

#    def find_
    def boot(self, fa_sel):
        self._add_estimates(fa_sel)

        df = pd.DataFrame()
        for ref_id,sub in fa_sel.groupby('ref_id'):
            for i in range(self.config['Nrand']):
#                part = sub.sample(n=len(sub), replace=True).groupby('band')['var_inv', 'w_flux'].sum()
                part = sub.sample(n=int(0.9*len(sub)), replace=False).groupby('band')['var_inv', 'w_flux'].sum()
                part['flux'] = part.w_flux / part.var_inv
##                part['flux_err'] = np.sqrt(1. / part.var_inv)

#                del part['var_inv']
                del part['w_flux']
                part['i'] = i
                part['gal'] = ref_id

                df = df.append(part)

        return df

    def chi2_min(self, f_mod, data_df, fa_sel):
        """Minimize the chi2 expression."""

        flux, flux_err, var_inv = self.get_arrays(data_df)

        t1 = time.time()
        A = np.einsum('gf,zfs,zft->gzst', var_inv, f_mod, f_mod)
        b = np.einsum('gf,gf,zfs->gzs', var_inv, flux, f_mod)
        print('time A+b',  time.time() - t1)

        Ap = np.where(A > 0, A, 0)
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

        vcoadd = v.copy()
        flux_real = self.boot(fa_sel)
        fluxXX = flux_real.reset_index().set_index(['i', 'gal','band']).to_xarray()
        fluxXX = fluxXX.fillna(0.)

        D = {}
        chi2L = []
        L = []
        Ngal = flux.shape[0]
        Nrand = self.config['Nrand']

        zx = np.zeros((Nrand, Ngal))
        zx = xr.DataArray(zx, dims=('i', 'gal'), coords={'gal': flux.gal, 'i':range(Nrand)})
        for i in range(Nrand): 
#        for i,sub in flux_real.groupby('i'):
            var_inv = fluxXX.sel(i=i).var_inv
            flux = fluxXX.sel(i=i).flux
            print('sum before', A.sum())
            np.einsum('gf,zfs,zft->gzst', var_inv, f_mod, f_mod, out=A)
            np.einsum('gf,gf,zfs->gzs', var_inv, flux, f_mod, out=b)
            print('sum after', A.sum())

            v = vcoadd.copy()
            for j in range(self.config['Niter_rand']):
                a = np.einsum('gzst,gzt->gzs', A, v)
                m0 = b / a
                vn = m0*v

                v = vn


#            ipdb.set_trace()             

            F = np.einsum('zfs,gzs->gzf', f_mod, v)
            F = xr.DataArray(F, coords=coords, dims=('gal', 'z', 'band'))

            D[i] = F
            chi2 = var_inv*(flux - F)**2

            pb = np.exp(-0.5*chi2.sum(dim='band'))
            pb = pb / (1e-100 + pb.sum(dim='z'))
            chi2 = chi2.sum(dim='band')

            chi2L.append(chi2)

#            ipdb.set_trace()
#            zx[i] = gal = flux.shape[0]
            zx[i] = pb.z[pb.argmin(dim='z')]
#            L.append(pb.z[pb.argmin(dim='z')])

        # UGLY!!!
        chi2_large = xr.concat(chi2L, dim='i')

        if True: #False:
            dims=('gal', 'z')
            coords={'gal': chi2.gal.values, 'z': chi2.z}

            Ngal = len(coords['gal'])
            Nz = len(coords['z'])

            X = np.zeros((Ngal, Nz))
            for i in range(Ngal):
                for j in range(Nz):
                    X[i,j] = chi2_large[:, i,j].median()

            chi2 = xr.DataArray(X, dims=dims, coords=coords)
            norm = xr.ones_like(chi2)

        if False:
            ipos = int(Nrand/2.+0.5)
            sel_i = []
            for i in range(Ngal): 
                X = zx[:,i].to_series()
                X = X.sort_values()

                sel_i.append(X.index[ipos])

            chi2 = chi2_large.sel_points(i=sel_i, gal=flux.gal)
            chi2 = xr.DataArray(chi2, dims=('gal', 'z'), coords={'gal': chi2.gal.values, 'z': chi2.z})

            norm = xr.ones_like(chi2)

        if False:
            F_coadd = np.einsum('zfs,gzs->gzf', f_mod, v_coadd)
            F_coadd = xr.DataArray(F, coords=coords, dims=('gal', 'z', 'band'))

            chi2 = var_inv*(flux - F)**2

            pb = np.exp(-0.5*chi2.sum(dim='band'))
            pb = pb / (1e-100 + pb.sum(dim='z'))
            chi2 = chi2.sum(dim='band')

            norm = xr.DataArray(v, coords=coords_norm, dims=\
                                ('gal','z','model'))

        return chi2, norm

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
        f_mod = fmod_in.to_xarray().f_mod
        f_mod = f_mod.stack(model=['sed', 'EBV'])

        f_mod_full = f_mod
        f_mod = f_mod_full.sel(band=self.config['filters'])

        return f_mod, f_mod_full
#        ipdb.set_trace()

    def run(self):
        self.check_conf()

        galcat = self.input.galcat.result
        f_mod, f_mod_full = self.fix_fmod_format(self.input.model.result)

        fa_store = self.input.flux_fa.get_store()
        galcat_store = self.input.galcat.get_store()
        chunksize = 10
        Rin = galcat_store.select('default', iterator=True, chunksize=chunksize)

        towrite = self.config['towrite']
        path = self.output.empty_file('default')
        store = pd.HDFStore(path)
        for i,galcat in enumerate(Rin):
            print('batch', i, 'tot', i*chunksize)

            fa_sel = fa_store.select('default', where='index in {}'.format(galcat.index.tolist()))

            index_name = galcat.index.name

            chi2, norm = self.chi2_min(f_mod, galcat, fa_sel)
            peaks, pz = self.photoz(chi2, norm)
#            best_model = self.best_model(norm, f_mod_full, peaks)

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
