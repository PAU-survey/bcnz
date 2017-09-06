#!/usr/bin/env python

from __future__ import print_function

import ipdb
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

descr = {
  'filters': 'Filters to use',
  'seds': 'SEDs to use',
  'zmin': 'Minimum redshift',
  'zmax': 'Maximum redshift',
  'dz': 'Grid width in redshift',
  'chi2_algo': 'The chi2 algorithm',
  'use_lines': 'If including emission lines'
}


class bcnz_fit:
    """Fitting the fluxes to a galaxy template."""

    # New version of the bcnz core code.
    version = 1.055
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
      'use_lines': True
    }

    def check_conf(self):
        assert self.config['filters'], 'Need to set filters'
        assert self.config['seds'], 'Need to set: seds'

    def _find_models(self, ab, seds):
        """Find a list of models."""

        models = []
        for key in ab.columns:
            if key == 'z':
                continue

            fname, sed, ext = key
            if not sed in seds:
                continue

            models.append((sed, ext))

        models = list(set(models))

        return models

    def _model_array(self, ab, zgrid, fL, seds):
        """Construct the model array."""

        models = self._find_models(ab, seds)

        f_mod = np.zeros((len(fL), len(models), len(zgrid)))
        for i,fname in enumerate(fL):
            for j,(sed,ext) in enumerate(models):
                key = (fname, sed, ext)
                spl = splrep(ab['z'], ab[key])
                f_mod[i,j,:] = splev(zgrid, spl)


        # Ok... this is not exactly pretty..
        model = list(map('_'.join, models))
        coords = {'f': fL, 'model': model, 'z': zgrid}
        f_mod = xr.DataArray(f_mod, coords=coords, dims=('f','model','z'))

        return f_mod


    def _model_normal(self, ab, zgrid, fL):
        """Array with the model."""

        seds = self.config['seds']
        f_mod = self._model_array(ab, zgrid, fL, seds)

        return f_mod

    def _model_lines(self, ab, zgrid, fL):
        return self._model_array(ab, zgrid, fL, ['lines'])

#        ipdb.set_trace()

    def model(self, ab, ab_el):

        C = self.config
        zgrid = np.arange(C['zmin'], C['zmax']+C['dz'], C['dz'])
        fL = self.config['filters']

        f_mod1 = self._model_normal(ab, zgrid, fL)
        if self.config['use_lines']:
            f_mod2 = self._model_lines(ab_el, zgrid, fL)
            f_mod = xr.concat([f_mod1, f_mod2], dim='model')
        else:
            f_mod = f_mod1

        return f_mod


    def get_arrays(self, data_df):
        """Read in the arrays and present them as xarrays."""

        # Seperating this book keeping also makes it simpler to write
        # up different algorithms.
        filters = self.config['filters']
        dims = ('gal', 'f')
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

    def chi2_min(self, f_mod, data_df, zs): #, mabs_df):
        """Minimize the chi2 expression."""

#        data_df = data_df[100:150]
        flux, flux_err, var_inv = self.get_arrays(data_df)


        t1 = time.time()
        A = np.einsum('fsz,ftz,gf->gzst', f_mod, f_mod, var_inv)
        print('time A',  time.time() - t1)

        t1 = time.time()
        b = np.einsum('gf,fsz,gf->gzs', flux, f_mod, var_inv)
        print('time b',  time.time() - t1)

        Ap = np.where(A > 0, A, 0)
        An = np.where(A < 0, -A, 0)


        v = 100*np.ones_like(b)

        gal_id = np.array(data_df.index)
        coords = {'gal': gal_id, 'f': f_mod.f, 'z': f_mod.z}
        coords_norm = {'gal': gal_id, 'z': f_mod.z, 'model': f_mod.model}

        t1 = time.time()
        for i in range(self.config['Niter']): #1000):
            print(i)
            a = np.einsum('gzst,gzt->gzs', Ap, v)
            c = np.einsum('gzst,gzt->gzs', An, v)

#            m0 = (-b + np.sqrt(b**2 + 4*a*c)) / (2*a)
#            ipdb.set_trace()
            m0 = b / a
            vn = m0*v

            # Comparing with chi2 would require evaluating this in each
            # iteration..
            adiff = np.abs(vn-v)

            v = vn

#        ipdb.set_trace()

        print('time minimize',  time.time() - t1)

        F = np.einsum('fsz,gzs->gfz', f_mod, v)
        F = xr.DataArray(F, coords=coords, dims=('gal', 'f', 'z'))

        chi2 = var_inv*(flux - F)**2

        pb = np.exp(-0.5*chi2.sum(dim='f'))
        pb = pb / (1e-100 + pb.sum(dim='z'))
        chi2 = chi2.sum(dim='f')

        norm = xr.DataArray(v, coords=coords_norm, dims=('gal','z','model'))


        return chi2, norm

    def chi2_marg(self, f_mod, data_df, zs): #, mabs_df):
        print('starting the right class')

#        data_df = data_df[100:150]
        flux, flux_err, var_inv = self.get_arrays(data_df)

        t1 = time.time()
        K = np.einsum('gf,gf->g', flux**2, var_inv)
#        B = np.einsum('gf,fsz->gzs', flux, f_mod)
#        A = np.einsum('fsz,ftz->zst', f_mod, f_mod)

        B = np.einsum('gf,fsz,gf->gzs', flux, f_mod, var_inv)
        A = np.einsum('fsz,ftz,gf->gzst', f_mod, f_mod, var_inv)
        print('time terms', time.time() - t1)

#        ipdb.set_trace()

        Ainv = np.zeros_like(A)
        ngal, nz = A.shape[:2]

        t1 = time.time()
        # - Also loop over galaxies.
        for i in range(ngal):
            for j in range(nz):
                Ainv[i,j] = np.linalg.inv(A[i,j])

        print('time inverting', time.time() - t1)

        # - Loop both over galaxies and redshifts...
        Adet = np.zeros((ngal, nz))
        for i in range(ngal):
            for j in range(nz):
                Adet[i,j] = np.linalg.det(A[i,j])

#        pb = np.zeros(nz)
        Fx = np.zeros((ngal, nz))
        pb = np.zeros((ngal, nz))
        for i in range(ngal):
            for j in range(nz):
                J = B[i,j]
                pre = np.sqrt(2*np.pi/Adet[i,j])
                F = np.dot(J, np.dot(Ainv[i,j], J))

#                ipdb.set_trace()
                print(F) #F-K[i]) 
                pb[i,j] = pre*np.exp(0.5*(F-K[i]))
                Fx[i,j] = F

        # Well, some tests...
        C = self.config
        zgrid = np.arange(C['zmin'], C['zmax']+C['dz'], C['dz'])


        for igal in range(len(Fx)):
            P = np.exp(Fx[igal] - Fx[igal].min())
            P = P / P.sum()

            plt.plot(zgrid, P)

            plt.axvline(zs.iloc[igal])

            plt.show()

        ipdb.set_trace()



    def odds_fast(self, pz, zb):

        # Very manual determination of the ODDS through the
        # cumsum function. xarray is n version 0.9.5 not
        # supporting integration.
        odds_lim = self.config['odds_lim']
        z1 = zb - odds_lim*(1.+zb)
        z2 = zb + odds_lim*(1.+zb)

        # When the galaxy is close to the end of the grid.
        z = pz.z.values
        z1 = np.clip(z1, z[0], z[-1])
        z2 = np.clip(z2, z[0], z[-1])


        # This assumes a regular grid.
        dz = self.config['dz']
        z0 = z[0]
        bins1 = (z1 - z0) / dz - 1 # Cumsum is estimated at the end
        bins2 = (z2 - z0) / dz - 1
        i1 = np.floor(bins1).astype(np.int)
        i2 = np.floor(bins2).astype(np.int)
        db1 = bins1 - i1
        db2 = bins2 - i2

        # Here the cdf is estimated using linear interpolation
        # between the points. This is done because the cdf is
        # changing rapidly for a large sample of galaxies.
        cumsum = pz.cumsum(dim='z')
        E = np.arange(len(pz))

        def C(zbins):
            return cumsum.isel_points(gal=E, z=zbins).values

        cdf1 = db1*C(i1+1) + (1.-db1)*C(i1)
        cdf2 = db2*C(i2+1) + (1.-db2)*C(i2)
        odds = cdf2 - cdf1

        # This version is similar to BPZ / BCNzv1
        old_odds = C(i2+1) - C(i1)

        return old_odds, odds

    def photoz(self, chi2, norm):
        pzcat = pd.DataFrame(index=chi2.gal)

        has_sed = 'sed' in chi2.dims
        dims = ['sed', 'z'] if has_sed else ['z']
        delta_chi2 = chi2 - chi2.min(dim=dims)
        pzt = np.exp(-0.5*delta_chi2)

        pz = pzt.sum('sed') if has_sed else pzt
        pz = pz / pz.sum(dim='z')

        izmin = pz.argmax(dim='z')
        zb = pz.z[izmin]
        pzcat['zb'] = zb

        old_odds, odds = self.odds_fast(pz, zb)
        pzcat['odds_old'] = old_odds
        pzcat['odds'] = odds

        # Ok, this should be written better...
        L = []
        for i,iz in enumerate(izmin):
            L.append(norm.values[i,iz].argmax())

        pzcat['tmax'] = np.array(L)
        pzcat['chi2'] = np.array(chi2.min(dim=dims))

        return pzcat

    def run(self):
        self.check_conf()

        algo = self.config['chi2_algo']
        key = 'chi2_{}'.format(algo)
        assert hasattr(self, key), 'No such key: {}'.format(key)
        f_algo = getattr(self, key)

#        galcat = self.job.galcat.result
        ab = self.job.ab.result.unstack()
        ab_el = self.job.ab_lines.result if hasattr(self.job, 'ab_lines') \
                else None
        f_mod = self.model(ab, ab_el)

        galcat_store = self.job.galcat.get_store()
        Rin = galcat_store.select('default', iterator=True, chunksize=10)

        zs = False
#        zs = self.job.zspec.result.zs

        path = self.job.empty_file('default')
        store = pd.HDFStore(path)
        for i,galcat in enumerate(Rin):
            chi2, norm = f_algo(f_mod, galcat, zs)
            peaks = self.photoz(chi2, norm)

            # Required by xarray..
            norm.name = 'norm'
            chi2.name = 'chi2'

            # This should be configurable somewhere. It takes a lot of storage..
            store.append('default', peaks.stack()) 
#            store.append('norm', norm.to_dataframe())
#            store.append('chi2', chi2.to_dataframe())
 

        store.close()
