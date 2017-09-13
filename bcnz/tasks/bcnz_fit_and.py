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
  'use_lines': 'If including emission lines',
  'use_ext': 'If including extinction',
  'nfeat': 'Number of features'
}

class bcnz_fit_and:
#    """Fitting the fluxes to a galaxy template."""
    """Testing adding extinction."""

    version = 1.61
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
      'use_lines': True,
      'use_ext': False,
      'nfeat': 10,
      'renorm': False
    }

    def check_conf(self):
        assert self.config['filters'], 'Need to set filters'
        assert self.config['seds'], 'Need to set: seds'

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

    def _model_array(self, ab, zgrid, fL, seds):
        """Construct the model array."""

        ab = ab.set_index(['band','sed', 'z', 'EBV'])
        f_mod = ab.to_xarray().flux
        f_mod = f_mod.stack(model=['sed', 'EBV'])

        f_mod = f_mod.sel(band=fL)
        f_mod = self.rebin_redshift(f_mod, zgrid)

        return f_mod

    def select_lines(self, ab_lines):

        # This method could be extended if wanting different templates
        # for the different emission lines.
        ab_lines = ab_lines[ab_lines.sed == 'lines']
        seds = ['lines']

        return ab_lines, seds

    def reduce_dim(self, f_mod):

#        A = fmod.sel(z=0.01).values
#        fmod2 = fmod.copy()
        t1 = time.time()
        A = np.einsum('zsf,ztf->zst', f_mod, f_mod)

        eig = np.zeros(A.shape[:-1])
        vec = np.zeros(A.shape)
        for i,zi in enumerate(f_mod.z):
            eig_i, vec_i = np.linalg.eig(A[i])

            eig[i] = eig_i
            vec[i] = vec_i

        print('time decomposing...', time.time() - t1)

#        N = 10
#        B = vec[:,:N,:N].shape

        return vec,eig
#        return B


    def model(self, ab_cont, ab_lines):

        if not self.config['use_ext']:
            ab_cont = ab_cont[ab_cont.EBV == 0.]
            ab_lines = ab_lines[ab_lines.EBV == 0.]

        # Testing out a new way to normalize the filters..
        C = self.config
        norm_filter = 'cfht_i'
        fL = C['filters'] + [norm_filter]

        zgrid = np.arange(C['zmin'], C['zmax']+C['dz'], C['dz'])
        fmod_cont = self._model_array(ab_cont, zgrid, fL, C['seds'])
        if self.config['renorm']:
            fmod_cont /= fmod_cont.sel(band=norm_filter)

        if self.config['use_lines']:
            ab_lines, seds_lines = self.select_lines(ab_lines)
            fmod_lines = self._model_array(ab_lines, zgrid, fL, seds_lines)

            # This is potentially dangerous [but might give better results..]
            if self.config['renorm']:
                fmod_lines /= fmod_lines.sel(band=norm_filter)

            fmod = xr.concat([fmod_cont, fmod_lines], dim='model')
        else: 
            fmod = fmod_cont

        fmod = fmod.sel(band=self.config['filters'])

        return fmod

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

    def dim_red(self, f_mod):
        nfeat = self.config['nfeat']

        nz, nsed, nf = f_mod.shape

        nfeat = min(min(nf,nsed), nfeat)

        f_out = np.zeros((nz, nfeat, nf))

        t1 = time.time()
        for i,zi in enumerate(f_mod.z):
            f_in = f_mod.values[i]


            U,s,V = np.linalg.svd(f_in, full_matrices=True)
            n = U.shape[0]
            m = V.shape[0]

            S = np.zeros((nfeat,m))
            S[:nfeat,:nfeat] = np.diag(s[:nfeat])

#            ipdb.set_trace()

            R = np.dot(U[:nfeat,:nfeat], np.dot(S, V))
            f_out[i] = R

        print('time dec2', time.time() - t1) 
        new_mod = np.arange(nfeat)
        cin = f_mod.coords
#        ipdb.set_trace()
        coords = {'z': f_mod.z, 'band': f_mod.band, 'model': new_mod}

        f_mod_out = xr.DataArray(f_out, dims=('z', 'model', 'band'), coords=coords)

        return f_mod_out

    def dim_nmf(self, f_mod):
        from sklearn.decomposition import NMF

        ncomp = self.config['nfeat']
        nz = len(f_mod.z)
        nf = len(f_mod.band)

        t1 = time.time()
        f_out = np.zeros((nz, ncomp, nf))
        for i,zi in enumerate(f_mod.z):
            f_in = f_mod.values[i]
            f_in = np.abs(f_in)

            inst = NMF(n_components=10)
            W = inst.fit_transform(f_in)
            H = inst.components_

            f_out[i] = H

        print('time factorizing', time.time() - t1)

        new_mod = np.arange(ncomp)
        coords = {'z': f_mod.z, 'band': f_mod.band, 'model': new_mod}
        fmod_out = xr.DataArray(f_out, dims=('z', 'model', 'band'), coords=coords)

        return fmod_out

    def chi2_min(self, f_mod, data_df, zs): #, mabs_df):
        """Minimize the chi2 expression."""

        flux, flux_err, var_inv = self.get_arrays(data_df)

#        f_mod = self.dim_red(f_mod)
#        vec,eig = self.reduce_dim(f_mod)

        t1 = time.time()
        A = np.einsum('gf,zsf,ztf->gzst', var_inv, f_mod, f_mod)

        print('time A',  time.time() - t1)

        t1 = time.time()
        b = np.einsum('gf,gf,zsf->gzs', var_inv, flux, f_mod)
        print('time b',  time.time() - t1)

        Ap = np.where(A > 0, A, 0)
        An = np.where(A < 0, -A, 0)
#        ipdb.set_trace()

        v = 100*np.ones_like(b)

        gal_id = np.array(data_df.index)
        coords = {'gal': gal_id, 'band': f_mod.band, 'z': f_mod.z}
        coords_norm = {'gal': gal_id, 'z': f_mod.z, 'model': f_mod.model}


        t1 = time.time()
        for i in range(self.config['Niter']):
            a = np.einsum('gzst,gzt->gzs', Ap, v)

            m0 = b / a
            vn = m0*v

            # Comparing with chi2 would require evaluating this in each
            # iteration..
            adiff = np.abs(vn-v)

            v = vn

        print('time minimize',  time.time() - t1)
        F = np.einsum('zsf,gzs->gzf', f_mod, v)
        F = xr.DataArray(F, coords=coords, dims=('gal', 'z', 'band'))

        chi2 = var_inv*(flux - F)**2

        pb = np.exp(-0.5*chi2.sum(dim='band'))
        pb = pb / (1e-100 + pb.sum(dim='z'))
        chi2 = chi2.sum(dim='band')

        norm = xr.DataArray(v, coords=coords_norm, dims=('gal','z','model'))

        return chi2, norm

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
        i1 = np.clip(np.floor(bins1), 0, np.infty).astype(np.int)
        i2 = np.clip(np.floor(bins2), 0, np.infty).astype(np.int)
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
        ab = self.job.ab.result #.unstack()
        ab_el = self.job.ab_lines.result if hasattr(self.job, 'ab_lines') \
                else None
        f_mod = self.model(ab, ab_el)

        # Hack to look at the model...
        ipdb.set_trace()

        X = f_mod.to_pandas()
        X.to_hdf('/home/eriksen/fmod.h5', 'fmod')

        import sys
        sys.exit(1)

        ipdb.set_trace()



        f_mod = self.dim_nmf(f_mod)

        galcat_store = self.job.galcat.get_store()
        chunksize = 10
        Rin = galcat_store.select('default', iterator=True, chunksize=chunksize)

        zs = False
#        zs = self.job.zspec.result.zs

        path = self.job.empty_file('default')
        store = pd.HDFStore(path)
        for i,galcat in enumerate(Rin):
            print('batch', i, 'tot', i*chunksize)

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
