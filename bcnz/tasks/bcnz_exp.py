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
#  'zmin': 'Minimum redshift',
#  'zmax': 'Maximum redshift',
#  'dz': 'Grid width in redshift',
  'chi2_algo': 'The chi2 algorithm',
#  'use_lines': 'If including emission lines',
#  'use_ext': 'If including extinction'
}

class bcnz_exp:
    # Experimenting with weighting only based on the emission
    # line template...

    # Some of these configuration options are no longer valid and 
    # moved into the flux_model code...
    version = 1.10
    config = {
      'filters': [],
      'seds': [],
      'zmin': 0.01,
      'zmax': 2.0,
      'dz': 0.01,
      'odds_lim': 0.01,
      'Niter': 200,
      'line_weight': 1.,
      'chi2_algo': 'min',
      'use_ext': False
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

    def chi2_min(self, f_mod, data_df, zs): #, mabs_df):
        """Minimize the chi2 expression."""

        flux, flux_err, var_inv = self.get_arrays(data_df)

        t1 = time.time()
#        A = np.einsum('gf,zsf,ztf->gzst', var_inv, f_mod, f_mod)
        A = np.einsum('gf,zfs,zft->gzst', var_inv, f_mod, f_mod)

        print('HERE...')

        # Bad hack..
#        A = np.clip(A, 1e-50, np.infty)
        print('time A',  time.time() - t1)

        t1 = time.time()
#        b = np.einsum('gf,gf,zsf->gzs', var_inv, flux, f_mod)
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

            # Comparing with chi2 would require evaluating this in each
            # iteration..
            adiff = np.abs(vn-v)

            v = vn

        print('time minimize',  time.time() - t1)
        # .. Also changed in last update of einsum

        F = np.einsum('zfs,gzs->gzf', f_mod, v)
#        ipdb.set_trace()
#        F = np.einsum('zsf,gzs->gzf', f_mod, v)

        F = xr.DataArray(F, coords=coords, dims=('gal', 'z', 'band'))


        chi2 = var_inv*(flux - F)**2

        # Testing only finding the chi^2 in bands where we have 
        # expect an emission line...
        model_lines = f_mod.sel(sed='lines', EBV=0.)
        model_lines = model_lines.drop('model')


        # Not sure which you should trust the most...
        has_lines = model_lines > 0.3*model_lines.max(dim='band')

        weight = self.config['line_weight']
        assert (0 <= weight) and (weight < 1.), 'Otherwise this is wrong..'

        ipdb.set_trace()

        chi2 = ~has_lines*(1.-weight)*chi2 + has_lines*weight*chi2


        pb = np.exp(-0.5*chi2.sum(dim='band'))
        pb = pb / (1e-100 + pb.sum(dim='z'))
        chi2 = chi2.sum(dim='band')

        norm = xr.DataArray(v, coords=coords_norm, dims=('gal','z','model'))

        return chi2, norm

    def best_model(self, norm, f_mod, peaks):
        """Estimate the best fit model."""

        # Moving this elsewhere is not as easy as it seems. The amplitude data
        # is huge and creates problems.
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

        # Ok, this is not pretty..
        L = []
        for gal,zb in zip(peaks.index, peaks.zb):
            L.append(norm.sel(gal=gal, z=zb).values)

        A = np.array(L)
        coords = {'gal': norm.gal, 'model': norm.model}
        best_model = xr.DataArray(A, dims=('gal', 'model'), coords=coords)

        return best_model


    def odds(self, pz, zb):
        """ODDS quality paramter."""

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
        z0 = z[0]
        dz = float(z[1] - z[0])
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

        E = np.arange(len(pz.gal))

        def C(zbins):
            return cumsum.isel_points(gal=E, z=zbins).values

        cdf1 = db1*C(i1+1) + (1.-db1)*C(i1)
        cdf2 = db2*C(i2+1) + (1.-db2)*C(i2)
        odds = cdf2 - cdf1

        return odds

    def pz_width(self, pz, zb):
        """Estimate the pz_width quality parameter."""

        # The redshift width with a fraction frac (see below) of the pdf on
        # either side. The ODDS require a different ODDS limit for different
        # magnitudes and fractions to be optimal, while this quality parameter
        # is more robust. The estimation use the first derivative of the
        # cumsum to estimate pz_width, since a discrete pz_width is problematic
        # when cutting.

        frac = 0.01
        cumsum = pz.cumsum(dim='z')
        ind1 = (cumsum > frac).argmax(axis=1) - 1
        ind2 = (cumsum > 1-frac).argmax(axis=1) -1

        igal = range(len(cumsum.gal))
        y1_a = cumsum.isel_points(z=ind1, gal=igal)
        dy1 = (cumsum.isel_points(z=ind1+1, gal=igal) - y1_a) / \
              (cumsum.z[ind1+1].values - cumsum.z[ind1].values)

        y2_a = cumsum.isel_points(z=ind2, gal=igal)
        dy2 = (cumsum.isel_points(z=ind2+1, gal=igal) - y2_a) / \
              (cumsum.z[ind2+1].values - cumsum.z[ind2].values)

        dz1 = (frac - y1_a) / dy1
        dz2 = ((1-frac) - y2_a) / dy2
        pz_width = 0.5*(cumsum.z[(cumsum > 1-frac).argmax(axis=1)].values \
                        + dz2.values \
                        - cumsum.z[(cumsum > frac).argmax(axis=1)].values \
                        - dz1.values)

        return pz_width


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

        pzcat['odds'] = self.odds(pz, zb)
        pzcat['pz_width'] = self.pz_width(pz, zb)

        # Ok, this should be written better...
        L = []
        for i,iz in enumerate(izmin):
            L.append(norm.values[i,iz].argmax())

        pzcat['tmax'] = np.array(L)
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

        algo = self.config['chi2_algo']
        key = 'chi2_{}'.format(algo)
        assert hasattr(self, key), 'No such key: {}'.format(key)
        f_algo = getattr(self, key)

        galcat = self.job.galcat.result
        f_mod, f_mod_full = self.fix_fmod_format(self.job.f_mod.result)

        galcat_store = self.job.galcat.get_store()
        chunksize = 10
        Rin = galcat_store.select('default', iterator=True, chunksize=chunksize)

        zs = False
#        zs = self.job.zspec.result.zs
        towrite = ['best_model', 'chi2', 'norm']
        path = self.job.empty_file('default')
        store = pd.HDFStore(path)
        for i,galcat in enumerate(Rin):
            print('batch', i, 'tot', i*chunksize)

#            ipdb.set_trace()
            chi2, norm = f_algo(f_mod, galcat, zs)
            peaks,pz = self.photoz(chi2, norm)

            #ipdb.set_trace()
            if 'best_model' in towrite:
                best_model = self.best_model(norm, f_mod_full, peaks)
                best_model.name = 'best_model'
                store.append('best_model', best_model.to_dataframe())

            if 'chi2' in towrite:
                chi2.name = 'chi2'
                store.append('chi2', chi2.to_dataframe())

            if 'norm' in towrite:
                best_norm = self.best_norm(norm, peaks)
                best_norm = best_norm.unstack(dim='model')
                store.append('best_norm', best_norm.to_dataframe('best_model'))

            # Required by xarray..
            norm.name = 'norm'
            pz.name = 'pz'

            # Storing with multiindex will give problems.
            norm = norm.unstack(dim='model')

            # This should be configurable somewhere. It takes a lot of storage..
            store.append('default', peaks.stack()) 
#            store.append('norm', norm.to_dataframe())
            store.append('pz', pz.to_dataframe())
 

        store.close()
