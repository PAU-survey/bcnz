#!/usr/bin/env python
# encoding: UTF8

import ipdb
import sys
import time
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

import libpzqual

#sys.path.append('/home/eriksen/papers/paupz/plots')
#import calcpz

descr = {'norm_nb': 'Normalize the NB p(z)',
         'norm_bb': 'Normalize the BB p(z)',
         'bb_minval': 'Minimum value added to the p(z)',
         'bb_exp': 'Minimum value added to the p(z)',
         'use_chi2': 'Convert to chi2',
         'z_bias': 'Bias to correct for.',
         'k': 'Weight between the NB and BB terms',
         'width_frac': 'Fraction on each side',
         'odds_lim': 'Limit within to calculate ODDS'}

class bcnz_comb_pz:
    """Combine the pz from different experiments."""

    # This code is on purpose written for combining two
    # catalogs.. beyond this the configuration of how
    # to postprocess the p(z) distributions become tricky.
    version = 1.11
    config = {'norm_nb': False,
              'norm_bb': False,
              'bb_minval': 0,
              'bb_exp': 1.,
              'use_chi2': True,
              'z_bias': 0.,
              'k': 0.5,
              'width_frac': 0.01,
              'odds_lim': 0.01}

    def get_pz(self, pzjob):
        with pzjob.get_store() as store:
            pz = store['pz'].to_xarray().pz
            cat = store['default'].unstack()

        return pz,cat

    def rescale_pz(self, pz_nb, pz_bb):
        if self.config['norm_nb']: 
            pz_nb /= pz_nb.max(dim='z')

        if self.config['norm_nb']: 
            pz_bb /= pz_bb.max(dim='z')

        pz_bb += pz_bb.max()*self.config['bb_minval']
        pz_bb = pz_bb**self.config['bb_exp']

        return pz_nb, pz_bb

    def regrid_pz(self, pz_old, z_highres):
        """Regrid the photo-z to a higher resolution grid."""

        z_lowres = pz_old.z.values
        z_eval = z_highres - self.config['z_bias']
        from scipy.interpolate import splrep, splev

        pz_out = np.zeros((len(pz_old), len(z_highres)))

        t1 = time.time()
        for i in range(len(pz_old)):
            spl = splrep(z_lowres, pz_old[i], k=1)
            pz_out[i] = splev(z_eval, spl, ext=1)

        print('time regridding', time.time() - t1)
        coords = {'gal': pz_old.gal, 'z': z_highres}
        pz_out = xr.DataArray(pz_out, dims=('gal','z'), coords=coords)

        norm = pz_out.sum(dim='z')
        norm = norm[norm != 0]
        pz_out = pz_out / norm

        return pz_out

    def combine_pz(self, pz_nb, pz_bb, cat_nb, cat_bb):
        """Combine the pdf from the broad and narrow bands."""

        pz_nb, pz_bb = self.rescale_pz(pz_nb, pz_bb)
        pz_bb = self.regrid_pz(pz_bb, pz_nb.z)

        if self.config['use_chi2']:
            cat_bb.index.name = 'gal'

            width = 0.001 # HACK..
            chi_bb = -2.*np.log(width*pz_bb+1e-100)
            chi_bb = chi_bb + cat_bb.chi2.to_xarray()

            chi_nb = -2.*np.log(width*pz_nb+1e-100)
            chi_nb = chi_nb + cat_nb.chi2.to_xarray()

            k = self.config['k']
            chi2 = 2*(1-k)*chi_nb + 2*k*chi_bb
            pz = np.exp(-0.5*(chi2))
        else:
            k = self.config['k']
            fac_nb = 2*(1.-k)
            fac_bb = 2*k

            #pz = pz_nb*pz_bb
            pz = (pz_nb**fac_nb)*(pz_bb**fac_bb)

        zb = pz.z[pz.argmax(dim='z')]
        pz /= pz.sum(dim='z')

        cat = pd.DataFrame(index=pz.gal)
        cat['zb'] = zb
        cat['pz_width'] = libpzqual.pz_width(pz, zb, self.config['width_frac'])
        cat['odds'] = libpzqual.odds(pz, zb, self.config['odds_lim'])

        return pz,cat

    def run(self):
        pz_nb,cat_nb = self.get_pz(self.job.pzcat_nb)
        pz_bb,cat_bb = self.get_pz(self.job.pzcat_bb)

        pz,cat = self.combine_pz(pz_nb, pz_bb, cat_nb, cat_bb)
        cat = cat.stack()

        self.job.result = cat
#        ipdb.set_trace()
