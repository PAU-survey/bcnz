#!/usr/bin/env python
# encoding: UTF8

import ipdb
import numpy as np
import pandas as pd
import xarray as xr

descr = {'use_pz': 'Combine the result using the full pdf',
         'flat_priors': True}

import sys
sys.path.append('/home/eriksen/source/bcnz/bcnz/tasks')
import libpzqual

class bcnz_comb_ext:
    version = 1.02
    config = {'use_pz': False, 'flat_priors': True,
              'odds_lim': 0.01, 'width_frac': 0.01}

    def load_catalogs(self):
        D = {}
        for key,dep in self.job.depend.items():
            # Since it also has to depend on the galaxy catalogs.
            if not key.startswith('pzcat_'):
                continue

            EBV = dep.f_mod.ab.config['EBV']

            D[EBV] = dep.result.unstack()

        cat_out = pd.concat(D, axis=0, names=['EBV']).reset_index()

        return cat_out

    def load_pdf(self):
        print('Start loading pz..')

        df = pd.DataFrame()
        for key,dep in self.job.depend.items():
            print('loading', key)

            # Since it also has to depend on the galaxy catalogs.
            if not key.startswith('pzcat_'):
                continue

            part = dep.get_store()['pz'].reset_index()
            part['EBV'] = dep.f_mod.ab.config['EBV']

            df = df.append(part, ignore_index=True)

        # This part will complain if the EBV is not unique among the runs...
        pz = df.set_index(['gal', 'z', 'EBV']).to_xarray().pz    

        print('Finish loading pz..')

        return pz


    def combine_pz(self, pz):
        if self.config['flat_priors']:
            priors = xr.DataArray(np.ones(len(pz.EBV)), dims=('EBV'))
        else:
            raise NotImplementedError('BLAH...')

        pz = pz.dot(priors)

        pz /= pz.sum(dim='z')
        izmin = pz.argmax(dim='z')
        zb = pz.z[izmin]

        pzcat = pd.DataFrame(index=pz.gal)
        pzcat['odds'] = libpzqual.odds(pz, zb, self.config['odds_lim'])
        pzcat['pz_width'] = libpzqual.pz_width(pz, zb, self.config['width_frac'])
        pzcat['zb'] = zb

        return pzcat


    def combine_cat(self, cat_in):

        cat_out = cat_in.loc[cat_in.groupby('gal').chi2.idxmin()]

        return cat_out

    def run(self):
        if not self.config['use_pz']:
            cat_in = self.load_catalogs()
            pzcat = self.combine_cat(cat_in)
            pzcat = pzcat.set_index('gal')
        else:
            pdf_in = self.load_pdf()
            pzcat = self.combine_pz(pdf_in)

        self.job.result = pzcat.stack()
