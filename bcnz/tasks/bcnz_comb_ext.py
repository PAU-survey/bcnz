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
    version = 1.16
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


    def combine_pz(self, pz_in, cat_in):
        if self.config['flat_priors']:
            priors = xr.DataArray(np.ones(len(pz_in.EBV)), dims=('EBV'))
        else:
            # COSMOS priors....
            S = [(0.00, 31334), (0.10, 12927), (0.15, 12131), (0.20, 11778), (0.25, 11025), \
                 (0.05, 10659), (0.30,  9055), (0.35,  6884), (0.40,  6067), (0.50,  5580)]

            E = pd.Series(dict(S), name='EBV').to_xarray()
            priors = E.rename({'index': 'EBV'})

            # Priors from the catalogue itself...
            E = cat_in.loc[cat_in.groupby('gal').chi2.idxmin()].EBV.value_counts().to_xarray()
            priors = E.rename({'index': 'EBV'})


        priors = priors / float(priors.sum())

        chi2_min = cat_in[['gal', 'EBV', 'chi2']].set_index(['gal', 'EBV']).to_xarray().chi2

        pz = np.clip(pz_in, 1e-100, np.infty)
        chi2_in = -2.*np.log(pz)

        chi2_tmp = chi2_in.min(dim=['z'])
        chi2 = chi2_in + (chi2_min - chi2_tmp)

        pz_ebv = np.exp(-0.5*chi2)

#        pz_ebv = pz_ebv / pz_ebv.sum(dim='z')
#        pz /= pz.sum(dim='')
        pz = (pz_ebv*priors).sum(dim='EBV')

        # Had 2 galaxies of 500 being Nan..
        pz[np.isnan(pz).all(axis=1)] = 1.
        pz /= pz.sum(dim='z')
        pz[np.isnan(pz).all(axis=1)] = 1./float(len(pz.z))

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

    def load_models(self):
        D = {}

        for key,dep in self.job.depend.items():
            print('loading', key)

            # Since it also has to depend on the galaxy catalogs.
            if not key.startswith('pzcat_'):
                continue

            best_model = dep.get_store()['best_model'] #.reset_index()

            EBV = dep.f_mod.ab.config['EBV']
            D[EBV] = best_model

        return D

    def get_best_model(self, cat_in, D): #best_models):
        F = pd.concat(D, names=['EBV']).reset_index().set_index(['gal', 'EBV'])

        tosel = cat_in.loc[cat_in.groupby('gal').chi2.idxmin()]
        tosel = tosel.set_index(['gal', 'EBV'])

        nbest = tosel[[]].join(F)
        nbest = nbest.reset_index().set_index(['gal', 'band'])

        return nbest

    def run(self):
        if not self.config['use_pz']:
            cat_in = self.load_catalogs()
            pzcat = self.combine_cat(cat_in)
            pzcat = pzcat.set_index('gal')
        else:
            cat_in = self.load_catalogs()
            pdf_in = self.load_pdf()
            pzcat = self.combine_pz(pdf_in, cat_in)


        # Estimate best model..
        _models = self.load_models()
        best_model = self.get_best_model(cat_in, _models)

        print('here...')
        path_out = self.job.empty_file('default')
        store = pd.HDFStore(path_out, 'w')
        store['default'] = pzcat.stack()
        store['best_model'] = best_model
        store.close()
