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
        for key,dep in self.input.depend.items():
            # Since it also has to depend on the galaxy catalogs.
            if not key.startswith('pzcat_'):
                continue

            D[key] = dep.result

        cat_out = pd.concat(D, axis=0, names=['run']).reset_index()

        return cat_out

    def load_pdf(self):
        print('Start loading pz..')

        df = pd.DataFrame()
        for key,dep in self.input.depend.items():
            print('loading', key)

            # Since it also has to depend on the galaxy catalogs.
            if not key.startswith('pzcat_'):
                continue

            part = dep.get_store()['pz'].reset_index()
#            EBV = self.job.depend[key].model.ab.config['EBV']
#            part['EBV'] = EBV
            part['run'] = key

            df = df.append(part, ignore_index=True)

        # This part will complain if the EBV is not unique among the runs...
        pz = df.set_index(['ref_id', 'z', 'run']).to_xarray().pz    

        print('Finish loading pz..')

        return pz


    def combine_pz(self, pz_in, cat_in):
        chi2_min = cat_in[['ref_id', 'run', 'chi2']].set_index(['ref_id', 'run']).to_xarray().chi2

        pz = np.clip(pz_in, 1e-100, np.infty)
        chi2_in = -2.*np.log(pz)

        chi2_tmp = chi2_in.min(dim=['z'])
        chi2 = chi2_in + (chi2_min - chi2_tmp)

        pz_runs = np.exp(-0.5*chi2)

#        pz = (pz_ebv*priors).sum(dim='EBV')
        pz = (pz_runs).sum(dim='run')

        # Had 2 galaxies of 500 being Nan..
        pz[np.isnan(pz).all(axis=1)] = 1.
        pz /= pz.sum(dim='z')
        pz[np.isnan(pz).all(axis=1)] = 1./float(len(pz.z))

        izmin = pz.argmax(dim='z')
        zb = pz.z[izmin]

        pzcat = pd.DataFrame(index=pz.ref_id)
        pz = pz.rename({'ref_id': 'gal'})
        pzcat['odds'] = libpzqual.odds(pz, zb, self.config['odds_lim'])
        pzcat['pz_width'] = libpzqual.pz_width(pz, zb, self.config['width_frac'])
        pzcat['zb'] = zb

        return pzcat

    def _pdf_iterator(self):

        RD = {}
        chunksize = 1000.
        for key,dep in self.input.depend.items():
            # Since it also has to depend on the galaxy catalogs.
            if not key.startswith('pzcat_'):
                continue

            store = dep.get_store()
            Rcat = store.select('default', iterator=True, chunksize=chunksize)
            Rpdf = store.select('pz', iterator=True, chunksize=chunksize)

            RD[key] = {'cat': iter(Rcat), 'pdf': iter(Rpdf)}

        rd_keys = list(RD.keys())
        rd_keys.sort()
        while True:
            part = {}

            for key in rd_keys:         
                cat = next(RD[key]['cat'])
                pdf = next(RD[key]['pdf'])

                part[key] = (cat, pdf)

            yield part

    def combine_pdf(self):
        R = self._pdf_iterator()


        ipdb.set_trace()

    def combine_cat(self, cat_in):

        cat_out = cat_in.loc[cat_in.groupby('ref_id').chi2.idxmin()]

        return cat_out

    def load_models(self):
        D = {}

        for key,dep in self.input.depend.items():
            print('loading', key)

            # Since it also has to depend on the galaxy catalogs.
            if not key.startswith('pzcat_'):
                continue

            best_model = dep.get_store()['best_model']
            D[key] = best_model

        return D

    def get_best_model(self, cat_in, D):
        F = pd.concat(D, names=['run']).reset_index()

        tosel = cat_in.loc[cat_in.groupby('ref_id').chi2.idxmin()]

        nbest = tosel[['run', 'ref_id']].merge(F, on=['run', 'ref_id'])
        nbest = nbest.set_index(['ref_id', 'band'])

        return nbest


    def run(self):
        if not self.config['use_pz']:
            cat_in = self.load_catalogs()
            pzcat = self.combine_cat(cat_in)
            pzcat = pzcat.set_index('ref_id')
        else:
#            cat_in = self.load_catalogs()
#            pdf_in = self.load_pdf()
            pzcat = self.combine_pdf()


        # Estimate best model..
        _models = self.load_models()
        best_model = self.get_best_model(cat_in, _models)

        print('here...')
        path_out = self.output.empty_file('default')
        store = pd.HDFStore(path_out, 'w')
        store['default'] = pzcat
        store['best_model'] = best_model
        store.close()
