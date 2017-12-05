#!/usr/bin/env python
# encoding: UTF8

import ipdb
import numpy as np
import pandas as pd
import xarray as xr

descr = {'use_pz': 'Combine the result using the full pdf',
         'flat_priors': True,
         'odds_lim': 'The limit withing to estimate the ODDS',
         'width_frac': 'Fraction (one-sided) outside pz_width',
         'Niter': 'Number of iteratios when estimating the priors'
        }

import sys
sys.path.append('/home/eriksen/source/bcnz/bcnz/tasks')
import libpzqual

class bcnz_comb_ext:
    """Combine the different extinction runs."""

    version = 1.19
    config = {'use_pz': False, 'flat_priors': True,
              'odds_lim': 0.01, 'width_frac': 0.01,
              'Niter': 1}

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


    def to_chi2(self, pdf_in, cat_in):
        """Convert the input back to the chi2 values."""

        pdf_in = pdf_in.to_xarray().pz
        chi2_min = cat_in.to_xarray().chi2
        pdf = np.clip(pdf_in, 1e-100, np.infty)
        chi2_in = -2.*np.log(pdf)

        chi2_tmp = chi2_in.min(dim=['z'])
        chi2 = chi2_in + (chi2_min - chi2_tmp)

        return chi2

    def _get_nz(self):
        """Estimate the number of redshift bins to not split a galaxy into different
           chunks.
        """

        sub = self.input.pzcat_0.get_store().select('pz', nrows=1e5)
        gal_ids = sub.index.get_level_values(0).unique()

        assert 1 < len(gal_ids), 'Internal error: loaded to few entries'

        nz = len(sub.loc[gal_ids[0]])

        return nz


    def _chi2_iterator(self):
        """Returns an iterator over the chi2 value for all the runs."""

        # Part of the complication here comes from not directly storing the
        # chi2 values.
        RD = {}

        ngal = 500
        nz = self._get_nz()
        chunksize = nz * ngal
        for key,dep in self.input.depend.items():
            # Since it also has to depend on the galaxy catalogs.
            if not key.startswith('pzcat_'):
                continue

            store = dep.get_store()
            Rpdf = store.select('pz', iterator=True, chunksize=chunksize)
            Rcat = store.select('default', iterator=True, chunksize=ngal)

            RD[key] = {'pdf': iter(Rpdf), 'cat': iter(Rcat)}

        rd_keys = list(RD.keys())
        rd_keys.sort()
        while True:
            chi2L = []
            for key in rd_keys:         
                print('Loading:', key)

                pdf_in = next(RD[key]['pdf'])
                cat_in = next(RD[key]['cat'])

                chi2_part = self.to_chi2(pdf_in, cat_in)
                chi2_part['run'] = key

                chi2L.append(chi2_part)
            
            chi2 = xr.concat(chi2L)
            chi2 = xr.concat(chi2L, dim='run')

            yield chi2

    def pzcat_part(self, chi2, priors):
        pz_runs = np.exp(-0.5*chi2)
        pz = (pz_runs*priors).sum(dim='run')

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


        # And then trying to get priors...
        A = pz_runs.isel_points(ref_id=range(len(izmin)), z=izmin)
        A = A / A.sum(dim='run')

        priors = A.sum(dim='points')

        return pzcat, priors


    def store_out(self):
        """Create the output store."""

        path = self.output.empty_file('default')
        store = pd.HDFStore(path, 'w')

        return store

    def init_priors(self):
        """Setting up the initial flat priors."""

        runs = list(filter(lambda x: x.startswith('pzcat_'), self.input.depend.keys()))
        priors = xr.DataArray(np.ones(len(runs)), dims='run', coords={'run': runs})

        return priors


    def combine_pdf(self):
        priors = self.init_priors()

        Rin, runs = self._chi2_iterator()
        store_out = self.store_out()

        Lpriors = []

        Niter = self.config['Niter']
        for i in range(Niter):
            for j,chi2 in enumerate(Rin):
                pzcat, priors = self.pzcat_part(chi2, priors)
                Lpriors.append(priors)

                if i == Niter - 1:
                    store_out.append('default', pzcat)

                if j == 2:
                    break

            priors = sum(Lpriors)
            priors /= priors.sum()

            print('priors')
            print(priors)


            ipdb.set_trace()


        return pzcat

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
            raise NotImplementedError('This is no longer in use..')
            cat_in = self.load_catalogs()
            pzcat = self.combine_cat(cat_in)
            pzcat = pzcat.set_index('ref_id')
        else:
#            cat_in = self.load_catalogs()
#            pdf_in = self.load_pdf()
            self.combine_pdf()


        # Estimate best model..
#        _models = self.load_models()
#        best_model = self.get_best_model(cat_in, _models)

        print('here...')
#        path_out = self.output.empty_file('default')
#        store = pd.HDFStore(path_out, 'w')
#        store['default'] = pzcat
#        store['best_model'] = best_model
#        store.close()
