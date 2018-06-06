#!/usr/bin/env python

from IPython.core import debugger as ipdb
import sys
import time
import numpy as np
import pandas as pd
import xarray as xr

sys.path.append('/home/eriksen/code/bcnz/bcnz/tasks')
import libpzqual

class new_comb_chunks:
    """Version which combined the different chunks."""

    version = 1.0
    config = {}

    def get_chi2(self, files, i, ngal, nz):
        """Estimate the chi2 for a chunk of galaxies."""

        L = []
        for j,hdf_file in enumerate(files):
            cat = hdf_file.select('default', start=i*ngal, stop=ngal*(i+1))
            pz = hdf_file.select('pz', start=ngal*nz*i, stop=ngal*nz*(i+1))

            bestchi2 = cat.chi2.to_xarray()
            pz = pz.to_xarray().pz
            if not len(pz):
                return []

            pz /= pz.max(dim='z')
            L.append(bestchi2 - 2.*np.log(pz))

        chi2_all = xr.concat(L, dim='chunk')
        chi2_all.coords['chunk'] = range(len(files))

        return chi2_all

    def get_pz(self, files):
        """Estimate the p(z)."""

        # Just because of how the indexes are stored..
        nz = len(files[0].pz.index.get_level_values(1).unique())

        i = 0
        ngal = 50
        K = []
        while True:
            chi2_all = self.get_chi2(files, i, ngal, nz)
            if not len(chi2_all):
                break

            chi2_all.coords['chunk'] = range(len(files))
            pz_marg = np.exp(-0.5*(chi2_all - chi2_all.min(dim=['z', 'chunk'])))
            pz_marg = pz_marg.sum(dim='chunk')
            pz_marg /= pz_marg.sum(dim='z')

            K.append(pz_marg)
            i += 1

        pz = xr.concat(K, dim='ref_id')

        return pz

    def get_cat(self, pz):
        """Create the photo-z catalogue."""

        pz = pz.rename({'ref_id': 'gal'})

        cat = pd.DataFrame({'zb': libpzqual.zb(pz)})
        cat['pz_width'] = libpzqual.pz_width(pz, cat.zb, 0.01)
        cat.index = pz.gal

        return cat

    def entry(self):
        nrcats = len(list(filter(lambda x: x.startswith('pzcat_'), dir(self.input))))
        files = [self.input.depend['pzcat_{}'.format(x)].get_store() for x in range(nrcats)]

        t1 = time.time()
        pz = self.get_pz(files)
        print('time', time.time() - t1)

        cat = self.get_cat(pz)

        return cat

    def run(self):
        self.output.result = self.entry()
