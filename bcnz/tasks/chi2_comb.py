#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import time
import numpy as np
import xarray as xr

descr = {}

class chi2_comb:

    version = 1.0
    config = {}

    def _chi2_part(self, files, i, ngal, nz):
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

    def get_chi2(self, files):
        """Estimate the chi2 array."""

        # Just because of how the indexes are stored..
        nz = len(files[0].pz.index.get_level_values(1).unique())

        i = 0
        ngal = 50
        K = []
        while True:
            chi2_all = self._chi2_part(files, i, ngal, nz)
            if not len(chi2_all):
                break

            chi2_all.coords['chunk'] = range(len(files))
            K.append(chi2_all)

            i += 1

        chi2 = xr.concat(K,dim='ref_id')
        return chi2

    def entry(self):
        nrcats = len(list(filter(lambda x: x.startswith('pzcat_'), dir(self.input))))
        files = [self.input.depend['pzcat_{}'.format(x)].get_store() for x in range(nrcats)]

        t1 = time.time()
        chi2 = self.get_chi2(files)
        print('time', time.time() - t1)

        chi2 = chi2.to_dataframe('chi2')

        return chi2

    def run(self):
        self.output.result = self.entry()

