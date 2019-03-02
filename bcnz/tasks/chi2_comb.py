#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import time
import numpy as np
import pandas as pd
import xarray as xr

descr = {}

class chi2_comb:
    """Combine the chi^2 from different photo-z runs."""

    version = 1.01
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

        # Number of redshift bins. Just because of how the
        # indices are stored.
        nz = len(files[0].pz.index.get_level_values(1).unique())

        i = 0
        ngal = 50 # Don't change this.. Seriously!!
        while True:
            chi2_all = self._chi2_part(files, i, ngal, nz)
            if not len(chi2_all):
                break

            chi2_all.coords['chunk'] = range(len(files))

            i += 1

            yield chi2_all


    def get_files(self):
        """Interface for opening the files."""

        nrcats = len(list(filter(lambda x: x.startswith('pzcat_'), dir(self.input))))

        # This part is more verbose since I need to debug...
        files = []
        for x in range(nrcats):
            job = self.input.depend['pzcat_{}'.format(x)]
            print('x', x)
            files.append(job.get_store())

        #files = [self.input.depend['pzcat_{}'.format(x)].get_store() for x in range(nrcats)]

        return files
    
    def entry(self):
        t1 = time.time()
        chi2 = self.get_chi2(files)
        print('time', time.time() - t1)

        return chi2

    def run(self):
        files = self.get_files()
        assert len(files), 'No input files'

        K = []
        for chi2 in iter(self.get_chi2(files)):
            print('Appending one..')
            K.append(chi2)

        # Be nice..
        for fb in files:
            fb.close()

        self.output.result = xr.concat(K, dim='ref_id')
