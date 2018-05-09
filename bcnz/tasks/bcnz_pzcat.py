#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import time
import numpy as np
import xarray as xr

from matplotlib import pyplot as plt

descr = {}

class bcnz_pzcat:
    """Catalogs for the photometric redshifts."""

    version = 1.0
    config = {}

    def entry(self, chi2):
#        chi2 = chi2.to_xarray().chi2

        pz = np.exp(-0.5*chi2)
        pz_norm = pz.sum(dim=['chunk', 'z'])
        pz_norm = pz_norm.clip(1e-200, np.infty)

        pz = pz / pz_norm

        hmm = pz.sum(dim='ref_id')


        ipdb.set_trace()

    def run(self):
        #chi2 = self.input.chi2.result
        path = '/home/eriksen/tmp/p540/chi2_v1.h5'

        chi2 = xr.open_dataset(path).chi2

        self.entry(chi2)
