#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import os
import sys
import time
import numpy as np
import pandas as pd
import xarray as xr

from matplotlib import pyplot as plt

# Yes, this is not exactly nice ...
sys.path.append('/home/eriksen/code/bcnz/bcnz/tasks')
sys.path.append(os.path.expanduser('~/Dropbox/pauphotoz/bcnz/bcnz/tasks'))
import libpzqual

descr = {'odds_lim': 'Limit within to estimate the ODDS',
         'width_frac': 'Fraction used when estimating the pz_width'}

class bcnz_pzcat:
    """Catalogs for the photometric redshifts."""

    version = 1.0
    config = {'odds_lim': 0.0035,
              'width_frac': 0.01,
              'priors': False}

    def entry(self, chi2):
        pz = np.exp(-0.5*chi2)
        pz_norm = pz.sum(dim=['chunk', 'z'])
        pz_norm = pz_norm.clip(1e-200, np.infty)

        pz = pz / pz_norm

        if self.config['priors']:
            prior_chunk = pz.sum(dim=['ref_id', 'z'])
            prior_chunk = prior_chunk / prior_chunk.sum()

            pz = pz*prior_chunk
            pz_norm = pz.sum(dim=['chunk', 'z'])
            pz = pz / pz_norm


        pz = pz.sum(dim='chunk')

        # Most of this should be moved into the libpzqual
        # library.
        pz = pz.rename({'ref_id': 'gal'})
        zb = libpzqual.zb(pz)
        odds = libpzqual.odds(pz, zb, self.config['odds_lim'])
        pz_width = libpzqual.pz_width(pz, zb, self.config['width_frac'])

        cat = pd.DataFrame()
        cat['zb'] = zb.values
        cat['odds'] = odds.values
        cat['pz_width'] = pz_width
        cat.index = pz.gal.values
        cat.index.name = 'ref_id'

        return cat

    def run(self):
        #chi2 = self.input.chi2.result
        path = '/home/eriksen/tmp/p540/chi2_v1.h5'

        chi2 = xr.open_dataset(path).chi2

        self.output.result = self.entry(chi2)
