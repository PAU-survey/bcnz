#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import os
import sys
import numpy as np
import pandas as pd


# Yes, this is not exactly nice ...
sys.path.append('/home/eriksen/code/bcnz/bcnz/tasks')
sys.path.append('/nfs/pic.es/user/e/eriksen/code/bcnz/bcnz/tasks')
sys.path.append(os.path.expanduser('~/Dropbox/pauphotoz/bcnz/bcnz/tasks'))

import chi2_comb
import libpzqual

descr = {'odds_lim': 'Limit within to estimate the ODDS',
         'width_frac': 'Fraction used when estimating the pz_width'}


class bcnz_direct(chi2_comb.chi2_comb):
    """Directly constructing the catalogs without priors."""

    version = 1.02
    config = {'odds_lim': 0.0035,
              'width_frac': 0.01}


    def entry(self):
        ipdb.set_trace()

    def calc_pzcat(self, chi2):
        pz = np.exp(-0.5*chi2)
        pz_norm = pz.sum(dim=['chunk', 'z'])
        pz_norm = pz_norm.clip(1e-200, np.infty)

        pz = pz / pz_norm

        pz = pz.sum(dim='chunk')

        # Most of this should be moved into the libpzqual
        # library.
        pz = pz.rename({'ref_id': 'gal'})
        zb = libpzqual.zb(pz)
        odds = libpzqual.odds(pz, zb, self.config['odds_lim'])
        pz_width = libpzqual.pz_width(pz, zb, self.config['width_frac'])
        zb_mean = libpzqual.zb_bpz2(pz)

        cat = pd.DataFrame()
        cat['zb'] = zb.values
        cat['odds'] = odds.values
        cat['pz_width'] = pz_width
        cat['zb_mean'] = zb_mean.values

        cat.index = pz.gal.values
        cat.index.name = 'ref_id'

        cat['chi2'] = chi2.min(dim=['chunk', 'z']).sel(ref_id=cat.index)

        # These are now in the "libpzqual" file. I could
        # consider moving them here..
        chi2_min = chi2.min(dim=['chunk', 'z'])
        cat['qual_par'] = (chi2_min*pz_width).values

        odds0p2 = libpzqual.odds(pz, zb, self.config['odds_lim'])
        cat['Qz'] = (chi2_min*pz_width / odds0p2.values).values

        # We need the chunk which contribute most to the redshift
        # peak..
        iz = pz.argmin(dim='z')
        points = chi2.isel_points(ref_id=range(len(chi2.ref_id)), z=iz)
        cat['best_chunk'] = points.argmin(dim='chunk')

        return cat

    def run(self):
        files = self.get_files()
        gen = self.get_chi2(files)

        path = self.output.empty_file('default')
        store = pd.HDFStore(path)
        for i,chi2 in enumerate(gen):
            print('batch', i)
            pzcat = self.calc_pzcat(chi2)

            store.append('default', pzcat)
