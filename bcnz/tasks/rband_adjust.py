#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import time
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

descr = {'ind': '',
         'synband': 'Which synthetic broad band to use',
         'scale_data': 'If scaling the data'}

class rband_adjust:
    version = 1.0
    config = {'ind': False, 'synband': 'R', 'scale_data': True}

    def fix_missing_data(self, cat_in):
        """Linear interpolation in magnitude space to replace missing data."""

        X = 455 + 10*np.arange(40)
        NB = list(map('NB{}'.format, X))

        def f_linear(x,a,b):
            return a*x + b

        pau_syn = cat_in.flux[NB].values

        # Not exactly good...
        pau_syn[pau_syn < 0.] = np.nan

        miss_ids = np.isnan(pau_syn).any(axis=1)
        miss_rows = np.arange(len(cat_in))[miss_ids]

        for i in miss_rows:
            touse = ~np.isnan(pau_syn[i])

            yfit = np.log10(pau_syn[miss_rows[i]][touse]) if self.config['ind'] \
                   else np.log10(pau_syn[miss_rows[0]][touse])
            
            try:       
                popt,pcov = curve_fit(f_linear, X[touse], yfit)
                pau_syn[i,~touse] = 10**f_linear(X[~touse], *popt)
            except ValueError:
                ipdb.set_trace()


        return pau_syn

    def find_synbb(self, pau_syn, bbsyn_coeff):
        bbsyn_coeff = bbsyn_coeff[bbsyn_coeff.bb == self.config['synband']]
 
#        X = 455 + 10*np.arange(40)
        NB = list(map('NB{}'.format, 455 + 10*np.arange(40)))

        vec = bbsyn_coeff.pivot('bb', 'nb', 'val')[NB].values[0]
        synbb = np.dot(pau_syn, vec)

        return synbb

    def scale_fluxes(self, cat_in, obs2syn):
        """Scale the fluxes between the systems."""

        # Here we scale the narrow bands without adding additional
        # errors. This might not be the most optimal.
        cat_out = cat_in.copy() 
        for band in cat_in.flux.columns:
            if not band.startswith('NB'):
                continue

            cat_out['flux', band] *= obs2syn
            cat_out['flux_err', band] *= obs2syn

        return cat_out

    def entry(self, cat_in, bbsyn_coeff):
        pau_syn = self.fix_missing_data(cat_in)
        synbb = self.find_synbb(pau_syn, bbsyn_coeff)

        synband = self.config['synband']
        obs2syn = cat_in.flux[synband] / synbb
        cat_out = self.scale_fluxes(cat_in, obs2syn)

        return obs2syn, cat_out


    def run(self):
        cat_in = self.input.galcat.result
        bbsyn_coeff = self.input.bbsyn_coeff.result

        obs2syn, cat_out = self.entry(cat_in, bbsyn_coeff)

        # Yes, I should really extend xdolphin to handle
        # this pattern.
        path = self.output.empty_file('default')
        store = pd.HDFStore(path, 'w')
        store['ratio'] = obs2syn
        store.append('default', cat_out)
        store.close()
