#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import time
import numpy as np
from scipy.optimize import curve_fit

descr = {'ind': '',
         'synband': 'Which synthetic broad band to use',
         'scale_data': 'If scaling the data'}

class rband_adjust:
    version = 1.0
    config = {'ind': False, 'synband': 'subaru_r', 'scale_data': True}

    def fix_missing_data(self, cat_in):
        """Linear interpolation in magnitude space to replace missing data."""

        X = 455 + 10*np.arange(40)
        NB = list(map('NB{}'.format, X))

        def f_linear(x,a,b):
            return a*x + b

        pau_syn = cat_in.flux[NB].values
        miss_ids = np.isnan(pau_syn).any(axis=1)
        miss_rows = np.arange(len(cat_in))[miss_ids]

        for i in miss_rows:
            touse = ~np.isnan(pau_syn[i])

            yfit = np.log10(pau_syn[miss_rows[i]][touse]) if self.config['ind'] \
                   else np.log10(pau_syn[miss_rows[0]][touse])
                   
            popt,pcov = curve_fit(f_linear, X[touse], yfit)
            pau_syn[i,~touse] = 10**f_linear(X[~touse], *popt)

        return pau_syn

    def find_synbb(self, pau_syn, bbsyn_coeff):
        bbsyn_coeff = bbsyn_coeff[bbsyn_coeff.bb == self.config['synband']]
 
#        X = 455 + 10*np.arange(40)
        NB = list(map('NB{}'.format, 455 + 10*np.arange(40)))

        vec = bbsyn_coeff.pivot('bb', 'nb', 'val')[NB].values[0]
        synbb = np.dot(pau_syn, vec)

        return synbb

    def entry(self, cat_in, bbsyn_coeff):
        pau_syn = self.fix_missing_data(cat_in)
        synbb = self.find_synbb(pau_syn, bbsyn_coeff)

        from matplotlib import pyplot as plt

        ipdb.set_trace()



    def run(self):
        cat_in = self.input.galcat.result
        bbsyn_coeff = self.input.bbsyn_coeff.result
        self.output.result = self.entry(cat_in, bbsyn_coeff)
