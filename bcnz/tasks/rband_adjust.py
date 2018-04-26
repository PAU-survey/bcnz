#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import time
import numpy as np
from scipy.optimize import curve_fit

descr = {'ind': ''}

class rband_adjust:
    version = 1.0
    config = {'ind': False}

    def fix_missing_data(self, cat_in):
        """Linear interpolation in magnitude space to replace missing data."""

        X = 455 + 10*np.arange(40)
        NB = list(map('NB{}'.format, X))

        def f_linear(x,a,b):
            return a*x + b

        pau_syn = cat_in.flux[NB].values
        miss_ids = np.isnan(pau_syn).any(axis=1)
        miss_rows = np.arange(len(cat_in))[miss_ids]

        X = 455 + 10*np.arange(40)
        t2 = time.time()
        for i in miss_rows:
            touse = ~np.isnan(pau_syn[i])

            yfit = np.log10(pau_syn[miss_rows[i]][touse]) if self.config['ind'] \
                   else np.log10(pau_syn[miss_rows[0]][touse])
                   
            popt,pcov = curve_fit(f_linear, X[touse], yfit)
            pau_syn[i,~touse] = 10**f_linear(X[~touse], *popt)

        return pau_syn

    def entry(self, cat_in):
        pau_syn = self.fix_missing_data(cat_in)

        ipdb.set_trace()



    def run(self):
        cat_in = self.input.galcat.result
        self.output.result = self.entry(cat_in)
