#!/usr/bin/env python

from __future__ import print_function

import os
import time
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from scipy.interpolate import splrep, splev
from scipy.integrate import simps

class emission_model:
    version = 1.051

    config = {'dz': 0.0005, 'ampl': 1e-16 }

    ratios = {
      'OII': 1.0,
      'OIII_1': 0.25*0.36,
      'OIII_2': 0.75*0.36,
      'Hbeta': 0.61,
      'Halpha': 1.77,
      'Lyalpha':  2.,
      'NII_1': 0.3 * 0.35 * 1.77, # Relative to Halpha.
      'NII_2': 0.35 * 1.77 # Relative to Halpha.
    }

    line_loc = {
      'OII': 3726.8,
      'OIII_1': 4959.,
      'OIII_2': 5007.,
      'Halpha': 6562.8,
      'Hbeta': 4861,
      'Lyalpha': 1215.7,
      'NII_1': 6548., 
      'NII_2': 6583.
    }


    def _to_spl(self, filters):
        """Convert filter curves to splines."""

        splD = {}
        rconstD = {}
        for fname in filters.index.unique():
            sub = filters.ix[fname]
            splD[fname] = splrep(sub.lmb, sub.y)

            rconstD[fname] = simps(sub.y/sub.lmb, sub.lmb)

        return splD, rconstD

    def find_flux(self, filters, ext):
        """Estimate the flux in the emission lines relative
           to the OII flux.
        """

        # This code rely on having continues filter curves.
        splD, rconstD = self._to_spl(filters)

        yD = {}
        t1 = time.time()
        z = np.arange(0., 2., self.config['dz'])
        for fname,spl in splD.iteritems():

            y = np.zeros_like(z)
            for key, ratio in self.ratios.iteritems():
                lmb0 = self.line_loc[key]
                y += ratio*splev(lmb0*(1.+z), spl, ext=1) 

            y /= rconstD[fname]
            # To have the same format as other SEDs.
            key = (fname, 'lines', 'none') #'/{}/lines'.format(fname)
            yD[key] = y

        flux = pd.DataFrame(yD)

        # This is to avoid having many orders of magnitude difference than
        # the normal SEDs.
        flux = self.config['ampl'] * flux

        t2 = time.time()
        print('time used...', t2-t1)

        # Perhaps not the best, but simple.
        flux['z'] = z

        return flux

    def run(self):
        filters = self.job.filters.result
        ext = self.job.ext.result if hasattr(self.job, 'ext') else None

        self.job.result = self.find_flux(filters, ext)
