#!/usr/bin/env python
# encoding: UTF8

from __future__ import print_function

from IPython.core import debugger as ipdb
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from scipy.interpolate import splrep, splev, interp1d
from scipy.integrate import trapz, simps

descr = {
  'broad_band': 'Get the coefficients to go from NB to this BB.'
}

class NB2BB:
    """The coefficients between narrow and broad bands which Alex used."""

    version = 1.02
    config = {\
      'broad_band': 'subaru_r'
    }

    # Note: This class is here temporary to exactly test the setup Alex
    # was using...

    def getcoef(self, WNB, WBB):
        NBNB= WNB.dot(WNB.T)
        BBNB= WNB.dot(WBB)
        iNBNB = np.linalg.inv(NBNB)

        return iNBNB.dot(BBNB)

    def calc_coeff(self, filt):
        ll = np.arange(3000,10000,5)
       
        #ipdb.set_trace() 
        #NB_filt = np.array([[filt.ix['pau_nb%s'%str(x)].lmb.values,filt.ix['pau_nb%s'%str(x)].response.values] for x in np.arange(455,850,10)])
        NB_filt = np.array([[filt.loc['pau_nb%s'%str(x)].lmb.values,filt.loc['pau_nb%s'%str(x)].response.values] for x in np.arange(455,850,10)])
        WNB = np.array([interp1d(NB_filt[i,0], NB_filt[i,1]/NB_filt[i,0], bounds_error=False, fill_value=(0,0))(ll) for i in range(40)])
        WNB = np.array([x/np.sqrt(x.dot(x)) for x in WNB]) # Normalize

        BB_name = self.config['broad_band']
        #BB_filt = filt.ix[BB_name].values.T
        BB_filt = filt.loc[BB_name].values.T
        WBB = interp1d(BB_filt[0],BB_filt[1]/BB_filt[0],  bounds_error=False, fill_value=(0,0))(ll)
        WBB = WBB/np.sqrt(WBB.dot(WBB)) # Normalize

        NBNB= WNB.dot(WNB.T)
        iNBNB = np.linalg.inv(NBNB)

        coeff = self.getcoef(WNB,WBB)

        coeff = coeff / np.sum(coeff)
        keys = ['pau_nb'+str(x)for x in np.arange(455,850,10)]
        coeff = pd.DataFrame(dict(zip(keys,coeff)), index=[0])
       
        # Fix the format to be more similar to what we use elsewhere...
        xcoeff = pd.DataFrame({'nb': coeff.columns, 'val': coeff.values[0]})
        xcoeff['bb'] = self.config['broad_band']

        return xcoeff

    def run(self):
        filt = self.input.filters.result
        ipdb.set_trace()

        coeff = self.calc_coeff(filt)

        self.output.result = coeff
