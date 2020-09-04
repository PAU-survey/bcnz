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

def _getcoef(WNB, WBB):
    NBNB= WNB.dot(WNB.T)
    BBNB= WNB.dot(WBB)
    iNBNB = np.linalg.inv(NBNB)

    return iNBNB.dot(BBNB)

def nb2bb(filt, broad_band):
    """The coefficients between narrow and broad bands which Alex used.
       Args:
           filt (dataframe): Filter response curves.
           broad_band (str): Which broad band to estimate the coefficient for.
    """

    ll = np.arange(3000,10000,5)
   
    NB_filt = np.array([[filt.loc['pau_nb%s'%str(x)].lmb.values,filt.loc['pau_nb%s'%str(x)].response.values] for x in np.arange(455,850,10)])
    WNB = np.array([interp1d(NB_filt[i,0], NB_filt[i,1]/NB_filt[i,0], bounds_error=False, fill_value=(0,0))(ll) for i in range(40)])
    WNB = np.array([x/np.sqrt(x.dot(x)) for x in WNB]) # Normalize

    BB_filt = filt.loc[broad_band].values.T
    WBB = interp1d(BB_filt[0],BB_filt[1]/BB_filt[0],  bounds_error=False, fill_value=(0,0))(ll)
    WBB = WBB/np.sqrt(WBB.dot(WBB)) # Normalize

    NBNB= WNB.dot(WNB.T)
    iNBNB = np.linalg.inv(NBNB)

    coeff = _getcoef(WNB,WBB)

    coeff = coeff / np.sum(coeff)
    keys = ['pau_nb'+str(x)for x in np.arange(455,850,10)]
    coeff = pd.DataFrame(dict(zip(keys,coeff)), index=[0])
   
    # Fix the format to be more similar to what we use elsewhere...
    xcoeff = pd.DataFrame({'nb': coeff.columns, 'val': coeff.values[0]})
    xcoeff['bb'] = broad_band

    return xcoeff
