#!/usr/bin/env python
# encoding: UTF8

import os
import numpy as np
import pandas as pd

from IPython.core import debugger as ipdb

descr = {'rm_stars': 'If removing stars here.'}

class cosmos_laigle:
    """Interface from reading the COSMOS Laigle catalogue."""

    version = 1.03
    config = {'rm_stars': True}

    # Note, this code does *not* apply zero-points, following Alex
    # latest notebook. The minimum error is applied later.

    d = '~/data/cosmos'
    fname_pdz ='pdz_cosmos2015_ip_MAG_AUTO_23.csv'
    fname_other = 'COSMOS2015_Laigle_ip_MAG_AUTO_23.csv'

    def read_cat(self):
        """Reads in the COSMOS catalog and convert to the internal
           format.
        """

        path = os.path.join(os.path.expanduser(self.d), self.fname_pdz)

        bands = ['cfht_u', 'subaru_B', 'subaru_V', 'subaru_r', 'subaru_i', 'subaru_z']
        mapD = {'cfht_u': 'U', 'subaru_B': 'B', 'subaru_V': 'V', 'subaru_r': 'R', 'subaru_i': 'I', \
                'subaru_z': 'ZN'}
        otherD = {'ID': 'id_laigle', 'RA': 'ra', 'DEC': 'dec'}

        flux_cols = [mapD[x] for x in bands]
        err_cols = ['D'+mapD[x] for x in bands]

        # Strictly not needed, but there are many columns.
        cols = list(otherD.keys()) + flux_cols + err_cols
        cat_in = pd.read_csv(path, sep=',', header=0, usecols=cols)

        flux = cat_in[flux_cols]
        flux.columns = bands
        flux_err = cat_in[flux_cols]
        flux_err.columns = bands

        cat = pd.concat({'flux': flux, 'flux_err': flux_err}, axis=1)

        # Yes, different definition and rounding error.
        cat[cat < -99] = np.nan
        for key_from, key_to in otherD.items():
            cat[key_to] = cat_in[key_from]

        return cat

    def other_cols(self, cat):
        """Add other columns not present in the first file."""

        path = os.path.join(os.path.expanduser(self.d), self.fname_other)

        fields = ['NUMBER', 'EBV', 'TYPE']
        other = pd.read_csv(path, usecols=fields)

        # Otherwise the hirarchical columns gives problems.
        other = other.rename(columns={'NUMBER': 'id_laigle', 'TYPE': 'type'})
        for key,val in other.items():
            cat[key] = val

    def entry(self):
        cat = self.read_cat()
        self.other_cols(cat)

        if self.config['rm_stars']:
            cat = cat[cat.type == 0]
 
        return cat

    def run(self):
        self.output.result = self.entry()
