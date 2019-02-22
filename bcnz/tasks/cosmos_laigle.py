#!/usr/bin/env python
# encoding: UTF8

import os
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits

from IPython.core import debugger as ipdb

descr = {'rm_stars': 'If removing stars here.'}

class cosmos_laigle:
    """Interface from reading the COSMOS Laigle catalogue."""

    version = 1.09
    config = {'rm_stars': True}

    # Note, this code does *not* apply zero-points, following Alex
    # latest notebook. The minimum error is applied later.

    # ftp://ftp.iap.fr/pub/from_users/hjmcc/COSMOS2015/
    d = '~/data/cosmos'
#    fname_pdz ='pdz_cosmos2015_ip_MAG_AUTO_23.csv'
#    fname_other = 'COSMOS2015_Laigle_ip_MAG_AUTO_23.csv'
    fname_pdz = 'pdz_cosmos2015_v1.3.fits'
    fname_other = 'COSMOS2015_Laigle+_v1.1.fits'

    def load_cat(self):
        path = os.path.join(os.path.expanduser(self.d), self.fname_pdz)
        dat = Table.read(path, format='fits')
        df = dat.to_pandas()

        return df

    def read_cat(self):
        """Reads in the COSMOS catalog and convert to the internal
           format.
        """


        bands = ['cfht_u', 'subaru_B', 'subaru_V', 'subaru_r', 'subaru_i', 'subaru_z']
        mapD = {'cfht_u': 'U', 'subaru_B': 'B', 'subaru_V': 'V', 'subaru_r': 'R', 'subaru_i': 'I', \
                'subaru_z': 'ZN'}
        otherD = {'ID': 'id_laigle', 'RA': 'ra', 'DEC': 'dec'}

        flux_cols = [mapD[x] for x in bands]
        err_cols = ['D'+mapD[x] for x in bands]

#        # Strictly not needed, but there are many columns.
        cols = list(otherD.keys()) + flux_cols + err_cols
#        cat_in = pd.read_csv(path, sep=',', header=0, usecols=cols)
        cat_in = self.load_cat()[cols]

        flux = cat_in[flux_cols]
        flux.columns = bands
        flux_err = cat_in[err_cols]
        flux_err.columns = bands

        cat = pd.concat({'flux': flux, 'flux_err': flux_err}, axis=1)

        # Yes, different definition and rounding error.
        cat[cat < -99] = np.nan

        # Scale to PAU fluxes.
        ab_factor = 10**(0.4*26)
        cosmos_scale = ab_factor * 10**(0.4*48.6)
        cat *= cosmos_scale

#        for key_from, key_to in otherD.items():
#            cat[key_to] = cat_in[key_from]

        return cat


    def OLDother_cols(self, cat):
        """Add other columns not present in the first file."""

        # DELETE THIS SOON...
        path = os.path.join(os.path.expanduser(self.d), self.fname_other)

        fields = ['NUMBER', 'EBV', 'TYPE']
        other = pd.read_csv(path, usecols=fields)

        # Otherwise the hirarchical columns gives problems.
        other = other.rename(columns={'NUMBER': 'id_laigle', 'TYPE': 'type'})
        for key,val in other.items():
            cat[key] = val

    def other_cols(self, cat):

        # Is it not great? Here I will need to join with another file to get
        # the galaxy type.
        path = os.path.join(os.path.expanduser(self.d), self.fname_other)
        hdul = fits.open(path)

        def f(x):
            return hdul[1].data[x]

        # This is needed, since the input file is very large. Astropy did not
        # have a suitable option for selecting columns.
        D = {}
        for key in ['NUMBER', 'TYPE', 'ALPHA_J2000', 'DELTA_J2000']:
            D[key] = f(key)

        other = pd.DataFrame(D)

        # Otherwise the hirarchical columns gives problems.
        other = other.rename(columns={'NUMBER': 'id_laigle', 'TYPE': 'type'})
        for key,val in other.items():
            cat[key] = val

    def fixes(self, cat):
        if self.config['rm_stars']:
            cat = cat[cat.type == 0]

        # This is probably not needed... 
        del cat['type']

        return cat

    def entry(self):
        cat = self.read_cat()
        self.other_cols(cat)
        cat = self.fixes(cat)

        ipdb.set_trace()

        return cat

    def run(self):
        self.output.result = self.entry()
