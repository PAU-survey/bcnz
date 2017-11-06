#!/usr/bin/env python

import ipdb
import numpy as np
import pandas as pd

descr = {'use_nondet': 'If including non-detections',
         'SN_lim': 'Limit in the estimated SN'}

class bcnz_select:
    """Selecting a subset of the fluxes to use in the fitting."""

    version = 1.02
    config = {'SN_lim': -100., 'SN_cap': 10000}

    def limit_SN(self, cat):
        """Limit based on SN."""

        SN = cat['flux'] / cat['flux_err']

        SN_lim = self.config['SN_lim']
        cat[SN < SN_lim] = np.nan

        SN_cap = self.config['SN_cap']
        new_SN = np.clip(SN, -np.infty, SN_cap)

        cat['flux_err'] = cat.flux / SN

        return cat

    def to_outformat(self, cat_in):
        """Convert to hirarchical columns and perform some weird temporary scaling."""
        
        cat_in = cat_in.reset_index()

        flux = cat_in.pivot('ref_id', 'band', 'flux')
        flux_err = cat_in.pivot('ref_id', 'band', 'flux_err')

        cat = pd.concat({'flux': flux, 'flux_err': flux_err}, axis=1)

        return cat

    def get_cat(self, cat):
        """Apply the different cuts."""

        if 2 < len(cat.columns):
            cat = cat[['flux', 'flux_err']].stack()

        cat = self.limit_SN(cat)
        cat = self.to_outformat(cat)

        return cat

    def run(self):
        cat = self.job.input.result

        self.job.result = self.get_cat(cat)
