#!/usr/bin/env python

import ipdb
import numpy as np
import pandas as pd

descr = {'use_nondet': 'If including non-detections',
         'SN_lim': 'Limit in the estimated SN'}

class bcnz_select:
    """Selecting a subset of the fluxes to use in the fitting."""

    version = 1.06
    config = {'SN_lim': -100., 'SN_cap': 10000}

    def limit_SN(self, cat):
        """Limit based on SN."""

        SN = cat['flux'] / cat['flux_err']

        SN_lim = self.config['SN_lim']
        cat['flux'] = cat['flux'][SN_lim < SN]


        SN_cap = self.config['SN_cap']
        new_SN = np.clip(SN, -np.infty, SN_cap)
        cat['flux_err'] = cat.flux / new_SN

        return cat

    def get_cat(self, cat):
        """Apply the different cuts."""

        cat = self.limit_SN(cat)

        return cat

    def run(self):
        cat = self.input.input.result

        self.output.result = self.get_cat(cat)
