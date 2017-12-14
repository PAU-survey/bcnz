#!/usr/bin/env python

from IPython.core import debugger
import numpy as np
import pandas as pd

descr = {'SN_lim': 'Limit in the estimated SN',
         'min_err': 'Minimum error of the measurement',
         'apply_mag': 'If applying the minimum error to magnitudes'}

class bcnz_select:
    """Selecting a subset of the fluxes to use in the fitting."""

    version = 1.09
    config = {'SN_lim': -100., 'min_err': 0.03,
              'apply_mag': False}

    def limit_SN(self, cat):
        """Limit based on SN."""

        SN = cat['flux'] / cat['flux_err']

        SN_lim = self.config['SN_lim']
        cat['flux'] = cat['flux'][SN_lim < SN]

        return cat

    def add_minerr(self, cat):
        """Add a minimum error in the flux measurements."""

        if self.config['apply_mag']:
            assert NotImplementedError('Not sure how todo this with negative magnitudes')
#        else:

        # Adding the error only to the narrow-band. The broad bands already has
        # a minimum error in the input.
        for band in cat.flux.columns:
            if not band.startswith('NB'):
                continue

            print(band)
            add_err = cat['flux', band] * self.config['min_err']
            cat['flux_err', band] = np.sqrt(cat['flux_err', band]**2 + add_err**2)

        return cat

    def entry(self, cat):
        """Apply the different cuts."""

        cat = self.limit_SN(cat)
        cat = self.add_minerr(cat)

        return cat

    def run(self):
        cat = self.input.input.result

        self.output.result = self.entry(cat)
