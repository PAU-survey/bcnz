#!/usr/bin/env python

from IPython.core import debugger as ipdb
import numpy as np
import pandas as pd

descr = {'SN_lim': 'Limit in the estimated SN',
         'min_err': 'Minimum error of the measurement',
         'apply_mag': 'If applying the minimum error to magnitudes'}

class bcnz_fix_noise:
    """Add noise floor and select fluxes for the fitting."""

    version = 1.15
    config = {'SN_lim': -100., 'min_err': 0.03,
              'apply_mag': False}

    def limit_SN(self, cat):
        """Limit based on SN."""

        SN = cat['flux'] / cat['flux_err']
        SN_lim = self.config['SN_lim']

        cat['flux'] = cat['flux'][SN_lim < SN]
        cat['flux_err'] = cat['flux_err'][SN_lim < SN]

        return cat

    def _flux_minerr(self, cat):
        """Apply minimum error to fluxes."""

        # Adding the error only to the narrow-band. The broad bands already has
        # a minimum error in the input.
        for band in cat.flux.columns:
            if not band.startswith('NB'):
                continue

            add_err = cat['flux', band] * self.config['min_err']
            cat['flux_err', band] = np.sqrt(cat['flux_err', band]**2 + add_err**2)

    def _mag_minerr(self, cat):
        """Apply minimum error to magnitudes."""

        # Adding the error only to the narrow-band. The broad bands already has
        # a minimum error in the input.
        for band in cat.flux.columns:
            if not band.startswith('NB'):
                continue

            # Some the absolute values are suspicious...
            SN = np.abs(cat['flux', band]) / cat['flux_err', band]

            mag_err = 2.5*np.log10(1+1./SN)
            mag_err = np.sqrt(mag_err**2 + self.config['min_err']**2)
            flux_err = np.abs(cat['flux', band])*(10**(0.4*mag_err) - 1.)

            cat[('flux_err', band)] = flux_err

    def add_minerr(self, cat):
        """Add a minimum error in the flux measurements."""

        # For the comparison with Alex.
        if self.config['apply_mag']:
            self._mag_minerr(cat)
        else:
            self._flux_minerr(cat)
        

    def entry(self, cat):
        """Apply the different cuts."""

        self.add_minerr(cat)
        cat = self.limit_SN(cat)

        return cat

    def run(self):
        cat = self.input.input.result

        self.output.result = self.entry(cat)
