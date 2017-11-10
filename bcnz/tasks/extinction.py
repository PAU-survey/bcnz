#!/usr/bin/env python
# encoding: UTF8

import numpy as np
import pandas as pd
from scipy.interpolate import splrep, splev, splint

from astropy import units
from specutils import extinction as extmod

descr = {
  'rv_cal': 'The Rv factor in k(lambda) for Calzetti',
  'rv_rest': 'The Rv factor in k(lambda) for the rest'
}

class extinction:
    """Estimate the expected extinction in different filters."""

    version = 1.24
    config = {'rv_cal': 4.05, 'rv_rest': 3.1}

    def _calzetti(self, lmb_in):
        """Calculate k(lambda) in Calzetti et.al."""

        lmb = lmb_in / 1e4

        rv = self.config['rv_cal']
        k_high = 2.659*(-1.857+1.040/lmb) + rv
        k_low = 2.659*(-2.156+1.509/lmb-0.198/lmb**2+0.011/lmb**3) + rv

        k = -np.ones_like(lmb)

        islow = lmb < 0.63
        k[islow] = k_low[islow]
        k[~islow] = k_high[~islow]

        assert not (k == -1).any(), 'Internal error'

        return k

    def _astropy(self, lmb_in):

        models = ['ccm89', 'od94', 'gcc09', 'f99', 'fm07','wd01','d03']

        lmb = lmb_in*units.angstrom

        r_v = self.config['rv_rest']
        a_v = r_v

        part = pd.DataFrame()
        for model in models:
            part[model] = extmod.extinction(lmb, model=model, a_v=a_v, r_v=r_v)

        return part

    def get_k(self):
        lmb = np.linspace(1200, 22000, 1000)

        df = pd.DataFrame()
        df['lmb'] = lmb
        df['calzetti'] = self._calzetti(lmb)

        part = self._astropy(lmb)
        df = pd.concat([df, part], axis=1)

        return df

    def run(self):
        self.job.result = self.get_k()
