#!/usr/bin/env python
# encoding: UTF8

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append('/nfs/pic.es/user/e/eriksen/source/paudm-nightly/paudm/nightly/cal2')

import ebv_skypos

class fix_extinction:
    version = 1.0
    config = {}


    def entry(self, cat):
        """Correct for the extinction in the narrow and broad bands."""

        # The correction factors.
        ext_fac = pd.read_csv('~/data/calib/ebv_corr_fac_v2.csv')
        ext_fac = ext_fac.set_index('band')
        rderiv = ext_fac.loc[cat.flux.columns].rderiv
        
        base = Path('/nfs/pic.es/user/e/eriksen/source/paudm-resources/paudm/resources/extinct')
        path = base / 'HFI_CompMap_ThermalDustModel_2048_R1.20.fits'

        EBV = ebv_skypos.ebv_skypos_wrapper(path, cat.ra, cat.dec)
        fac = 1 - pd.DataFrame(np.outer(EBV, rderiv), index=cat.index, columns=rderiv.index)
        
        cat = cat.copy()
        cat['flux'] = fac * cat.flux
        cat['flux_err'] = fac * cat.flux_err

        return cat

    def run(self):
        cat = self.input.input.result
        self.output.result = self.entry(cat)
