#!/usr/bin/env python

from __future__ import print_function

import ipdb
import os
import time
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from scipy.interpolate import splrep, splev
from scipy.integrate import simps

descr = {
  'dz': 'Redshift steps',
  'ampl': 'Added amplitude',
  'EBV': 'Extinction coefficient',
  'ext_law': 'Extinction law'}

class emission_lines:
    """The model flux for the emission lines."""

    version = 1.057

    config = {'dz': 0.0005, 'ampl': 1e-16, 'EBV': 0.,
              'ext_law': 'SB_calzetti'}

    ratios = {
      'OII': 1.0,
      'OIII_1': 0.25*0.36,
      'OIII_2': 0.75*0.36,
      'Hbeta': 0.61,
      'Halpha': 1.77,
      'Lyalpha':  2.,
      'NII_1': 0.3 * 0.35 * 1.77, # Paper gave lines relative to Halpha.
      'NII_2': 0.35 * 1.77
      'SII_1': 0.35,
      'SII_2': 0.35
    }

    line_loc = {
      'OII': 3726.8,
      'OIII_1': 4959.,
      'OIII_2': 5007.,
      'Halpha': 6562.8,
      'Hbeta': 4861,
      'Lyalpha': 1215.7,
      'NII_1': 6548., 
      'NII_2': 6583.,
      'SII_1': 6716.44,
      'SII_2': 6730.82
    }


    def _filter_spls(self, filters):
        """Convert filter curves to splines."""

        splD = {}
        rconstD = {}
        for fname in filters.index.unique():
            sub = filters.ix[fname]
            splD[fname] = splrep(sub.lmb, sub.response)

            rconstD[fname] = simps(sub.response/sub.lmb, sub.lmb)

        return splD, rconstD

    def ext_spl(self, ext):
        """Spline for the extinction."""

        sub = ext[ext.ext_law == self.config['ext_law']]
        ext_spl = splrep(sub.lmb, sub.k)

        return ext_spl

    def _find_flux(self, z, f_spl, rconst, ext_spl):
        """Estimate the flux in the emission lines relative
           to the OII flux.
        """

        EBV = self.config['EBV']
        ampl = self.config['ampl']

        flux = {}
        for key, ratio in self.ratios.items():
            lmb = self.line_loc[key]*(1+z)
            y_f = splev(lmb, f_spl, ext=1) 

            k_ext = splev(lmb, ext_spl, ext=1) 
            y_ext = 10**(-0.4*EBV*k_ext)

            flux[key] = (ampl*ratio*y_f*y_ext) / rconst

        flux['lines'] = sum(flux.values())

        return flux

    def _new_fix(self, oldD, z, band, ext_law, EBV):
        """Dictionary suitable for concatination."""

        df = pd.DataFrame()

        # Ok, there is probably a much better way of doing this...
        F = pd.DataFrame(oldD)
        F.index = z
        F.index.name = 'z'
        F.columns.name = 'sed'

        F = F.stack()
        F.name = 'flux'
        F = F.reset_index()
        F = F.rename(columns={0: 'flux'})
        F['band'] = band
        F['ext'] = ext_law
        F['EBV'] = EBV

        return F


    def get_model(self, filtersD, rconstD, ext_spl):
        """Get the model fluxes for the emission lines."""

        ext_law = self.config['ext_law']
        bandL = filtersD.keys()

        z = np.arange(0., 2., self.config['dz'])

        df = pd.DataFrame()
        for fname, f_spl in filtersD.items():
            rconst = rconstD[fname]     

            part = self._find_flux(z, f_spl, rconst, ext_spl)
            part = self._new_fix(part, z, fname, ext_law, self.config['EBV'])

            df = df.append(part, ignore_index=True)

        return df


    def entry(self, filters, extinction):
        filtersD, rconstD = self._filter_spls(filters)
        ext_spl = self.ext_spl(extinction)
        model = self.get_model(filtersD, rconstD, ext_spl)

        return model

    def run(self):
        filters = self.input.filters.result
        extinction = self.input.extinction.result

        self.output.result = self.entry(filters, extinction)
