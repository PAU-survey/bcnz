#!/usr/bin/env python
# encoding: UTF8

from __future__ import print_function

import ipdb
import time
import numpy as np
import pandas as pd

from scipy.interpolate import splrep, splev
from scipy.integrate import trapz, simps

descr = {
  'zmax_ab': 'Maximum redshift in the AB files',
  'dz_ab': 'Redshift resolution in the AB files',
  'int_dz': 'Resolution when integrating',
  'int_method': 'Integration method',
  'ext_law': 'Extinction law',
  'EBV': 'Extinction amplitude'
}

class ab_cont:
    """The model fluxes for the continuum."""

    version = 1.12
    config = {\
      'zmax_ab': 12.,
      'dz_ab': 0.001,
      'int_dz': 1.,
      'int_method': 'simps',
      'ext_law': 'SB_calzetti',
      'EBV': 0.0
    }

    def r_const(self, filters):
        """Normalization factor for each filter."""

        # 2.5*log10(clight_AHz) = 46.19205, which you often see applied to
        # magnitudes.
        clight_AHz = 2.99792458e18

        r_const = pd.Series()
        fL = filters.index.unique()
        for fname in fL:
            sub = filters.ix[fname]
            r_const[fname] = 1./simps(sub.response / sub.lmb, sub.lmb) / clight_AHz

        return r_const

    def sed_spls(self, seds):
        """Create a spline of all the SEDs."""

        sedD = {}
        for sed in seds.index.unique():
            sub_sed = seds.ix[sed]
            spl_sed = splrep(sub_sed.lmb, sub_sed.response)

            sedD[sed] = spl_sed

        return sedD

    def ext_spl(self, ext):
        """Spline for the extinction."""

        ext_spl = splrep(ext.lmb, ext[self.config['ext_law']])

        return ext_spl


    def calc_ab(self, filters, seds, ext, r_const):
        """Estimate the fluxes for all filters and SEDs."""

        sedD = self.sed_spls(seds)
        ext_spl = self.ext_spl(ext)
        z = np.arange(0., self.config['zmax_ab'], self.config['dz_ab'])

        # Test...
        df = pd.DataFrame()

        int_method = self.config['int_method']
        a = 1./(1+z)
        for i,band in enumerate(filters.index.unique()):
            print('# band', i, 'band', band)

            sub_f = filters.ix[band]

            # Define a higher resolution grid.
            _tmp = sub_f.lmb
            int_dz = self.config['int_dz']
            lmb = np.arange(_tmp.min(), _tmp.max(), int_dz)

            # Evaluate the filter on this grid.
            spl_f = splrep(sub_f.lmb, sub_f.response)
            y_f = splev(lmb, spl_f, ext=1)

            X = np.outer(a, lmb)

            for sed in seds.index.unique():
                t1 = time.time()

                y_sed = splev(X, sedD[sed])
                k_ext = splev(X, ext_spl)
                EBV = self.config['EBV']
                y_ext = 10**(-0.4*EBV*k_ext)

                Y = y_ext*y_sed*y_f*lmb

                if int_method == 'simps':
                    ans = r_const[band]*simps(Y, lmb, axis=1)
                elif int_method == 'sum':
                    ans = r_const[band]*int_dz*Y.sum(axis=1)

                # This might be overkill in terms of storage, but information in
                # the columns is a pain..
                part = pd.DataFrame({'z': z, 'flux': ans})
                part['band'] = band 
                part['sed'] = sed
                part['ext_law'] = self.config['ext_law']
                part['EBV'] = EBV

                df = df.append(part, ignore_index=True)

                t2 = time.time()

        return df

    def convert_flux(self, ab):
        """Convert fluxes into the PAU units."""

        # Later we might want to store fluxes in different systems, but
        # this is all we currently need.

        # Here f is the flux estimated by the integral. The factor of 48.6
        # comes from assuming the templates being in erg s^-1 cm^-2 Hz^-1.
        # m_ab = -2.5*log10(f) - 48.6
        # m_ab = -2.5*log10(f_PAU) - 26.

#        if self.config['flux_unit'] == 'PAU':
#        fac = 10**(0.4*(48.6-26.))
#        ab['flux'] *= fac

#        ipdb.set_trace()

        return ab

    def entry(self, filters, seds, ext):
        """Estimate model fluxes."""

        r_const = self.r_const(filters)
        ab = self.calc_ab(filters, seds, ext, r_const)
        ab = self.convert_flux(ab)

        return ab, r_const

    def run(self):
        filters = self.input.filters.result
        seds = self.input.seds.result
        extinction = self.input.extinction.result

        ab, r_const = self.entry(filters, seds, extinction)

        path_out = self.output.empty_file('default')
        store_out = pd.HDFStore(path_out, 'w')
        store_out['default'] = ab
        store_out['r_const'] = r_const
        store_out.close()
