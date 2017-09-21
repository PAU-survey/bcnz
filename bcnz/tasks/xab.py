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
  'EBV': 'Extinction amplitude'
}

class xab:
    """The model fluxes."""

    version = 1.10
    config = {\
      'zmax_ab': 12.,
      'dz_ab': 0.001,
      'int_dz': 1.,
      'int_method': 'simps',
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
            r_const[fname] = 1./simps(sub.y / sub.lmb, sub.lmb) / clight_AHz

        return r_const

    def sed_spls(self, seds):
        """Create a spline of all the SEDs."""

        sedD = {}
        for sed in seds.index.unique():
            sub_sed = seds.ix[sed]
            spl_sed = splrep(sub_sed.lmb, sub_sed.y)

            sedD[sed] = spl_sed

        return sedD

    def ext_spls(self, ext):
        """Splines for the extinction."""

        extD = {}

        # Bad hack...
        if isinstance(ext, type(None)):
            lmb = np.linspace(3000, 15000)
            ones = np.zeros_like(lmb)
            extD['none'] = splrep(lmb, ones)

            return extD

        for law in ['calzetti']:
            extD[law] = splrep(ext.lmb, ext[law])

        ones = np.zeros_like(ext.lmb)
        extD['none'] = splrep(ext.lmb, ones)

        return extD


    def calc_ab(self, filters, seds, ext, r_const):
        """Estimate the fluxes for all filters and SEDs."""

        sedD = self.sed_spls(seds)
        extD = self.ext_spls(ext)
        z = np.arange(0., self.config['zmax_ab'], self.config['dz_ab'])

        # Test...
        df = pd.DataFrame()

        int_method = self.config['int_method']
        a = 1./(1+z)
        for band in filters.index.unique():
            sub_f = filters.ix[band]

            # Define a higher resolution grid.
            _tmp = sub_f.lmb
            int_dz = self.config['int_dz']
            lmb = np.arange(_tmp.min(), _tmp.max(), int_dz)

            # Evaluate the filter on this grid.
            spl_f = splrep(sub_f.lmb, sub_f.y)
            y_f = splev(lmb, spl_f, ext=1)

            X = np.outer(a, lmb)

            for sed in seds.index.unique():
                t1 = time.time()
                print('calc', band, sed)

                y_sed = splev(X, sedD[sed])
                for ext_key, ext_spl in iter(extD.items()):
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
                    part['ext'] = ext_key
                    part['EBV'] = EBV

                    df = df.append(part, ignore_index=True)

                    t2 = time.time()
                    print('time', t2-t1)

        return df

    def convert_flux(self, ab):
        """Convert fluxes into the PAU units."""

        # Later we might want to store fluxes in different systems, but
        # this is all we currently need.

        # Here f is the flux estimated by the integral. The factor of 48.6
        # comes from assuming the templates being in erg s^-1 cm^-2 Hz^-1.
        # m_ab = -2.5*log10(f) - 48.6
        # m_ab = -2.5*log10(f_PAU) - 26.

        fac = 10**(0.4*(48.6-26.))
        ab['flux'] *= fac

        ipdb.set_trace()

        return ab

    def ab(self, filters, seds, ext):
        """Estimate model fluxes."""

        r_const = self.r_const(filters)
        ab = self.calc_ab(filters, seds, ext, r_const)
        ab = self.convert_flux(ab)

        return ab, r_const

    def run(self):
        filters = self.job.filters.result
        seds = self.job.seds.result
        ext = self.job.ext.result if hasattr(self.job, 'ext') else None

        ab, r_const = self.ab(filters, seds, ext)

        path_out = self.job.empty_file('default')
        store_out = pd.HDFStore(path_out, 'w')
        store_out['default'] = ab.stack()
        store_out['r_const'] = r_const
        store_out.close()
