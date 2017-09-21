#!/usr/bin/env python
# encoding: UTF8

from __future__ import print_function

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
  'normalize': 'DEPRECATED???',
  'EBV': 'Extinction amplitude'
}

class xab:
    """The model fluxes."""

    version = 1.06
    config = {\
      'zmax_ab': 12.,
      'dz_ab': 0.001,
      'int_dz': 1.,
      'int_method': 'simps',
      'normalize': True,
      'EBV': 0.0
    }

    def _calc_r_const(self, filters):
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

    def r_const(self, filters):
        """TODO: Remove."""

        # Testing if the normalization actually makes a difference.
        clight_AHz=2.99792458e18

        if self.config['normalize']:
            r_const = self._calc_r_const(filters)
        else:
            assert NotImplementedError('Why is this implemented? Remove if possible!')

            r_const = {}

            for f in filters.index.unique():
                r_const[f] = 1. / clight_AHz

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

        df = pd.DataFrame()
        df['z'] = z

        int_method = self.config['int_method']
        a = 1./(1+z)
        for fname in filters.index.unique():
            sub_f = filters.ix[fname]

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
                print('calc', fname, sed)

                y_sed = splev(X, sedD[sed])
                for ext_key, ext_spl in iter(extD.items()):
                    k_ext = splev(X, ext_spl)
                    EBV = self.config['EBV']
                    y_ext = 10**(-0.4*EBV*k_ext)

                    Y = y_ext*y_sed*y_f*lmb

                    if int_method == 'simps':
                        ans = r_const[fname]*simps(Y, lmb, axis=1)
                    elif int_method == 'sum':
                        ans = r_const[fname]*int_dz*Y.sum(axis=1)

                    key = (fname, sed, ext_key)
                    df[key] = ans

                    t2 = time.time()
                    print('time', t2-t1)

        return df


    def run(self):
        filters = self.job.filters.result
        seds = self.job.seds.result
        ext = self.job.ext.result if hasattr(self.job, 'ext') else None

        r_const = self.r_const(filters)

        ab = self.calc_ab(filters, seds, ext, r_const)

        path_out = self.job.empty_file('default')
        store_out = pd.HDFStore(path_out, 'w')
        store_out['default'] = ab.stack()
        store_out['r_const'] = r_const
        store_out.close()

#        ipdb.set_trace()

#        self.job.result = ab.stack()
