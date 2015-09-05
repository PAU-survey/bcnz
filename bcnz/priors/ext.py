#!/usr/bin/env python
# encoding: UTF8

import ipdb
import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import zoom

class ext:
    # The default values equals the one in the catalog generation. Should be
    # moved elsewhere.
    sedl = [0,17,55,65]
    sed_types = [1, 2, 3]

    def __init__(self, config, zdata, prior_cat):
        self.config = config
        self.zdata = zdata
        self.zbins, self.mbins, self.pr_hist = self.generate_hist(prior_cat)

    def generate_hist(self, prior_cat):
        """Splits into 2D histograms for different SED ranges. Only done
           once.
        """

        # Bins for splitting in magnitudes. For redshift bins, use
        # the same values as for the binning.
        m_int = self.config['m_int']
        mN = self.config['mN']
        mbins = np.linspace(*m_int+[mN+1])
        z = self.zdata['z']

        # Should not be needed. Need the edges.
        dz = self.config['dz']
        assert np.allclose(z[1:] - z[:-1], dz)
        zbins = np.hstack([z - 0.5*dz, z[-1] + 0.5*dz])

        priorD = {}
        gr = pd.cut(prior_cat['sed'], right=False, bins=self.sedl)
        for i,(label, sub) in enumerate(pd.groupby(prior_cat, gr)):
            H,b1,b2 = np.histogram2d(sub['zs'], sub['m0'], bins=[zbins,mbins])

            priorD[i] = H

        # Stacking the priors. A big clumsy, but works.
        L = []
        for i, n in enumerate(self.sed_types):
            L.extend(n*[priorD[i]])

        pr_hist = np.dstack(L)
        pr_hist = pr_hist.swapaxes(0,1)

        # Using the same interpolation as the chi^2 values.
        t0 = pr_hist.shape[2]
        interp = self.config['interp']
        t1 = 1 + (t0 - 1) * (interp + 1)
        fac = (1, 1, float(t1) / t0)
        pr_hist = zoom(pr_hist, fac, order=1)

        return zbins, mbins, pr_hist

    def add_priors(self, m, lh):
        # Clip to the lower bin. Could be consider having a bin collecting the
        # results.
        mind = np.digitize(m, self.mbins)
        mind = np.clip(mind - 1, 0, len(self.mbins)-2)

        ngal = lh.shape[0]
        for i in range(ngal):
            lh[i] *= self.pr_hist[mind[i]]

        return lh
