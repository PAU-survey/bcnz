#!/usr/bin/env python
# encoding: UTF8

import ipdb
import numpy as np
import pandas as pd

class ext:
    # The default values equals the one in the catalog generation. Should be
    # moved elsewhere.
    sedl = [0,17,55,65]

    def __init__(self, config, zdata, prior_cat):
        self.config = config
        self.zdata = zdata
        self.z, self.mbins, self.prior_hist = self.generate_hist(prior_cat)

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

        prior_hist = {}
        gr = pd.cut(prior_cat['sed'], right=False, bins=self.sedl)
        for i,(label, sub) in enumerate(pd.groupby(prior_cat, gr)):
            H,b1,b2 = np.histogram2d(sub['zs'], sub['m0'], bins=[z,mbins])

            prior_hist[i] = H

        return z, mbins, prior_hist

    def add_priors(self, m, lh):
        ipdb.set_trace()
