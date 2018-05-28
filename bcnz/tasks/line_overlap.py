#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
from scipy.interpolate import splrep, splev
import time
import pandas as pd
import numpy as np

descr = {'min_curve': 'Minimum values of the filter curve'}


class line_overlap:
    """Estimates the overlap between an emission lines and a filter
       in the observed frame.
    """

    version = 1.0
    config = {'min_curve': 0.01}


    def entry(self, cat, F, lines):
        # We only use the subset with a spectra.
        zs = cat.zspec
        zs = zs[zs != 0.0]

        t1 = time.time()
        L = []

        from matplotlib import pyplot as plt

        min_curve = self.config['min_curve']
        for band in F.index.unique():
            sub = F.loc[band]
            resp = sub.response / sub.response.max()
            spl = splrep(sub.lmb, resp)

            for line,lmb in lines.lmb.items():
                val = splev(lmb*(1+zs), spl, ext=1)
                S = pd.Series(val)

                tosel = min_curve < val
                part = pd.DataFrame({'ref_id': zs.index[tosel], 'k': val[tosel]})
                part['band'] = band
                part['line'] = line

                L.append(part)

        print('time', time.time() - t1)

        df = pd.concat(L, axis=0)


        ipdb.set_trace()

        return df 

    def run(self):
        galcat = self.input.galcat.result
        F = self.input.filters.result
        lines = self.input.lines.result

        self.output.result = self.entry(galcat, F, lines)
