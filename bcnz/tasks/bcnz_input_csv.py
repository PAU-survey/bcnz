#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import os
import numpy as np
import pandas as pd

class bcnz_input_csv:
    """Input as a csv file."""

    version = 1.01
    config = {'file_name': 'pau821_onlyzs35_003_NB2BB.csv'}

    # This should not be in the configuration, since it
    # will change the taskid. We can agree on a location.

    d = '/home/eriksen/papers/paupz/from_alex'
    def entry(self):
        path = os.path.join(self.d, self.config['file_name'])

        cat_in = pd.read_csv(path)
        bbmapD = {'cfht_u': 'U', 'subaru_B': 'B', 'subaru_V': 'V', \
                  'subaru_r': 'R', 'subaru_i': 'I', 'subaru_z': 'ZN'}

        X = 455 + 10*np.arange(40)
        NB = list(map('NB{}'.format, X))
        NB2 = list(map('NB_{}'.format, X))

        def sel(pre, L):
            return [pre+x for x in L]

        flux_cols = sel('flux_', NB2)
        err_cols = sel('flux_err_', NB2)
        flux = cat_in[flux_cols].rename(columns=dict(zip(flux_cols, NB)))
        err = cat_in[err_cols].rename(columns=dict(zip(err_cols, NB)))

        cat = pd.concat({'flux': flux, 'flux_err': err}, axis=1)
        cat['ref_id'] = cat_in.ref_id

        fields = ['ra', 'dec', 'I_auto', 'zspec', 'conf', 'r50']
        for field in fields:
            if not field in cat_in:
                continue

            cat[field] = cat_in[field]

        cat = cat.rename(columns={'zspec': 'zs'})
        cat = cat.set_index('ref_id')

        return cat

    def run(self):
        self.output.result = self.entry()
