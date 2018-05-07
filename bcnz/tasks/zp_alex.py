#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import os
import pandas as pd

descr = {'file_name': 'File name to load the catalogs',
         'Nrounds': 'Number of calibration rounds'}

class zp_alex:
    """Converts the zero-points Alex sent to a multiplicative
       factor.
    """

    version = 1.0
    config = {'file_name': 'sys_50iter_pau821_onlyzs35_003_NB2BB.csv',
              'Nrounds': -1}

    def check_config(self):
        assert self.config['file_name']

    d = '/home/eriksen/papers/paupz/from_alex'
    def entry(self):
        path = os.path.join(self.d, self.config['file_name'])

        zp_all = pd.read_csv(path)
        if 0 < self.config['Nrounds']:
            zp_mag = zp_all.iloc[:self.config['Nrounds']].sum(axis=0)
        else:
            zp_mag = zp_all.sum(axis=0)

        R = 10**(0.4*zp_mag)

        return R

    def run(self):
        self.output.result = self.entry()
