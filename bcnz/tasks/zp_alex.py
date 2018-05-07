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

    version = 1.02
    config = {'file_name': 'sys_50iter_pau821_onlyzs35_003_NB2BB.csv',
              'Nrounds': -1}

    def check_config(self):
        assert self.config['file_name']

    d = '/home/eriksen/papers/paupz/from_alex'

    def convert_names(self, cat):
        """Convert between different namings of the bands."""

        BB_in = ['u_cfht', 'B_Subaru', 'V_Subaru', 'r_Subaru', 'i_Subaru', 'suprime_FDCCD_z']
        BB_out = ['cfht_u', 'subaru_B','subaru_V', 'subaru_r', 'subaru_i', 'subaru_z']

        cat = cat.rename(columns=dict(zip(BB_in, BB_out)))
        cat = cat.rename(columns=lambda x: x.replace('NB_', 'NB'))

        return cat

    def entry(self):
        path = os.path.join(self.d, self.config['file_name'])
        zp_all = pd.read_csv(path)
        zp_all = self.convert_names(zp_all)


        if 0 < self.config['Nrounds']:
            zp_mag = zp_all.iloc[:self.config['Nrounds']].sum(axis=0)
        else:
            zp_mag = zp_all.sum(axis=0)

        R = 10**(-0.4*zp_mag)

        return R

    def run(self):
        self.output.result = self.entry()
