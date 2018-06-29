#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import os
import pandas as pd

descr = {'name': 'Name of the stored file'}

class zp_saved:
    version = 1.02
    config = {'name': 'v1'}

    d = '/home/eriksen/papers/paupz/calib'
    def run(self):
        fname = '{}.csv'.format(self.config['name'])
        path = os.path.join(self.d, fname)

        zp = pd.read_csv(path, names=['band', 'zp'])
        zp = zp.set_index('band').zp

        self.output.result = zp
