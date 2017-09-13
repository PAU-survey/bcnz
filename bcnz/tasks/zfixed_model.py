#!/usr/bin/env python
# encoding: UTF8

import ipdb
import numpy as np
import pandas as pd

#import bcnz_fit

class zfixed_model:
    version = 1.0
    config = {}

    def run(self):
        store = self.job.pzfit.get_store()

        norm = store['norm'].to_xarray().norm
        pzcat = store['default'].unstack()

#        inst = bcnz_fit.bcnz_fit()


        ipdb.set_trace()
