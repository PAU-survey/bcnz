#!/usr/bin/env python
# encoding: UTF8

# Calibrating the priors.

import pdb
import numpy as np

class priors:
    def __init__(self, myconf):
        self.myconf = myconf

    def run(self):
        # Only working with ASCII by now.
        cat_file = self.myconf['cat']

        ngal_priors = 100

        cols = (44, 4, 16)

        cat_all = np.loadtxt(cat_file, usecols=cols)
       
        if False: 
            # Then select a random subsample.
            ngal = cat_all.shape[0]
            inds = np.random.randint(0, ngal, ngal_priors)

            cat = cat_all[inds]

        np.savetxt('mzt_bcnz.txt', cat_all)
