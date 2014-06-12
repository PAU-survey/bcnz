#!/usr/bin/env python
# encoding: UTF8

import pdb
import time
import itertools as it

import numpy as np
from scipy.stats import gaussian_kde

class kde(object):
    """Priors using a gaussian kernel density estimator
       calculated directly from the mocks.
    """

    def __init__(self, conf, zdata, z, m0):

        msg_test = 'This is a test implementation.'

        assert len(conf['obs_files']) == 1, msg_test
        obs_file = conf['obs_files'][0]
        cols = [1,2,5]

        cat = np.loadtxt(obs_file, usecols=cols, unpack=True)

        self.gkde = gaussian_kde(cat)
        self.z = z
        self.m0 = m0
#        ndes = 1 # Number of decimals

    def add_priors(self, m, lh):
        t = np.linspace(0,65, lh.shape[2]) # HACK

        t1 = time.time()
        A = it.product(m, self.z, t)
        dt1 = time.time() - t1

        #A = it.product(
        t2 = time.time()
        points = np.fromiter(A,dtype='f8,f8,f8')
        points = np.array(zip(*points))
        dt2 = time.time() - t2

        t3 = time.time()
        res = self.gkde.evaluate(points)
        dt3 = time.time() - t3
