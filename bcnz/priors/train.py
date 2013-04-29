#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import pdb
import sys
import time
import itertools as it

from scipy.interpolate import RectBivariateSpline as Bspl
from scipy.interpolate import UnivariateSpline
# How I define the values here are extremely 
# Temporary configuration before deciding if the idea should
# permanently be included in bcnz..
#file_name = "/Users/marberi/pau/photoz/config/dc6_types"
gal_types = [(0,19), (20, 51),(52, 65)]
#nell, nspr, nirr = [20, 32, 14]
nell, nspr, nirr = 3,6,7
types = [3,6,7]

train_zs = 0.1
train_dm = .1
smooth = 0.5

class train(object):
    def __init__(self, conf, zdata, m_0, m_step, ninterp):
        self.conf = conf

        self.read_training_set(conf)
        self.tinds = self.type_inds(self.t)
        self.spl_p_mt = self.type_priors(self.tinds)
        self.find_edges()

        self.z = zdata['z']

    def read_training_set(self, conf):
        """Read in training set."""

        assert len(conf['obs_files']) == 1
        train_file = conf['obs_files'][0]

        cols = (1,2,5)
        zs, t, m = np.loadtxt(train_file, usecols=cols, unpack=True)

        t = t.astype(np.int)

        self.zs = zs
        self.t = t
        self.m = m


    def type_inds(self, t):
        """Find which galaxies that belongs to each type."""

        tinds = []
        for t_low,t_high in gal_types:
            tinds.append(np.logical_and(t_low <= t, t <= t_high))

        assert len(self.t) == np.sum([np.sum(x) for x in tinds])

        return tinds

    def type_priors(self, inds):
        noseen = (self.m == self.conf['undet'])
        seen = np.logical_not(noseen)

        m_min = np.min(self.m)
        m_max = np.max(self.m[seen])

        m_edges = np.arange(m_min, m_max+train_dm, train_dm)
        m_c = 0.5*(m_edges[1:] + m_edges[:-1])

        tm_arr = np.zeros((len(inds), len(m_c)))

        spl_p_mt = {}
        for i,ind in enumerate(inds):
            H,_ = np.histogram(self.m[ind], m_edges)
            tm_arr[i] = H

#            H = H.astype(np.float)/np.sum(H)
#            spl_p_mt[i] = UnivariateSpline(m_c, H, s=smooth)

        tm_arr = tm_arr / tm_arr.sum(axis=0)
        for i in range(tm_arr.shape[0]):
            spl_p_mt[i] = UnivariateSpline(m_c, tm_arr[i], s=smooth)

        #pdb.set_trace()
        return spl_p_mt

    def find_edges(self):
        """Define magnitude and redshift bins."""
        
        noseen = (self.m == 99)
        seen = np.logical_not(noseen)

        m_min = np.min(self.m)
        m_max = np.max(self.m[seen])
        zs_min = 0.
        zs_max = np.max(self.zs)
        self.m_edges = np.arange(m_min, m_max+train_dm, train_dm)
        self.zs_edges = np.arange(zs_min, zs_max+train_zs, train_zs)


        self.spls = self.prior_spls(self.tinds)


    def find_spl(self, inds):
        """Construct splines for the priors."""

        H,_,_ = np.histogram2d(self.m[inds], self.zs[inds], \
                               [self.m_edges, self.zs_edges])

        # Centrer of the magnitude and redshift bins.
        m_c = 0.5*(self.m_edges[1:] + self.m_edges[:-1])
        zs_c = 0.5*(self.zs_edges[1:] + self.zs_edges[:-1])

        m_sum = H.sum(axis=0)
        H = H/m_sum

        H = np.nan_to_num(H)
        #pdb.set_trace()
        spl = Bspl(m_c, zs_c, H, s=smooth)

        return spl

    def prior_spls(self, tinds):
        """Priors splines for all galaxy types."""

        res = {}
        for i, inds in enumerate(tinds):
            res[i] = self.find_spl(inds)

        return res

    def add_priors(self, m, lh):
        """Add the priors."""

        z = self.z 
        nm = len(m)
        nz = len(self.z)

        # The spline code expect accending order of the arguments.
        m_ind_sort = m.argsort()
        m_sorted = m.copy()
        m_sorted.sort()


        pre = {}
        for spl_key in np.arange(len(gal_types)):
            spl = self.spls[spl_key]
            mt_spl = self.spl_p_mt[spl_key]

            pr_type = spl(m_sorted, z)
            pr_type[m_ind_sort] = pr_type

            p_mt = mt_spl(m) #, ext=0) #splev(m, mt_spl, ext=0)
            pr_type = (p_mt*pr_type.T).T / types[spl_key]

            pr_type = np.clip(pr_type, 0., np.inf)
           # pr_type = (p_mt*pr_type.T).T
            pr_type = (pr_type.T / pr_type.sum(axis=1)).T

            pre[spl_key] = pr_type

#            pdb.set_trace()

        pr = np.dstack(nell*[pre[0]]+nspr*[pre[1]]+nirr*[pre[2]])

        lh *= pr

        return lh
