#!/usr/bin/env python
# encoding: UTF8
import numpy as np
import pdb
import time

from scipy.spatial import Delaunay
from scipy.interpolate import splrep, splev, UnivariateSpline

# Temporary configuration...
gal_types = [(0,19), (20, 51),(52, 65)]
file_all_templ = 'data/all_templates.txt'
types = [3,6,7]
nell, nspr, nirr = types
train_dm = .1
smooth = 3

class tess(object):
    def __init__(self, conf, zdata, m0, m_step, ninterp):
        self.conf = conf
        self.zdata = zdata

#        self.type_map()
        self.read_training_set()

        inds = self.type_inds(self.t)
        self.spl_p_mt = self.type_priors(inds)
        self.tess, self.vol_inv = self.construct_tri(conf, inds)

    def read_training_set(self):
        """Read in training set."""

        assert len(self.conf['obs_files']) == 1
        train_file = self.conf['obs_files'][0]

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

        return spl_p_mt

    def m_zs_tri(self, ind):
        """Triangulation for one galaxy type."""

        points = np.vstack([self.m[ind], self.zs[ind]]).T
        return Delaunay(points)

    def tri_volume(self, tess):
        """Volumes for simplices in triangulation."""

        vertices = tess.points[tess.vertices]
        q = vertices[:,:-1,:] - vertices[:,-1,None,:]

        volume = np.array([np.linalg.det(q[x,:,:]) for x in 
                           range(tess.nsimplex)])

        volume = np.abs(volume)


       # neighbors
        return volume

    def smooth_volume(self, tess, volume):
        pass

    def construct_tri(self, conf, inds):
        """Construct triangulation of the m,z space."""

        all_tess = {}
        vol_inv = {}
        for i, ind in enumerate(inds):
            tess = self.m_zs_tri(ind)
            volume = self.tri_volume(tess)
            self.smooth_volume(tess, volume)


            all_tess[i] = tess
            vol_inv[i] = 1./volume

        return all_tess, vol_inv
#        for ind


    def add_priors(self, m, lh):
        #tess.find_simplex(points)

        t1 = time.time()
        z = self.zdata['z']

        # Secon index is the fastest changing.
        nm = len(m)
        nz = len(z)
        points = np.vstack([np.repeat(m, nz), np.tile(z, nm)]).T

        pre = {}
        for i in range(len(gal_types)):
            tess = self.tess[i]
            simpl = self.tess[i].find_simplex(points)
            vol_inv = self.vol_inv[i][simpl]

            vol_inv[simpl == -1] = 0.

#            vol_inv = np.clip(vol_inv, 0.05*np.max(vol_inv), np.inf)

            pr_type = vol_inv.reshape((nm,nz))
            pr_type = (pr_type.T / pr_type.sum(axis=1)).T

            mt_spl = self.spl_p_mt[i]

            # Extrapolation or zero???? By now extrapolation...
            #¢p_mt = splev(m, mt_spl, ext=0)
            p_mt = mt_spl(m) #, ext=0) #splev(m, mt_spl, ext=0)
            
            pr_type = (p_mt*pr_type.T).T / types[i]

            pr_type = np.clip(pr_type, 0.01*np.max(pr_type), np.inf)
            pr_type = (pr_type.T / pr_type.sum(axis=1)).T


            #pre[i] = vol_inv.reshape((nm,nz))


            pre[i] = pr_type

        pr = np.dstack(nell*[pre[0]]+nspr*[pre[1]]+nirr*[pre[2]])

        lh *= pr

        t2 = time.time()

        return lh


def main():
    file_name = '/Users/marberi/pau/photoz/bpz/photoz/mock_faint.cat'

    conf = {}
    conf['obs_files'] = [file_name]


    inst = prior_tess(conf, False, False, False, False)

if __name__ == '__main__':
    main()
