#!/usr/bin/env python
# encoding: UTF8

import sys
import numpy as np

np.seterr(under='ignore')

class pau(object):
    """Priors calibrated to mocks used for the PAU survey."""

    # Ok, here something really escalated!

    def __init__(self, conf, zdata, z, m0):
        ndes = 1 # Number of decimals

        self.conf = conf
        nn = (conf['nell'], conf['nsp'], conf['nsb'])

        for param in ['a', 'zo', 'km']:
            val = np.repeat(conf['pr_'+param], nn)
            setattr(self, param, val)

        # This factors is only set for Elliptical and Spiral
        # galaxies.
        fo_t = np.array(conf['pr_fo_t'])
        self.fo_t = np.repeat(fo_t / np.array(nn[:2]), nn[:2])
        self.k_t = np.repeat(conf['pr_k_t'], nn[:2])
        self.nn = nn

        m_step = conf['m_step']
        m_min = np.floor(10**ndes*min(m0)) / 10.**ndes
        steps = np.ceil((max(m0) - m_min) / m_step).astype(int)

        m = m_min + m_step * np.arange(steps+1)
        #self.pr = np.ascontiguousarray(self.prior_precalc(z, m, ninterp))
        self.pr = self.prior_precalc(conf, z, m)


        self.z = z
        self.m0 = m0
        self.m_min = m_min

    def prior_basis(self, z, m):
        """Priors without interpolation between types."""

        nt = len(self.a)
        nm = len(m)

        # Indexes magnitude, type. Both add and pow works over
        # the last index.
        m_min = 20.0

        m = np.clip(m, 20., 32.)
        dm = m - m_min

        h = self.zo + np.outer(dm, self.km)
        zmt = np.clip(h, 0.01, 15.)
        zmt_at_a = zmt**self.a
        zmt_at_a += self.conf['hh_fac']
        # See Benitez 2000 for the functional form.
        # The probabilities of the Irregulars is given by the
        # normalization.
        f_t = np.zeros((nm, nt))

        nsplit = sum(self.nn[:2])
        f_t[:,:nsplit] = self.fo_t*np.exp(-np.outer(dm, self.k_t))

        f_irr = (1-f_t.sum(axis=1)) / self.nn[-1]
        f_t[:,nsplit:] = np.tile(f_irr, (self.nn[-1],1)).T
        assert np.allclose(f_t.sum(axis=1), 1.0), 'Internal error'

#        f_t[:,:3] = self.fo_t*np.exp(-np.outer(dm, self.k_t))
#        h = (1.-np.sum(f_t[:,:3], axis=1))/3.
#        f_t[:,3:] = np.tile(h, (nt-3,1)).T


        # redshift - type
        zt_at_a = np.exp(np.outer(np.log(z), self.a))
        zt_at_a += self.conf['hz_fac']

        # In all the summations, m-mag,z-redshift, t-type.
        p = np.einsum('zt,mt->mtz', zt_at_a, 1./zmt_at_a)
        p = np.exp(-np.clip(p, 0., 700.))
        p = np.einsum('zt,mtz->mtz', zt_at_a, p)

        inv_norm = 1./np.sum(p, axis=2)

        p = np.einsum('mt,mtz,mt->mtz', f_t, p, inv_norm)
        p = np.swapaxes(p, 1, 2)

        return p

    def prior_precalc(self, conf, z, m):
        """Precalculating the priors with interpolated types."""

        nt = len(self.a)
        temp_types = np.arange(nt, dtype=float)
        all_types = np.linspace(0.,nt-1., nt + conf['interp']*(nt-1))

        p = self.prior_basis(z, m)

        # Derivative with respect to type. 
        d = (p[:,:,1:] - p[:,:,:-1]) / (temp_types[1:] - temp_types[:-1])

        inds = np.searchsorted(temp_types, all_types) - 1
        inds = np.clip(inds, 0, nt-2)

        pr = p[:,:,inds] + d[:,:,inds]*(all_types-temp_types[inds])

        return pr

    def inds(self, m):
        """Indexes for the magnitude to use."""

        return np.round((m  - self.m_min)/self.conf['m_step']).astype(int)

    def add_priors(self, m, lh):
        """Add priors to the likelihood."""

        inds = self.inds(m)
        for i in range(lh.shape[0]):
            lh[i] *= self.pr[inds[i]]

        return lh
