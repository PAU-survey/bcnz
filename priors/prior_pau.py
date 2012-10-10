import numpy as np
import sys

class prior_pau:
    def __init__(self, conf, zdata, m_0):
        ndes = 1 # Number of decimals

        # Ok, these might be passed later on..
        self.a = np.array((2.369,1.843,1.843,1.340,1.340,1.340))
        self.zo = np.array((0.387,0.390,0.390,0.208,0.208,0.208))
        self.km = np.array((0.119,0.093,0.093,0.130,0.130,0.130))
        self.fo_t = np.array((0.48,0.22,0.22))
        self.k_t = np.array((0.186,0.038,0.038))

        m_step = conf['m_step']
        m_min = np.floor(10**ndes*min(m_0)) / 10.**ndes
        steps = np.ceil((max(m_0) - m_min) / m_step).astype(int)

        m = m_min + m_step * np.arange(steps+1)
        #self.pr = np.ascontiguousarray(self.prior_precalc(z, m, ninterp))
        self.pr = self.prior_precalc(conf, zdata, m)

        self.conf = conf

        self.z = zdata['z']
        self.m_0 = m_0
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

        # Defined in BPZ..
        f_t = np.zeros((nm, nt))
        f_t[:,:3] = self.fo_t*np.exp(-np.outer(dm, self.k_t))
        h = (1.-np.sum(f_t[:,:3], axis=1))/3.
        f_t[:,3:] = np.tile(h, (nt-3,1)).T

        # redshift - type
        zt_at_a = np.exp(np.outer(np.log(z), self.a))

        # In all the summations, m-mag,z-redshift, t-type.
        p = np.einsum('zt,mt->mtz', zt_at_a, 1./zmt_at_a)
        p = np.exp(-np.clip(p, 0., 700.))
        p = np.einsum('zt,mtz->mtz', zt_at_a, p)

        inv_norm = 1./np.sum(p, axis=2)

        p = np.einsum('mt,mtz,mt->mtz', f_t, p, inv_norm)
        p = np.swapaxes(p, 1, 2)

        return p

    def prior_precalc(self, conf, zdata, m):
        """Precalculating the priors with interpolated types."""

        nt = len(self.a)
        temp_types = np.arange(nt, dtype=float)
        all_types = np.linspace(0.,nt-1., nt + conf['interp']*(nt-1))

        p = self.prior_basis(zdata['z'], m)

        # Derivative with respect to type. 
        d = (p[:,:,1:] - p[:,:,:-1]) / (temp_types[1:] - temp_types[:-1])

        inds = np.searchsorted(temp_types, all_types) - 1
        inds = np.clip(inds, 0, nt-2)

        pr = p[:,:,inds] + d[:,:,inds]*(all_types-temp_types[inds])
#        import pdb; pdb.set_trace()
        return pr

    def inds(self, m):
        """Indexes for the magnitude to use."""

        return np.round((m  - self.m_min)/self.conf['m_step']).astype(int)

    def add_priors(self, m, lh):

        inds = self.inds(m)
        for i in range(lh.shape[0]):
            lh[i] *= self.pr[inds[i]]

        return lh
 
#        import time
#        t1 = time.time()
##        lh = lh.reshape(lh.shape[0], self.nz*self.nt)
##        ans = lh*self.pr[inds]
##        ans = lh*self.pr[inds]
##        lh *= self.pr[inds]
#       
#        priors = self.pr[inds]
#        t2 = time.time()
##        import pdb; pdb.set_trace()
##        return lh
##        return ans        
#        return priors