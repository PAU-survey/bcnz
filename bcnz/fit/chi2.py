#!/usr/bin/env python
# encoding: UTF8
# First iteration on a code to do Bayesian template 
# fitting to determine galaxy type and redshift.
import pdb
import time
import sys
import numpy as np
from scipy.ndimage.interpolation import zoom
import scipy.stats

import bcnz

# Currently the speedup with numexpr is not high
# enough to make it mandatory.
try:
    import numexpr as ne
    use_numexpr = True
except ImportError:
    use_numexpr = False

np.seterr(under='ignore')

def prob_interval(p,x,plim):
    """Limits enclosing probabilities plim. Handles
       stacks of probabilities.
    """

    # Upper and lower limits.
    qmin = 0.5*(1-plim)
    qmax = 1. - qmin

    cdf = p.cumsum(axis=1)
    jmin = np.apply_along_axis(np.searchsorted,1,cdf,qmin)
    jmax = np.apply_along_axis(np.searchsorted,1,cdf,qmax)
    jmax = np.minimum(jmax, p.shape[1] - 1)

    return x[jmin], x[jmax]


def find_odds(p,x,xmin,xmax):
    """Probabilities in the given intervals."""

    cdf = p.cumsum(axis=1)

    def K(xlim):
        return np.searchsorted(x, xlim)

    imin = np.apply_along_axis(K, 0, xmin) - 1
    imax = np.apply_along_axis(K, 0, xmax)
    imax = np.minimum(imax, p.shape[1] - 1)
    gind = np.arange(p.shape[0])

    return (cdf[gind,imax] - cdf[gind,imin])/cdf[:,-1]

class chi2_calc:
    def __init__(self, conf, zdata, data, pop='', dz='',min_rms=''):
        assert data['mag'].shape[0], 'No galaxies'
        self.conf = conf
        self.zdata = zdata
        self.data = data

        self.dz = dz if dz else conf['dz']
        self.min_rms = min_rms if min_rms else conf['min_rms']

        if conf['use_split']:
            self.f_mod = zdata['{0}.f_mod'.format(pop)]
            z = zdata['{0}.z'.format(pop)]
        else:
            self.f_mod = zdata['f_mod']
            z = zdata['z']

        self.z = z
        self.cols, self.dtype = self.cols_dtype()

        # Priors
        self.load_priors(conf, z)
        self.set_values()
        self.calc_P1()

    def set_values(self):
        f_obs = self.data['f_obs']
        ef_obs = self.data['ef_obs']

        obs = np.logical_and(ef_obs <= 1., ef_obs*1e-4 < f_obs)
        self.h = obs / ef_obs ** 2.

        self.nf = self.f_mod.shape[2]

        # Working with 2D is faster.
        nz,nt,nf = self.f_mod.shape
        f_mod2 = self.f_mod.swapaxes(0,2)
        f_mod2 = f_mod2.swapaxes(1,2)
        f_mod2 = f_mod2.reshape((nf,nz*nt))

        self.nz = nz
        self.nf = nf
        self.nt = nt
        self.f_mod2 = f_mod2

        q = 0.5*(1.-self.conf['odds'])
        oi = np.abs(scipy.stats.norm.ppf(q))

        self.odds_pre = oi * self.min_rms 
#        self.z = z

        # Precalculate values.

    def cols_dtype(self):
        """Columns and block dtype."""

        int_cols = ['id']
        cols = self.conf['order']+self.conf['others']
        dtype = [(x, 'float' if not x in int_cols else 'int') for x in cols]

        return cols, dtype

    msg_import = """\
To import priors, you need the following:
- Module to estimate priors.
- Import statement in priors/__init__.py. Use a "priors_" prefix.
- Set the option priors without the "priors_" prefix.
"""

    def load_priors(self, conf, z):
        """Load the prior module."""

        prior_name = 'prior_%s' % conf['prior']
        try:
            self.priors = getattr(bcnz.priors, conf['prior'])(conf, \
                          self.zdata, z, self.data['m_0'])

        except ImportError:
            raise ImportError(self.msg_import)

#    @profile
    def calc_P1(self):
        """Term in chi**2."""

        l1 = self.data['f_obs']**2*self.h
        self.P1 = np.sum(l1, axis=1)

        self.l2 = self.data['f_obs']*self.h
        self.r3 = self.f_mod2**2.

    def _block(self, n):
        """Term in chi**2."""

        imin = n*self.conf['ncalc']
        imax = imin + self.conf['ncalc']

        l2 = self.l2[imin:imax,:]
        h = self.h[imin:imax,:]

        P2 = np.dot(l2, self.f_mod2)
        P3 = np.dot(h, self.r3)

        if use_numexpr:
            D = ne.evaluate("P2**2 / (P3 + 2.0e-300)")
        else:
            D = P2**2 / (P3 + 2.0e-300)

        D = D.reshape((h.shape[0], self.nz, self.nt))

        # Order after: type - redshift - ig
        D = D.swapaxes(0,2) 

        chi2_ig_last = self.P1[imin:imax] - D
        chi2 = chi2_ig_last.swapaxes(0,2)

        chi_argmin = np.array([np.argmin(x) for x in chi2])
        iz, it = np.unravel_index(chi_argmin,  chi2.shape[1:]) #self.z_t_shape)
        min_chi2 = chi2[range(chi2.shape[0]), iz,it]
        red_chi2 = min_chi2 / float(self.nf-1.) 

        # Using numexpr does not improve this evaluation.
        pb = -0.5*(chi2_ig_last - min_chi2)
        pb = np.exp(pb).swapaxes(0,2)
        
        # Add priors.
        if self.conf['use_priors']:
            m = self.data['m_0'][imin:imax]
            pb = self.priors.add_priors(m, pb) # pb now include priors.

        p_bayes = pb.sum(axis=2)

        # Only to be compatible with BPZ. The ideal case would be a
        # absolute value...
        pmin = self.conf['p_min']*p_bayes.max(axis=1.)
        for i in range(len(pmin)):
            p_bayes[i] = np.where(pmin[i] < p_bayes[i], p_bayes[i], 0.)

        norm = p_bayes.sum(axis=1)
        norm = np.clip(norm, 1e-300, np.inf)
        p_bayes = (p_bayes.T / norm).T

        iz_b = p_bayes.argmax(axis=1)
        zb = self.z[iz_b]

#        pdb.set_trace()

        # Calculate odds.
        dz = self.odds_pre*(1.+zb)
        zo1 = zb - dz
        zo2 = zb + dz

        odds = find_odds(p_bayes, self.z, zo1, zo2)

        it_b = pb[range(len(zb)),iz_b].argmax(axis=1)


#        import pdb; pdb.set_trace()
        interp = self.conf['interp']
        tt_b = it_b / (1. + interp)
        tt_ml = it / (1. + interp)

        A = pb[range(len(zb)),it_b]
        test = pb[range(len(zb)),:,it_b].max(axis=1) < 1e-300
        it_b = np.where(test,  -1., it_b)
        tt_b = np.where(test,  -1., tt_b)

        zb_min, zb_max = prob_interval(p_bayes, self.z, self.conf['odds'])

        # TODO: Find a better approach.
        z_ml = self.z[iz]
        m_0 = self.data['m_0'][imin:imax]
        chi2 = red_chi2
        t_b = tt_b+1
        t_ml = tt_ml+1
        if 'z_s' in self.data:
            z_s = self.data['z_s'][imin:imax]

        # Added right before the LBNL DES conference.
        ra = self.data['ra'][imin:imax]
        dec = self.data['dec'][imin:imax]
        spread_model_i = self.data['spread_model_i'][imin:imax]

#        pdb.set_trace()

        id = self.data['id'][imin:imax]
        loc = locals()
        A = [loc[x] for x in self.cols]

#        pdb.set_trace()
        return np.rec.fromarrays(A, self.dtype)

    def blocks(self):
        """Iterate over the different blocks."""

        ngal = len(self.data['id'])
        ncalc = self.conf['ncalc']
        nblocks = int(np.ceil(float(ngal) / ncalc))

        for n in np.arange(nblocks):
            yield self._block(n)


class chi2_combined:
    """Interface for combining the results for several populations."""

    def __init__(self, conf, zdata, f_obs, ef_obs, m_0, \
                 z_s, inds, ngal_calc):

        filters = zdata['filters']
        filter_id = filters.index('Fi')
        mag_lim = conf['mag_split']

        f_lim = 10.**(-0.4*mag_lim)

        ind_bright = f_lim < f_obs[:,filter_id]
        ind_faint = np.logical_not(ind_bright)

        self.conf = conf
        self.zdata = zdata
        self.f_obs = f_obs
        self.ef_obs = ef_obs
        self.m_0 = m_0
        self.z_s = z_s
        self.inds = inds
        self.ind_pop = {'bright': ind_bright, 'faint': ind_faint}

        # use_ind gives the index within chi2_bright and chi2_faint to use.
        use_ind = np.zeros(f_obs.shape[0], dtype=np.int)
        use_ind[ind_bright] = np.array(np.arange(np.sum(ind_bright)))
        use_ind[ind_faint] = np.array(np.arange(np.sum(ind_faint)))

        self.ngal = len(z_s)
        self.ngal_calc = ngal_calc
        self.ind_bright = ind_bright
        self.ind_faint= ind_faint
        self.use_ind = use_ind

#        pdb.set_trace()

    def blocks(self):
        """Iterate over blocs."""

        for pop in ['bright', 'faint']:
            dz = self.conf['dz_{}'.format(pop)]
            min_rms = self.conf['min_rms_{}'.format(pop)]
            ind = self.ind_pop[pop]

            if not len(self.m_0[ind]):
                continue

            chi2_pop = chi2_calc(self.conf, self.zdata, self.f_obs[ind], self.ef_obs[ind], \
                                 self.m_0[ind], self.z_s[ind], self.inds[ind], \
                                 self.ngal_calc, dz, min_rms, pop)

            if not 'new_block' in locals():
                new_block = np.zeros(self.ngal, dtype=chi2_pop.dtype)

            nparts = int(np.ceil(len(ind)/ float(self.ngal_calc)))
            s = self.ngal_calc*np.arange(self.ngal_calc)

            splitted = np.split(ind.nonzero()[0], s[1:])

            for i, p_block in enumerate(chi2_pop.blocks()):
#                pdb.set_trace()
                new_block[splitted[i]] = p_block

        yield new_block

def chi2(conf, zdata, data):
    """Select which chi2 object to use depending on splitting in magnitudes
       or not.
    """

    if conf['use_split']:
        #return chi2_combined(conf, zdata, f_obs, ef_obs, m_0, z_s, ids, ngal_calc)
        return chi2_combined(conf, zdata, data)
    else:
#        dz = conf['dz']
#        min_rms = conf['min_rms']

        return chi2_calc(conf, zdata, data)
