#!/usr/bin/env python
# encode: UTF8
# First iteration on a code to do Bayesian template 
# fitting to determine galaxy type and redshift.
import pdb
import time
import sys
import numpy as np
from scipy.ndimage.interpolation import zoom

import priors
import bpz_useful
import bpz_min_tools as bpz_tools
import bcnz_mintest

# Currently the speedup with numexpr is not high
# enough to make it mandatory.
try:
    import numexpr as ne
    use_numexpr = True
except ImportError:
    use_numexpr = False

np.seterr(under='ignore')
test_min = False
class chi2:
#    @profile
    def __init__(self, conf, zdata, f_obs, ef_obs, m_0, \
                 z_s, ngal_calc=100):

        self.conf = conf
        self.zdata = zdata
        self.f_obs = f_obs
        self.ef_obd = ef_obs
        self.f_mod = zdata['f_mod']
        self.ngal_calc = ngal_calc

        self.z = zdata['z']
        self.m_0 = m_0
        self.z_s = z_s

        # Priors
        self.load_priors(conf)
        obs = np.logical_and(ef_obs <= 1., 1e-4 < f_obs /ef_obs)
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

        oi = bpz_useful.inv_gauss_int(float(conf['odds']))
        self.odds_pre = oi * conf['min_rms']

        # Precalculate values.
        self.calc_P1()
        self.calc_D(0)

    def load_priors(self, conf):
        prior_name = 'prior_%s' % conf['prior']
        try:
            self.priors = getattr(priors, prior_name)(conf, \
                          self.zdata, self.m_0)

        except ImportError:
            msg_import = """\
To import priors, you need the following:
- Module to estimate priors.
- Import statement in priors/__init__.py. Use a "priors_" prefix.
- Set the option priors without the "priors_" prefix.
"""
            raise ImportError, msg_import

#    @profile
    def calc_P1(self):
        l1 = self.f_obs*self.f_obs*self.h
        self.P1 = np.sum(l1, axis=1)

        self.l2 = self.f_obs*self.h
        self.r3 = self.f_mod2**2.

#    @profile
    def calc_D(self, imin):
        imax = imin + self.ngal_calc

        l2 = self.l2[imin:imax,:]
        h = self.h[imin:imax,:]

        P2 = np.dot(l2, self.f_mod2)
        P3 = np.dot(h, self.r3)

        if use_numexpr:
            D = ne.evaluate("P2**2 / (P3 + 2.0e-300)")
        else:
            D = P2**2 / (P3 + 2.0e-300)

        if not self.conf['opt']:
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
        use_priors = True
        if use_priors:
            m =  self.m_0[imin:imax]
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

        # Calculate odds.
        dz = self.odds_pre*(1.+zb)
        zo1 = zb - dz
        zo2 = zb + dz

        odds = [bpz_tools.odds(p_bayes[i], self.z, zo1[i], zo2[i]) for i \
                in range(len(zb))]
        odds = np.array(odds)

        it_b = pb[range(len(zb)),iz_b].argmax(axis=1)

        if test_min:
            iz_b, it_b = bcnz_mintest.mintest(pb, pb_without, self.z, 
                                              self.z_s[imin:imax], iz_b, it_b)


#        import pdb; pdb.set_trace()
        interp = self.conf['interp']
        tt_b = it_b / (1. + interp)
        tt_ml = it / (1. + interp)

        A = pb[range(len(zb)),it_b]
        test = pb[range(len(zb)),:,it_b].max(axis=1) < 1e-300
        it_b = np.where(test,  -1., it_b)
        tt_b = np.where(test,  -1., tt_b)

        _z_odds = [bpz_tools.interval(x,self.z,self.conf['odds']) for x in p_bayes]
        z1, z2 = zip(*_z_odds)
 
        # Set values.
        self.iz = iz
        self.it = it
        self.min_chi2 = min_chi2
        self.red_chi2 = red_chi2
        self.pb = pb
        self.p_bayes = p_bayes
        self.iz_b = iz_b
        self.zb = zb
        self.odds = odds
        self.it_b = it_b

        self.tt_b = tt_b
        self.tt_ml = tt_ml
        self.z1 = z1
        self.z2 = z2

        self.imin = imin
        self.imax = imax

#        self.opt_type = opt_type

#    @profile
    def find_zp_odds(self):
        self.calc_D(0)

        return self.zb, self.odds

    def __call__(self, ig):

        corr_imin = int(float(ig)/self.ngal_calc)*self.ngal_calc
        ind = ig - corr_imin

        if not self.imin <= ig < self.imax:
            self.calc_D(corr_imin)

        # Note that P1 contains information about all galaxies
        # and is therefore indexed by ig and *not* ind.

        #chi2 = self.chi2[ind]
        iz = self.iz[ind]
        it = self.it[ind]

        red_chi2 = self.red_chi2[ind]
        pb = self.pb[ind]
        p_bayes = self.p_bayes[ind]
        iz_b = self.iz_b[ind]
        zb = self.zb[ind]
        odds = self.odds[ind]
        it_b = self.it_b[ind]

        tt_b = self.tt_b[ind]
        tt_ml = self.tt_ml[ind]

        z1 = self.z1[ind]
        z2 = self.z2[ind]

#        opt_type = self.opt_type[ind]
        opt_type = 0.

        return iz, it, red_chi2, pb, p_bayes, iz_b, zb, odds, it_b, tt_b, tt_ml, z1, z2, opt_type
