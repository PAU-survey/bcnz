#!/usr/bin/env python
# encoding: UTF8
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

# <testing>
import math
def texp(n):
    if n == 0:
        return '1'
    else:
        part = '+ (-pb2)**{0}/{1}.'.format(n, math.factorial(n))

    return texp(n-1)+part
# </testing>


np.seterr(under='ignore')
test_min = False
class chi2_calc:
#    @profile
    def __init__(self, conf, zdata, f_obs, ef_obs, m_0, \
                 z_s, ids, ngal_calc, dz, min_rms, pop=''):

#        pdb.set_trace()
        assert f_obs.shape[0], 'No galaxies'
        self.conf = conf
        self.zdata = zdata
        self.f_obs = f_obs
        self.ef_obd = ef_obs
        if conf['use_split']:
            self.f_mod = zdata['{0}.f_mod'.format(pop)]
            z = zdata['{0}.z'.format(pop)]
        else:
            self.f_mod = zdata['f_mod']
            z = zdata['z']

        self.ngal_calc = ngal_calc

#        pdb.set_trace()
#        self.z = zdata['z']
#        self.z = np.arange(conf['zmin'],conf['zmax']+dz,dz)

        self.m_0_data = m_0
        self.z_s_data = z_s
        self.ids_data = ids
        self.min_rms = min_rms

        self.cols, self.dtype = self.cols_dtype()

        # Priors
        self.load_priors(conf, z)
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
        self.odds_pre = oi * self.min_rms #conf['min_rms']
        self.z = z

        # Precalculate values.
        self.calc_P1()

    def cols_dtype(self):
        """Columns and block dtype."""

        int_cols = ['id']
        cols = ['id']+self.conf['order']+self.conf['others']
        dtype = [(x, 'float' if not x in int_cols else 'int') for x in cols]

        return cols, dtype



    def load_priors(self, conf, z):
        prior_name = 'prior_%s' % conf['prior']
        try:
            self.priors = getattr(priors, prior_name)(conf, \
                          self.zdata, z, self.m_0_data)

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

#        pdb.set_trace()

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
            m =  self.m_0_data[imin:imax]
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
        self.z_ml = self.z[iz]
        self.m_0 = self.m_0_data[imin:imax]
        self.z_s = self.z_s_data[imin:imax]
        self.id = self.ids_data[imin:imax]

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

#    def find_zp_odds(self):
#        self.calc_D(0)
#
#        return self.zb, self.odds

    def _block(self, n):
        imin = n*self.ngal_calc
        self.calc_D(imin)

        self.zb_min = np.array(self.z1)
        self.zb_max = np.array(self.z2)
        self.t_b = self.tt_b+1
        self.chi2 = self.red_chi2
        self.t_ml = self.tt_ml+1


        A = [getattr(self, x) for x in self.cols]

        return np.rec.fromarrays(A, self.dtype)

    def blocks(self):
        """Iterate over the different blocks."""

#        pdb.set_trace()
        ngal = len(self.z_s_data)
        nblocks = int(np.ceil(float(ngal) / self.ngal_calc))

#        pdb.set_trace()
        for n in np.arange(nblocks):
            yield self._block(n)

#        yield

#        pdb.set_trace()
#        return np.array(np.vstack(A).T, dtype)
#
#    def __call__(self, ig):
#
#        corr_imin = int(float(ig)/self.ngal_calc)*self.ngal_calc
#        ind = ig - corr_imin
#
#        if not self.imin <= ig < self.imax:
#            self.calc_D(corr_imin)
#
#        # Note that P1 contains information about all galaxies
#        # and is therefore indexed by ig and *not* ind.
#
#        iz = self.iz[ind]
#        it = self.it[ind]
#
#        red_chi2 = self.red_chi2[ind]
#        pb = self.pb[ind]
#        p_bayes = self.p_bayes[ind]
#        iz_b = self.iz_b[ind]
#        zb = self.zb[ind]
#        odds = self.odds[ind]
#        it_b = self.it_b[ind]
#
#        tt_b = self.tt_b[ind]
#        tt_ml = self.tt_ml[ind]
#
#        z1 = self.z1[ind]
#        z2 = self.z2[ind]
#
#        
#        opt_type = 0.
#
#        return iz, it, red_chi2, pb, p_bayes, iz_b, zb, odds, it_b, tt_b, tt_ml, z1, z2, opt_type
#
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

        """
        self.chi2_bright = chi2_calc(conf, zdata, f_obs[ind_bright], ef_obs[ind_bright], \
                                     m_0[ind_bright], z_s[ind_bright], inds[ind_bright], \
                                     ngal_calc, conf['dz_bright'], conf['min_rms_bright'], 'bright')

        self.chi2_faint = chi2_calc(conf, zdata, f_obs[ind_faint], ef_obs[ind_faint], \
                                    m_0[ind_faint], z_s[ind_faint], inds[ind_faint], \
                                    ngal_calc, conf['dz_faint'], conf['min_rms_faint'], 'faint')
        """

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
            chi2_pop = chi2_calc(self.conf, self.zdata, self.f_obs[ind], self.f_obs[ind], \
                                 self.m_0[ind], self.z_s[ind], self.inds[ind], \
                                 self.ngal_calc, dz, min_rms, pop)

            if not hasattr(self, 'new_block'):
                new_block = np.zeros(self.ngal, dtype=chi2_pop.dtype)


            nparts = int(np.ceil(len(ind)/ float(self.ngal_calc)))
            s = self.ngal_calc*np.arange(self.ngal_calc)

            splitted = np.split(ind.nonzero()[0], s[1:])
            for i, p_block in enumerate(chi2_pop.blocks()):
#                if not p_
                new_block[splitted[i]] = p_block

        yield new_block

def chi2_inst(conf, zdata, f_obs, ef_obs, m_0, z_s, ids, ngal_calc=100):
    """Select which chi2 object to use depending on splitting in magnitudes
       or not.
    """

    if conf['use_split']:
        return chi2_combined(conf, zdata, f_obs, ef_obs, m_0, z_s, ids, ngal_calc)
    else:
        dz = conf['dz']
        min_rms = conf['min_rms']

        return chi2_calc(conf, zdata, f_obs, ef_obs, m_0, z_s, ids, ngal_calc, dz,min_rms)
