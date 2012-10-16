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
        if conf['split_pop']:
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
        self.calc_D(0)

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

#        if not self.conf['opt']:
        D = D.reshape((h.shape[0], self.nz, self.nt))

#        pdb.set_trace()
        # Order after: type - redshift - ig
        D = D.swapaxes(0,2) 

        print('D', imin, D.shape)
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

#        pdb.set_trace()

#        self.opt_type = opt_type

#    @profile
    def find_zp_odds(self):
        self.calc_D(0)

        return self.zb, self.odds

#    @profile
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

    def __call__(self, ig):

        corr_imin = int(float(ig)/self.ngal_calc)*self.ngal_calc
        ind = ig - corr_imin

        if not self.imin <= ig < self.imax:
            self.calc_D(corr_imin)

        # Note that P1 contains information about all galaxies
        # and is therefore indexed by ig and *not* ind.

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

class chi2_combined:
#(conf, zdata, f_obs, ef_obs, m_0, z_s, ids, ngal_calc)
    def __init__(self, conf, zdata, f_obs, ef_obs, m_0, \
                 z_s, inds, ngal_calc):

        filters = zdata['filters']
        filter_id = filters.index('i')
        mag_lim = conf['mag_split']

        f_lim = 10.**(-0.4*mag_lim)

        ind_bright = f_lim < f_obs[:,filter_id]
        ind_faint = np.logical_not(ind_bright)

        self.chi2_bright = chi2_calc(conf, zdata, f_obs[ind_bright], ef_obs[ind_bright], \
                                     m_0[ind_bright], z_s[ind_bright], inds[ind_bright], \
                                     ngal_calc, conf['dz_bright'], conf['min_rms_bright'], 'bright')

        self.chi2_faint = chi2_calc(conf, zdata, f_obs[ind_faint], ef_obs[ind_faint], \
                                    m_0[ind_faint], z_s[ind_faint], inds[ind_faint], \
                                    ngal_calc, conf['dz_faint'], conf['min_rms_faint'], 'faint')

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

        new_block = np.zeros(self.ngal, dtype=self.chi2_bright.dtype)
        to_iter = [(self.ind_bright, self.chi2_bright),\
                   (self.ind_faint, self.chi2_faint)]

        print('in blocks')
        for ind, pop_chi2 in to_iter:
            nparts = int(np.ceil(len(ind)/ float(self.ngal_calc)))
            s = self.ngal_calc*np.arange(self.ngal_calc)

            splitted = np.split(ind.nonzero()[0], s[1:])
#            pdb.set_trace()
            for i, p_block in enumerate(pop_chi2.blocks()):
                new_block[splitted[i]] = p_block

#            for ind_part, p_block in zip(splitted, pop_chi2.blocks()):
#                print('here')
#                pdb.set_trace()
#
#                imin = n*self.ngal_calc
#                imax = min((n+1)*self.ngal_calc, self.ngal)
#                part_ind = imin + ind[imin:imax].nonzero()[0]
#
#                new_block[part_ind] = p_block

        yield new_block
#                print('here!!!!')
#                pdb.set_trace()

#    def OLD_blocks(self):
#        """Iterate over galaxy blocks merged from the bright and faint."""
#
#        # Not the most elegant conde, but it was difficult getting it
#        # right. Abstractions are badly needed.
#        nblocks = int(np.ceil(float(self.ngal) / self.ngal_calc))
#        ind_bright = self.ind_bright
#        ind_faint = self.ind_faint
#        ncalc = self.ngal_calc
#
#        block_bright = self.chi2_bright._block(0)
#        block_faint = self.chi2_faint._block(0)
#
#        b_blocknr, f_blocknr, nb, nf = 0, 0, 0, 0
#        for n in np.arange(nblocks):
#            imin = n*self.ngal_calc
#            imax = min((n+1)*self.ngal_calc, self.ngal)
#            new_block = np.zeros(imax-imin, dtype=block_bright.dtype)
#
#            b_ind = ind_bright[imin:imax].nonzero()[0]
#            f_ind = ind_faint[imin:imax].nonzero()[0]
#
#            # Number of galaxies in the current and next block.
#            b_sum,f_sum = len(b_ind), len(f_ind)
#            b1 = min(b_sum, ncalc)
#            f1 = min(f_sum, ncalc)
#            b2 = b_sum - b1
#            f2 = f_sum - f1
#
#            # It complains on empty indices.
#            if b1: new_block[b_ind[:b1]] = block_bright[nb:nb+b1]
#            if f1: new_block[f_ind[:f1]] = block_faint[nf:nf+f1]
#
#            nb += b1
#            nf += f1
#
#            if b2 or b1 == ncalc:
#                # Fetch new bright block.
#                b_blocknr += 1
#                block_bright = self.chi2_bright._block(b_blocknr)
#                nb = 0
#                if b2:
#                    #new_block[b_ind[b1:b2]] = block_bright[:b2] 
#                    new_block[b_ind[b1:]] = block_bright[:b2] 
#                    nb = b2
#
#            if f2 or f1 == ncalc:
#                # Fetch new faint block.
#                f_blocknr += 1
#                block_faint = self.chi2_faint._block(f_blocknr)
#                nf = 0
#                if f2:
#                    new_block[b_ind[f1:]] = block_bright[:f2] 
#                    nf = f2
#            
#            if 13069 in new_block['id']:
#                pdb.set_trace()
#
#            if 1 < np.sum(new_block['id'] == 0):
#                pdb.set_trace()
##            nb = b2
##            nf = f2
#
#            yield new_block[:imax-imin]

#    def blocks(self):
#        """Iterate over blocks of galaxies."""
#
#        nblocks = int(np.ceil(float(self.ngal) / self.ngal_calc))
#        self.blah()
#        for n in np.arange(nblocks):
#            yield self._block(n)


#@def chi2_inst(conf, zdata, f_obs, ef_obs, m_0, z_s, ngal_calc=100):
def chi2_inst(conf, zdata, f_obs, ef_obs, m_0, z_s, ids, ngal_calc=100):
    """Select which chi2 object to use depending on splitting in magnitudes
       or not.
    """

    if conf['split_pop']:
        return chi2_combined(conf, zdata, f_obs, ef_obs, m_0, z_s, ids, ngal_calc)
    else:
        dz = conf['dz']
        min_rms = conf['min_rms']

        return chi2_calc(conf, zdata, f_obs, ef_obs, m_0, z_s, ids, ngal_calc, dz,min_rms)
