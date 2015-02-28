#!/usr/bin/env python
# encoding: UTF8
# First iteration on a code to do Bayesian template 
# fitting to determine galaxy type and redshift.

import ipdb
import pdb
import time
import sys
import numpy as np
from scipy.ndimage.interpolation import zoom
import scipy.stats

import bcnz
import bcnz.priors

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
    jmin = np.minimum(jmin, p.shape[1] - 1)
    jmax = np.minimum(jmax, p.shape[1] - 1)

    # These galaxies are very uncertain. Also set a large probability
    # range in case someone use this to cut.
    useless = cdf[:,-1] < 0.999
    xa = x[jmin]
    xb = x[jmax]
    xa[useless] = 0.
    xb[useless] = 10.

    return xa, xb


def find_odds(p,x,xmin,xmax):
    """Probabilities in the given intervals."""

    cdf = p.cumsum(axis=1)

    def K(xlim):
        return np.searchsorted(x, xlim)

    # The clipping here is needed because the pdf is not always 0 
    # at zero redshift, which cause funny problems in the cdf, including
    # negative indices.
    imin = np.apply_along_axis(K, 0, xmin) - 1
    imin = np.clip(imin, 0, np.infty).astype(np.int)

#    ipdb.set_trace()
    imax = np.apply_along_axis(K, 0, xmax)
    imax = np.minimum(imax, p.shape[1] - 1)
    gind = np.arange(p.shape[0])

    # The ones not being able to properly calculate the cdf normalization
    # is known to have very high chi^2 values.
    odds = (cdf[gind,imax] - cdf[gind,imin])/cdf[:,-1]
    odds[cdf[:,-1] < 0.999] = 0.


    return odds

class chi2_calc(object):
    def __init__(self, conf, zdata, data, priors=None):
        assert data['mag'].shape[0], 'No galaxies'
        self.conf = conf
        self.zdata = zdata
        self.data = data

        # These was previously passed.
        self.dz = conf['dz']
        self.min_rms = conf['min_rms']

        self.f_mod = zdata['f_mod']
        z = zdata['z']

        self.z = z
        self.cols, self.dtype = self.cols_dtype()

        # Priors
        self.priors = self.load_priors(conf, z) if priors is None else priors

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

        # This should be passed to the module. I have changed this in the 
        # xdolphin wrapper, but not in general.
        prior_name = 'prior_%s' % conf['prior']
        try:
            return getattr(bcnz.priors, conf['prior'])(conf, self.zdata, z, self.data['m0'])

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

        output = {}
        if self.conf['out_pdf']:

            dzbin = self.conf['dz']

            pdf = pb.sum(axis=2)
            norm = 1./(dzbin*pdf.sum(axis=1))

            pdf = (pdf.T * norm).T

            output['pzpdf'] = pdf

        if self.conf['out_pdftype']:
            norm = np.einsum('ijk,j->i', pb, self.zdata['dz'])
            pdf_type = np.einsum('ijk,i->ijk', pb, 1/norm)
       
            output['pzpdf_type'] = pdf_type 

        # Add priors.
        if self.conf['use_priors']:
            m = self.data['m0'][imin:imax]
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
        odds = find_odds(p_bayes, self.z, zb-dz, zb+dz)


        it_b = pb[range(len(zb)),iz_b].argmax(axis=1)
        interp = self.conf['interp']
        tt_b = it_b / (1. + interp)
        tt_ml = it / (1. + interp)

        A = pb[range(len(zb)),it_b]
        test = pb[range(len(zb)),:,it_b].max(axis=1) < 1e-300
        it_b = np.where(test,  -1., it_b)
        tt_b = np.where(test,  -1., tt_b)

        zb_min, zb_max = prob_interval(p_bayes, self.z, self.conf['odds'])

        # Actual number of galaxies in the block.
        ngal = min(imax, self.data['id'].shape[0]) - imin
        peaks = np.zeros(ngal, self.dtype)
        peaks['id'] = self.data['id'][imin:imax]
        peaks['zb'] = self.z[iz_b]
        peaks['zb_min'] = zb_min
        peaks['zb_max'] = zb_max
        peaks['t_b'] = tt_b + 1 
        peaks['odds'] = odds
        peaks['z_ml'] = self.z[iz]
        peaks['t_ml'] = tt_ml + 1
        peaks['chi2'] = red_chi2

        for key in ['zs', 'ra', 'dec', 'spread_model_i', 'm0']:
            if key in self.conf['order']:
                peaks[key] = self.data[key][imin:imax]

        output['pzcat'] = peaks

        return output

    def blocks(self):
        """Iterate over the different blocks."""

        ngal = len(self.data['id'])
        ncalc = self.conf['ncalc']
        nblocks = int(np.ceil(float(ngal) / ncalc))

        for n in np.arange(nblocks):
            yield self._block(n)

def chi2(config, zdata, data, priors=None):
    """Select which chi2 object to use depending on splitting in magnitudes
       or not.
    """

    assert not config['use_split'], 'Splitting the catalog is no longer supported.'

    return chi2_calc(config, zdata, data, priors)
