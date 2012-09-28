#!/usr/bin/env python
from __future__ import print_function
# Project into the different filters.
import numpy as np
import os
import pdb
import re
import time
from scipy.interpolate import splrep, splev
from scipy.integrate import simps, trapz, quad

np.seterr('raise')

class model_mag:
    def set_ninterp(self, filters):
        """Number of interpolation points for each filter."""

        self.ninterp = dict((name,200) for name in filters)

#        nri = [(30 if re.match('PAU/\d+.res', name) else 200)  for name in self.all_filters]
#        self.ninterp = dict((name, x) for name, x in zip(self.all_filters, nri))


    def __init__(self, conf, zdata):
        self.conf = conf
        self.zdata = zdata

        self.sed_spls = zdata['sed_spls']
        self.set_ninterp(zdata['filters'])

    def proj_filter(self, d, z_ab, filter_name):
        """Project into filter."""

        xrlim = self.zdata['rlim'][filter_name]
        xedges = np.linspace(xrlim[0], xrlim[1],
                             self.ninterp[filter_name])
        
        xm = .5*(xedges[:-1] + xedges[1:])
        xw = xedges[1:] - xedges[:-1]
        
        phi_spl = self.zdata['resp_spls'][filter_name]
        xpart = self.zdata['r_const'][filter_name]*xw*splev(xm, phi_spl)
   
 
        a_ab = 1./(1.+z_ab)

        res = {}
        for sed in self.zdata['spectra']:
            sed_spl = self.sed_spls[sed]

            ker = np.outer(a_ab, xm)
            ker_shape = ker.shape
            
            hmm = splev(ker.flatten(), sed_spl).reshape(ker_shape)
            bla = np.sum(xpart * hmm, axis=1)
            AB = np.vstack([z_ab, bla]).T

            file_path = os.path.join(d, '%s.%s.AB' % (sed, filter_name))
            np.savetxt(file_path, AB)

        return res

    def all_proj(self):
        filters = self.zdata['filters']
        zmax_ab = self.conf['zmax_ab']
        dz_ab = self.conf['dz_ab']

        z_ab = np.arange(0., zmax_ab, dz_ab)
        d = self.conf['ab_tmp']
        for filter_name in filters:
            self.proj_filter(d, z_ab, filter_name)

#            pdb.set_trace()

    def f_mod(self):
        d = self.conf['ab_tmp']
        spectra = self.zdata['spectra']
        filters = self.zdata['filters']
        z = self.zdata['z']

        # This method is not the fastest, but works slightly faster
        # than the linear interpolation in BPZ!

        f_mod = np.zeros((len(z), len(spectra), len(filters)))
        t1 = time.time()
        for i,sed in enumerate(spectra):
            for j,filter_name in enumerate(filters):
                file_name = '%s.%s.AB' % (sed, filter_name)
                file_path = os.path.join(d, file_name)

                x,y = np.loadtxt(file_path, unpack=True)
                spl = splrep(x,y)
                y_new = splev(z, spl)

                f_mod[:,i,j] = y_new

        t2 = time.time()

        print('time splev', t2-t1)
#        pdb.set_trace()
        return f_mod

    def interp(self, conf, f_mod, z, filters, spectra):
        """Interpolation between spectras."""

        ninterp = conf['interp']
        if not ninterp:
            return f_mod

        nz = len(z)
        nt = len(spectra)
        nf = len(filters)

        # Index of the first type in the linear interpolation
        ftype = np.repeat(np.arange(nt-1), ninterp+1)
        btype = np.array(list(ftype)+[nt-1])

        f_new = f_mod[:,btype,:]
        df = f_mod[:,1:,:] - f_mod[:,:-1,:]
        df_part = df[:,ftype,:] 

        # Weights for each of the interpolation points. 
        w = np.tile(np.arange(ninterp+1), nt-1)/(ninterp+1.)

        for i, wi in enumerate(w):
            f_new[:,i,:] += wi*df_part[:,i,:]

        return f_new

    def __call__(self):
        self.all_proj()
