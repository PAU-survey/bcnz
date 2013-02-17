#!/usr/bin/env python
# encoding: UTF8
# Project into the different filters.

from __future__ import print_function
import numpy as np
import os
import pdb
import re
import time
from scipy.interpolate import splrep, splev
from scipy.integrate import simps, trapz, quad
from scipy.ndimage.interpolation import zoom

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
        for sed in self.zdata['seds']:
            sed_spl = self.sed_spls[sed]

            ker = np.outer(a_ab, xm)
            ker_shape = ker.shape
            
            hmm = splev(ker.flatten(), sed_spl).reshape(ker_shape)
            bla = np.sum(xpart * hmm, axis=1)
            AB = np.vstack([z_ab, bla]).T

            file_path = os.path.join(d, '%s.%s.AB' % (sed, filter_name))
            np.savetxt(file_path, AB)

        return res

    def f_mod(self, z):
        """Model frequencies."""

        dir_ab = os.path.join(self.conf['cache_dir'], 'ab')

        seds = self.zdata['seds']
        filters = self.zdata['filters']
        z = self.zdata['z']

        # HACK. Does not handle new filters added..
        if not os.listdir(dir_ab):
            self.all_proj()

        # This method is not the fastest, but works slightly faster
        # than the linear interpolation in BPZ!
        f_mod = np.zeros((len(z), len(seds), len(filters)))
        for i,sed in enumerate(seds):
            for j,filter_name in enumerate(filters):
                file_name = '%s.%s.AB' % (sed, filter_name)
                file_path = os.path.join(dir_ab, file_name)

                x,y = np.loadtxt(file_path, unpack=True)
                spl = splrep(x,y)
                y_new = splev(z, spl)

                f_mod[:,i,j] = y_new


        return f_mod

    def all_proj(self):
        filters = self.zdata['filters']
        zmax_ab = self.conf['zmax_ab']
        dz_ab = self.conf['dz_ab']

        z_ab = np.arange(0., zmax_ab, dz_ab)
        d = os.path.join(self.conf['cache_dir'], self.conf['ab_dir'])
        for filter_name in filters:
            self.proj_filter(d, z_ab, filter_name)


    def interp(self, f_mod):
        """Interpolation between spectras."""

        ntypes_orig = len(self.zdata['seds'])
        nt = self.conf['interp']*(ntypes_orig - 1) + ntypes_orig
        zoom_fac = (1, float(nt) / ntypes_orig, 1)

        f_new = zoom(f_mod, zoom_fac, order=1)

#        pdb.set_trace()
        return f_new

    def __call__(self):
        self.all_proj()
