##!/usr/bin/env python
# encoding: UTF8
# Project into the different filters.

from __future__ import print_function
import numpy as np
import os
import re
import time
from scipy.interpolate import splrep, splev
from scipy.integrate import simps, trapz, quad
from scipy.ndimage.interpolation import zoom

import tables

np.seterr('raise')
np.seterr(under='ignore') # Caused a problem. We are only interested in the amplitude.

# Functions used when loading the results.
def _load_ab(ab_path):
    """Load the raw arrays from the AB files."""

    assert os.path.exists(ab_path), 'The import AB file does not exists.'

    fb = tables.openFile(ab_path)
    ab_grp = fb.getNode('/')

    ab = {}
    for fname_store, ab_filter_grp in ab_grp._v_children.iteritems():
        filter_name = fname_store if not fname_store.startswith('NB') \
                      else int(fname_store.replace('NB', ''))

        for sed_name, ab_obj in ab_filter_grp._v_children.iteritems():
            ab[filter_name, sed_name] = ab_obj.read()

    fb.close()

    return ab


class model_mag(object):
    def set_ninterp(self, filters):
        """Number of interpolation points for each filter."""

        # Quite high accuracy. It can be reduced, but in practice
        # you don't care since its only going to be calculated once.

        # Ok, becoming paranoid. Changing 200 to 1000.
        self.ninterp = dict((name, 1000) for name in filters)

    def __init__(self, conf, zdata):
        self.conf = conf
        self.zdata = zdata

        if 'sed_spls' in zdata:
            self.sed_spls = zdata['sed_spls']
       
        self.set_ninterp(zdata['filters'])


    def _ab_path(self):
        """Directory where the AB files are stored."""

        file_name = '{0}.hdf5'.format(self.conf['proj'])
        ab_path = os.path.join(os.environ['HOME'], self.conf['cache_dir'], \
                               self.conf['ab_dir'], file_name)

        return ab_path

    def _proj_filter(self, z_ab, seds, filter_name):
        """Project into filter."""

        xrlim = self.zdata['rlim'][filter_name]
        xedges = np.linspace(xrlim[0], xrlim[1], self.ninterp[filter_name])
        
        xm = .5*(xedges[:-1] + xedges[1:])
        xw = xedges[1:] - xedges[:-1]
        
        phi_spl = self.zdata['resp_spls'][filter_name]
        xpart = self.zdata['r_const'][filter_name]*xw*splev(xm, phi_spl, ext=1)
   
        a_ab = 1./(1.+z_ab)

        res = {}
        for sed in seds:
            sed_spl = self.sed_spls[sed]

            ker = np.outer(a_ab, xm)
            ker_shape = ker.shape
           
            # Not completely safe... 
            hmm = splev(ker.flatten(), sed_spl, ext=1).reshape(ker_shape)
            xpart = np.clip(xpart, 0, np.infty)
            hmm = np.clip(hmm, 0, np.infty)

            bla = np.sum(xpart * hmm, axis=1)
            AB = np.vstack([z_ab, bla]).T

#            file_path = os.path.join(d, '%s.%s.AB' % (sed, filter_name))
#            np.savetxt(file_path, AB)
            res[sed] = AB

            assert (0 <= AB).all(), 'Negative fluxes'

        return res


    def _all_proj(self):
        """Project into the combination of all filters and SEDS."""

        filters = self.zdata['filters']
        zmax_ab = self.conf['zmax_ab']
        dz_ab = self.conf['dz_ab']

        z_ab = np.arange(0., zmax_ab, dz_ab)

        # This will cause problems if switching the set of templates.
        # A proper solution should first detect all available templates
        # on the system.
        seds = self.zdata['seds'] 

        ab = {}
        for filter_name in filters:
            print('proj filter', filter_name)
            ab[filter_name] = self._proj_filter(z_ab, seds, filter_name)

        return ab

    def ensure_cached(self, ab_in=False):
        """Create cache of the AB files unless it exists."""

        ab_path = self._ab_path()
        if os.path.exists(ab_path):
            return

        ab = self._all_proj()

        descr = {'z': tables.FloatCol(pos=0),
                 'ab': tables.FloatCol(pos=1)}

        # Start writing the results to a file. The format is (in theory)
        # documented elsewhere.
        fb = tables.openFile(ab_path, 'w')
        for filter_name, abD in ab.iteritems():
            # A change of name is needed sine HDF5 does not support
            # storing nodes with numerical names.
            try:
                int(filter_name)
                fname_store = 'NB{0}'.format(filter_name)
            except ValueError:
                fname_store = filter_name

            fb.createGroup('/', fname_store)
            for sed_name, AB in abD.iteritems():
                grname = '/{0}'.format(fname_store)
                ab_table = fb.createTable(grname, sed_name, descr)
                ab_table.append(AB)
                ab_table.close()


        fb.close()


    def f_mod(self, z):
        """Create the array with expected fluxes for different SEDS,
           filters and redshifts.
        """

        seds = self.zdata['seds']
        filters = self.zdata['filters']
        z = self.zdata['z']

        if 'ab_df' in self.zdata:
            ab_df = self.zdata['ab_df']
        else:
            raise NotImplemented('Changed format.')
            abD = _load_ab(self._ab_path())

        # This method is not the fastest, but works slightly faster
        # than the linear interpolation in BPZ!
        ab_z = ab_df['z'] 
        f_mod = np.zeros((len(z), len(seds), len(filters)))
        for i,sed in enumerate(seds):
            for j,filter_name in enumerate(filters):
                val = ab_df['/{0}/{1}'.format(filter_name, sed)]
                spl = splrep(ab_z, val)

                y_new = splev(z, spl)

                f_mod[:,i,j] = y_new

        return f_mod

    def interp(self, f_mod):
        """Interpolation between spectras."""

        ntypes_orig = len(self.zdata['seds'])
        nt = self.conf['interp']*(ntypes_orig - 1) + ntypes_orig
        zoom_fac = (1, float(nt) / ntypes_orig, 1)

        f_new = zoom(f_mod, zoom_fac, order=1)

        return f_new

    def __call__(self):
        raise NotImplemented, 'If using this again, implement properly.'

        self.ensure_cached()
