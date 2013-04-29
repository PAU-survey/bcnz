#!/usr/bin/env python
# encoding: UTF8

from __future__ import print_function

import glob
import os
import pdb
import numpy as np

import bcnz

class zdata(dict):
    msg_filters = 'Missing filter files.'
    msg_seds = 'Missing sed files.'

    def __init__(self, conf):
        self.conf = conf

        self['z_model'] = self.z_binning()
        self['filters'] = self.use_filters()
        self['seds'] = self.conf['seds']

        self.check_filenames()
        self.add_texp()

    def z_binning(self):
        """Binning in redshift when calculating the model."""

        zmin = self.conf['zmin']
        zmax = self.conf['zmax']
        dz = self.conf['dz']

        return np.arange(zmin,zmax+dz,dz)

    def use_filters(self):
        """Filters to use."""

        filters = self.conf['filters']
        exclude = self.conf['exclude']

        filters = [x for x in filters if not x in exclude]
        assert filters, 'No filters specified.'

        return filters


    def check_filenames(self):
        """Test if the filter and sed files are there."""

        def sel_files(self, d, suf):
            g = os.path.join(self.conf['data_dir'], d,
                             '*.{0}'.format(suf))

            names = map(os.path.basename, glob.glob(g))
            names = map(lambda x: os.path.splitext(x)[0], names)

            return names

        filters_db = sel_files(self, self.conf['filter_dir'], 'res')
        seds_db = sel_files(self, self.conf['sed_dir'], 'sed')

        assert set(self['filters']).issubset(set(filters_db)), self.msg_filters
        assert set(self['seds']).issubset(set(seds_db)), self.msg_seds

    def add_texp(self):
        # To not require tray configurations to always be
        # present.
        if self.conf['add_noise']:
            zdata['texp'] = bcnz_exposure.texp(conf, zdata)

        return zdata
