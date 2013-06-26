#!/usr/bin/env python
# encoding: UTF8

from __future__ import print_function

import glob
import os
import pdb
import numpy as np

import bcnz

class zdata(dict, object):
    msg_filters = 'Missing filter files: {0}'
    msg_seds = 'Missing sed files: {0}'

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

        def sel_files(self, tofind, d, fmt):
            g = os.path.join(self.conf['data_dir'], d, fmt.replace('{0}', '*'))

            file_names = map(os.path.basename, glob.glob(g))
            missing_files = []
            for o in tofind:
                file_name = fmt.format(o)
                if not file_name in file_names:
                    missing_files.append(file_name)

            if missing_files:
                raise Exception('Missing files: {0}'.format(missing_files))

        filters_db = sel_files(self, self['filters'], self.conf['filter_dir'], self.conf['res_fmt'])
        seds_db = sel_files(self, self['seds'], self.conf['sed_dir'], self.conf['sed_fmt'])


    def add_texp(self):
        # To not require tray configurations to always be
        # present.
        if self.conf['add_noise']:
            zdata['texp'] = bcnz_exposure.texp(conf, zdata)

        return zdata
