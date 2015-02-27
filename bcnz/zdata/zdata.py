#!/usr/bin/env python
# encoding: UTF8

from __future__ import print_function

import glob
import os
import ipdb
import numpy as np

import bcnz
import bcnz.model

class zdata(dict, object):
    msg_filters = 'Missing filter files: {0}'
    msg_seds = 'Missing sed files: {0}'

    def __init__(self, conf, minimal=False):
        self.conf = conf
        self['z_model'] = self.z_binning()
        self['filters'] = self.use_filters()
        self['seds'] = self.conf['seds']

        # Should be calculated at the same time as the binning, but
        # this is one step forward!
        self['dz'] = self.conf['dz']*np.ones_like(self['z_model'])

        if not minimal:
            self.check_filenames()
            self.update(bcnz.model.sed_filters()(conf, self))
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
                msg = '\nMissing files: {missing}\nFound: {found}\nGlob: {g}'.\
                      format(missing=missing_files, found=file_names, g=g)

                raise Exception(msg)

        filters_db = sel_files(self, self['filters'], self.conf['filter_dir'], self.conf['res_fmt'])
        seds_db = sel_files(self, self['seds'], self.conf['sed_dir'], self.conf['sed_fmt'])


    def add_texp(self):
        # To not require tray configurations to always be
        # present.
        if self.conf['add_noise']:
            self['t_exp'] = bcnz.zdata.noise.texp(self.conf, self['filters'])

            # Passing the object itself to another function is not really normal
            self.update(bcnz.zdata.noise.sn_spls(self.conf, self))
