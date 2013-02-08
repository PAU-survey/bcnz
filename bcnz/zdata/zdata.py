#!/usr/bin/env python
# encoding: UTF8

from __future__ import print_function

import glob
import os
import pdb
import numpy as np

class zdata(dict):
    msg_filters = 'Missing filter files.'
    msg_seds = 'Missing sed filtes.'

    def __init__(self, conf):
        self.conf = conf

        self['z'] = self.z_binning()

        self.read_files()
        self.add_noise()

    def z_binning(self):
        zmin = self.conf['zmin']
        zmax = self.conf['zmax']
        dz = self.conf['dz']

        self['z'] = np.arange(zmin,zmax+dz,dz)

    def use_filters(self):
        return ['up', 'g']

    def read_files(self):

        def sel_files(self, d, suf):
            g = os.path.join(self.conf['data_dir'], 
                             self.conf['filter_dir'],
                             '*.{0}'.format(suf))

            names = map(os.path.basename, glob.glob(g))
            names = map(lambda x: os.path.splitext(x)[0], names)

            return names

        filters_db = sel_files(self, self.conf['filter_dir'], 'res')
        sed_db = sel_files(self, self.conf['sed_dir'], 'sed')

        filters = self.use_filters()

        assert set(filters_db).issubset(set(filters)), self.msg_filters
        assert set(seds_db).issubset(set(seds)), self.msg_seds

        pdb.set_trace()
        filters = bcnz_div.find_filters(conf)
        # TODO: CONTINUE HERE...
        spectra_file = bcnz_div.spectra_file(conf)
        seds = bcnz_div.find_spectra(spectra_file)

        self['filters'] = filters
        self['spectra'] = spectra

        pdb.set_trace()
    def add_noise(self):
        # To not require tray configurations to always be
        # present.
        if conf['add_noise']:
            zdata['texp'] = bcnz_exposure.texp(conf, zdata)

        return zdata




def catalogs(conf):
    """File names with input catalogs."""

    input_file = conf['catalog']
    cat_files = glob.glob(input_file)
    msg_noinput = "Found no input files for: %s" % input_file

    assert len(cat_files), msg_noinput

    return cat_files


