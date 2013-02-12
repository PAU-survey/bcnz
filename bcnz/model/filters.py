#!/usr/bin/env python
# encoding: UTF8

import os
import pdb
import time

import numpy as np

class mag_model:
    def __init__(self, conf, zdata):
        self.conf = conf
        self.zdata = zdata

    def tja(self, zdata):
        conf = self.conf
#        self.z = zdata['z']
        self.filters = zdata['filters']
        self.spectra = zdata['spectra']
        self.ab_db = zdata['ab_db']

        def binning(pop):
            dz = conf['dz_{0}'.format(pop)]

            return np.arange(conf['zmin'],conf['zmax']+dz,dz)

        if conf['use_split']:
            for pop in ['bright', 'faint']:

                zdata['{0}.z'.format(pop)] = binning(pop)

        # New approach..
        sed_filt_inst = bcnz_sedf.sed_filters()
        zdata = sed_filt_inst(conf, zdata)

        if conf['use_split']:
            to_iter = [('bright.f_mod', zdata['bright.z']), ('faint.f_mod', zdata['faint.z'])]
        else:
            to_iter = [('f_mod', zdata['z'])]

        model = bcnz_model.model_mag(conf, zdata)
        for key, z in to_iter:
            f_mod2 = model.f_mod(z)
            f_mod2 = model.interp(conf, f_mod2, z, self.filters, self.spectra)
            zdata[key] = f_mod2

        f_mod = f_mod2

        col_path = os.path.join(conf['data_dir'], conf['columns'])
        col_pars = bcnz_div.find_columns(col_path)

        col_data = bcnz_input.split_col_pars(col_pars, self.filters)
        zdata.update(col_data)

        if conf['zp_offsets']:
            zdata['zp_offsets'] += conf['zp_offsets']

        zdata['col_pars'] = col_pars

        return zdata

    def __call__(self, conf, zdata):
        self.conf = conf
        zdata = self.tja(zdata)

        return zdata
