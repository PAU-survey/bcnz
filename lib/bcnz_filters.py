#!/usr/bin/env python
# encoding: UTF8
import pdb
import time

import numpy as np

import bpz_color_interp
import bpz_model

import bcnz_div
import bcnz_input
import bcnz_model
import bcnz_norm
import bcnz_sedf
#import bcnz_interp

class filter_and_so:
    def blah(self, zdata):
        conf = self.conf

        filters_db = bcnz_div.sel_files(conf['filter_dir'], '.res')
        sed_db = bcnz_div.sel_files(conf['sed_dir'], '.sed')
        ab_db = bcnz_div.sel_files(conf['ab_dir'], '.AB')

        filters = bcnz_div.find_filters(conf)

        spectra_file = bcnz_div.spectra_file(conf)
        spectra = bcnz_div.find_spectra(spectra_file)

        bcnz_div.check_found('Filters', filters, filters_db)
        bcnz_div.check_found('Spectras', spectra, sed_db)

        zdata['filters'] = filters
        zdata['spectra'] = spectra
        zdata['ab_db'] = ab_db
        zdata['sed_db'] = sed_db

        return zdata

    def tja(self, zdata):
        conf = self.conf
        self.z = zdata['z']
        self.filters = zdata['filters']
        self.spectra = zdata['spectra']
        self.ab_db = zdata['ab_db']

        if conf['old_model']:
            # Old approach..
            t1 = time.time()
            f_mod = bpz_model.gen_model(conf, zdata)
            f_mod = bpz_color_interp.interp(conf, f_mod, self.z, self.filters, self.spectra)

            t2 = time.time()
#            print('time bpz', t2-t1)
        else:
            # New approach..
            sed_filt_inst = bcnz_sedf.sed_filters()
            zdata = sed_filt_inst(conf, zdata)

            t3 = time.time()
            inst = bcnz_model.model_mag(conf, zdata)
            f_mod2 = inst.f_mod()
            f_mod2 = inst.interp(conf, f_mod2, self.z, self.filters, self.spectra)
            t4 = time.time()

            f_mod = f_mod2
#            print('time bcnz', t4-t3)


        col_pars = bcnz_div.find_columns(conf['col_file'])

#        flux_cols, eflux_cols, cals, zp_errors, zp_offsets = \
#        bcnz_input.split_col_pars(col_pars, filters)

        col_data = bcnz_input.split_col_pars(col_pars, self.filters)
        zdata.update(col_data)

        if conf['zp_offsets']:
            zdata['zp_offsets'] += conf['zp_offsets']

        zdata['f_mod'] = f_mod
        zdata['col_pars'] = col_pars

        return zdata

    def __call__(self, conf, zdata):
        self.conf = conf

        zdata = self.blah(zdata)
        zdata = self.tja(zdata)

        return zdata
