#!/usr/bin/env python
# encoding: UTF8

import os
import pdb
import time

import numpy as np
import bcnz

def add_model(conf, zdata):
    """Add model predictions."""

    def binning(pop):
        """Model redshift binning for population."""

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

    model = bcnz.model.model_mag(conf, zdata)
    for key, z in to_iter:
        f_mod2 = model.f_mod(z)
        f_mod2 = model.interp(conf, f_mod2, z, self.filters, self.spectra)
        zdata[key] = f_mod2

    f_mod = f_mod2

    return zdata
