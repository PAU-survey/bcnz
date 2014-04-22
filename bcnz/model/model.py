#!/usr/bin/env python
# encoding: UTF8

import os
import pdb
import time

import numpy as np
import bcnz

def add_binning(conf, zdata):
    """Model redshift binning for populations."""

    # Why is this here...

    if conf['use_split']:
        for pop in ['bright', 'faint']:
            dz = conf['dz_{0}'.format(pop)]
            binning = np.arange(conf['zmin'],conf['zmax']+dz,dz)

            zdata['{0}.z'.format(pop)] = binning
    else:
        dz = conf['dz']
        zdata['z'] = np.arange(conf['zmin'],conf['zmax']+dz,dz)

    return zdata

def add_model(conf, zdata, only_initial=False):
    """Add model predictions."""

    zdata = add_binning(conf, zdata)
    resp_inst = bcnz.model.sed_filters()
    zdata.update(resp_inst(conf,zdata))
    
    if only_initial:
        return zdata

    model = bcnz.model.model_mag(conf, zdata)
    model()
    to_iter = [('bright.f_mod', zdata['bright.z']), ('faint.f_mod', zdata['faint.z'])] \
              if conf['use_split'] else [('f_mod', zdata['z'])]

    for key, z in to_iter:
        f_mod = model.f_mod(z)
        f_mod = model.interp(f_mod)
        zdata[key] = f_mod

    return zdata
