#!/usr/bin/env python
# encoding: UTF8
from __future__ import print_function

def z_binning(conf):
    zmin = conf['zmin']
    zmax = conf['zmax']
    dz = conf['dz']

    return np.arange(zmin,zmax+dz,dz)

def find_zdata(conf):
    zdata = {}
    zdata['z'] = bcnz_div.z_binning(conf)

    zdata = bcnz_filters.filter_and_so()(conf, zdata)

    # To not require tray configurations to always be
    # present.
    if conf['add_noise']:
        zdata['texp'] = bcnz_exposure.texp(conf, zdata)

    return zdata
