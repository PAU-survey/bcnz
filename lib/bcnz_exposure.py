#!/usr/bin/env python
# encoding: UTF8
# Relating exposure times in different trays to exposure
# in filters. Very PAU specific and is therefore located
# in a separate file.

from __future__ import print_function
import numpy as np

import pdb

def find_tray_config(tray_file):
    """Read in tray configuration from file."""

    A = np.loadtxt(tray_file, dtype='string')

    return A

def find_texp_layout(conf, tray_conf):
    """Exposure time including vignetting for all the filters
       in the layout.
    """

    texp_tray = conf['exp_trays']
    texp_layout = np.zeros(tray_conf.shape)

    vign_cent, vign_perh, vign_corn = conf['vign']
    n_tray, _ = tray_conf.shape
    for i in range(n_tray):
        texp_layout[i][:8]   = vign_cent*texp_tray[i]
        texp_layout[i][8:14] = vign_perh*texp_tray[i]
        texp_layout[i][14:]  = vign_corn*texp_tray[i]

    return texp_layout

def exp_in_filters(tray_conf, texp_layout, filters):
    """Find the exposure time in each filter."""

    all_filters = tray_conf.flatten().tolist()

    texpD = dict((x, 0) for x in all_filters)
    for f, texp in zip(tray_conf.flatten(), \
                       texp_layout.flatten()):

        texpD[f] += texp

    texp = [texpD[x] for x in filters]

    return np.array(texp)

def texp(conf, zdata):
    """Exposure time for each filter."""

    tray_path = conf['trays']
    filters = zdata['filters']

    tray_conf = find_tray_config(tray_path)
    texp_layout = find_texp_layout(conf, tray_conf)
    res = exp_in_filters(tray_conf, texp_layout, filters)

    return res

