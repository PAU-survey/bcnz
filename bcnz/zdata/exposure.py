#!/usr/bin/env python
# encoding: UTF8
# Relating exposure times in different trays to exposure
# in filters. Very PAU specific and is therefore located
# in a separate file.

from __future__ import print_function
import numpy as np

import pdb

def texp(conf, filters):
    """Exposure time for each filter."""

    tall = {}
    # Narrow band exposures.
    for i,ftray in enumerate(conf['trays']):
        t_tray = conf['exp_t{0}'.format(i+1)]

        for f in ftray:
            tall[f] = t_tray

    # Sorry...
    for f in ['up', 'g', 'r', 'i', 'z', 'y']:
        tall[f] = conf['exp_{0}'.format(f)]

    texp = [tall[f] for f in filters]
