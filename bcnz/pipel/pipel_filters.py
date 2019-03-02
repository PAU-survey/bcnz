#!/usr/bin/env python
# encoding: UTF8

import os
import xdolphin as xd

def filters(fjc_filters, new_bbfilters):
    """The filter curves."""

    # Here the default is intentionally the same wrong filter
    # curves which Alex is using...
    if fjc_filters:
        pau_filters = xd.Job('fjc_filters')
    else:
        pau_filters = xd.Job('paudm_filters')

    F = xd.Job('join_output')
    F.depend['pau'] = pau_filters

    # Scaled down version of the function I use for the
    # calibration ...
    if new_bbfilters:
        F.depend['cosmos_bb'] = xd.Job('cosmos_filters')
    else:
        F.depend['cfht_filters'] = xd.Job('cfht_filters')
        F.depend['subaru_filters'] = xd.Job('subaru_filters')


    return F
