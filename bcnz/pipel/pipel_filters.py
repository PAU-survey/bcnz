#!/usr/bin/env python
# encoding: UTF8

import os
import xdolphin as xd

def filters(): #fjc_filters):
    """The filter curves."""

    # Scaled down version of the function I use for the
    # calibration ...
    cfht_filters = xd.Job('cfht_filters')
    subaru_filters = xd.Job('subaru_filters')

    # TODO: Replace this with the official ones from PAUdm..
    pau_filters = xd.Job('pau_filters')

    F = xd.Job('join_output')
    F.depend['pau'] = pau_filters
    F.depend['cfht'] = cfht_filters
    F.depend['subaru'] = subaru_filters

    return F
