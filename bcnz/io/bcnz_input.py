#!/usr/bin/env python
# encoding: UTF8

import os
import glob
import pdb
import sys

import tables
import numpy as np

def mapping_templ_prior(file_name):
    """Read file with mapping between templates
       and priors names.
    """

    data = basic_read(file_name)   
    d = dict([x.split() for x in data])

    return d

def spectras(file_name):
    """Read in list of spectras."""

    data = basic_read(file_name)

    return data

def templ_prior(f_templ_pr, specs):
    """List of priors corresponding to the spectras 
       we use.
    """

    d = mapping_templ_prior(f_templ_pr)

    return [d[x.replace('.sed','')] for x in specs]

def DM_filter_names(filters):
    """The notation which PAU-DM is using for the filters."""

    known_broadband = ['i', 'up', 'g', 'r', 'z', 'y']
    new_names = []
    i = 0
    for f in filters:
        if f in known_broadband:
            new_names.append(f)
        else:
            new_names.append(str(i))
            i += 1

    return new_names
