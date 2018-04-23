#!/usr/bin/env python
# encoding: UTF8

import copy
import ipdb
import numpy as np

import xdolphin as xd
import libcommon

def get_ab():
    """The input to the flux model."""

    seds = xd.Job('curves')
    seds.config.update({'input_dir': '/home/eriksen/data/photoz/seds/cosmos_self', 'suf': 'sed', 'min_val': 0.0})

    extinction = xd.Job('extinction_lagaile')

    ab_cont = xd.Job('ab_cont')
    ab_cont.config.update({'dz_ab': 0.001, 'zmax_ab': 2.05})
    ab_cont.depend['seds'] = seds
    ab_cont.depend['filters'] = libcommon.Common('filters')
    ab_cont.depend['extinction'] = extinction

    ab_lines = xd.Job('emission_lines')
    ab_lines.depend['filters'] = libcommon.Common('filters')
    ab_lines.depend['extinction'] = extinction

    return ab_cont, ab_lines

def get_pzcat_config():
    """The default photo-z configuration."""

    config = {
      'dz': 0.001,
      'zmax': 1.0,
      'Niter': 200,
    #  'Niter': 400,
    }

    config['filters'] = list(map('NB{}'.format, range(455, 845+1, 10)))
    # Add the SEDs..
    ell = ['ell2','ell5','ell13']
    sp = ['s0','sa','sb','sc','sd']
    sb = sb = map('sb{}'.format, range(11))
    seds = ell + sp + list(sb)
    config['seds'] = seds

    return config

def get_model():
    """The part returning the model."""

    model = xd.Job('flux_model')
    model.config.update({'dz': 0.001, 'zmax': 1.0})
    ab, ab_lines = get_ab()

    model.depend['ab'] = ab
    model.depend['ab_lines'] = ab_lines

    return model

def get_galcat():
    """Two steps for first selecting a subset of galaxies and
       then one selection internally in the 
    """

    nbsubset = xd.Job('nbpzsubset')
    nbsubset.depend['input'] = libcommon.Common('galcat')
    nbsubset.depend['ref_cat'] = libcommon.Common('ref_cat')

    select_data = xd.Job('bcnz_select')
    select_data.depend['input'] = nbsubset
    select_data.config['SN_lim'] = 2.6

    return select_data

def pipel():
    """The main entry point for getting the pipeline."""

    pzcat = xd.Job('bcnz_fit')
    pzcat.config.update(get_pzcat_config())
    pzcat.depend['model'] = get_model()
    pzcat.depend['galcat'] = get_galcat()

    pzcat.model.config['seds'] = pzcat.config['seds'] # yes...

    return pzcat
