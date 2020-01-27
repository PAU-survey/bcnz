#!/usr/bin/env python
# encoding: UTF8

import copy
from IPython.core import debugger as ipdb
import numpy as np

import xdolphin as xd

def get_ab():
    """The input to the flux model."""

    seds = xd.Job('curves')
    seds.config.update({'input_dir': '/home/eriksen/data/photoz/seds/cosmos_self', 'suf': 'sed', 'min_val': 0.0})

    extinction = xd.Job('extinction_laigle')

    ab_cont = xd.Job('ab_cont')
    ab_cont.config.update({'dz_ab': 0.001, 'zmax_ab': 2.05})
    ab_cont.depend['seds'] = seds
    ab_cont.depend['filters'] = xd.Common('filters')
    ab_cont.depend['extinction'] = extinction

    ab_lines = xd.Job('emission_lines')
    ab_lines.depend['filters'] = xd.Common('filters')
    ab_lines.depend['extinction'] = extinction
    ab_lines.depend['ratios'] = xd.Job('line_ratios')

    return ab_cont, ab_lines

def get_pzcat_config():
    """The default photo-z configuration."""

    config = {
      'dz': 0.001,
      'zmax': 1.0,
      'Niter': 1000,
    }

    return config

def get_model():
    """The part returning the model."""

    ab_cont, ab_lines = get_ab()

    model = xd.Job('fmod_adjust')
    model.depend['bbsyn_coeff'] = xd.Common('bbsyn_coeff')
    model.depend['model_cont'] = ab_cont
    model.depend['model_lines'] = ab_lines

    model_rebinned = xd.Job('model_rebin')
    model_rebinned.config.update({'dz': 0.001, 'zmax': 1.2})
    model_rebinned.depend['model'] = model

    return model_rebinned

def get_galcat():
    """Two steps for first selecting a subset of galaxies and
       then one selection internally in the 
    """

    cat1 = xd.Job('bcnz_fix_noise')
    cat1.depend['input'] = xd.Common('galcat')

    # Added later...
    cat1x = xd.Job('fix_extinction')
    cat1x.depend['input'] = cat1

    cat2 = xd.Job('nbpzsubset')
    cat2.depend['input'] = cat1x
    cat2.depend['ref_cat'] = xd.Common('ref_cat')

    # Here we make the r-band adjustment after selecting
    # galaxies with many narrow bands. Saves some trouble.
    cat3 = xd.Job('rband_adjust')
    cat3.depend['bbsyn_coeff'] = xd.Common('bbsyn_coeff')
    cat3.depend['galcat'] = cat2

    return cat3

def get_bbsyn_coeff():
    """The synthetic broad band coefficients."""

    bbsyn_coeff = xd.Job('NB2BB')
    bbsyn_coeff.depend['filters'] = xd.Common('filters')

    return bbsyn_coeff

def pipel():
    """The main entry point for getting the pipeline."""

    pzcat = xd.Job('bcnz_fit')
    pzcat.config.update(get_pzcat_config())
    pzcat.depend['model'] = get_model()
    pzcat.depend['galcat'] = get_galcat()

    #pzcat.model.config['seds'] = pzcat.config['seds'] # yes...

    return pzcat
