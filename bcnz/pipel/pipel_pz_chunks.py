#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import numpy as np

import pipel_pz_basic
import def_chunks

import xdolphin as xd

def get_model(part, model, bbsyn_coeff):
    """Model for a specific chunk."""

    # The shallow copy here ends up being very confusing...
    xmodel = model.shallow_copy()
    xab = model.ab.shallow_copy()
    xab_lines = model.ab_lines.shallow_copy()
    xmodel.config['sep_lines'] = part['sep_lines']

    seds = xab.seds.shallow_copy()
    seds.config['input_dir'] = part.sed_dir
    xab.depend['seds'] = seds

    xmodel.config['use_lines'] = part['use_lines']
    conf = {'EBV': part['EBV'], 'ext_law': part['ext_law']}

    xab.config.update(conf)
    xab_lines.config.update(conf)
    xmodel.config['seds'] = part['seds']

    xmodel.depend['ab'] = xab 
    xmodel.depend['ab_lines'] = xab_lines

    ymodel = xd.Job('fmod_adjust')
    ymodel.config['norm_band'] = 'subaru_r'
    ymodel.depend['bbsyn_coeff'] = bbsyn_coeff
    ymodel.depend['model'] = xmodel

    return ymodel


def pipel(chunks=False, prevot_calib=True, prevot_pzrun=False, bands=False):
    """Pipeline for BCNZv2 when running with many chunks."""

    # chunks - Dataframe specifying the chunks to use. If not specified, use
    #          default.
    # prevot_calib - If including Prevot extinction for the calibration run.
    # prevot_pzrun - If including Prevot extinction for the photo-z run.


    if not bands:
        # Default bands.
        NB = list(map('NB{}'.format, 455 + 10*np.arange(40)))
        BB = ['cfht_u', 'subaru_B', 'subaru_V', 'subaru_r', 'subaru_i', 'subaru_z']
        bands = NB + BB

    if not chunks:
        chunks = def_chunks.pz_chunks()

    pzcat_orig = pipel_pz_basic.pipel()
    pzcat_orig.config['bands'] = bands

    # Synthetic broad band coefficients used to scale the 
    # broad band fluxes.
    bbsyn_coeff = xd.Job('bbsyn_coeff')
    bbsyn_coeff.depend['filters'] = xd.Common('filters')

    # These are needed both for calibration and the photo-z.
    modelD = {}
    model = pzcat_orig.model
    for key, row in chunks.iterrows():
        modelD[key] = get_model(row, model, bbsyn_coeff)

    # Calibration by comparing the model fit to the observations.
    inter_calib = xd.Job('inter_calib')
    inter_calib.depend['bbsyn_coeff'] = bbsyn_coeff
    inter_calib.depend['galcat'] = pzcat_orig.galcat

    # These should be elsewhere...
    inter_calib.config.update({\
      'Nrounds': 15,
      'learn_rate': 1.0,
      'fit_bands': NB + BB,
      'bb_norm': 'subaru_r',
      'zp_min': 'flux2'})

    for key, row in chunks.iterrows():
        if (not prevot_calib) and (row.ext_law == 'SMC_prevot'):
            continue

        inter_calib.depend['model_{}'.format(key)] = modelD[key]

    # Linking the photo-z runs.
    bcnz_comb_ext = xd.Job('bcnz_comb_ext')
    for key,row in chunks.iterrows():
        if (not prevot_pzrun) and (row.ext_law == 'SMC_prevot'):
            continue

        pzcat = pzcat_orig.shallow_copy()
        pzcat.depend['model'] = modelD[key]
        pzcat.depend['galcat'] = inter_calib

        bcnz_comb_ext.depend['pzcat_{}'.format(key)] = pzcat

    return bcnz_comb_ext
