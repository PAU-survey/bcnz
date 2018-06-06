#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import numpy as np

import pipel_pz_basic
import def_chunks

import xdolphin as xd

def get_model(part):
    """Get the model."""

    model_rebinned = pipel_pz_basic.get_model()
    model = model_rebinned.model

    conf = {'EBV': part.EBV, 'ext_law': part.ext_law}
    model.model_cont.config.update(conf)
    model.model_lines.config.update(conf)

    # If using emission lines.
    model.config['use_lines'] = part.use_lines

    # If having a different template for the OIII lines.
    model.model_lines.config['sep_OIII'] = part.sep_OIII

    # Where and which continuum SEDs to use.
    model.model_cont.seds.config['input_dir'] = part.sed_dir
    model.model_cont.config['seds'] = part.seds

    return model_rebinned

def pipel(chunks=False, prevot_calib=True, prevot_pzrun=False, bands=False,
          ngal=0, Niter=1000):
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
    pzcat_orig.config['filters'] = bands
    pzcat_orig.config['Niter'] = Niter

    # Synthetic broad band coefficients used to scale the 
    # broad band fluxes.
    bbsyn_coeff = xd.Job('NB2BB')
    bbsyn_coeff.depend['filters'] = xd.Common('filters')

    # These are needed both for calibration and the photo-z.
    modelD = {}
    model = pzcat_orig.model
    for key, row in chunks.iterrows():
        #modelD[key] = get_model(row, model, bbsyn_coeff)
        modelD[key] = get_model(row) #, model, bbsyn_coeff)

    # Calibration by comparing the model fit to the observations.
    inter_calib = xd.Job('inter_calib')
    inter_calib.depend['galcat'] = pzcat_orig.galcat

    # These should be elsewhere...
    inter_calib.config.update({\
      'Nrounds': 15,
      'learn_rate': 1.0,
      'fit_bands': NB + BB})

    for key, row in chunks.iterrows():
        if (not prevot_calib) and (row.ext_law == 'SMC_prevot'):
            continue

        inter_calib.depend['model_{}'.format(key)] = modelD[key]

    # Add the zero-points.
    galcat_zp = xd.Job('apply_zp')
    galcat_zp.depend['zp'] = inter_calib
    galcat_zp.depend['galcat'] = inter_calib.galcat

    # Just limit the number of galaxies...
    galcat_lim = xd.Job('limit_ngal')
    galcat_lim.depend['galcat'] = galcat_zp
    galcat_lim.config['ngal'] = ngal

    # Linking the photo-z runs.
    chi2_comb = xd.Job('chi2_comb')
    for key,row in chunks.iterrows():
        if (not prevot_pzrun) and (row.ext_law == 'SMC_prevot'):
            continue

        pzcat = pzcat_orig.shallow_copy()
        pzcat.depend['model'] = modelD[key]
        pzcat.depend['galcat'] = galcat_lim

        chi2_comb.depend['pzcat_{}'.format(key)] = pzcat

    bcnz_pzcat = xd.Job('bcnz_pzcat')
    bcnz_pzcat.depend['chi2'] = chi2_comb

    return bcnz_pzcat
