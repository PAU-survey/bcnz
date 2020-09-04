#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import pandas as pd
import xarray as xr

descr = {'norm_band': 'Band used for the normalization',
         'funky_limit': 'If adding limits to match Alex code.',
         'use_lines': 'If using emission lines',
         'scale_synband': 'If applying the scaling. Useful shortcut.'}

def_config = {'norm_band': 'subaru_r', 'funky_limit': True,
              'lines_upper': 0.1226,
              'use_lines': True, 
              'scale_synband': False} 

def funky_hack(config, syn2real, sed, model_norm):
    """Exactly mimic the cuts Alex was making."""

    ratio = syn2real.sel(sed=sed).copy()
    if sed == 'OIII':
        ratio[(ratio.z < 0.1) | (0.45 < ratio.z)] = 1.
    elif sed == 'lines': 
        flux = model_norm.sel(sed=sed)
        ratio.values[flux < 0.001*flux.max()] = 1.

        # Yes, this actually happens in an interval.
        upper = config['lines_upper']
        ratio[(ratio.z>0.1025) & (ratio.z<upper)] = 1.
    else:
        # The continuum ratios are better behaved.
        pass

    return ratio

def scale_model(config, coeff, model):
    """Directly scale the model as with the data."""

    # Transform the coefficients.
    norm_band = config['norm_band']
    coeff = coeff[coeff.bb == norm_band][['nb', 'val']]
    coeff = coeff.set_index(['nb']).to_xarray().val

    inds = ['band', 'z', 'sed', 'ext_law', 'EBV']
    model = model.set_index(inds)
    model = model.to_xarray().flux
    model_norm = model.sel(band=norm_band).copy() # A copy is needed for funky_hack..
    model_NB = model.sel(band=coeff.nb.values)

    # Scaling the fluxes..
    synbb = (model_NB.rename({'band': 'nb'})*coeff).sum(dim='nb')
    syn2real = model_norm / synbb

    for j,xsed in enumerate(model.sed):
        sed = str(xsed.values) 
        syn2real_mod = funky_hack(config, syn2real, sed, model_norm)

        for i,xband in enumerate(model.band):
            # Here we scale the narrow bands into the broad band system.
            if str(xband.values).startswith('NB'):
                model[i,:,j,:,:] *= syn2real_mod

    # Since we can not directly store xarray.
    model = model.to_dataframe()

    return model

def fmod_adjust(model_cont, model_lines, coeff=False, **myconf):
    """Adjust the model to reflect that the syntetic narrow bands is not
       entirely accurate.
    """

    config = def_config.copy()
    config.update(myconf)

    # Here the different parts will have a different zbinning! For the
    # first round I plan adding another layer doing the rebinning. After
    # doing the comparison, we should not rebin at all.

    # Special case of not actually doing anything ...
    if not config['scale_synband']:
        config['norm_band'], 'Need to specify the normalization band'
        comb = pd.concat([model_cont, model_lines])
        comb = comb.set_index(['band', 'z', 'sed', 'ext_law', 'EBV'])

        return comb

    out_cont = scale_model(coeff, model_cont)
    if config['use_lines']:
        out_lines = scale_model(coeff, model_lines)
        out = pd.concat([out_cont, out_lines])
    else:
        out = out_cont

    return out
