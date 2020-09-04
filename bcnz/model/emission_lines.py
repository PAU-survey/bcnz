#!/usr/bin/env python

from __future__ import print_function

from IPython.core import debugger as ipdb
import os
import time
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from scipy.interpolate import splrep, splev
from scipy.integrate import simps

# This could be set as a config...
descr = {
  'dz': 'Redshift steps',
  'ampl': 'Added amplitude',
  'EBV': 'Extinction coefficient',
  'ext_law': 'Extinction law',
  'sep_OIII': 'If keeping OIII separate', 
  'funky_OIII_norm': 'Separate normalization for the OIII lines.'}


def_config = {'dz': 0.0005, 'ampl': 1e-16, 'EBV': 0.,
          'ext_law': 'SB_calzetti', 'sep_OIII': True,
          #'zcut_OIII': 0.705,
          'zcut_OIII': 10.,
          'funky_OIII_norm': True}

def _filter_spls(filters):
    """Convert filter curves to splines."""

    splD = {}
    rconstD = {}
    for fname in filters.index.unique():
        # Drop the PAUS BB, since they have repeated measurements.
        if fname.startswith('pau_') and not fname.startswith('pau_nb'):
            continue

        try:
            sub = filters.loc[fname]
            splD[fname] = splrep(sub.lmb, sub.response)
            rconstD[fname] = simps(sub.response/sub.lmb, sub.lmb)
        except ValueError:
            ipdb.set_trace()

    return splD, rconstD

def create_ext_spl(config, ext):
    """Spline for the extinction."""

    sub = ext[ext.ext_law == config['ext_law']]
    ext_spl = splrep(sub.lmb, sub.k)

    return ext_spl

def _find_flux(config, z, f_spl, ratios, rconst, ext_spl, band):
    """Estimate the flux in the emission lines relative
       to the OII flux.
    """

    EBV = config['EBV']
    ampl = config['ampl']

    # Note: This is not completely general.
    fluxD = {'lines': 0., 'OIII': 0.}
    for line_name, line in ratios.iterrows():
        lmb = line.lmb*(1+z)
        y_f = splev(lmb, f_spl, ext=1) 

        k_ext = splev(lmb, ext_spl, ext=1) 
        y_ext = 10**(-0.4*EBV*k_ext)

        isOIII = line_name.startswith('OIII')
        dest = 'OIII' if (isOIII and config['sep_OIII']) else 'lines'

        # Since Alex had a different normalization for the OIII lines.
        # This is not needed...
        ratio = line.ratio
        if isOIII and config['funky_OIII_norm']:
            ratio /= ratios.loc['OIII_2'].ratio

        flux_line = ampl*(ratio*y_f*y_ext) / rconst

        # Testing not having a free OIII above a certain redshift... 
        if isOIII:
            zcutOIII = config['zcut_OIII']
            flux_line[zcutOIII < z] = 0.


        fluxD[dest] += flux_line

    if not config['sep_OIII']:
        del fluxD['OIII']

    if band == 'pau_g':
        ipdb.set_trace() # Should not happen.

    return fluxD

def _to_df(oldD, z, band):
    """Dictionary suitable for concatination."""


    # I have tried finding better ways, but none wored very well..
    F = pd.DataFrame(oldD)
    F.index = z
    F.index.name = 'z'
    F.columns.name = 'sed'

    F = F.stack()
    F.name = 'flux'
    F = F.reset_index()
    F = F.rename(columns={0: 'flux'})
    F['band'] = band


    if band == 'pau_g':
        ipdb.set_trace()


    return F

def line_model(config, ratios, filters, extinction):
    """Find the emission line model."""

    filtersD, rconstD = _filter_spls(filters)
    ext_spl = create_ext_spl(config, extinction)

    z = np.arange(0., 2., config['dz'])

    df = pd.DataFrame()
    for band, f_spl in filtersD.items():
        rconst = rconstD[band]
        part = _find_flux(config, z, f_spl, ratios, rconst, ext_spl, band)
        part = _to_df(part, z, band)

        df = df.append(part, ignore_index=True)

    df['ext_law'] = config['ext_law']
    df['EBV'] = config['EBV']

    return df


def emission_lines(ratios, filters, extinction, **myconf):
    """The model flux for the emission lines."""

    config = def_config.copy()
    config.update(myconf)
    model = line_model(config, ratios, filters, extinction)

    return model
