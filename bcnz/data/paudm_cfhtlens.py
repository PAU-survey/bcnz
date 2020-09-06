#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import numpy as np
import pandas as pd

def query(engine, field):
    """The query against the PAUdm postgresql database."""

    field = field.lower()
    if field == 'w3':
        coords = {'ra_min': 200, 'ra_max': 230, 'dec_min': 50, 'dec_max': 60}
    elif field == 'w1':
        coords = {'ra_min': 20, 'ra_max': 50, 'dec_min': -7, 'dec_max': 3}


    # Here we directly select W3 to save time!
    sql = """SELECT paudm_id as ref_id, alpha_j2000 as ra, delta_j2000 as dec, \
                    mag_u, mag_g, mag_r, mag_i, mag_z, mag_y,  \
                    magerr_u, magerr_g, magerr_r, magerr_i, magerr_z, magerr_y \
             FROM cfhtlens \
             WHERE alpha_j2000 > {ra_min} AND alpha_j2000 < {ra_max} \
             AND   delta_j2000 > {dec_min} AND delta_j2000 < {dec_max} \
             AND   mag_i < 23
          """.format(**coords)


    cat = pd.read_sql_query(sql, engine)

    return cat

def change_format(cat_in):
    """Change format to be compatible with the rest of the pipeline."""
    
    fnameL = ['u','g','r','i','z','y']
    new_names = ['cfht_'+x for x in fnameL]

    # Convert the fluxes.
    F = list(map('mag_{}'.format, fnameL))
    mag = cat_in[F].rename(columns=dict(zip(F, new_names)))
    mag = mag.replace(-99, np.nan)
    flux = 10**(-0.4*(mag-26))

    # Convert the flux errors.
    F = list(map('magerr_{}'.format, fnameL))
    mag_error = cat_in[F].rename(columns=dict(zip(F, new_names)))
    mag_error = mag_error.replace(-99, np.nan)
    mag_error = mag_error.replace(99, np.nan)

    SN = 1. / (10**(0.4*mag_error) - 1.)

    flux_error = flux / SN

    cat = pd.concat({'flux': flux, 'flux_error': flux_error}, axis=1)
    for field in ['ref_id', 'ra', 'dec']:
        cat[field] = cat_in[field]

    cat = cat.set_index('ref_id')

    return cat

def select_i(cat_in):
    """Select which of the columns to use."""

    use_y = np.isnan(cat_in.flux.cfht_i)
    cat_in[('flux', 'cfht_i')] = np.where(use_y, cat_in.flux.cfht_y, cat_in.flux.cfht_i)
    cat_in[('flux_error', 'cfht_i')] = np.where(use_y, cat_in.flux_error.cfht_y, cat_in.flux_error.cfht_i)

    del cat_in[('flux', 'cfht_y')]
    del cat_in[('flux_error', 'cfht_y')]

    return cat_in

def paudm_cfhtlens(engine, field):
    """Getting the parent catalogue from CFHT lens.
       Args:
           engine (obj): Connection to PAUdb.
           field (str): The observed field.
    """

    cat_in = query(engine, field)
    cat = change_format(cat_in)
    cat = select_i(cat)

    return cat
