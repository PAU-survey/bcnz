# Copyright (C) 2020 Martin B. Eriksen
# This file is part of BCNz <https://github.com/PAU-survey/bcnz>.
#
# BCNz is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BCNz is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BCNz.  If not, see <http://www.gnu.org/licenses/>.
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
                    magerr_u, magerr_g, magerr_r, magerr_i, magerr_z, magerr_y, \
                    star_flag, mask as mask_cfhtlens \
             FROM cfhtlens \
             WHERE alpha_j2000 > {ra_min} AND alpha_j2000 < {ra_max} \
             AND   delta_j2000 > {dec_min} AND delta_j2000 < {dec_max} \
             AND   mag_i < 23 
          """.format(**coords)

    cat = pd.read_sql_query(sql, engine)

    return cat


def change_format(cat_in):
    """Change format to be compatible with the rest of the pipeline."""


    fnameL = ['u', 'g', 'r', 'i', 'z', 'y']
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
    for field in ['ref_id', 'ra', 'dec', 'star_flag', 'mask_cfhtlens']:
        cat[field] = cat_in[field]

    cat = cat.set_index('ref_id')

    return cat


def select_i(cat, cat_down):
    """Select which of the columns to use."""

    use_y = np.isnan(cat.flux.cfht_i)
    cat[('flux', 'cfht_i')] = np.where(
        use_y, cat.flux.cfht_y, cat.flux.cfht_i)
    cat[('flux_error', 'cfht_i')] = np.where(
        use_y, cat.flux_error.cfht_y, cat.flux_error.cfht_i)

    del cat[('flux', 'cfht_y')]
    del cat[('flux_error', 'cfht_y')]

    # Also add the i-band to later have this for plotting.
    cat['use_y'] = use_y

    cat['mag_i'] = np.where(use_y, cat_down.mag_y, cat_down.mag_i)
    cat['mag_i'] = cat.mag_i.replace(-99, np.nan)

    return cat


def paudm_cfhtlens(engine, field):
    """Getting the parent catalogue from CFHT lens.
       Args:
           engine (obj): Connection to PAUdb.
           field (str): The observed field.
    """

    cat_down = query(engine, field)
    cat = change_format(cat_down)
    cat = select_i(cat, cat_down)

    return cat
