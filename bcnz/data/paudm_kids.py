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
# !/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import numpy as np
import pandas as pd


def query(engine, field):
    """The query against the PAUdm postgresql database."""

    field = field.lower()
    if field == 'w2':
        coords = {'ra_min': 125, 'ra_max': 143, 'dec_min': -10, 'dec_max': 10}

    # Here we directly select W2 to save time!
    sql = """SELECT paudm_id as ref_id, alpha_j2000 as ra, delta_j2000 as dec, \
                    mag_gaap_u as mag_u, mag_gaap_g as mag_g, mag_gaap_r as mag_r, \
                    mag_gaap_i as mag_i, mag_gaap_z as mag_z, mag_gaap_y as mag_y,  \
                    mag_gaap_j as mag_j, mag_gaap_h as mag_h, mag_gaap_ks as mag_ks,  \
                    magerr_gaap_u as magerr_u, magerr_gaap_g as magerr_g, \
                    magerr_gaap_r as magerr_r, magerr_gaap_i as magerr_i, \
                    magerr_gaap_z as magerr_z, magerr_gaap_y as magerr_y, \
                    magerr_gaap_j as magerr_j, magerr_gaap_h as magerr_h, \
                    magerr_gaap_ks as magerr_ks \
             FROM kids \
             WHERE alpha_j2000 > {ra_min} AND alpha_j2000 < {ra_max} \
             AND   delta_j2000 > {dec_min} AND delta_j2000 < {dec_max} \
             AND   mag_gaap_i < 23
          """.format(**coords)

    cat = pd.read_sql_query(sql, engine)

    return cat


def change_format(cat_in):
    """Change format to be compatible with the rest of the pipeline."""

    fnameL = ['u', 'g', 'r', 'i', 'z', 'y', 'j', 'h', 'ks']
    #new_names = ['kids_'+x for x in fnameL]
    new_names = ['kids_u', 'kids_g', 'kids_r', 'kids_i', 'kids_z', 'vista_y', 'vista_j', 'vista_h', 'vista_ks']

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


def paudm_kids(engine, field):
    """Getting the parent catalogue from KiDS 
       Args:
           engine (obj): Connection to PAUdb.
           field (str): The observed field.
    """

    cat_in = query(engine, field)
    cat = change_format(cat_in)
    cat = cat

    return cat
