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
import time
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def fix_missing_data(cat_in, ind):
    """Linear interpolation in magnitude space to replace missing data."""

    X = 455 + 10*np.arange(40)
    NB = list(map('pau_nb{}'.format, X))

    def f_linear(x, a, b):
        return a*x + b

    pau_syn = cat_in.flux[NB].values

    # Not exactly good...
    pau_syn[pau_syn < 0.] = np.nan

    miss_ids = np.isnan(pau_syn).any(axis=1)
    miss_rows = np.arange(len(cat_in))[miss_ids]

    for i in miss_rows:
        touse = ~np.isnan(pau_syn[i])

        yfit = np.log10(pau_syn[miss_rows[i]][touse]) if ind \
            else np.log10(pau_syn[miss_rows[0]][touse])

        try:
            popt, pcov = curve_fit(f_linear, X[touse], yfit)
            pau_syn[i, ~touse] = 10**f_linear(X[~touse], *popt)
        except ValueError:
            ipdb.set_trace()

    return pau_syn


def find_synbb(pau_syn, bbsyn_coeff, synband):
    """Just because of the different naming..."""

    bbsyn_coeff = bbsyn_coeff[bbsyn_coeff.bb == synband]
    NB = list(map('pau_nb{}'.format, 455 + 10*np.arange(40)))

    vec = bbsyn_coeff.pivot('bb', 'nb', 'val')[NB].values[0]
    synbb = np.dot(pau_syn, vec)

    return synbb


def scale_fluxes(cat_in, obs2syn):
    """Scale the fluxes between the systems."""

    # Here we scale the narrow bands without adding additional
    # errors. This might not be the most optimal.
    cat_out = cat_in.copy()
    for band in cat_in.flux.columns:
        if not band.startswith('pau_nb'):
            continue

        cat_out['flux', band] *= obs2syn
        cat_out['flux_error', band] *= obs2syn

    return cat_out


def synband_scale(cat_in, bbsyn_coeff, ind=False, synband='subaru_r',
                  scale_data=False):
    """Adjust the data based on a synthetic band.
       Args:
           cat_id (df): Galaxy catalogue.
           bbsyn_coeff (df): The synthetic broad band coefficients.
           ind (bool): No idea.
           synband (str): Synthetic band to normalize.
           scale_data (bool): If actually scaling the data.
    """

    if not scale_data:
        return cat_in

    pau_syn = fix_missing_data(cat_in, ind)
    synbb = find_synbb(pau_syn, bbsyn_coeff, synband)

    obs2syn = cat_in.flux[synband] / synbb
    cat_out = scale_fluxes(cat_in, obs2syn)

    return cat_out
