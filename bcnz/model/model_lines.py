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

from __future__ import print_function

from IPython.core import debugger as ipdb
import os
import time
import numpy as np
import pandas as pd

from scipy.interpolate import splrep, splev
from scipy.integrate import simps


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

        fluxD[dest] += flux_line

    if not config['sep_OIII']:
        del fluxD['OIII']

    if band == 'pau_g':
        ipdb.set_trace()  # Should not happen.

    return fluxD


def _to_df(oldD, z, band):
    """Dictionary suitable for concatination."""

    # I have tried finding better ways, but none worked very well..
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


def model_lines(ratios, filters, extinction, ext_law, EBV, dz=0.0005, ampl=1e-16,
                sep_OIII=True, funky_OIII_norm=True):
    """The model flux for the emission lines.

       Args:
           ratios (df): Emission line ratios.
           filters (df): Filter response curves.
           extinction (df): Extinction law relations.
           ext_law (str): Extinction law.
           EBV (float): Extinction strength.
           dz (float): Redshift steps.
           ampl (float): Arbitrary scaling of the flux amplitude.
           sep_OIII (bool): If keeping the OIII line model separately.
           funky_OIII_norm (bool): Normalization to match Alexes code.
    """

    config = {'ext_law': ext_law, 'EBV': EBV, 'dz': dz, 'ampl': ampl,
              'sep_OIII': sep_OIII, 'funky_OIII_norm': funky_OIII_norm}

    filtersD, rconstD = _filter_spls(filters)
    ext_spl = create_ext_spl(config, extinction)

    z = np.arange(0., 2.05, config['dz'])

    df = pd.DataFrame()
    for band, f_spl in filtersD.items():
        rconst = rconstD[band]
        part = _find_flux(config, z, f_spl, ratios, rconst, ext_spl, band)
        part = _to_df(part, z, band)

        df = df.append(part, ignore_index=True)

    df['ext_law'] = config['ext_law']
    df['EBV'] = config['EBV']

    return df
