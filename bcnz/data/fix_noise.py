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

from IPython.core import debugger as ipdb
import numpy as np
import pandas as pd


def limit_SN(cat, SNR_lim):
    """Limit based on SN."""

    SNR = cat['flux'] / cat['flux_error']

    cat['flux'] = cat['flux'][SNR_lim < SNR]
    cat['flux_error'] = cat['flux_error'][SNR_lim < SNR]

    return cat


def _flux_minerr(cat):
    """Apply minimum error to fluxes."""

    # By now applying 3% minimum error to all the different fluxes.
    for band in cat.flux.columns:
        add_err = cat['flux', band] * self.config['min_err']
        cat['flux_error', band] = np.sqrt(
            cat['flux_error', band]**2 + add_err**2)


def _mag_minerr(cat, min_err):
    """Apply minimum error to magnitudes."""

    # By now applying 3% minimum error to all the different fluxes.
    for band in cat.flux.columns:
        # Some the absolute values are suspicious...
        SN = np.abs(cat['flux', band]) / cat['flux_error', band]

        mag_err = 2.5*np.log10(1+1./SN)
        mag_err = np.sqrt(mag_err**2 + min_err**2)
        flux_error = np.abs(cat['flux', band])*(10**(0.4*mag_err) - 1.)

        cat[('flux_error', band)] = flux_error


def _add_minerr(cat, min_err, apply_mag):
    """Add a minimum error in the flux measurements."""

    # For the comparison with Alex.
    if apply_mag:
        _mag_minerr(cat, min_err)
    else:
        _flux_minerr(cat, min_err)


def fix_noise(cat, SNR_lim=-2, min_err=0.03, apply_mag=True):
    """Cut on SNR and add a minmum error.

       Args:
           cat (df): The galaxy catalogue.
           SNR_lim (float): Minimum limit for SNR.
           min_err (float): Minimum error added.
           apply_mag (bool): If the minimum is applied to magnitudes.

       Returns:
           The modified galaxy catalogue
    """

    _add_minerr(cat, min_err, apply_mag)
    cat = limit_SN(cat, SNR_lim)

    return cat
