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


def _normalize_zp(zp, norm_filter):
    """One need to normalize the zero-points for the broad bands
       when the calibration has been run with a free amplitude.
    """

    print('In normalize..')
    BBlist = list(filter(lambda x: not x.startswith('NB'), zp.index))
    norm_val = zp.loc[norm_filter]

    for band in BBlist:
        zp.loc[band] /= norm_val


def apply_zp(galcat, zp, norm_bb=True, norm_filter=''):
    """Apply zero-points per band.

       Args:
           galcat (df): The galaxy catalogue.
           zp (series): Zero-points.
           norm_bb (bool): If separately normalizing the broad bands.
           norm_filter (str): Band to normalize to.
    """

    # Since some operations are performed in-place.
    galcat = galcat.copy()

    if norm_bb:
        assert norm_filter, 'You need to specify norm_filter'
        _normalize_zp(zp, norm_filter)

    # Applying this inline is simpler.
    for band, zp_val in zp.items():
        galcat[('flux', band)] *= zp_val
        galcat[('flux_error', band)] *= zp_val

    return galcat
