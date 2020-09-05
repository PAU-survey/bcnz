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


def apply_zp(galcat, zp, norm_bb=True, norm_filter='subaru_r'):
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
        _normalize_zp(zp, norm_filter)

    # Applying this inline is simpler.
    for band, zp_val in zp.items():
        galcat[('flux', band)] *= zp_val
        galcat[('flux_error', band)] *= zp_val

    return galcat
