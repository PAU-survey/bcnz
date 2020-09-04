#!/usr/bin/env python
# encoding: UTF8

import os
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits

from IPython.core import debugger as ipdb

descr = {'rm_stars': 'If removing stars here.'}

cfg = [['NUV', 'DNUV', 'galex_nuv'],
       ['U', 'DU', 'cfht_u'],
       ['B', 'DB', 'subaru_b'],
       ['V', 'DV', 'subaru_v'],
       ['R', 'DR', 'subaru_r'],
       ['I', 'DI', 'subaru_i'],
       ['ZN', 'DZN', 'subaru_z'],
       ['YHSC', 'DYHSC', 'subaru_y'],
       ['Y', 'DY', 'vista_y'],
       ['J', 'DJ', 'vista_j'],
       ['H', 'DH', 'vista_h'],
       ['K', 'DK', 'vista_ks'],
#       ['KW', 'DKW', 'wircam_Ks'], # Not available in this catalogue.
#       ['HW', 'DHW', 'wircam_H'],
       ['IA427', 'DIA427', 'subaru_ia427'],
       ['IA464', 'DIA464', 'subaru_ia464'],
       ['IA484', 'DIA484', 'subaru_ia484'],
       ['IA505', 'DIA505', 'subaru_ia505'],
       ['IA527', 'DIA527', 'subaru_ia527'],
       ['IA574', 'DIA574', 'subaru_ia574'],
       ['IA624', 'DIA624', 'subaru_ia624'],
       ['IA679', 'DIA679', 'subaru_ia679'],
       ['IA709', 'DIA709', 'subaru_ia709'],
       ['IA738', 'DIA738', 'subaru_ia738'],
       ['IA767', 'DIA767', 'subaru_ia767'],
       ['IA827', 'DIA827', 'subaru_ia827'],
       ['NB711', 'DNB711', 'subaru_nb711'],
       ['NB816', 'DNB816', 'subaru_nb816'],
       ['CH1', 'DCH1', 'irac_ch1'],
       ['CH2', 'DCH2', 'irac_ch2'],
       ['CH3', 'DCH3', 'irac_ch3'],
       ['CH4', 'DCH4', 'irac_ch4']]

# Note, this code does *not* apply zero-points, following Alex
# latest notebook. The minimum error is applied later.

# The input data can be downloaded from:
# ftp://ftp.iap.fr/pub/from_users/hjmcc/COSMOS2015/

def load_cat(d_cosmos):
    """Load the main FITS file."""

    # The joy of FITS files.
    fname_pdz = 'pdz_cosmos2015_v1.3.fits'
    path = os.path.join(os.path.expanduser(d_cosmos), fname_pdz)
    dat = Table.read(path, format='fits')
    df = dat.to_pandas()

    return df

def read_cat(d_cosmos):
    """Reads in the COSMOS catalog and convert to the internal
       format.
    """

    cat_in = load_cat(d_cosmos)

    # Being a bit lazy, I currently don't convert to the internal
    # format.
    flux_cols, err_cols, names = zip(*cfg)

    flux = cat_in[list(flux_cols)].rename(columns=dict(zip(flux_cols, names)))
    flux_error = cat_in[list(err_cols)].rename(columns=dict(zip(err_cols, names)))

    cat = pd.concat({'flux': flux, 'flux_error': flux_error}, axis=1)

    # Yes, different definition and rounding error.
    cat[cat < -99] = np.nan

    # Scale to PAU fluxes.
    ab_factor = 10**(0.4*26)
    cosmos_scale = ab_factor * 10**(0.4*48.6)
    cat *= cosmos_scale

    # Adding this here, since otherwise it gets scaled.
    cat['id_laigle'] = cat_in.ID

    return cat


def other_cols(d_cosmos, cat):

    # Is it not great? Here I will need to join with another file to get
    # the galaxy type.
    fname_other = 'COSMOS2015_Laigle+_v1.1.fits'
    path = os.path.join(os.path.expanduser(d_cosmos), fname_other)
    hdul = fits.open(path)

    def f(x):
        return hdul[1].data[x]

    # Since I don't know a general way to change the endianness.
    fields = [
    ('NUMBER', 'id_laigle', '<i8'),
    ('TYPE', 'type', '<i2'), 
    ('ALPHA_J2000', 'ra', '<f8'), 
    ('DELTA_J2000', 'dec', '<f8')]

    D = {}
    for key, to_field, dtype in fields:
        D[to_field] = f(key).astype(dtype)

    other = pd.DataFrame(D)

    # Otherwise the hirarchical columns gives problems.
    assert (cat.id_laigle == other.id_laigle).all()
    for key,val in other.items():
        cat[key] = val

def fixes(cat, rm_stars=True):
    if rm_stars:
        cat = cat[cat.type == 0]

    # This is probably not needed... 
    del cat['type']

    return cat

def cosmos_laigle(d_cosmos, rm_stars=True):
    """Interface from reading the COSMOS Laigle catalogue."""

    cat = read_cat(d_cosmos)
    other_cols(d_cosmos, cat)
    cat = fixes(cat, rm_stars)

    return cat
