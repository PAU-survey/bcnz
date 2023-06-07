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

# The galaxy selection. This code works with COSMOS and CFHTlens.
from IPython.core import debugger as ipdb
import numpy as np
import pandas as pd


def set_other_fields(cat, other):
    """Setting the spectroscopic redshift as coming from another catalogue."""

    cat['zs'] = other.zspec

    # A bit of gluing together...
    cosmos_fields = ['type', 'conf']
    cfht_fields = ['zquality', 'obj_type', 'zspec']
    kids_fields = ['z_quality','zspec']
    vipers_fields = ['zflg']
    for field in cosmos_fields + cfht_fields + kids_fields + vipers_fields:
        if not field in other.columns:
            continue

        cat[field] = other[field]


def change_format(cat_in):
    """The photo-z code expects a different format."""

    flux = cat_in.pivot(index='ref_id', columns='band', values='flux')
    flux_err = cat_in.pivot(index='ref_id', columns='band', values='flux_err')

    nexp = cat_in.pivot(index='ref_id', columns='band', values='nexp')
    cat = pd.concat({'flux': flux, 'flux_err': flux_err, 'nexp': nexp}, axis=1)

    return cat


def limit_nb(df, min_nb):
    """Select based on a minimum number of filters."""

    def isnb(x):
        return x.startswith('pau_nb')

    # So we are only including the narrow band filters when
    # counting.
    filters = [x for x in df.columns.get_level_values(1).unique()
               if isnb(x)]

    df = df.copy()
    df['nrobs'] = (~np.isnan(df['flux'][filters])).sum(axis=1)
    df = df[min_nb <= df.nrobs]

    return df


def limit_spec(cat, only_specz, secure_spec):
    """Only return the galaxies with a spectroscopic redshift."""
    #only_specz = False
   
    if only_specz == False:
    
        return cat

    # Here we only make the selection for the COSMOS field. (WHY??)
    cat = cat[cat.zs != 0]
    cat = cat[~np.isnan(cat.zs)]

    if secure_spec:
        if 'conf' in cat.columns:
            cat = cat[(3. <= cat.conf) & (cat.conf <= 5.)]
        elif 'zflg' in cat.columns:
            # Highly (99%) secure Vipers spectra.
            cat = cat[(3. <= cat.zflg) & (cat.zflg < 5.)]
        elif 'zquality' in cat.columns:
            # Deep2 spectra.
            cat = cat[(3. <= cat.zquality) & (cat.zquality <= 4)]
        elif 'z_quality' in cat.columns:
            # SDSS and GAMA spectra. 
            cat = cat[3. <= cat.z_quality]
        else:
            raise NotImplementedError('Which field is this??')

    return cat


def limit_isgal(sub, field, sel_gal, sel_gal_specz):
    """If limiting ourself to only the galaxies."""

    if not sel_gal:
        return sub
    
    if sel_gal_specz:
        if 'obj_type' in sub.columns:
            sub = sub[sub.obj_type == 'GALAXY']
        elif 'type' in sub.columns:
            sub = sub[sub.type == 0]
        else:
            # W1 and W2 selection.
            sub = sub[sub.zspec != 0]
    
    elif not sel_gal_specz:
        if field.lower() == 'w1' or field.lower() == 'w3':
            sub = sub[sub.star_flag == 0]
        elif field.lower() == 'w2':
            sub = sub[sub.sg_flag == 1]
        else:
            raise NotImplementedError('Which flag do I use?')

    return sub


def limit_mask(sub, field, apply_mask):
    """If applying a mask"""
    
    if not apply_mask:
        return sub
    
    if field.lower() == 'w1' or field.lower() == 'w3':
        sub = sub[sub['mask_cfhtlens'] <= 1]
    elif field.lower() == 'w2':
        Mask = sub['mask_kids']
        masked_index = (Mask & 32764)>0
        Mask[masked_index]=1.
        Mask[Mask>1.]=0.
        sub['new_mask']=Mask
        sub = sub[sub.new_mask == 0]
    else:
        raise NotImplementedError('Which mask do I use?')

    return sub


def limit_has_bb(sub, has_bb, test_band):
    """Test is a specific broad band is present in the
       catalogue.
    """

    if has_bb:
        sub = sub[~np.isnan(sub.flux[test_band])]

    return sub


def limit_ngal(cat, ngal):
    """Limit based on the number of galaxies."""

    if ngal == 0:
        return cat

    assert ngal <= len(cat), 'Not sufficient galaxies'

    sub = cat.sample(n=ngal)

    return sub


def limit_zmax(cat, zmax):
    """Limit based on redshift."""

    if not zmax:
        return cat

    cat = cat[cat.zspec < zmax]

    return cat


def gal_subset(galcat, ref_cat, field = 'COSMOS', min_nb=39, only_specz=True, ngal=0, has_bb=False,
               secure_spec=True, sel_gal=True, sel_gal_specz=True, apply_mask = False, zmax=0., test_band='subaru_r'):
    """Selects a subset of galaxies, either for calibration or running the photoz.
       Args:
           galcat (df): Galaxy catalogue.
           ref_cat (df): Reference catalogue including specz.
           field (str): Field to download.
           min_nb (int): Minimum number of narrow bands.
           only_specz (bool): If only using galaxies with spectra.
           ngal (int): Random selection of ngal galaxies.
           has_bb (bool): Ensure the galaxy has a broad band.
           secure_spec (bool): Restrict to secure spectra.
           sel_gal (bool): Select galaxies.
           sel_gal_specz (bool): Select galaxies from the specz or the parent catalogue.
           apply_mask (bool): Apply mask from the parent catalogue.
           zmax (float): Maximum reshift.
           test_band (str): Band to test if broad bands are available.
    """

    print('Selecting a subset...')

    # Ok, here I first cut based on one format...
    if not isinstance(galcat.columns, pd.MultiIndex):
        galcat = change_format(galcat)

    set_other_fields(galcat, ref_cat)

    print('Total', len(galcat))
    sub = limit_isgal(galcat, field, sel_gal, sel_gal_specz)
    print('Limit galaxies', len(sub))
    
    sub = limit_mask(sub, field, apply_mask)
    print('Limit mask', len(sub))

    sub = limit_nb(sub, min_nb)
    print('Limit #NB', len(sub))

    sub = limit_spec(sub, only_specz, secure_spec)
    print('Limit spec', len(sub))

    sub = limit_has_bb(sub, has_bb, test_band)
    print('Has BB', len(sub))

    sub = limit_zmax(sub, zmax)
    print('Limit zmax', len(sub))

    sub = limit_ngal(sub, ngal)
    print('Limit ngal', len(sub))

    return sub
