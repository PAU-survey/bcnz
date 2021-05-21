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
# encoding: UTF8

import functools
from IPython.core import debugger as ipdb


def rband(field):
    """The rband name in different fields."""
    
    if field.lower() == 'cosmos':
        r_band = 'subaru_r'
    elif field.lower() == 'w2':
        r_band = 'kids_r'
    else:
        r_band = 'cfht_r'
        
    return r_band


def paus(engine, memba_prod, field, d_cosmos='~/data/cosmos',
         d_filters='~/data/photoz/all_filters/v3', min_nb=35,
         only_specz=False, secure_spec=False, has_bb=False, sel_gal=True,
         coadd_file=None):
    """Load the PAUS data from PAUdm and perform the required
       transformation.

       Args:
           engine (obj): SqlAlchemy engine connected to PAUdb.
           memba_prod (int): The MEMBA production.
           field (str): Field to download.
           d_cosmos (str): Directory with downloaded COSMOS files.
           d_filters (str): Directory with filter curves.
           min_nb (int): Minimum number of narrow bands.
           only_specz (bool): Only selecting galaxy with spectroscopic redshifts.
           secure_spec (bool): Selecting secure spectroscopic redshifts.
           has_bb (bool): Select galaxies with broad bands data.
           sel_gal (bool): Select galaxies.
           coadd_file (str): Path to file containing the coadds.'
    """

    import bcnz

    if field.lower() == 'cosmos':
        # The parent catalogue require positional matching.
        paudm_cosmos = bcnz.data.paudm_cosmos(engine)
        cosmos_laigle = bcnz.data.cosmos_laigle(d_cosmos)
        parent_cat = bcnz.data.match_position(paudm_cosmos, cosmos_laigle)

        # In the parent catalogue, but needed if using other coadds.
        specz = bcnz.specz.zcosmos(engine)
    elif field.lower() == 'w3':
        parent_cat = bcnz.data.paudm_cfhtlens(engine, 'w3')
        specz = bcnz.specz.deep2(engine)
    elif field.lower() == 'w2':
        parent_cat = bcnz.data.paudm_kids(engine, 'w2')
        specz = bcnz.specz.sdss_gama(engine)
    else:
        raise ValueError(f'No spectroscopy defined for: {field}')

    if coadd_file is None:
        paudm_coadd = bcnz.data.paudm_coadd(engine, memba_prod, field)
    else:
        paudm_coadd = bcnz.data.load_coadd_file(coadd_file)
        
    data_in = paudm_coadd.join(parent_cat, how='inner')

    # Add some minimum noise.
    data_noisy = bcnz.data.fix_noise(data_in)

    # Select a subset of the galaxies.
    conf = {'min_nb': min_nb, 'only_specz': only_specz, 'secure_spec': secure_spec,
            'has_bb': has_bb, 'sel_gal': sel_gal}

    conf['test_band'] = rband(field)

    nbsubset = bcnz.data.gal_subset(data_noisy, specz, **conf)

    # Synthetic narrow band coefficients.
    synband = rband(field)
    filters = bcnz.model.all_filters(d_filters=d_filters)
    coeff = bcnz.model.nb2bb(filters, synband)

# Disabling this scaling by now. Some tests showed this scaling is not needed.
#    data_scaled = bcnz.data.synband_scale(nbsubset, coeff, synband=synband, scale_data=True)

    return nbsubset 

paus_calib_sample = functools.partial(
    paus, min_nb=39, only_specz=True, has_bb=True, secure_spec=False) 

# The entries for which we run the photo-z.
paus_main_sample = functools.partial(
    paus, min_nb=39, only_specz=False, has_bb=True, sel_gal=False, secure_spec=False)
