# encoding: UTF8

import functools
from IPython.core import debugger as ipdb


def rband(field):
    """The rband name in different fields."""

    return 'subaru_r' if field.lower() == 'cosmos' else 'cfht_r'


def paus(engine, memba_prod, field, d_cosmos='~/data/cosmos', min_nb=35,
         only_specz=False, secure_spec=False, has_bb=False, sel_gal=True):
    """Load the PAUS data from PAUdm and perform the required
       transformation.

       Args:
           engine (obj): SqlAlchemy engine connected to PAUdb.
           memba_prod (int): The MEMBA production.
           field (str): Field to download.
           d_cosmos (str): Directory with downloaded COSMOS files.
           min_nb (int): Minimum number of narrow bands.
           only_specz (bool): Only selecting galaxy with spectroscopic redshifts.
           secure_spec (bool): Selecting secure spectroscopic redshifts.
           has_bb (bool): Select galaxies with broad bands data.
           sel_gal (bool): Select galaxies.
    """

    import bcnz

    if field.lower() == 'cosmos':
        # The parent catalogue require positional matching.
        paudm_cosmos = bcnz.data.paudm_cosmos(engine)
        cosmos_laigle = bcnz.data.cosmos_laigle(d_cosmos)
        parent_cat = bcnz.data.match_position(paudm_cosmos, cosmos_laigle)

        specz = parent_cat # Contains zCOSMOS DR 3
    elif field.lower() == 'w3':
        parent_cat = bcnz.data.paudm_cfhtlens(engine, 'w3')
        specz = bcnz.specz.deep2(engine)
    else:
        raise ValueError(f'No spectroscopy defined for: {field}')

    paudm_coadd = bcnz.data.paudm_coadd(engine, memba_prod, field)
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
    filters = bcnz.model.all_filters()
    coeff = bcnz.model.nb2bb(filters, synband)

    data_scaled = bcnz.data.synband_scale(nbsubset, coeff, synband=synband,
                                          scale_data=True)

    return data_scaled

paus_calib_sample = functools.partial(
    paus, min_nb=39, only_specz=True, has_bb=True, secure_spec=True)

# The entries for which we run the photo-z.
paus_main_sample = functools.partial(
    paus, min_nb=39, only_specz=False, has_bb=True, sel_gal=False, secure_spec=False)
