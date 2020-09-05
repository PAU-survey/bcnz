# encoding: UTF8

import functools
from IPython.core import debugger as ipdb

def paus(engine, memba_prod, field, d_cosmos = '~/data/cosmos', min_nb=35,
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
    paudm_cosmos = bcnz.data.paudm_cosmos(engine)
    cosmos_laigle = bcnz.data.cosmos_laigle(d_cosmos)

    match_cosmos = bcnz.data.match_position(paudm_cosmos, cosmos_laigle)
    paudm_coadd = bcnz.data.paudm_coadd(engine, memba_prod, field)

    data_in = paudm_coadd.join(match_cosmos, how='inner')
    data_noisy = bcnz.data.fix_noise(data_in)

    conf = {'min_nb': min_nb, 'only_specz': only_specz, 'secure_spec': secure_spec,
            'has_bb': has_bb, 'sel_gal': sel_gal}

    nbsubset = bcnz.data.gal_subset(data_noisy, paudm_cosmos, **conf)

    # Synthetic narrow band coefficients.
    filters = bcnz.model.all_filters()
    coeff = bcnz.model.nb2bb(filters, 'subaru_r')

    data_scaled = bcnz.data.synband_scale(nbsubset, coeff, scale_data=True)

    return data_scaled


paus_calib_sample = functools.partial(paus, min_nb=39, only_specz=True, secure_spec=True)


#def paus_calib_sample(engine, meba_prod, di**kwrdargs):
#    data = paus(