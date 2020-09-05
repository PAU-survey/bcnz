# encoding: UTF8

def paus(engine, memba_prod, field, d_cosmos = '~/data/cosmos'):
    """Load the PAUS data from PAUdm and perform the required
       transformation.

       Args:
           engine (obj): SqlAlchemy engine connected to PAUdb.
           memba_prod (int): The MEMBA production.
           field (str): Field to download.
           d_cosmos (str): Directory with downloaded COSMOS files.
    """

    import bcnz
    paudm_cosmos = bcnz.data.paudm_cosmos(engine)
    cosmos_laigle = bcnz.data.cosmos_laigle(d_cosmos)

    match_cosmos = bcnz.data.match_position(paudm_cosmos, cosmos_laigle)
    paudm_coadd = bcnz.data.paudm_coadd(engine, memba_prod, field)

    data_in = paudm_coadd.join(match_cosmos, how='inner')
    data_noisy = bcnz.data.fix_noise(data_in)

    nbsubset = bcnz.data.gal_subset(data_noisy, paudm_cosmos)

    # Synthetic narrow band coefficients.
    filters = bcnz.model.all_filters()
    coeff = bcnz.model.nb2bb(filters, 'subaru_r')

    data_scaled = bcnz.data.synband_scale(nbsubset, coeff, scale_data=True)

    return data_scaled
