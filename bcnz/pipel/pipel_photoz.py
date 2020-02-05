#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import xdolphin as xd

import pipel_filters
import pipel_galcat
import pipel_pz_chunks

import pipel_pz_basic

def pipel(memba, fjc_filters=True, new_bbfilters=False, **kwargs):
    """Photo-z pipeline for the PAU data."""

    # Here we intentionally don't set a default memba
    # production.
#    coadd = xd.Job('paudm_coadd')
#    coadd.config['prod_memba'] = memba

    # HACK: Testing new coadds from a flie...
    coadd = xd.Job('coadd_new_calib')
#    coadd.config['path'] = '~/data/calib/coadd_newcal_v1.csv'
#    coadd.config['path'] = '~/data/calib/coadd_w3_v2.csv'
    coadd.config['path'] = '~/data/calib/coadd_w3_v3.csv'


    # Here the galaxy
    D = {'coadd': coadd, 
         'bbsyn_coeff': pipel_pz_basic.get_bbsyn_coeff(),
         'ref_cat': xd.Job('paudm_cosmos'),
         'galcat': pipel_galcat.galcat(),
         'filters': pipel_filters.filters(fjc_filters, new_bbfilters)}

    xpipel = xd.Job()
    xpipel.depend['pzcat'] = pipel_pz_chunks.pipel(**kwargs)
    xpipel.depend['ref_cat'] = xd.Common('ref_cat')
    xpipel.replace_common(D)

    xpipel.set_session(xd.session())

    return xpipel
