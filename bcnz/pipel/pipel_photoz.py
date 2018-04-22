#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import xdolphin as xd

import pipel_filters
import pipel_galcat
import pipel_pz_chunks

# Move this away...
import libcommon

def pipel(memba):
    """Photo-z pipeline for the PAU data."""

    # Here we intentionally don't set a default memba
    # production.
    coadd = xd.Job('paudm_coadd')
    coadd.config['prod_memba'] = memba

    D = {'coadd': coadd, 
         'galcat': pipel_galcat.galcat(),
         'filters': pipel_filters.filters()}

    xpipel = xd.Job()
    xpipel.depend['pzcat'] = pipel_pz_chunks.pipel()


    ipdb.set_trace()
