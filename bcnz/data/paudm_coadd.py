#!/usr/bin/env python
# encoding: UTF8

from __future__ import print_function

from IPython.core import debugger as ipdb
import os
import numpy as np
import pandas as pd
import psycopg2

desc = {
  'prod_memba': 'Production number',
  'run': 'Something Santi introduced. I have no idea what it is...',
  'ilim': 'Iband magnitude limit'}

def query_cosmos(engine, prod_memba, ilim, run):
    """Query for a catalogue in the COSMOS field."""

    sql = 'SELECT fac.* FROM forced_aperture_coadd AS fac JOIN cosmos AS cm ON cm.paudm_id=fac.ref_id \
           WHERE fac.production_id={0} AND cm."{1}"<{2} and fac.run={3}'

    sql = sql.format(prod_memba, 'I_auto', ilim, run)

    cat = pd.read_sql_query(sql, engine)

    return cat

def query_cfht(engine, prod_memba, ilim, run):
    """Query for a catalogue in a CFHT field."""

    sql = 'SELECT fac.* FROM forced_aperture_coadd AS fac JOIN cfhtlens AS refcat ON refcat.paudm_id=fac.ref_id \
               WHERE fac.production_id={0} AND refcat."{1}"<{2} and fac.run={3}'
    sql = sql.format(prod_memba, 'mag_i', ilim, run)

    cat = pd.read_sql_query(sql, engine)

    return cat


def paudm_coadd(engine, prod_memba, field, ilim=25, run=1):
    """Query the coadd with a possible selection."""

    args = prod_memba, ilim, run
    if field.lower() == 'cosmos':
        cat = query_cosmos(engine, *args)
    else:
        cat = query_cfht(engine, *args)

    # Since we later combine with Subary, which also has narrow bands.
    cat['band'] = cat.band.apply(lambda x: 'pau_{}'.format(x.lower())) 

    return cat
