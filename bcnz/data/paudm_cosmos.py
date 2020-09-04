#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger
import os
import pdb
import numpy as np
import pandas as pd

def query(engine, table):
    """The query against the PAUdm postgresql database."""

    # Because the 2015 catalog is huge and this otherwise would not work...
    if table == 'COSMOS':
        sql = """SELECT * from COSMOS"""
    elif 'laigle' in table:
        sql = """SELECT number, alpha_j2000, delta_j2000, ip_mag_auto,EBV,id2006,id2008,id2013 \
                 FROM cosmos2015_laigle_v1_1 \
                 WHERE ip_mag_auto < 23""".format(tbl)

    print('Starting to query..', table)
    cat = pd.read_sql_query(sql, engine)

    return cat

def paudm_cosmos(engine, table='COSMOS'):
    """Download the Ilbert catalogue. This is only to have the position."""

    cat_in = query(engine, table)
    if 'paudm_id' in cat_in.columns:
        cat = cat_in.rename(columns={'paudm_id': 'ref_id'}).set_index('ref_id')
    else:
        cat = cat_in

    return cat
