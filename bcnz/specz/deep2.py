#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import os
import pandas as pd
import psycopg2

def _query_deep2(engine):
    """Query to get the deep2 catalogue."""

    sql = """SELECT ra,dec,z as zspec,zquality,obj_type,magi
             FROM deep2
             WHERE 0 < z"""

    cat = pd.read_sql_query(sql, engine)
    cat = cat[cat.obj_type == 'GALAXY']

    return cat

def _query_refcat(engine):
    """Query to find the coordinates."""

    # Similar query to the reference catalogue, but only with the
    # position to reduce the data volume.
    coords = {'ra_min': 200, 'ra_max': 230, 'dec_min': 50, 'dec_max': 60}

    # Here we directly select W3 to save time!
    sql = """SELECT paudm_id as ref_id, alpha_j2000 as ra, delta_j2000 as dec
             FROM cfhtlens \
             WHERE alpha_j2000 > {ra_min} AND alpha_j2000 < {ra_max} \
             AND   delta_j2000 > {dec_min} AND delta_j2000 < {dec_max} \
             AND   mag_i < 23
          """.format(**coords)

    cat = pd.read_sql_query(sql, engine)
    cat = cat.set_index('ref_id')

    return cat

def deep2(engine):
    """Downloads the full deep2 catalogue and match to get
       PAUdm reference IDs.

       Args:
           engine (obj): Connection to PAUdb.
    """

    import bcnz

    # Specz catalogue needs positional matching.
    parent_cat = _query_refcat(engine)
    deep2_in = _query_deep2(engine)
    deep2 = bcnz.data.match_position(parent_cat, deep2_in)

    return deep2    
