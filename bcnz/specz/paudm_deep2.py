#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import os
import pandas as pd
import psycopg2

def deep2(engine):
    """Downloads the full deep2 catalogue.
       Args:
           engine (obj): Connection to PAUdb.
    """

    sql = """SELECT ra,dec,z as zspec,zquality,obj_type,magi
             FROM deep2
             WHERE 0 < z"""

    cat = pd.read_sql_query(sql, conn)
    cat = cat[cat.obj_type == 'GALAXY']

    return cat
