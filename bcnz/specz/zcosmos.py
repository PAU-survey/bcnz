#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger
import os
import pdb
import numpy as np
import pandas as pd

def zcosmos(engine):
    """Download the Ilbert catalogue. This is only to have the position.

       Args:
           engine (obj): Connection to PAUdb.
    """

    # Similar to the COSMOS query, but here restricted to only get the
    # columns relevant for validation.
    sql = """SELECT paudm_id AS ref_id, zspec, "I_auto", r50, conf
             FROM cosmos
             WHERE zspec > 0
          """

    print('Starting to query..', table)
    cat = pd.read_sql_query(sql, engine)
    cat = cat.set_index('ref_id')

    return cat
