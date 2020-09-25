# Copyright (C) 2020 Martin B. Eriksen
# This file is part of BCNz <https://github.com/PAU-survey/bcnz>.
#
# BCNz is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BCNz is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BCNz.  If not, see <http://www.gnu.org/licenses/>.
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
    """Download the Ilbert catalogue. This is only to have the position.
       Args:
           engine (obj): Connection to PAUdb.
           table (str): Table in the data base.
    """

    cat_in = query(engine, table)
    if 'paudm_id' in cat_in.columns:
        cat = cat_in.rename(columns={'paudm_id': 'ref_id'}).set_index('ref_id')
    else:
        cat = cat_in

    return cat
