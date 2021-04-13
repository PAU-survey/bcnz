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
# !/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import os
import pandas as pd
import psycopg2


def _query_sdss(engine):
    """Open csv file to get the sdss catalogue. FOR W3 !!!!"""
    #sdss_file = "/nfs/pic.es/user/a/awittje/src/bcnz/bcnz/specz/sdss_dr12_w3_calib.csv"
    
    cat = pd.read_csv(sdss_file,header = 1)#### 1 !!
    cat = cat.rename(columns = {'redshift':'zspec', 'zwarning':'zquality', 'class':'obj_type', 'i':'magi'}) 

    cat = cat
    print(cat)
    
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

def sdss(engine):
    """Downloads the full sdss_spec (for W3) catalogue and match to get
       PAUdm reference IDs.

       Args:
           engine (obj): Connection to PAUdb.
    """

    import bcnz

    # Specz catalogue needs positional matching.
    parent_cat = _query_refcat(engine)
    sdss_in = _query_sdss(engine)
    sdss = bcnz.data.match_position(parent_cat, sdss_in)

    return sdss    
