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

    cat = pd.read_sql_query(sql, engine)
    cat = cat.set_index('ref_id')

    return cat
