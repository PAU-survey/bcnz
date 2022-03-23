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

import os
import sys

import numpy as np
import pandas as pd


def load_seds(input_dir):
    """Load seds from files.

       Args:
           input_dir: Directory where the SEDs are stored.
    """

    input_dir = os.path.expanduser(input_dir) 
    suf = 'sed'
    min_val = 0

    df = pd.DataFrame()
    for fname in os.listdir(input_dir):
        if not fname.endswith(suf):
            continue

        path = os.path.join(input_dir, fname)
        lmb, response = np.loadtxt(path).T
        name = fname.replace('.'+suf, '')

        # This is to avoid numerical aritifacts which can
        # be important if the spectrum is steep.
        if min_val:
            y[y < min_val] = 0

        part = pd.DataFrame()
        part['lmb'] = lmb
        part['response'] = response
        part['sed'] = name

        df = df.append(part, ignore_index=True)

    # The SEDs are sometimes defined with duplicate entries that ends
    # up creating technical problems later.
    df = df.drop_duplicates()

    df = df.set_index('sed')

    return df
