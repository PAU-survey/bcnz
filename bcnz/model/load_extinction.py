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
import glob
import numpy as np
from pathlib import Path
import pandas as pd
from IPython.core import debugger as ipdb


def load_extinction(d, suff):
    """Load extinction curves from path."""

    g = os.path.join(os.path.expanduser(d), "*.{}".format(suff))

    df = pd.DataFrame()
    for x in glob.glob(g):
        path = Path(x)
        sep = "," if path.suffix == ".csv" else " "
        part = pd.read_table(x, sep=sep, names=["lmb", "k"], comment="#")
        part["ext_law"] = path.with_suffix("").name

        df = df.append(part, ignore_index=True)

    assert len(df), "No extinction curves found"

    return df
