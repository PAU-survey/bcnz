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
from pathlib import Path
import pandas as pd
from glob import glob


def all_filters():
    """Create a dataframe joining all filters."""

    # Upgraded directory to include the CFHT + KiDS filters.
    dfilters = '~/data/photoz/all_filters/v4'
    dfilters = os.path.expanduser(dfilters)

    L = []
    for x in glob(str(Path(dfilters) / '*')):
        path = Path(x)

        sep = ',' if path.suffix == '.csv' else ' '
        part = pd.read_csv(x, names=['lmb', 'response'], comment='#', sep=sep)
        part['band'] = path.with_suffix('').name

        L.append(part)

    assert len(L), 'Filters not found.'
    df = pd.concat(L, ignore_index=True)
    df = df.set_index('band')

    return df
