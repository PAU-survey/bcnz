#!/usr/bin/env python
# encoding: UTF8

import os
from pathlib import Path
import pandas as pd
from glob import glob

def all_filters():
    """Create a dataframe joining all filters."""

    # Upgraded directory to include the CFHT filters.
    dfilters = '~/data/photoz/all_filters/v3'
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
