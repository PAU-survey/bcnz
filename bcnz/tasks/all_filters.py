#!/usr/bin/env python
# encoding: UTF8

import os
from pathlib import Path
import pandas as pd
from glob import glob

class all_filters:
    config = {}
    version = 1.11

    def entry(self):

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


    def run(self):
        result = self.entry()
        self.output.result = result
