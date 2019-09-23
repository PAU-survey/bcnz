#!/usr/bin/env python
# encoding: UTF8

import os
from pathlib import Path
import pandas as pd
from glob import glob

class all_filters:
    config = {}
    version = 1.01

    def entry(self):

        dfilters = '~/data/photoz/all_filters/v2'
        dfilters = os.path.expanduser(dfilters)

        L = []
        for path in glob(str(Path(dfilters) / '*')):
            part = pd.read_csv(path, names=['lmb', 'response'])
            part['band'] = Path(path).with_suffix('').name
            
            L.append(part)

        assert len(L), 'Filters not found.'
        df = pd.concat(L, ignore_index=True)
        df = df.set_index('band')

        return df

    def run(self):
        result = self.entry()
        self.output.result = result
