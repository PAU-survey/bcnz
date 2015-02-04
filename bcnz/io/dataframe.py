#!/usr/bin/env python
# encoding: UTF8

import ipdb
import pandas as pd

class write_cat:
    def __init__(self, conf, zdata, out_paths, nz, nt):
        self.conf = conf
        self.zdata = zdata
        self.out_paths = out_paths

        self.nz = nz
        self.nt = nt

    def open(self):
        store = pd.HDFStore(self.out_paths['pzcatdf'], 'w')
        store['z'] = pd.Series(self.zdata['z'])
        store['seds'] = pd.Series(self.zdata['seds'])

        self._store = store

    def close(self):
        self._store.close()

    def append(self, output):
        # Assumes all the entries are recordarrays. Valid for
        # now.
        for key,val in output.iteritems():
            self._store.append(key, pd.DataFrame(val))
