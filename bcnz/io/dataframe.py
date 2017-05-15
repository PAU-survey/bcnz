#!/usr/bin/env python
# encoding: UTF8

from __future__ import print_function
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
        store['dz'] = pd.Series(self.zdata['dz'])
        store['seds'] = pd.Series(self.zdata['seds'])

        self._store = store

    def close(self):
        self._store.close()

    def append(self, output):
        # Assumes all the entries are recordarrays. Valid for
        # now.
        galid = output['pzcat']['id']

        for key,val in output.iteritems():
            if key in ['pzpdf_type', 'chi2']:
                panel = pd.Panel(val, items=galid)
                df = panel.to_frame().stack()
            else:

                df = pd.DataFrame(val)
                df = df.set_index(galid)
                df = df.stack()

            dfname = 'default' if key == 'pzcat' else key

            self._store.append(dfname, df)
