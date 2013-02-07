#!/usr/bin/env python

import pdb
import tables

class read_cat:
    def __init__(self, obs_file, nmax, cols_keys, cols, filters):
        self.catalog = tables.openFile(obs_file)

    def __iter__(self):
        i = 0
        yield self.catalog.read(start=i*nmax, stop=(i+1)*nmax)
        i += 1

class write_cat:
    def __init__(self, conf, out_file):
        pass

    def append(self, cat):
        pdb.set_trace()
