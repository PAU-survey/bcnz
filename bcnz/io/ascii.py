#!/usr/bin/env python
# encoding: UTF8

import pdb
import numpy as np

import filebase

class read_cat(filebase.filebase):
    """Read an ascii catalogs. The columns file reading is directly
       included here since I only want to use that format together
       with the ascii files.
    """

    msg_notsupported = 'The feature in not implemented.'
    def __init__(self, conf, zdata, file_name):
        self.conf = conf
        self.zdata = zdata
        self.file_name = file_name
        self.cols = self._columns()

        assert (self.cols['zp_errors'] == 0.).all(), self.msg_notsupported
        assert (self.cols['zp_offsets'] == 0.).all(), self.msg_notsupported

    def _read_columns_file(self, file_name):
        """Convert each line in the columns file to a touple."""

        res = {}
        for line in open(file_name):
            spld = line.strip().split()
            spld = [x.split(',') for x in spld]
            spld = sum(spld, [])

            key = spld[0]
            val = spld[1] if len(spld) == 2 else tuple(spld[1:])
            res[key] = val

        return res

    def _columns(self):
        """Split the input from the columns file in different parts."""

        cols_input = self._read_columns_file(self.conf['columns'])
        A = zip(*[cols_input[x] for x in self.zdata['filters']])

        cols = {}
        cols['mag_cols'] = (np.array(A[0]).astype(np.int) - 1).tolist()
        cols['emag_cols'] = (np.array(A[1]).astype(np.int) - 1).tolist()
        cols['cals'] = A[2]
        cols['zp_errors'] = np.array(A[3]).astype(np.float)
        cols['zp_offsets'] = np.array(A[4]).astype(np.float)

        return cols

    def __iter__(self):
        return self

    def next(self):
        data = {
          'mag': np.loadtxt(self.file_name, usecols=self.cols['mag_cols']),
          'emag': np.loadtxt(self.file_name, usecols=self.cols['emag_cols']),
        }

        return data

class write_cat:
    def __init__(self, conf, out_file):
        self.out_file = out_file
