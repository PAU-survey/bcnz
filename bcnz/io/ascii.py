#!/usr/bin/env python
# encoding: UTF8

import pdb
import filebase

class columns_file:
    def _columns_file(self):
        """Name of the columns file."""

        obs_files = self.conf['obs_files']

        if 'columns' in self.conf:
            return self.conf['columns']
        elif len(obs_files) == 1:
            root = os.path.splitext(obs_files[0])[0]
            file_name = "%s.%s" % (root, 'columns')

            return file_name
        else:
            raise ValueError

    def find_columns(file_name):
        res = {}
        for line in open(file_name):
            spld = line.strip().split()
            spld = [x.split(',') for x in spld]
            spld = sum(spld, [])

    #        key, val = spld[0], tuple(spld[1:])
            key = spld[0]
            val = spld[1] if len(spld) == 2 else tuple(spld[1:])
            res[key] = val

        return res

class read_cat(filebase.filebase):
    def __init__(self, file_name):
        self.file_name = file_name

    def basic_read(file_name):
        """Remove empty and commented lines."""

        a = [line.strip() for line in open(file_name)]
        a = [x for x in a if not x.startswith('#')]
        a = [x for x in a if x]

        return a

    def split_col_pars(col_pars, filters):
        """Split the input from the columns file in different parts."""

        A = zip(*[col_pars[x] for x in filters])

        out = {}
        out['flux_cols'] = (np.array(A[0]).astype(np.int) - 1).tolist()
        out['eflux_cols'] = (np.array(A[1]).astype(np.int) - 1).tolist()
        out['cals'] = A[2]
        out['zp_errors'] = np.array(A[3]).astype(np.float)
        out['zp_offsets'] = np.array(A[4]).astype(np.float)

        return out

    def __iter__(self):
        return self

    def next(self):
        pdb.set_trace()

class write_cat:
    def __init__(self, conf, out_file):
        self.out_file = out_file
