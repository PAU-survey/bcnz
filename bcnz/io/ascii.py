##!/usr/bin/env python
# encoding: UTF8

import pdb
import numpy as np

from bcnz.io import filebase

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
        self.col_other, self.cols = self._columns()
        self.has_read = False

        assert (self.cols['zp_errors'] == 0.).all(), self.msg_notsupported
        assert (self.cols['zp_offsets'] == 0.).all(), self.msg_notsupported

    def open(self):
        """Not relevant for the ascii files."""

        pass

    def close(self):
        """Not relevant for the ascii files."""

        pass


    def _read_columns_file(self, file_name):
        """Convert each line in the columns file to a tuple."""

        col2,col4 = {},{}
        for line in open(file_name):
            spld = line.strip().split()
            spld = [x.split(',') for x in spld]
            spld = sum(spld, [])

            key = spld[0]
            if len(spld) == 2:
                col2[key] = int(spld[1])-1
            else:
                col4[key] = tuple(spld[1:])


        return col2, col4

    def _columns(self):
        """Split the input from the columns file in different parts."""

        # Grouping similar information for different filters together.
        col2, col4 = self._read_columns_file(self.conf['columns'])
        A = list(zip(*[col4[x] for x in self.zdata['filters']]))

        cols = {}
        cols['mag_cols'] = (np.array(A[0]).astype(np.int) - 1).tolist()
        cols['emag_cols'] = (np.array(A[1]).astype(np.int) - 1).tolist()
        cols['cals'] = A[2]
        cols['zp_errors'] = np.array(A[3]).astype(np.float)
        cols['zp_offsets'] = np.array(A[4]).astype(np.float)

        return col2, cols

    def _read_all(self):
        """Read all entries."""

        try:
            data = {
              'mag': np.loadtxt(self.file_name, usecols=self.cols['mag_cols']),
              'emag': np.loadtxt(self.file_name, usecols=self.cols['emag_cols']),
            }
        except IndexError:
            raise IndexError('Could not load columns from file: {}'.format(self.file_name))

        keys, cols = zip(*self.col_other.items())
        A = np.loadtxt(self.file_name, usecols=cols, unpack=True)
        for i,key in enumerate(keys):
            data[key] = A[i]

        return data

    def __iter__(self):
        return self

    def __next__(self):
        if not self.has_read:
            self.has_read = True
            return self._read_all()

        raise StopIteration

    # Python 2.x compatability.
    next = __next__

class write_cat(filebase.filebase):
    """Write ascii catalog to file."""

    def __init__(self, conf, zdata, out_peaks, out_pdf, nz, zt):
        self.conf = conf
        self.out_name = out_name

        assert not conf['out_pdf'], 'Writing pdfs in ascii is not supported.'

    def open(self):

        self.setup()
        self.fb_out = open(self.out_path, 'w')

    def close(self):
        self.fb_out.close()

    def fix_format(self, dtype):
        """Format string for output."""

        def cfmt(i, dt):
            if dt == '<i8':
                return '{'+str(i)+'}'
            elif dt == '<f8':
                return '{'+str(i)+':.4f}' 

        fmt = [cfmt(i,dt) for i,(key,dt) in enumerate(dtype.descr)]
        self.fmt = ' '.join(fmt)+'\n'

    def append(self, block):
        """Append record array to file."""

        if not hasattr(self, 'fmt'):
            self.fix_format(block.dtype)

        lines = [self.fmt.format(*x) for x in block]
        self.fb_out.writelines(lines)
        self.fb_out.flush()
