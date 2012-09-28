#!/usr/bin/env python
# 
import os
import sys
import numpy as np

class file_part:
    """Only returns lines nmax lines from the current position
       at initializing.
    """

    def __init__(self, fobj, nmax):
        self.fobj = fobj
        self.nmax = nmax

#    @profile
    def __iter__(self):
        n = 0
        while n < self.nmax:
            line = self.fobj.readline()
            if not line: raise StopIteration
            if not line.startswith('#'):
                n += 1
                yield line


class loadparts:
    """Creates an iterator over parts of the file."""

#    @profile
    def __init__(self, file_name, nmax, cols_keys, cols):
        self.nmax = nmax
        self.fobj = open(file_name)

        self.cols_keys = cols_keys
        self.ncols, self.all_cols = self.getvars(cols)

#    @profile
    def getvars(self, cols):
        all_cols = []
        for x in cols:
            if isinstance(x, list) or isinstance(x,tuple):
                all_cols.extend(x)
            else:
                all_cols.append(x)
    
        ncols = [(len(x) if hasattr(x, '__len__') else 1) for x in cols]

        return ncols, all_cols

    def __iter__(self):
        while True:
            tmp_obj = file_part(self.fobj, self.nmax)
            ans = np.loadtxt(tmp_obj, usecols=self.all_cols)
            if not len(ans):
                raise StopIteration

            data = {}
            ind = 0
            for col_key, ncol in zip(self.cols_keys, self.ncols):
                if ncol == 1:
                    data[col_key] = ans[:,ind]
                else:
                    data[col_key] = ans[:,ind:ind+ncol]
        
                ind += ncol
        
            yield data

