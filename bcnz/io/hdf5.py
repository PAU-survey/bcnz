#!/usr/bin/env python
# encoding: UTF8

import os
import pdb
import time
try:
    import tables
except ImportError:
    pass

import numpy as np


import filebase

class read_cat(filebase.filebase):
    def __init__(self, conf, zdata, file_name):
        self.conf = conf
        self.file_name = file_name

        filters = zdata['filters']
        mag_fmt = self.conf['mag_fmt']
        err_fmt = self.conf['err_fmt']

        self.mag_fields = [mag_fmt.format(x) for x in self.conf['filters']]
        self.err_fields = [err_fmt.format(x) for x in self.conf['filters']]

        fields_in = dict(
            (x, x) for x in self.conf['order'] if not x in self.conf['from_bcnz']
        )

        fields_in['m_0'] = mag_fmt.format(self.conf['prior_mag'])
        self.fields_in = fields_in 

    def open(self):
        self.i = 0 
        self.catalog = tables.openFile(self.file_name)

        self.nmax = self.conf['nmax']
        self.cat = self.catalog.getNode(self.conf['hdf5_node'])
        self.nf = len(self.conf['filters'])

    def close(self):
        self.catalog.close()

    def __iter__(self):
        return self

    def next(self):
        i = self.i
        nmax = self.nmax
        nf = self.nf

        tbl_array = self.cat.read(start=i*nmax, stop=(i+1)*nmax)
        ngal_read = tbl_array.shape[0]

        if not ngal_read:
            raise StopIteration

        data = {}
        for to_field, from_field in self.fields_in.iteritems():
            data[to_field] = tbl_array[from_field]

        mag = np.zeros((ngal_read, nf))
        err = np.zeros((ngal_read, nf))
        mag_fields = self.mag_fields
        err_fields = self.err_fields
        for j in range(nf):
            mag[:,j] = tbl_array[mag_fields[j]]
            err[:,j] = tbl_array[err_fields[j]]

        data['mag'] = mag
        data['emag'] = err

        names = tbl_array.dtype.names
        for key in ['z_s', 'ra', 'dec', 'spread_model_i', 'm_0']:
            if key in names:
                data[key] = tbl_array[key]

        self.i += 1

        return data


    # Python 2.x compatability
    __next__ = next


class write_cat(filebase.filebase):

    def __init__(self, conf, out_name):
        self.conf = conf
        self.out_name = out_name


    def create_descr(self, cols):
        def colobj(i, col):
            int_cols = ['id']
            if col in int_cols:
                return tables.Int64Col(pos=i)
            else:
                return tables.Float64Col(pos=i)
    
        descr = dict((col, colobj(i,col)) for i,col in enumerate(cols))
    
        return descr

    def create_hdf5(self, file_path):

        assert not os.path.exists(file_path), 'File already exists: {0}'.format(file_path)


        cols = self.conf['order'] + self.conf['others']
        descr = self.create_descr(cols)

        fb = tables.openFile(file_path, 'w')
        fb.createGroup('/', 'photoz')
        self. out_cat = fb.createTable('/photoz', 'photoz', descr, 'BCNZ photo-z')

        self.fb = fb

    def open(self):
        self.setup()

        self.create_hdf5(self.out_path)

    def append(self, cat):
        self.out_cat.append(cat)

    def close(self):
        self.relink()

def load_pzcat(file_path):
    """Load the photo-z catalog to a record array."""

    fb = tables.openFile(file_path)
    cat = fb.getNode('/photoz/photoz').read()
    fb.close()

    return cat
