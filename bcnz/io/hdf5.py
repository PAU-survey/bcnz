#!/usr/bin/env python
# encoding: UTF8
import pdb
import time
try:
    import tables
except ImportError:
    pass

import numpy as np

"""
def create_descr(cols):
    def colobj(i, col):
        int_cols = ['id']
        if col in int_cols:
            return tables.Int64Col(pos=i)
        else:
            return tables.Float64Col(pos=i)

    descr = dict((col, colobj(i,col)) for i,col in enumerate(cols))

    return descr

def create_hdf5(conf, file_path):

    # Move away existing file
    if os.path.exists(file_path):
        dst = '%s.bak' % file_path
        shutil.move(file_path, dst)

        print('File %s exists. Moving it to %s.' % (file_path, dst))


    cols = conf['order']+conf['others']
    descr = create_descr(cols)

    f = tables.openFile(file_path, 'w')
    f.createGroup('/', 'bcnz')
    f.createTable('/bcnz', 'bcnz', descr, 'BCNZ photo-z')

    return f
"""

import filebase

class read_cat(filebase.filebase):
    def __init__(self, conf, zdata, file_name):
        self.conf = conf
        self.file_name = file_name

        filters = zdata['filters']
        pre_mag = self.conf['pre_mag']
        pre_err = self.conf['pre_err']

#        self.conv_mag = {'{}{}'.format(pre_mag, x):x for x in filters}
#        self.conv_err = {'{}{}'.format(pre_err, x):x for x in filters}
        self.mag_fields = [pre_mag+x for x in self.conf['filters']]
        self.err_fields = [pre_err+x for x in self.conf['filters']]

        fields_in = {x:x for x in self.conf['order'] if not x in self.conf['from_bcnz']}
        fields_in['m_0'] = pre_mag+self.conf['prior_mag']
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

        self.i += 1

        return data


    # Python 2.x compatability
    __next__ = next

"""
        i = 0
        yield self.catalog.read(start=i*nmax, stop=(i+1)*nmax)
        i += 1
"""

class write_cat:
    def __init__(self, conf, out_file):
        pass

    def append(self, cat):
        pdb.set_trace()
