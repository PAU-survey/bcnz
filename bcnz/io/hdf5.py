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

            # In the case of adding errors, this field is not always
            # present in the input catalog.
            if not self.conf['add_noise']:
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

    def __init__(self, conf, out_peaks, out_pdfs, nz, nt):
        self.conf = conf
        self.out_peaks = out_peaks
        self.out_pdfs = out_pdfs
        self.nz = nz
        self.nt = nt

        self._write_pdfs = self.conf['out_pdf']

    def create_descr(self, cols):
        def colobj(i, col):
            int_cols = ['id']
            if col in int_cols:
                return tables.Int64Col(pos=i)
            else:
                return tables.Float64Col(pos=i)
    
        descr = dict((col, colobj(i,col)) for i,col in enumerate(cols))
    
        return descr

    def _create_file_peaks(self, file_path):
        """Initialize empty file for storing the photo-z peaks."""

        assert not os.path.exists(file_path), 'File already exists: {0}'.format(file_path)


        cols = self.conf['order'] + self.conf['others']
        descr = self.create_descr(cols)

        fb = tables.openFile(file_path, 'w')
        fb.createGroup('/', 'photoz')
        peaks = fb.createTable('/photoz', 'photoz', descr, 'BCNZ photo-z')

        return peaks, fb

    def _create_file_pdfs(self, file_path):
        """Initialize empty file for storing the photo-z pdfs."""

        assert not os.path.exists(file_path), 'File already exists: {0}'.format(file_path)

        fb = tables.openFile(file_path, 'w')
        shape = (0, self.nz, self.nt) if self.conf['pdf_type'] else (0, self.nz)
        pdfs = fb.createEArray('/', 'pdfs', tables.FloatAtom(), shape)

        return pdfs, fb

    def open(self):
        self.setup()

        if self.conf['use_cache']:
            peaks_path = self._obj_path_peaks
            pdfs_path = self._obj_path_pdfs
        else:
            peaks_path = self.out_peaks
            pdfs_path = self.out_pdfs

        self._peaks, self._peaks_file = self._create_file_peaks(peaks_path)
        if self._write_pdfs:
            self._pdfs, self._pdfs_file = self._create_file_pdfs(pdfs_path)

    def append(self, output):
        self._peaks.append(output['peaks'])

        if self._write_pdfs:
            self._pdfs.append(output['pdfs'])

    def close(self):
        """Make sure the files are properly closed."""

        self._peaks_file.close()
        if self._write_pdfs:
            self._pdfs_file.close()

def load_pzcat(file_path):
    """Load the photo-z catalog to a record array."""

    fb = tables.openFile(file_path)
    cat = fb.getNode('/photoz/photoz').read()
    fb.close()

    return cat
