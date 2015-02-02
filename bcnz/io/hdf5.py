#!/usr/bin/env python
# encoding: UTF8

import ipdb
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

        fields_in['m0'] = mag_fmt.format(self.conf['prior_mag'])
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
        for key in ['zs', 'ra', 'dec', 'spread_model_i', 'm0']:
            if key in names:
                data[key] = tbl_array[key]

        self.i += 1

        return data


    # Python 2.x compatability
    __next__ = next


class write_cat(filebase.filebase):

    def __init__(self, conf, zdata, out_paths, nz, nt):
        self.conf = conf
        self.zdata = zdata
        self.out_paths = out_paths
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

        return {'node': peaks, 'fb': fb}

    def _base_pdf(self, file_path, shape):
        """Initialize empty file for storing the photo-z pdfs."""

        assert not os.path.exists(file_path), 'File already exists: {0}'.format(file_path)

        fb = tables.openFile(file_path, 'w')

        z_mean = self.zdata['z_model']

        # Technically the pdfs could have been estimated in bins with
        # different widths.
        dz_arr = z_mean[1:] - z_mean[:-1]
        assert np.allclose(dz_arr, dz_arr[0])
        z_width = dz_arr[0]*np.ones_like(z_mean)

        fb.create_array('/', 'z_mean', z_mean)
        fb.create_array('/', 'z_width', z_width)

        pdfs = fb.createEArray('/', 'pdfs', tables.FloatAtom(), shape)

        return {'node': pdfs, 'fb': fb}


    def _create_file_pdfs(self, file_path):
        """Initialize empty file for storing the photo-z pdfs."""

        shape = (0, self.nz)
        return self._base_pdf(file_path, shape)

    def _create_file_pdfs_type(self, file_path):
        """Initialize empty file for storing the photo-z pdfs with type info."""

        shape = (0, self.nz, self.nt)
        return self._base_pdf(file_path, shape)

    def open(self):
        f_out = {'pzcat': self._create_file_peaks,
                 'pzpdf': self._create_file_pdfs,
                 'pzpdf_type': self._create_file_pdfs_type}

        nodes = {}
        for out, file_path in self.out_paths.iteritems():
            nodes[out] = f_out[out](file_path)

        self.nodes = nodes

    def append(self, output):
        # Assumes all opened nodes has a corresponding output. Hiding failure
        # here can mask other errors.
        for key, node in self.nodes.iteritems():
            node['node'].append(output[key])


    def close(self):
        """Make sure the files are properly closed."""

        # One could also delete self.nodes, but the error messages would
        # probably be too cryptic.
        for node in self.nodes.itervalues():
            node['fb'].close()


def load_pzcat(file_path):
    """Load the photo-z catalog to a record array."""

    fb = tables.openFile(file_path)
    cat = fb.getNode('/photoz/photoz').read()
    fb.close()

    return cat
