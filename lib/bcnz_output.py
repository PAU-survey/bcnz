#!/use/bin/env python
# encoding: UTF8

import os
import pdb
import shutil
import tables
import time

def create_hdf5(conf, file_path):
    """Empty HDF5 to with a tables to store the photo-z results."""

    # Move away existing file
    if os.path.exists(file_path):
        dst = '%s.bak' % file_path
        shutil.move(file_path, dst)

        print('File %s exists. Moving it to %s.' % (file_path, dst))


    int_cols = ['id']
    cols = conf['order']+conf['others']
    def colobj(i, col):
        if col in int_cols:
            return tables.Int64Col(pos=i)
        else:
            return tables.Float64Col(pos=i)


    descr = dict((col, colobj(i,col)) for i,col in enumerate(cols))

    f = tables.openFile(file_path, 'w')
    f.createGroup('/', 'bcnz')
    f.createTable('/bcnz', 'bcnz', descr, 'BCNZ photo-z')

    return f
