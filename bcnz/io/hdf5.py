#!/usr/bin/env python
# encoding: UTF8
import pdb

try:
    import tables
except ImportError:
    pass

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
    """Empty HDF5 to with a tables to store the photo-z results."""

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
