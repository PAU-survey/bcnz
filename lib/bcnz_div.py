#!/usr/bin/env
import os
import pdb
import sys
import time
import re

import bcnz_input

import numpy as np

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

def set_min_rms(conf):
    # Only for full compatability with BPZ.

    if conf['spectra'] == 'CWWSB.list':
        conf['min_rms'] = 0.067

def test_conf(conf):

    msg_z = 'zmax <= zmin is not allowed.'
    assert conf['zmin'] < conf['zmax'], msg_z

#    pdb.set_trace()

def seglist(vals, mask=None):
   # Sent to Txitxo... not update to absolute imports of numpy.

   if mask == None:
       mask = greater(vals, 0)

   # up_ste - Indices where mask goes from False to True
   # down_step - Indices where mask goes from True to False
   up_step = 1+nonzero(logical_and(logical_not(mask[:-1]), mask[1:]))[0]
   down_step = 1+nonzero(logical_and(mask[:-1], logical_not(mask[1:])))[0]

   # Adding False on both ends. This ensures the ends are included if
   # the values are positive there. Unless the if condition is evaluated to
   # false, the last part is not calculated.
   up_step = up_step if not mask[0] else [0] + list(up_step)
   down_step = down_step if not mask[-1] else list(down_step) + [len(vals)]

   res = [list(vals[i:j]) for i,j in zip(up_step, down_step)]
   return res

def z_binning(conf):
    zmin = conf['zmin']
    zmax = conf['zmax']
    dz = conf['dz']

    return np.arange(zmin,zmax+dz,dz)

def sel_files(d, suf):

    files = os.listdir(d)
    files = [x.replace(suf,'') for x in files if x.endswith(suf)]

    return files

def find_filters(conf):

    # Demand that at least the magnitude system is specified.
    filters = []
    for line in open(conf['col_file']):
        spl = line.split()
        if not 2 < len(spl):
            continue

        filters.append(spl[0])

    filters = [re.sub('.res$','',x) for x in filters]
    filters = [x for x in filters if not x in conf['exclude']]

#    pdb.set_trace()
    return filters

def find_spectra(spectra_file):
    spectra = np.loadtxt(spectra_file, usecols=[0], unpack=True, dtype=np.str)
    spectra = spectra.tolist()
    spectra = [re.sub('.sed$', '', x) for x in spectra]

    return spectra

def check_found(what, elems, elem_db):
    elem_nf = set(elems) - set(elem_db)
    msg = '%s not found in database: %s' % (what, list(elem_nf))

    assert not elem_nf, msg

def print_parameters(conf):
    """Print the value of all parameters in conf."""

    keys = conf.keys()
    keys.sort()

    for key in keys:
        print('%s   =   %s' % (key.upper(), conf[key]))

def root(rm_dir='bin'):
    """Find root directory."""

    cmd = sys.argv[0]
    path = os.path.abspath(os.path.dirname(cmd))

    if os.path.split(path)[1] == rm_dir:
        path = os.path.join(path, '..') 
        path = os.path.normpath(path)


    return path

def spectra_file(conf):
    """Detect the right path of spectra file."""

    for d in [conf['root'], conf['sed_dir']]:
        file_path = os.path.join(d, conf['spectra'])
        if os.path.exists(file_path):
            return file_path

    raise ValueError, 'No spectra file found.'

def update_conf(conf):
    """Update some of the configuration values."""

    conf['root'] = root()
    conf['obs_files'] = bcnz_input.catalogs(conf)
    conf['col_file'] = bcnz_input.columns_file(conf)
    set_min_rms(conf)

    if conf['verbose']:
        print('Current parameters')
        print_parameters(conf)

    return conf


class watch:
    """Stopwatch that can disply time intervals."""

    def __init__(self, fmt='%H:%M:%S'):
        self.fmt = fmt
        self.start_time = time.time()

    def print_difference(self):
        diff = time.gmtime(time.time() - self.start_time)

        days = diff[2]
        if 1 < days:
            print('The runtime is in addition %s days' % str(days-1))

        time_str = time.strftime(self.fmt, diff)
 
        print('\nElapsed time %s\n' % time_str)

