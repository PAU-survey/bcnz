#!/usr/bin/env python
# encoding: UTF8

import os
import glob
import pdb
import sys

import tables
import numpy as np

# Move elsewhere.
def check_collision(conf):
    """Check for collision between different input files."""

    file_keys = ['obs_files', 'col_file']
    file_names = [conf[x] for x in file_keys]

    all_files = []
    for file_name in file_names:
        if isinstance(file_name, list):
            all_files.extend(file_name)
        else:
            all_files.append(file_name)

    msg_overwrite = 'Overwriting some input files.'
    assert len(set(all_files)) == len(all_files), msg_overwrite

def catalogs(conf):
    """File names with input catalogs."""

    input_file = conf['catalog']
    cat_files = glob.glob(input_file)
    msg_noinput = "Found no input files for: %s" % input_file

    assert len(cat_files), msg_noinput

    return cat_files

#def columns_file(conf):
#    """Name of the columns file."""
#
#    obs_files = conf['obs_files']
#
#    if 'columns' in conf:
#        return conf['columns']
#    elif len(obs_files) == 1: 
#        root = os.path.splitext(obs_files[0])[0]
#        file_name = "%s.%s" % (root, 'columns')
#
#        return file_name
#    else:
#        raise ValueError
#    
#    pdb.set_trace()

#def basic_read(file_name):
#    """Remove empty and commented lines."""
#
#    a = [line.strip() for line in open(file_name)]
#    a = [x for x in a if not x.startswith('#')]
#    a = [x for x in a if x]
#
#    return a
 
def mapping_templ_prior(file_name):
    """Read file with mapping between templates
       and priors names.
    """

    data = basic_read(file_name)   
    d = dict([x.split() for x in data])

    return d

def spectras(file_name):
    """Read in list of spectras."""

    data = basic_read(file_name)

    return data

def templ_prior(f_templ_pr, specs):
    """List of priors corresponding to the spectras 
       we use.
    """

    d = mapping_templ_prior(f_templ_pr)

    return [d[x.replace('.sed','')] for x in specs]

#def split_col_pars(col_pars, filters):
#    """Split the input from the columns file in different parts."""
#
#    A = zip(*[col_pars[x] for x in filters])
#  
#    out = {} 
#    out['flux_cols'] = (np.array(A[0]).astype(np.int) - 1).tolist() 
#    out['eflux_cols'] = (np.array(A[1]).astype(np.int) - 1).tolist()
#    out['cals'] = A[2]
#    out['zp_errors'] = np.array(A[3]).astype(np.float)
#    out['zp_offsets'] = np.array(A[4]).astype(np.float)
#
#    return out

def DM_filter_names(filters):
    """The notation which PAU-DM is using for the filters."""

    known_broadband = ['i', 'up', 'g', 'r', 'z', 'y']
    new_names = []
    i = 0
    for f in filters:
        if f in known_broadband:
            new_names.append(f)
        else:
            new_names.append(str(i))
            i += 1

    return new_names
