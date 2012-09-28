#!/usr/bin/env python
# encoding: UTF8
import os
import glob
import pdb

import numpy as np

import config
import descr
import bcnz_config
import bcnz_compat
import bcnz_descr
import bcnz_div
import bcnz_parser

def parse_arguments():
    """Parse input arguments."""

    first_parser = bcnz_parser.first_parser()
    config_file = first_parser()

    try: 
        m = getattr(config, config_file)
        def_conf = getattr(m, 'conf')
    except AttributeError:
        raise

    def_descr = descr.standard.descr

    arg_parser = bcnz_parser.argument_parser(def_conf, def_descr)
    conf = arg_parser.parse_args()

    bcnz_compat.assert_compat(conf)
    bcnz_div.test_conf(conf)

    return conf

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

def test_config(conf):
    check_collision(conf)

def catalogs(conf):
    """File names with input catalogs."""

    input_file = conf['catalog']
    cat_files = glob.glob(input_file)
    msg_noinput = "Found no input files for: %s" % input_file

    assert len(cat_files), msg_noinput

    return cat_files

def columns_file(conf):
    obs_files = conf['obs_files']

    if len(obs_files) == 1: 
#    if os.path.exists(obs_file):
        root = os.path.splitext(obs_files[0])[0]
        file_name = "%s.%s" % (root, 'columns')

        return file_name
    elif 'columns' in conf:
        return conf['columns']

    raise ValueError

def basic_read(file_name):
    """Remove empty and commented lines."""

    a = [line.strip() for line in open(file_name)]
    a = [x for x in a if not x.startswith('#')]
    a = [x for x in a if x]

    return a
 
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

def split_col_pars(col_pars, filters):
    """Split the input from the columns file in different parts."""

    A = zip(*[col_pars[x] for x in filters])
  
    out = {} 
    out['flux_cols'] = (np.array(A[0]).astype(np.int) - 1).tolist() 
    out['eflux_cols'] = (np.array(A[1]).astype(np.int) - 1).tolist()
    out['cals'] = A[2]
    out['zp_errors'] = np.array(A[3]).astype(np.float)
    out['zp_offsets'] = np.array(A[4]).astype(np.float)

    return out
#    return flux_cols, eflux_cols, cals, zp_errors, zp_offsets
