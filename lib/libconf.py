#!/usr/bin/env python
# encoding: UTF8

import os
import sys

import config
import descr
import bcnz_compat
import bcnz_input
import bcnz_parser

def root(rm_dir='bin'):
    """Find root directory."""

    cmd = sys.argv[0]
    path = os.path.abspath(os.path.dirname(cmd))

    if os.path.split(path)[1] == rm_dir:
        path = os.path.join(path, '..')
        path = os.path.normpath(path)

    return path

def set_min_rms(conf):
    # Only for full compatability with BPZ.

    if conf['spectra'] == 'CWWSB.list':
        conf['min_rms'] = 0.067

def parse_arguments():
    """Parse input arguments."""

    # Detects which configuration file to use. This enable
    # b
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
    test_conf(conf)

    return conf

def print_parameters(conf):
    """Print the value of all parameters in conf."""

    keys = conf.keys()
    keys.sort()

    for key in keys:
        print('%s   =   %s' % (key.upper(), conf[key]))

def update_conf(conf):
    """Update some of the configuration values."""

    conf['root'] = root()
    conf['obs_files'] = bcnz_input.catalogs(conf)
    conf['col_file'] = bcnz_input.columns_file(conf)
    set_min_rms(conf)

    if conf['verbose']:
        print('Current parameters') #z
        print_parameters(conf)

    return conf

def test_config(conf):
    bcnz_input.check_collision(conf)

def test_conf(conf):
    msg_z = 'zmax <= zmin is not allowed.'
    assert conf['zmin'] < conf['zmax'], msg_z


