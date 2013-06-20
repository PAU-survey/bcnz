#!/usr/bin/env python
# encoding: UTF8
# Parse arguments from the command line.
import argparse
import pdb
import sys
import numpy as np

import bcnz
import bcnz.config
import bcnz.descr

# is_true  - Values accepted as Bool true
# is_false - Values accepted as Bool false
is_true = ['yes', 'true', '1']
is_false = ['no', 'false', '0']
convert_directly = [int, float, str, bool, complex]
comb_types = [list, tuple, np.ndarray]

# Fuctions for converting types.

class conv_type(argparse.Action, object):
    """Convert more complex types to the same type as defined in
       the configuration.
    """

    def _conv_bool(self, x):
        """Convert from string to bool."""
    
        assert not set(is_true) & set(is_false)
    
        if x.lower() in is_true:
            return True
        elif x.lower() in is_false:
            return False
        else:
            msg = '\nNot convertible to bool: %s' % x            
            raise argparse.ArgumentError(self, msg)
    
    def _find_converter(self, key):
        """Function to convert the type of key."""
    
        def_type = type(self.def_conf[key])
        if def_type == bool:
            return self._conv_bool
        else:
            return def_type
    
    def convert_combined(self, type_conv, key, values):
        assert len(values) == 1

        try:
            val = [x.strip() for x in values[0].split(',')]
            elem_type = type(self.def_conf[key][0])
            val = type_conv([elem_type(x) for x in val])
        except ValueError:
            msg = "\nInvalid value: %s" % str(values)
            raise argparse.ArgumentError(self, msg)

        return val

    def __call__(self, parser, namespace, values, option_string=None):
        key = option_string.lstrip('-')


        # When converting numpy arrays, you can not use the type.
        type_conv = self._find_converter(key)

        # Convert basic types
        def_type = type(self.def_conf[key])
        if def_type in convert_directly:
            val = type_conv(values)
            setattr(namespace, self.dest, val)        
            return

        # Convert list arguments
        val = self.convert_combined(type_conv, key, values)
        setattr(namespace, self.dest, val)

def test_defaults(def_conf, descr):

    missing_descr = set(def_conf.keys()) - set(descr.keys())
    if missing_descr:
        msg = 'Missing descriptions for: %s.' % list(missing_descr)

        raise ValueError(msg)

class bcnz_parser(object):
    def __init__(self, def_conf, descr):
        """Create parser for the BCNZ command line options."""

        test_defaults(def_conf, descr)
        self.parser = self._create_parser(def_conf, descr)

        self.def_conf = def_conf
        self.descr = descr

    def _create_parser(self, def_conf, descr):
        """Add each option to the parser."""

        parser = argparse.ArgumentParser()
        action = conv_type
        action.def_conf = def_conf

        for var, def_val in def_conf.items():
            arg = '-%s' % var
            nargs = '*' if (type(def_val) in comb_types) else '?'
            h = descr[var]

            parser.add_argument(arg, action=conv_type, help=h, nargs=nargs)

        return parser

    def parse_args(self):
        args = self.parser.parse_args()

        keys = self.def_conf.keys()
        vals = [getattr(args, x) for x in keys]

        # Remove entries set to None.
        is_set = lambda X: not isinstance(X[1], type(None))
        to_update = dict(filter(is_set, zip(keys,vals)))

        return to_update

def parse_arguments():
    """Parse input arguments."""

    def_conf = bcnz.config.conf['standard']
    def_descr = bcnz.descr.standard
    arg_parser = bcnz.lib.parser.bcnz_parser(def_conf, def_descr)
    conf = arg_parser.parse_args()

    return conf

