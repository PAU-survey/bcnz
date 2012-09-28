#!/usr/bin/env python
# encoding: UTF8
# Parse arguments from the command line.
import argparse
import pdb
import sys
import numpy as np

is_true = ['yes', 'true', '1']
is_false = ['no', 'false', '0']
convert_directly = [int, float, str, bool, complex]
comb_types = [list, tuple, np.ndarray]

# Fuctions for converting types.

class conv_type(argparse.Action):
    """Convert more complex types to the same type as defined in
       the configuration.
    """

    def conv_bool(self, x):
        """Convert from string to bool."""
    
        assert not set(is_true) & set(is_false)
    
        if x.lower() in is_true:
            return True
        elif x.lower() in is_false:
            return False
        else:
            msg = '\nNot convertible to bool: %s' % x            
            raise argparse.ArgumentError(self, msg)
    
    def find_converter(self, key):
        """Function to convert the type of key."""
    
        def_type = type(self.def_conf[key])
        if def_type == np.ndarray:
            return np.array
        elif def_type == bool:
            return self.conv_bool
        else:
            return def_type
    
    def convert_combined(self, type_conv, key, values):
        try:
            val = [x.rstrip(',') for x in values]
            elem_type = type(self.def_conf[key][0])
            val = type_conv([elem_type(x) for x in val])
        except ValueError:
            msg = "\nInvalid value: %s" % str(values)
            raise argparse.ArgumentError(self, msg)

        return val

    def __call__(self, parser, namespace, values, option_string=None):
        key = option_string.lstrip('-')


        # When converting numpy arrays, you can not use the type.
        type_conv = self.find_converter(key)

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

        raise ValueError, msg

class argument_parser:
    def __init__(self, def_conf, descr):
        """Create parser for the BCNZ command line options."""

        test_defaults(def_conf, descr)
        self.parser = self._create_parser(def_conf, descr)

        self.def_conf = def_conf
        self.descr = descr

    def _create_parser(self, def_conf, descr):
        """Actual work for creating the parser."""

        parser = argparse.ArgumentParser()
        action = conv_type
        action.def_conf = def_conf

        parser.add_argument('catalog', help='Galaxy catalog.')
        for var, def_val in def_conf.iteritems():
            arg = '-%s' % var
            nargs = '*' if (type(def_val) in comb_types) else '?'
            h = descr[var]

            parser.add_argument(arg, action=conv_type, help=h, nargs=nargs)

        parser.add_argument('-c')

        return parser

    def parse_args(self):
        args = self.parser.parse_args()

#        pdb.set_trace()
        keys = self.def_conf.keys()
        vals = [getattr(args, x) for x in keys]

        # Remove entries set to None.
        is_set = lambda (x,y): not isinstance(y, type(None))
        to_update = dict(filter(is_set, zip(keys,vals)))
        
        conf = self.def_conf.copy()
        conf.update(to_update)

        return conf

class first_parser:
    """Detect which configuration file to use."""

    def __init__(self):
        parser = argparse.ArgumentParser()
        #parser.add_argument('-c', type=str, nargs=1, dest="config")
        parser.add_argument('-c', type=str, dest="config", default="standard")
        self.parser = parser

    def __call__(self):
        args = self.parser.parse_known_args() 
        config_file = args[0].config

        return config_file
