#!/usr/bin/env python
# encoding: UTF8
# Author: Martin Eriksen

import os
import pdb
import sys

sys.path.append('/Users/marberi/photoz/bcnz')
import bcnz

def parse_arguments():
    """Parse input arguments."""

    # Detects which configuration file to use. This enable
    # b
    first_parser = bcnz.lib.parser.first_parser()
    config_file = first_parser()

    try:
        m = getattr(bcnz.config, config_file)
        def_conf = getattr(m, 'conf')
    except AttributeError:
        raise

    def_descr = bcnz.descr.standard.descr

    arg_parser = bcnz.lib.parser.argument_parser(def_conf, def_descr)
    conf = arg_parser.parse_args()

    return conf

def main():
    watch = bcnz.lib.timer.watch()

    # Read the initial configuration..
    myconf = parse_arguments()
    conf = bcnz.libconf(myconf)
    
    # Estimate the photoz
    zdata = bcnz.zdata.zdata(conf)
    zdata = bcnz.lib.bcnz_filters.filter_and_so()(conf, zdata)
    bcnz.modes.bcnz_main.wrapper(conf, zdata)

    print(watch)


if __name__ == '__main__':
    main()
