#!/usr/bin/env python
# encoding: UTF8
# Author: Martin Eriksen
import os
import pdb
import sys

def root_dir(rm_dir='bin'):
    """Find root directory."""

    # In case the program is only as symlink.
    prog_file = os.path.realpath(sys.argv[0])
    path = os.path.dirname(prog_file)

    if os.path.split(path)[1] == rm_dir:
        path = os.path.join(path, '..')
        path = os.path.normpath(path)

    return path

root = root_dir()
for d in ['config', 'descr', 'lib', 'modes', 'priors','']:
    p = os.path.join(root, d)
    sys.path.append(p)

import config
import descr
import bcnz_div
import bcnz_input
import bcnz_filters
import bcnz_main
import bcnz_zdata

def main():
    watch = bcnz_div.watch()

    # Read the initial configuration..
    conf = bcnz_input.parse_arguments()
    conf = bcnz_div.update_conf(conf)
    bcnz_input.test_config(conf)

    # Estimate the photoz
    zdata = bcnz_zdata.find_zdata(conf)
    zdata = bcnz_filters.filter_and_so()(conf, zdata)
    bcnz_main.wrapper(conf, zdata)

    watch.print_difference()

if __name__ == '__main__':
    main()
