#!/usr/bin/env python
# encoding: UTF8

# Calibrate the priors. Possible temporary solution.
import os
import pdb
import sys

def setup_path():
    """To avoid name collisions between the program name."""

    rm_pathes = lambda x: not (x.endswith('bcnz/bin') or x == '')
    sys.path = list(filter(rm_pathes, sys.path))

    # To find the module when running from the repository.
    dir_path = os.path.join(os.path.dirname(__file__), '../..')
    sys.path.insert(0, os.path.abspath(dir_path))

def main():
    setup_path()
    import bcnz
    myconf = bcnz.lib.parser.parse_arguments()

    task = bcnz.tasks.priors(myconf)
    task.run()

if __name__ == '__main__':
    main()
