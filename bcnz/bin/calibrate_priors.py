#!/usr/bin/env python
# encoding: UTF8

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

setup_path()
pdb.set_trace()
import bcnz

bcnz.tasks.priors()
