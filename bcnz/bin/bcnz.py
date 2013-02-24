#!/usr/bin/env python
# encoding: UTF8
# Author: Martin Eriksen

import os
import pdb
import sys

# To avoid name collisions between the program name.
rm_pathes = lambda x: not (x.endswith('bcnz/bin') or x == '')
sys.path = list(filter(rm_pathes, sys.path))

# To find the module when running from the repository.
dir_path = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, os.path.normpath(dir_path))
import bcnz

def main():
    watch = bcnz.lib.timer.watch()

    myconf = bcnz.lib.parser.parse_arguments()
    task = bcnz.tasks.pzcat_local(myconf)
    task.run()

    print(watch)

if __name__ == '__main__':
    main()
