#!/usr/bin/env python
# encoding: UTF8
# Author: Martin Eriksen

import os
import pdb
import sys

sys.path.append('/Users/marberi/photoz/bcnz')
import bcnz

def main():
    watch = bcnz.lib.timer.watch()

    # Read the initial configuration..
    myconf = bcnz.lib.parser.parse_arguments()
    task = bcnz.tasks.pzcat(myconf)
    task.run()

    print(watch)


if __name__ == '__main__':
    main()
