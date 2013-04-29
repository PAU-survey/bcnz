#!/usr/bin/env python
# encoding: UTF8
# Author: Martin Eriksen

import os
import pdb
import sys

def setup_path():
    """To avoid name collisions between the program name."""

    # To find the module when running from the repository.
    dir_path = os.path.join(os.path.dirname(__file__), '../..')
    sys.path.insert(0, os.path.normpath(dir_path))


class bcnz_cmd(object):
    """Run bcnz from the command line."""

    def __call__(self):
        watch = bcnz.lib.timer.watch()

        myconf = bcnz.lib.parser.parse_arguments()
        task = bcnz.tasks.pzcat_local(myconf)
        task.run()

        print(watch)

if __name__ == '__main__':
    setup_path()
    import bcnz
    import bcnz.tasks

    cmd = bcnz_cmd()
    cmd()
