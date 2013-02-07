#!/usr/bin/env python
# encoding: UTF8

import os
import pdb
import sys

import bcnz

class conf:
    msg_z = 'zmax <= zmin is not allowed.'
    msg_overwrite = 'Overwriting some input files.'

    def __init__(self, myconf):
        self.data = bcnz.config.standard.conf.copy() # HACK
        self.data.update(myconf)

        self._test_zrange()
        self._test_collisions()
        pdb.set_trace()

    def _test_zrange(self):

        assert self.data['zmin'] < self.data['zmax'], self.msg_z

    def _test_collisions(self):
        """Check for collisions between different input files."""

        file_keys = ['obs_files', 'col_file']
        file_names = [self.data[x] for x in file_keys]

        all_files = []
        for file_name in file_names:
            if isinstance(file_name, list):
                all_files.extend(file_name)
            else:
                all_files.append(file_name)

        assert len(set(all_files)) == len(all_files), self.msg_overwrite


