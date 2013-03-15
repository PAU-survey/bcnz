#!/usr/bin/env python
# encoding: UTF8

import os
import pdb
import sys

import bcnz

class conf(dict):
    msg_z = 'zmax <= zmin is not allowed.'
    msg_overwrite = 'Overwriting some input files.'

    def __init__(self, myconf):
        self.update(bcnz.config.conf['standard'].copy()) # HACK
        self.update(myconf)
        self._test_zrange()

    def _test_zrange(self):

        assert self['zmin'] < self['zmax'], self.msg_z
