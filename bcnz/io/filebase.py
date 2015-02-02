#!/usr/bin/env python
# encoding: UTF8

import os
import pdb
import shutil

import bcnz.lib

class filebase(object):
# This code was introduced as an experiment, allowing the photo-z code to store 
# many different catalogs by providing unique names. This functionality is by
# now assumed to be provided by an external code.
# 

    def setup(self):
        pass

    @property
    def out_path(self):
        return self.obj_path if self.conf['use_cache'] else \
               self.out_name
