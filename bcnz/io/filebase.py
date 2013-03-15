#!/usr/bin/env python
# encoding: UTF8

import os
import pdb
import shutil

class filebase:
    def setup(self):
        obj_name = bcnz.lib.obj_hash.hash_structure(self.conf)

        if self.conf['cache_dir']:
            self.cache_dir = os.path.join(self.conf['cache_dir'],\
                                          'pzcat')
        else:
            root_dir = self.conf['root']
            self.cache_dir = os.path.join(root_dir, 'cache')

        self.obj_path = os.path.join(self.cache_dir, obj_name)

    def relink(self):
        """Link output file name to the object file."""

        if os.path.isfile(self.out_name):
            shutil.move(self.out_name, "%s.bak" % self.out_name)

        if os.path.islink(self.out_name):
            os.remove(self.out_name)

        os.symlink(self.obj_path, self.out_name)

    def run_file(self):
        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)

        if self.conf['use_cache']:
            if os.path.isfile(self.obj_path):
                self.relink()
                return

            self.estimate_photoz(self.obj_path)
            self.relink()
        else:
            self.estimate_photoz(self.out_name)

