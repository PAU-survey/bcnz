#!/usr/bin/env python
# encoding: UTF8

import os
import pdb
import shutil

import bcnz.lib

class filebase(object):
    def setup(self):
        obj_name = bcnz.lib.obj_hash.hash_structure(self.conf)

        if self.conf['cache_dir']:
            self.cache_tdir = os.path.join(self.conf['cache_dir'],\
                                          'pzcat')
        else:
            root_dir = self.conf['root']
            self.cache_dir = os.path.join(root_dir, 'cache')

        cache_dir = self.conf['cache_dir'] if 'cache_dir' in self.conf \
                    else os.path.join(self.conf['root'], 'cache')

        self._obj_path_peaks = os.path.join(cache_dir, 'pzcat', obj_name)
        self._obj_path_pdfs = os.path.join(cache_dir, 'pzpdf', obj_name)

    def run_file(self):
        assert False, 'Is this actually in use????'

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

    @property
    def out_path(self):
        return self.obj_path if self.conf['use_cache'] else \
               self.out_name
