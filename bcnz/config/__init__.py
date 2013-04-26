#!/usr/bin/env python
# encoding: UTF8

import glob
import os
import pdb
import yaml

def comb(A,B):
    assert set(A.keys()) == set(B.keys())
    for key in A.keys():
        assert A[key] == B[key], key

conf_dir = os.path.dirname(__file__)

conf = {}
for file_path in glob.glob(os.path.join(conf_dir, '*.yaml')):
    loaded_conf = yaml.load(open(file_path))

    overlap = set(conf.keys()) & set(loaded_conf.keys())
    assert not overlap, 'Config: {0} is defined twice.'.format(overlap)

    conf.update(loaded_conf)
