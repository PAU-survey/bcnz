#!/usr/bin/env python
# encoding: UTF8
import pdb

import os
import yaml

def comb(A,B):
    assert set(A.keys()) == set(B.keys())
    for key in A.keys():
        assert A[key] == B[key], key

# Horrible code. In the process of converting to YAML.
d = '/Users/marberi/photoz/bcnz/bcnz/config'
conf = {}
for pop_name in ['standard', 'bright', 'faint']:
    file_path = os.path.join(d, '{}.yaml'.format(pop_name))
    conf[pop_name] = yaml.load(open(file_path))[pop_name]

for pop_name in ['bright', 'faint']:
    pop = conf['standard'].copy()
    pop.update(conf[pop_name])

    conf[pop_name] = pop
