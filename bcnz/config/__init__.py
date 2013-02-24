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
d = os.path.dirname(__file__)
all_pop = ['bright', 'faint', 'mice', 'des']
conf = {}
for pop_name in all_pop+['standard']:
    file_path = os.path.join(d, '{}.yaml'.format(pop_name))
    conf[pop_name] = yaml.load(open(file_path))[pop_name]

for pop_name in all_pop:
    pop = conf['standard'].copy()
    pop.update(conf[pop_name])

    conf[pop_name] = pop
