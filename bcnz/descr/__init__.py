#!/usr/bin/env python
# encoding: UTF8

import os
import pdb
import yaml

d = os.path.dirname(__file__)
file_path = os.path.join(d, 'descr.yaml')
standard = yaml.load(file(file_path))['descr']
