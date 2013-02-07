#!/usr/bin/env python
# encoding: UTF8
import bcnz
import copy

import standard
conf = copy.deepcopy(standard.conf)

conf['interp'] = 2
conf['prior'] = 'pau'
conf['dz'] = 0.005
conf['odds'] = 0.68
conf['min_rms'] = 0.055
