#!/usr/bin/env python
# encoding: UTF8
import bcnz_config
import copy

conf = copy.deepcopy(bcnz_config.conf)
conf['interp'] = 2
conf['prior'] = 'pau'
conf['dz'] = 0.001
conf['odds'] = 0.68
conf['min_rms'] = 0.0055
#conf['col_file'] = 'cat_test/mock_bright.columns' # HACK...
#conf['spectra'] = 'spectras.txt'
