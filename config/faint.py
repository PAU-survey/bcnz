#!/usr/bin/env python
import bcnz_config
import copy

conf = copy.deepcopy(bcnz_config.conf)

conf['interp'] = 2
conf['prior'] = 'pau'
conf['dz'] = 0.005
conf['odds'] = 0.68
conf['min_rms'] = 0.055
#conf['spectra'] = 'spectras.txt'

#from bcnz_con
#-interp 2 -prior pau -dz 0.005 -odds 0.68 -min_rms 0.055 -spectra spectras.txt
