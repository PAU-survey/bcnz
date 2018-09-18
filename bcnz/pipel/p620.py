#!/usr/bin/env python
# encoding: UTF8

def has_bb(pipel):
    F = pipel.pzcat.chi2.pzcat_0.config['filters']
    res = True in set(map(lambda x: not x.startswith('NB'), F))

    return res

def free_fit(pipel):
    algo = pipel.pzcat.chi2.pzcat_0.config['chi2_algo']
    res = algo == 'min_free'

    return res

def fmod_adjust(pipel):
    res = pipel.get('fmod_adjust').config['actually_scale']

    return res

def free_calib(pipel):
    res = pipel.get('inter_calib_free').config['free_ampl']
    
    return res

S = pd.Series()
S['has_BB'] = has_bb(pipel)
S['free_fit'] = free_fit(pipel)
S['fmod_adjust'] = fmod_adjust(pipel)
S['free_calib'] = free_calib(pipel)

descr = {'has_bb': 'If including broad band in photo-z',
         'free_fit': 'If having a free amplitude in the photo-z fit',
         'fmod_adjust': 'Correcting the model from a syn r-band',
         'free_calib': 'Free amplitude in the calibration'}
