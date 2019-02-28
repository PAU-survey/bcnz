#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
from pathlib import Path

import numpy as np
import pandas as pd

cfg = [['NUV', 'DNUV', 'galex2500_nuv'],
       ['U', 'DU', 'u_cfht'],
       ['B', 'DB', 'B_Subaru'],
       ['V', 'DV', 'V_Subaru'],
       ['R', 'DR', 'r_Subaru'],
       ['I', 'DI', 'i_Subaru'],
       ['ZN', 'DZN', 'suprime_FDCCD_z'],
       ['YHSC', 'DYHSC', 'yHSC'],
       ['Y', 'DY', 'Y_uv'],
       ['J', 'DJ', 'J_uv'],
       ['H', 'DH', 'H_uv'],
       ['K', 'DK', 'K_uv'],
       ['KW', 'DKW', 'wircam_Ks'],
       ['HW', 'DHW', 'wircam_H'],
       ['IA427', 'DIA427', 'IA427.SuprimeCam'],
       ['IA464', 'DIA464', 'IA464.SuprimeCam'],
       ['IA484', 'DIA484', 'IA484.SuprimeCam'],
       ['IA505', 'DIA505', 'IA505.SuprimeCam'],
       ['IA527', 'DIA527', 'IA527.SuprimeCam'],
       ['IA574', 'DIA574', 'IA574.SuprimeCam'],
       ['IA624', 'DIA624', 'IA624.SuprimeCam'],
       ['IA679', 'DIA679', 'IA679.SuprimeCam'],
       ['IA709', 'DIA709', 'IA709.SuprimeCam'],
       ['IA738', 'DIA738', 'IA738.SuprimeCam'],
       ['IA767', 'DIA767', 'IA767.SuprimeCam'],
       ['IA827', 'DIA827', 'IA827.SuprimeCam'],
       ['NB711', 'DNB711', 'NB711.SuprimeCam'],
       ['NB816', 'DNB816', 'NB816.SuprimeCam'],
       ['CH1', 'DCH1', 'irac_ch1'],
       ['CH2', 'DCH2', 'irac_ch2'],
       ['CH3', 'DCH3', 'irac_ch3'],
       ['CH4', 'DCH4', 'irac_ch4']]

filter_dir = Path('/home/eriksen/data/photoz/COSMOS2015_filters')

class cosmos_filters:
    """Combine all the filters coming from COSMOS."""

    version = 1.0
    config = {}

    def entry(self):
        L = []
        for band, _, fname in cfg:
            path = filter_dir / (fname+'.res')
            suf = '.res' if not ('SuprimeCam' in str(fname) or band == 'YHSC') else '.pb'    
            path = filter_dir / (fname + suf)

            lmb, y = np.loadtxt(path).T
            part = pd.DataFrame({'band': band, 'lmb': lmb, 'reponse': y})
            L.append(part)

        df = pd.concat(L, ignore_index='True')
        
        return df

    def run(self):
        df = self.entry()
        self.output.result = df
