#!/usr/bin/env python
# encoding: UTF8

import ipdb
import os
import glob
import numpy as np
import pandas as pd


class extinction_lagaile:
    """Extinction laws in the Lagaile paper."""

    version = 1.0
    config = {}

    def entry(self):
        d = '/home/eriksen/data/photoz/ext_laws'
        g = os.path.join(d, '*.dat')
       
        df = pd.DataFrame() 
        for path in glob.glob(g):
            part = pd.read_csv(path, names=['lmb', 'k'], sep='\s+')
            part['ext_law'] = os.path.basename(path).replace('.dat','')

            df = df.append(part, ignore_index=True)

        return df

    def run(self):
        self.output.result = self.entry()
