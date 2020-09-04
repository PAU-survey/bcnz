#!/usr/bin/env python
# encoding: UTF8

import os
import sys

import numpy as np
import pandas as pd

def seds(input_dir):
    """Load seds from files."""

    input_dir = os.path.expanduser(input_dir) #self.config['input_dir'])
    suf = 'sed'
    min_val = 0
#        suf = self.config['suf']
#        min_val = self.config['min_val']

    df = pd.DataFrame()
    for fname in os.listdir(input_dir):
        if not fname.endswith(suf):
            continue

        path = os.path.join(input_dir, fname)
        lmb,response = np.loadtxt(path).T
        name = fname.replace('.'+suf,'')

        # This is to avoid numerical aritifacts which can
        # be important if the spectrum is steep.
        if min_val:
            y[y < min_val] = 0

        part = pd.DataFrame()
        part['lmb'] = lmb
        part['response'] = response
        part['sed'] = name

        df = df.append(part, ignore_index=True)

    # The SEDs are sometimes defined with duplicate entries that ends
    # up creating technical problems later.
    df = df.drop_duplicates()

    df = df.set_index('sed')

    return df

