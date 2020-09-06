#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger
import os
import glob
import numpy as np
import pandas as pd
from IPython.core import debugger as ipdb

def extinction_laigle():
    """The extinction files used in the Laigle paper."""

    d = '~/data/photoz/ext_laws'
    g = os.path.join(os.path.expanduser(d), '*.csv')
       
    df = pd.DataFrame() 
    for path in glob.glob(g):
        part = pd.read_csv(path, names=['lmb', 'k'], comment='#') #, sep='\s+')
        part['ext_law'] = os.path.basename(path).replace('.csv','')

        df = df.append(part, ignore_index=True)

    assert len(df), 'No extinction curves found'

    return df
