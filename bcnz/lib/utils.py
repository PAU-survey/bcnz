#!/usr/bin/env python
# encoding: UTF8

import pandas as pd

def flatten(df):
    flux = df.flux.rename(columns=lambda x: f'flux_{x}')
    flux_err = df.flux_err.rename(columns=lambda x: f'flux_err_{x}')

    comb = pd.concat([flux, flux_err], 1)
    
    return comb
