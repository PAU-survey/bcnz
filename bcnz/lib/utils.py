#!/usr/bin/env python
# encoding: UTF8

import pandas as pd
from tqdm import tqdm

def flatten(df):
    """Flattens the input dataframe."""

    flux = df.flux.rename(columns=lambda x: f'flux_{x}')
    flux_err = df.flux_err.rename(columns=lambda x: f'flux_err_{x}')

    comb = pd.concat([flux, flux_err], 1)
    
    return comb

def load_models(job, bands):
    """Load the models corresponding to a photo-z job."""
    
    modelD = {}
    for key,dep in tqdm(job.depend.items()):
        part = dep.model.result.to_xarray().flux.squeeze()
        part = part.sel(band=bands).transpose('z', 'band', 'sed')
        part = part.rename(sed='model')
        nr = int(key.replace('pzcat_', ''))
        
        modelD[nr] = part
        
    return modelD
