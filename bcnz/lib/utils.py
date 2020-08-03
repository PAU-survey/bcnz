#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import pandas as pd
from tqdm import tqdm

def flatten(df):
    """Flattens the input dataframe."""

    flux = df.flux.rename(columns=lambda x: f'flux_{x}')
    flux_error = df.flux_error.rename(columns=lambda x: f'flux_error_{x}')

    comb = pd.concat([flux, flux_error], 1)
    
    return comb

def load_models(job, bands):
    """Load the models corresponding to a photo-z job."""
    
    modelD = {}
    for key,dep in tqdm(job.depend.items()):
        part = dep.model.result.to_xarray().flux.squeeze()
        part = part.sel(band=bands).transpose('z', 'band', 'sed')

#        # Testing setting the OIII emission lines to zero above some redshift.
#        part.loc[0.7 < part.z, :, 'OIII'] = 1e-3 #0.

        part = part.rename(sed='model')
        nr = int(key.replace('pzcat_', ''))
        
        modelD[nr] = part
        
    return modelD
