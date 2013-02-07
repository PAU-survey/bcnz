#!/usr/bin/env python
import pdb
import numpy as np

def interp(conf, f_mod, z, filters, spectra):
    """Interpolation between spectras."""

    ninterp = conf['interp']
    if not ninterp:
        return f_mod

    nz = len(z)
    nt = len(spectra)
    nf = len(filters)

    # Index of the first type in the linear interpolation
    ftype = np.repeat(np.arange(nt-1), ninterp+1)
    btype = np.array(list(ftype)+[nt-1])

    f_new = f_mod[:,btype,:]
    df = f_mod[:,1:,:] - f_mod[:,:-1,:]
    df_part = df[:,ftype,:] 

    # Weights for each of the interpolation points. 
    w = np.tile(np.arange(ninterp+1), nt-1)/(ninterp+1.)

    for i, wi in enumerate(w):
        f_new[:,i,:] += wi*df_part[:,i,:]

    return f_new
