#!/usr/bin/env python
import sys
import numpy as np

import bpz_useful

def interp(conf, f_mod, z, filters, spectra):
    nf=len(filters)
    nt=len(spectra)
    nz=len(z)

    #Here goes the interpolacion between the colors
    ninterp = int(conf['interp'])

    
    ntypes = conf['ntypes']
#    if ntypes == None:
    """
    if not ntypes:
        nt0 = nt
    else:
        nt0 = list(ntypes)
        for i, nt1 in enumerate(nt0):
            print(i, nt1)
            nt0[i] = int(nt1)
        if (len(nt0) <> 3) or (sum(nt0) <> nt):
            print
            print('%d ellipticals + %d spirals + %d ellipticals' % tuple(nt0))
            print('does not add up to %d templates' % nt)
            print('USAGE: -NTYPES nell,nsp,nsb')
            print('nell = # of elliptical templates')
            print('nsp  = # of spiral templates')
            print('nsb  = # of starburst templates')
            print('These must add up to the number of templates in the SPECTRA list')
            print('Quitting BPZ.')
            sys.exit(1)
    """
 
    if ninterp:
        nti = nt+(nt-1)*ninterp
        buffer = np.zeros((nz,nti,nf))*1.
        tipos = np.arange(0.,float(nti),float(ninterp)+1.)
        xtipos = np.arange(float(nti))
        for iz in np.arange(nz):
            for jf in range(nf):
                buffer[iz,:,jf] = bpz_useful.match_resol(tipos,f_mod[iz,:,jf],xtipos)
    
        nt=nti
        f_mod=buffer

    return f_mod
