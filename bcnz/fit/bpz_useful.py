#!/usr/bin/env python
# encoding: UTF8
import time
import numpy as np

import pdb

def X(p,x,x1,x2):
    """Estimate the fraction of the total probability p(x)
    enclosed by the interval x1,x2"""
    cp=np.add.accumulate(p)
    i1=np.searchsorted(x,x1)-1
    i2=np.searchsorted(x,x2)

    try:
        if i1<0:
            return cp[i2]/cp[-1]
        if i2>len(x)-1:
            return 1.-cp[i1]/cp[-1]
    except FloatingPointError:
        return 0.

    return (cp[i2]-cp[i1])/cp[-1]

# <bcnz>

def find_odds(p,x,xmin,xmax):
    """Probabilities in the given intervals."""

    p = np.vstack([p, p])
    cdf = p.cumsum(axis=1)

    imin = np.searchsorted(x, xmin) - 1
    imax = np.searchsorted(x, xmax)
    gind = np.arange(p.shape[0])

    return (cdf[gind,imax] - cdf[gind,imin])/cdf[:,-1]


    print(cdf)
    print('imin', imin, 'imax', imax)
    pdb.set_trace()
    return (cdf[gind,imax] - cdf[gind,imin])/cdf[:,-1]


def odds(p,x,xmin,xmax):
    A = X(p,x,xmin,xmax)
#    print(A)
#    B = Y(p,x,xmin,xmax)
#    print(B)
#    pdb.set_trace()

    return A

# </bcnz>
