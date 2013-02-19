#!/usr/bin/env python
# encoding: UTF8
import time
import numpy as np

import pdb

def X(p,x,ci=.99):
    """Gives the limits of the confidence interval
       enclosing ci of the total probability
       i1,i2=limits(p,0.99)
    """
    q1=(1.-ci)/2.
    q2=1.-q1
    cp=np.add.accumulate(p)
    try:
        if cp[-1]<>1.:
            cp=cp/cp[-1]
    except FloatingPointError:
        return 0.,0.

    print('ci', ci)
    print('q1', q1, 'q2', q2)
    i1=np.searchsorted(cp,q1)-1
    i2=np.searchsorted(cp,q2)
    i2=np.minimum(i2,len(p)-1)
    i1=np.maximum(i1,0)

    return x[i1],x[i2]

# <bcnz>
def prob_interval(p,x,plim):
    """Limits enclosing probabilities plim. Handles
       stacks of probabilities.
    """

    p = np.vstack([p,p])

    # Upper and lower limits.
    q1 = 0.5*(1-ci)
    q2 = 1. - q1

    cdf = p.cumsum(axis=1)
    j1 = np.apply_along_axis(np.searchsorted,1,cdf,q1)
    j2 = np.apply_along_axis(np.searchsorted,1,cdf,q2)
    j2 = np.minimum(j2, p.shape[1] - 1)

    pdb.set_trace()
    return x[j1], x[j2]

def interval(p,z,ci=.99):
    return X(p,z,ci)
#    B = Y(p,z,ci)

#    pdb.set_trace()
# </bcnz>

def odds(p,x,x1,x2):
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
