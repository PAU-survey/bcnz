#!/usr/bin/env python
# encoding: UTF8
import time
import numpy as np

import pdb

def interval(p,x,ci=.99):
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

    i1=np.searchsorted(cp,q1)-1
    i2=np.searchsorted(cp,q2)
    i2=np.minimum(i2,len(p)-1)
    i1=np.maximum(i1,0)

    return x[i1],x[i2]

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
