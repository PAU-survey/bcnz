import os
from numpy import *
import numpy as np
import bpz_useful

"""
clight_AHz=2.99792458e18

zmax_ab=12.
dz_ab=0.01
ab_clip=1e-6

import bcnz_div
fil_dir, sed_dir, ab_dir = bcnz_div.paths()
"""

def interval(p,x,ci=.99):
    """Gives the limits of the confidence interval
       enclosing ci of the total probability
       i1,i2=limits(p,0.99)
    """
    q1=(1.-ci)/2.
    q2=1.-q1
    cp=add.accumulate(p)
    try:
        if cp[-1]<>1.:
            cp=cp/cp[-1]
    except FloatingPointError:
        return 0.,0.

    i1=searchsorted(cp,q1)-1
    i2=searchsorted(cp,q2)
    i2=minimum(i2,len(p)-1)
    i1=maximum(i1,0)

    return x[i1],x[i2]

def odds(p,x,x1,x2):
    """Estimate the fraction of the total probability p(x)
    enclosed by the interval x1,x2"""
    cp=add.accumulate(p)
    i1=searchsorted(x,x1)-1
    i2=searchsorted(x,x2)
    try:
        if i1<0:
            return cp[i2]/cp[-1]
        if i2>len(x)-1:
            return 1.-cp[i1]/cp[-1]
    except FloatingPointError:
        return 0.

    return (cp[i2]-cp[i1])/cp[-1]

def e_mag2frac(errmag):
    """Convert mag error to fractionary flux error"""
    return 10.**(.4*errmag)-1.
