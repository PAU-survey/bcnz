#!/usr/bin/env python
# encoding: UTF8
import time
import numpy as np

def inv_gauss_int(p):
    #Brute force approach. Limited accuracy for >3sigma
    #find something better 
    #DO NOT USE IN LOOPS (very slow)
    """
    Calculates the x sigma value corresponding to p
    p=int_{-x}^{+x} g(x) dx
    """
    if p<0. or p>1.:
        print 'Wrong value for p(',p,')!'
        sys.exit()

    step=.00001
    xn = np.arange(0.,4.+step,step)
    gn=1./np.sqrt(2.*np.pi)*np.exp(-xn**2/2.)
    cgn=np.add.accumulate(gn)*step
    p=p/2.
    ind=np.searchsorted(cgn,p)
    return xn[ind]

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
