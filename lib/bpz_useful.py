import time
import numpy as np

def view_keys(dict):
    """Prints sorted dictionary keys"""

    claves=dict.keys()
    claves.sort()
    for line in claves:
        print line.upper(),'  =  ',dict[line]

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

