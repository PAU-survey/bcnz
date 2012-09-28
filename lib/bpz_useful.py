import time
import numpy as np

def view_keys(dict):
    """Prints sorted dictionary keys"""

    claves=dict.keys()
    claves.sort()
    for line in claves:
        print line.upper(),'  =  ',dict[line]

def match_resol(xg,yg,xf,method="linear"):
    """ 
    Interpolates and/or extrapolate yg, defined on xg, onto the xf coordinate set.
    Options are 'linear' or 'spline' (uses spline.py from Johan Hibscham)
    Usage:
    ygn=match_resol(xg,yg,xf,'spline')
    """

    if method<>"spline":
        if type(xf)==type(1.):
            xf=array([xf])

        ng = len(xg)
        d = (yg[1:]-yg[0:-1])/(xg[1:]-xg[0:-1])

        #Get positions of the new x coordinates
        ind = np.clip(np.searchsorted(xg,xf)-1,0,ng-2)
        ygn = np.take(yg,ind)+np.take(d,ind)*(xf-np.take(xg,ind))
        if len(ygn)==1: ygn=ygn[0]
        return ygn
    else:
        low_slope=(yg[1]-yg[0])/(xg[1]-xg[0])
        high_slope=(yg[-1]-yg[-2])/(xg[-1]-xg[-2])
        sp=Spline(xg,yg,low_slope,high_slope)

    return sp(xf)

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

