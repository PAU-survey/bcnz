import os
from numpy import *
import numpy as np
import bpz_useful

clight_AHz=2.99792458e18

zmax_ab=12.
dz_ab=0.01
ab_clip=1e-6

import bcnz_div
fil_dir, sed_dir, ab_dir = bcnz_div.paths()

def ascend(x):
    """True if vector x is monotonically ascendent, false otherwise 
       Recommended usage: 
       if not ascend(x): sort(x) 
    """
    return alltrue(greater_equal(x[1:],x[0:-1]))

def etau_madau(wl,z):
    """
    Madau 1995 extinction for a galaxy spectrum at redshift z 
    defined on a wavelenght grid wl
    """

    raise

    n=len(wl)
    l=array([1216.,1026.,973.,950.])
    xe=1.+z
   
    #If all the spectrum is redder than (1+z)*wl_lyman_alfa 
    if wl[0]> l[0]*xe: return zeros(n)+1.
   
    #Madau coefficients
    c=array([3.6e-3,1.7e-3,1.2e-3,9.3e-4])
    ll=912.
    tau=wl*0.
    i1=searchsorted(wl,ll)
    i2=n-1
    #Lyman series absorption
    for i in range(len(l)):
        i2=searchsorted(wl[i1:i2],l[i]*xe)
        tau[i1:i2]=tau[i1:i2]+c[i]*(wl[i1:i2]/l[i])**3.46

    if ll*xe < wl[0]:
        return exp(-tau)

    #Photoelectric absorption
    xe=1.+z
    i2=searchsorted(wl,ll*xe)
    xc=wl[i1:i2]/ll
    xc3=xc**3
    tau[i1:i2]=tau[i1:i2]+\
                (0.25*xc3*(xe**.46-xc**0.46)\
                 +9.4*xc**1.5*(xe**0.18-xc**0.18)\
                 -0.7*xc3*(xc**(-1.32)-xe**(-1.32))\
                 -0.023*(xe**1.68-xc**1.68))

    tau = clip(tau, 0, 700)
    return exp(-tau)

def ABflux(sed,filter,madau):
    """
    Calculates a AB file like the ones used by bpz
    It will set to zero all fluxes
    which are ab_clip times smaller than the maximum flux.
    This eliminates residual flux which gives absurd
    colors at very high-z
    """

    print sed, filter
    ccd='yes'
    units='nu'
    madau=madau
    z_ab=arange(0.,zmax_ab,dz_ab) #zmax_ab and dz_ab are def. in bpz_tools
    
    #Figure out the correct names
    if sed[-4:]<>'.sed':
        sed = "%s.sed" % sed

    sed = os.path.join(sed_dir, sed)

    if filter[-4:]<>'.res':filter=filter+'.res'

    filter = os.path.join(fil_dir, filter)

    #Get the data
    x_sed,y_sed = np.loadtxt(sed, usecols=[0,1], unpack=True)
    nsed=len(x_sed)
    x_res,y_res = np.loadtxt(filter, usecols=[0,1], unpack=True)
    nres=len(x_res)
    
    if not ascend(x_sed):
        print
        print 'Warning!!!'
        print 'The wavelenghts in %s are not properly ordered' % sed
        print 'They should start with the shortest lambda and end with the longest'        
        print 'This will probably crash the program'

    if not ascend(x_res):
        print
        print 'Warning!!!'
        print 'The wavelenghts in %s are not properly ordered' % filter
        print 'They should start with the shortest lambda and end with the longest'
        print 'This will probably crash the program'

    if x_sed[-1]<x_res[-1]: #The SED does not cover the whole filter interval
        print 'Extrapolating the spectrum'
        #Linear extrapolation of the flux using the last 4 points
        #slope=mean(y_sed[-4:]/x_sed[-4:])
        d_extrap=(x_sed[-1]-x_sed[0])/len(x_sed)
        x_extrap=arange(x_sed[-1]+d_extrap,x_res[-1]+d_extrap,d_extrap)
        extrap=lsq(x_sed[-5:],y_sed[-5:])
        y_extrap=extrap.fit(x_extrap)
        y_extrap=clip(y_extrap,0.,max(y_sed[-5:]))
        x_sed=concatenate((x_sed,x_extrap))
        y_sed=concatenate((y_sed,y_extrap))
        #connect(x_sed,y_sed)
        #connect(x_res,y_res)

    #Wavelenght range of interest as a function of z_ab
    wl_1=x_res[0]/(1.+z_ab)
    wl_2=x_res[-1]/(1.+z_ab)
    #print 'wl', wl_1, wl_2
    #print 'x_res', x_res
    print 'x_res[0]', x_res[0]
    print 'x_res[-1]', x_res[-1]
    n1=clip(searchsorted(x_sed,wl_1)-1,0,100000)
    n2=clip(searchsorted(x_sed,wl_2)+1,0,nsed-1)
    
    #Typical delta lambda
    delta_sed=(x_sed[-1]-x_sed[0])/len(x_sed)
    delta_res=(x_res[-1]-x_res[0])/len(x_res)


    #Change resolution of filter
    if delta_res>delta_sed:
        x_r=arange(x_res[0],x_res[-1]+delta_sed,delta_sed)
        print 'Changing filter resolution from %.2f AA to %.2f' % (delta_res,delta_sed)
        r=match_resol(x_res,y_res,x_r)
        r=where(less(r,0.),0.,r) #Transmission must be >=0
    else:
        x_r,r=x_res,y_res

    #Operations necessary for normalization and ccd effects
    if ccd=='yes': r=r*x_r
    norm_r=trapz(r,x_r)
    if units=='nu': const=norm_r/trapz(r/x_r/x_r,x_r)/clight_AHz
    else: const=1.

    const=const/norm_r
    
    nz_ab=len(z_ab)
    f=zeros(nz_ab)*1.
    for i in range(nz_ab):
        i1,i2=n1[i],n2[i]
	#if (x_sed[i1] > max(x_r/(1.+z_ab[i]))) or (x_sed[i2] < min(x_r/(1.+z_ab[i]))):
	if (x_sed[i1] > x_r[-1]/(1.+z_ab[i])) or (x_sed[i2-1] < x_r[0]/(1.+z_ab[i])) or (i2-i1<2):
	    print 'bpz_tools.ABflux:'
	    print "YOUR FILTER RANGE DOESN'T OVERLAP AT ALL WITH THE REDSHIFTED TEMPLATE"
	    print "THIS REDSHIFT IS OFF LIMITS TO YOU:"
	    print 'z = ', z_ab[i]
            print i1, i2
            print x_sed[i1], x_sed[i2]
            print y_sed[i1], y_sed[i2]
            print min(x_r/(1.+z_ab[i])), max(x_r/(1.+z_ab[i]))
	    # NOTE: x_sed[i1:i2] NEEDS TO COVER x_r(1.+z_ab[i])
	    # IF THEY DON'T OVERLAP AT ALL, THE PROGRAM WILL CRASH
	    #sys.exit(1)
	else:
            try:
                ys_z = bpz_useful.match_resol(x_sed[i1:i2],y_sed[i1:i2],x_r/(1.+z_ab[i]))
            except:
                print i1, i2
                print x_sed[i1], x_sed[i2-1]
                print y_sed[i1], y_sed[i2-1]
                print min(x_r/(1.+z_ab[i])), max(x_r/(1.+z_ab[i]))
                print x_r[1]/(1.+z_ab[i]), x_r[-2]/(1.+z_ab[i])
                print x_sed[i1:i2]
                print x_r/(1.+z_ab[i])
                import pdb; pdb.set_trace()

	    if madau: ys_z=etau_madau(x_r,z_ab[i])*ys_z

	    f[i]=trapz(ys_z*r,x_r)*const        

#    import pdb; pdb.set_trace()
    file_name = sed.split('/')[-1][:-4]+'.'+filter.split('/')[-1][:-4]+'.AB'
    ABoutput = os.path.join(ab_dir, file_name)
#    ABoutput=ab_dir+split(sed,'/')[-1][:-4]+'.'+split(filter,'/')[-1][:-4]+'.AB'

    #print "Clipping the AB file"
    #fmax=max(f)
    #f=where(less(f,fmax*ab_clip),0.,f)

    print 'Writing AB file ',ABoutput
#    put_data(ABoutput,(z_ab,f))

    np.savetxt(ABoutput, np.vstack([z_ab,f]).T)

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
