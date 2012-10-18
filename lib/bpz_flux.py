#!/usr/bin/env python
# encoding: UTF8
import pdb
import sys
import numpy as np
from bpz_min_tools import e_mag2frac

def mega1(conf, zdata, data):

    # I don't really feel like reading the details yet!
    undet = conf['undet']
    unobs = conf['unobs']
 
    f_obs = data['f_obs']
    ef_obs = data['ef_obs']

    #Convert them to arbitrary fluxes if they are in magnitudes
    if conf['mag']:
        seen=np.greater(f_obs,0.)*np.less(f_obs,undet) #*(ef_obs != -1.)
        no_seen=np.equal(f_obs,undet)
        no_observed=np.equal(f_obs, unobs)
#        no_observed=np.equal(ef_obs, -1)
        todo=seen+no_seen+no_observed
        #The minimum photometric error is 0.01
        #ef_obs=ef_obs+seen*np.equal(ef_obs,0.)*0.001
        ef_obs=np.where(np.greater_equal(ef_obs,0.),np.clip(ef_obs,conf['min_magerr'],1e10),ef_obs)
        if np.add.reduce(np.add.reduce(todo))<>todo.shape[0]*todo.shape[1]:
            print('Objects with unexpected magnitudes!')
            print("""Allowed values for magnitudes are 0<m<"""+`undet`+" m="+`undet`+"(non detection), m="+`unobs`+"(not observed)")
            for i in range(len(todo)):
                if not np.alltrue(todo[i,:]):
                    import pdb; pdb.set_trace()
                    print i+1,f_obs[i,:],ef_obs[i,:]

            sys.exit()
     
        #Detected objects
        try:
            f_obs=np.where(seen,10.**(-.4*f_obs),f_obs)
        except OverflowError:
            print 'Some of the input magnitudes have values which are >700 or <-700'
            print 'Purge the input photometric catalog'
            print 'Minimum value',min(f_obs)
            print 'Maximum value',max(f_obs)
            print 'Indexes for minimum values',argmin(f_obs,0.)
            print 'Indexes for maximum values',argmax(f_obs,0.)
            print 'Bye.'
            sys.exit()
    
        try:
            ef_obs=np.where(seen,(10.**(.4*ef_obs)-1.)*f_obs,ef_obs)
        except OverflowError:
            print 'Some of the input magnitude errors have values which are >700 or <-700'
            print 'Purge the input photometric catalog'
            print 'Minimum value',min(ef_obs)
            print 'Maximum value',max(ef_obs)
            print 'Indexes for minimum values',argmin(ef_obs,0.)
            print 'Indexes for maximum values',argmax(ef_obs,0.)
            print 'Bye.'
            sys.exit()
   
        to_high = 700 < ef_obs 
        seen = np.logical_and(seen, np.logical_not(to_high))
        no_seen = np.logical_or(no_seen, to_high)

#        import pdb; pdb.set_trace() 
#        print('x3', np.max(ef_obs))
        #Looked at, but not detected objects (mag=99.)
        #We take the flux equal to zero, and the error in the flux equal to the 1-sigma detection error.
        #If m=99, the corresponding error magnitude column in supposed to be dm=m_1sigma, to avoid errors
        #with the sign we take the absolute value of dm 
        f_obs=np.where(no_seen,0.,f_obs)
        ef_obs=np.where(no_seen,10.**(-.4*abs(ef_obs)),ef_obs)
    
        #Objects not looked at (mag=-99.)
        f_obs=np.where(no_observed,0.,f_obs)
        ef_obs=np.where(no_observed,0.,ef_obs)
    
    return f_obs, ef_obs

def mega2(conf, zdata, f_obs, ef_obs):
    zp_errors = zdata['zp_errors']
    zp_offsets = zdata['zp_offsets']

    #Flux codes:
    # If f>0 and ef>0 : normal objects
    # If f==0 and ef>0 :object not detected
    # If f==0 and ef==0: object not observed
    #Everything else will crash the program
    
    #Check that the observed error fluxes are reasonable
    #if sometrue(less(ef_obs,0.)): raise 'Negative input flux errors'
    if np.less(ef_obs,0.).any(): raise 'Negative input flux errors'
    
    f_obs=np.where(np.less(f_obs,0.),0.,f_obs) #Put non-detections to 0
    ef_obs=np.where(np.less(f_obs,0.),np.maximum(1e-100,f_obs+ef_obs),ef_obs) # Error equivalent to 1 sigma upper limit
    
    #if sometrue(np.less(f_obs,0.)) : raise 'Negative input fluxes'
    seen=np.greater(f_obs,0.)*np.greater(ef_obs,0.)
    no_seen=np.equal(f_obs,0.)*np.greater(ef_obs,0.)
    no_observed=np.equal(f_obs,0.)*np.equal(ef_obs,0.)
    
    todo=seen+no_seen+no_observed
    if np.add.reduce(np.add.reduce(todo))<>todo.shape[0]*todo.shape[1]:
        print 'Objects with unexpected fluxes/errors'
    
    #Convert (internally) objects with zero flux and zero error(non observed)
    #to objects with almost infinite (~1e108) error and still zero flux
    #This will yield reasonable likelihoods (flat ones) for these objects
    ef_obs=np.where(no_observed,1e108,ef_obs)
    
    #Include the zero point errors
    zp_errors=np.array(map(float,zp_errors))
    zp_frac=e_mag2frac(zp_errors)
    #zp_frac=10.**(.4*zp_errors)-1.
#    pdb.set_trace() 
    ef_obs=np.where(seen,np.sqrt(ef_obs*ef_obs+(zp_frac*f_obs)**2),ef_obs)
    ef_obs=np.where(no_seen,np.sqrt(ef_obs*ef_obs+(zp_frac*(ef_obs/2.))**2),ef_obs)
   
    #Add the zero-points offset
    #The offsets are defined as m_new-m_old
    zp_offsets=np.array(map(float,zp_offsets))
    zp_offsets=np.where(np.not_equal(zp_offsets,0.),10.**(-.4*zp_offsets),1.)
    f_obs=f_obs*zp_offsets
    ef_obs=ef_obs*zp_offsets
    
    return f_obs,ef_obs
