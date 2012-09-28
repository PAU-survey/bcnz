#!/usr/bin/env python
import os
import pdb
import time

def add_header(conf, output, out_name, sxhdr):
    """Add header to BPZ output file."""

    time_stamp=time.ctime(time.time())
    output.write('## File '+out_name+'  '+time_stamp+'\n')


    output.write("""##
##Parameters used to run BPZ:
##
""")


    claves = conf.keys()
    claves.sort()
    for key in claves:
        if type(conf[key])==type((1,)):
            cosa=join(list(conf[key]),',')
        else:
            cosa=str(conf[key])

        output.write('##'+key.upper()+'='+cosa+'\n')



    output.write(sxhdr+'##\n')

def find_something(conf, zdata):
#Use a empirical prior?
    tipo_prior=conf['prior']
    col_pars = zdata['col_pars']
    useprior=0
    
    if 'M_0' in col_pars: has_mags=1
    else: has_mags=0
    if has_mags and tipo_prior<>'none' and tipo_prior<>'flat': useprior=1
    

    # Output format
    #format='%'+`maximum(5,len(ids[0]))`+'s' #ID format
    format='%'+'5'+'s' #ID format
    format=format+conf['n_peaks']*' %.3f %.3f  %.3f %.3f %.5f'+' %.3f %.3f %10.3f'
    
    #Add header with variable names to the output file
    sxhdr="""##
    ##Column information
    ##
    # 1 ID"""
    k=1
    
    if conf['n_peaks']>1:
        for j in range(conf['n_peaks']):
            sxhdr+="""
    # %i Z_B_%i
    # %i Z_B_MIN_%i
    # %i Z_B_MAX_%i
    # %i T_B_%i
    # %i ODDS_%i""" % (k+1,j+1,k+2,j+1,k+3,j+1,k+4,j+1,k+5,j+1)
            k+=5
    else:
        sxhdr+="""
    # %i Z_B
    # %i Z_B_MIN
    # %i Z_B_MAX
    # %i T_B
    # %i ODDS""" % (k+1,k+2,k+3,k+4,k+5)
        k+=5
    
    sxhdr+="""
    # %i Z_ML
    # %i T_ML
    # %i CHI-SQUARED\n""" % (k+1,k+2,k+3)
    
    nh=k+4
    if 'Z_S' in col_pars: #.d.has_key('Z_S'):
        sxhdr=sxhdr+'# %i Z_S\n' % nh
        format=format+'  %.3f'
        nh+=1
    if has_mags:
        format=format+'  %.3f'
        sxhdr=sxhdr+'# %i M_0\n' % nh
        nh+=1
    if 'OTHER' in col_pars: #.d.has_key('OTHER'):
        sxhdr=sxhdr+'# %i OTHER\n' % nh
        format=format+' %s'
        nh+=n_other

    return format,sxhdr,has_mags
