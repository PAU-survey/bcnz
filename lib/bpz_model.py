import os
import string
import sys
import numpy as np

import bpz_min_tools
import bpz_useful

def gen_model(conf, zdata):

    z = zdata['z']
    filters = zdata['filters']
    spectra = zdata['spectra']
    ab_db = zdata['ab_db']
    sed_db = zdata['sed_db']

    nf=len(filters)
    nt=len(spectra)
    nz=len(z)

    #Get the model fluxes
    f_mod = np.zeros((nz,nt,nf))*0.
    abfiles=[]
    
    for it in range(nt):
        for jf in range(nf):
            if filters[jf][-4:]=='.res':
                filtro = filters[jf][:-4]
            else:
                filtro = filters[jf]


            model = string.join([spectra[it],filtro,'AB'],'.')
            model_path = os.path.join(conf['ab_dir'],model)
            abfiles.append(model)
        #Generate new ABflux files if not present
        # or if new_ab flag on
            if conf['new_ab'] or model[:-3] not in ab_db:
                if spectra[it] not in sed_db:
                    x = spectra[it]
                    print('SED %s is not in the database %s' % (x, sed_dir)) # BCNZ
    #                import pdb; pdb.set_trace()
    
                    sys.exit(1)
                #print spectra[it],filters[jf]
                print('     Generating ',model,'....')
                bpz_min_tools.ABflux(spectra[it],filtro,madau=conf['madau'])

            zo,f_mod_0 = np.loadtxt(model_path, usecols=[0,1], unpack=True)

            #Rebin the data to the required redshift resolution
            f_mod[:,it,jf] = bpz_useful.match_resol(zo,f_mod_0,z)
            #if sometrue(less(f_mod[:,it,jf],0.)):

            if np.less(f_mod[:,it,jf],0.).any():
                print('Warning: some values of the model AB fluxes are <0')
                print('due to the interpolation ')
                print('Clipping them to f>=0 values')
               
                #To avoid rounding errors in the calculation of the likelihood
                f_mod[:,it,jf] =np.clip(f_mod[:,it,jf],0.,1e300)

    return f_mod
