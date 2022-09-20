from __future__ import division
import numpy as np
import scipy as sp
from scipy import special

cimport numpy as np
cimport cython
from libc.math cimport erf, sqrt, fabs, log, pi, exp
from libc.stdlib cimport rand, RAND_MAX

DTYPE = np.float
ctypedef np.float_t DTYPE_t
ctypedef np.int_t DTYPE_int_t

cdef DTYPE_t SQRT2 = sqrt(2)
cdef DTYPE_t LOG10 = log(10)

@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef DTYPE_t sign(DTYPE_t x):
    if x < 0: return -1.
    elif x==0: return 0.
    else: return 1.

@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef DTYPE_t inverse_erf(DTYPE_t x):
    cdef DTYPE_t ax = fabs(x)
    cdef DTYPE_t t = 0
    if ax <= 0.75:
        t = x*x-0.75*0.75
        return x*(-13.0959967422+t*(26.785225760+t*(-9.289057635)))/(-12.0749426297+t*(30.960614529+t*(-17.149977991+t)))
    elif ((ax >= 0.75) & (ax <= 0.9375)):
        t = x*x - 0.9375*0.9375
        return  x*(-.12402565221+t*(1.0688059574+t*(-1.9594556078+t*.4230581357))) / (-.08827697997+t*(.8900743359+t*(-2.1757031196+t)))
    else:
        t=1.0/sqrt(-log(1.0-ax));
        return sign(x)*(.1550470003116/t+1.382719649631+t*(.690969348887+t*(-1.128081391617+t*(.680544246825+t*(-.16444156791)))))/(.155024849822+t*(1.385228141995+t))

@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef DTYPE_t phiinv3(DTYPE_t z):
    return SQRT2 * inverse_erf(2.*z-1.)



@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def integral_MH(np.ndarray[DTYPE_t, ndim=1] delta1,np.ndarray[DTYPE_t, ndim=1] delta2, DTYPE_t z, np.ndarray[DTYPE_t, ndim=1] FUV, int N, np.ndarray[DTYPE_t, ndim=1] prior_mus, np.ndarray[DTYPE_t, ndim=1] prior_sigmas, DTYPE_t proposal_norm, DTYPE_t DM):

    cdef DTYPE_t alpha_0 = 0
    cdef DTYPE_t alpha_1 = 0
    cdef DTYPE_t alpha_2 = 0
    cdef DTYPE_t proposal = 0
    cdef DTYPE_t prior_value = 0
    cdef DTYPE_t I = 0

    for i in xrange(N):
         alpha_0 = rand() / float(RAND_MAX)
         alpha_0 = alpha_0*(delta2[0]-delta1[0]) + delta1[0]
         alpha_1 = rand() / float(RAND_MAX)
         alpha_1 = alpha_1*(delta2[1]-delta1[1]) + delta1[1]
         alpha_2 = rand() / float(RAND_MAX)
         alpha_2 = phiinv3(alpha_2)*prior_sigmas[0] + prior_mus[0]
         while (alpha_2<0) or (alpha_2>delta2[2]):
             alpha_2 = rand() / float(RAND_MAX)
             alpha_2 = phiinv3(alpha_2)*prior_sigmas[0] + prior_mus[0]

         proposal = (1./proposal_norm)*gaussian1d(alpha_2,prior_mus[0],prior_sigmas[0])
         prior_value = prior(alpha_0, alpha_1, alpha_2, z, FUV, DM)
         I = I + prior_value/proposal

    I = I/N
    return I


@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef DTYPE_t prior(DTYPE_t alpha_0, DTYPE_t alpha_1, DTYPE_t alpha_OII, DTYPE_t z, np.ndarray[DTYPE_t, ndim=1] FUV, DTYPE_t DM):
    cdef DTYPE_t log10_alpha_OII = 0
    cdef DTYPE_t MUV = 0
    cdef DTYPE_t prior_OII = 0

    MUV = -2.5*log(alpha_0*FUV[0]+alpha_1*FUV[1]+alpha_OII*FUV[2])/LOG10 - 48.6 - 2.5*log(1+z)/LOG10 - DM
    MUV = 0.7497566 *MUV -33.38391793

    log10_alpha_OII = -2.5*log(alpha_OII)/LOG10 - DM

    prior_OII = gaussian1d(MUV, log10_alpha_OII, 0.32682459)

    return prior_OII



@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef DTYPE_t gaussian1d(DTYPE_t x, DTYPE_t mu, DTYPE_t sigma):
    return (1./(sigma*sqrt(2*pi))) * exp(-0.5*(x-mu)**2/sigma**2)
