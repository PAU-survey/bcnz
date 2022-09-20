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
cdef DTYPE_t phi3(DTYPE_t y):
    return 0.5 * (erf(y/SQRT2) + 1.)


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
cdef DTYPE_t gaussian1d(DTYPE_t x, DTYPE_t x0, DTYPE_t sigma):
    return (1./(sigma*sqrt(2*pi))) * exp(-0.5*(x-x0)**2/sigma**2)


@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def get_posterior_2lims_with_prior_OII_new(np.ndarray[DTYPE_t, ndim=2] a, np.ndarray[DTYPE_t, ndim=2] b, np.ndarray[DTYPE_t, ndim=3] C,int Nmax, np.ndarray[DTYPE_t, ndim=2] mu, np.ndarray[DTYPE_t, ndim=1] muv_model_0 , np.ndarray[DTYPE_t, ndim=1] z_array, np.ndarray[DTYPE_t, ndim=1] DM):
    cdef DTYPE_int_t iz = 0
    cdef DTYPE_int_t k = 0
    cdef DTYPE_int_t kk = 0
    cdef DTYPE_int_t Nmodels = a.shape[1]
    cdef DTYPE_int_t nz = a.shape[0]
    cdef DTYPE_int_t N = 0
    cdef DTYPE_int_t sign = 0
    cdef DTYPE_t z = 0
    cdef DTYPE_t f = 1
    cdef DTYPE_t delta = 0
    cdef DTYPE_t value = 0
    cdef DTYPE_t asum = 0
    cdef DTYPE_t idW = 0
    cdef DTYPE_t g = 0
    cdef DTYPE_t muv_model_0_sum = 0
    cdef DTYPE_t MUV = 0
    cdef DTYPE_t model_FOII = 0
    cdef DTYPE_t prior_OII = 0
    cdef DTYPE_t log_FOII = 0
    cdef np.ndarray[DTYPE_t, ndim=1] d = np.empty(Nmodels, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] e = np.empty(Nmodels, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] y = np.empty(Nmodels, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] theta = np.empty(Nmodels, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] post = np.empty(nz, dtype=DTYPE)

    for iz in xrange(nz):
        value = 0
        d[0] = phi3(min(max(a[iz,0]/C[iz,0,0], -5), 5))
        e[0] = phi3(min(max(b[iz,0]/C[iz,0,0], -5), 5))
        for N in xrange(1,Nmax+1):
            f = e[0] - d[0]
            for k in xrange(1,Nmodels):
                idW = rand() / float(RAND_MAX)
                z = d[k-1]+idW*(e[k-1]-d[k-1])
                y[k-1] = phiinv3(z)
                asum = 0
                for kk in xrange(k):
                    asum = asum + C[iz,k,kk]*y[kk]
                d[k] = phi3(min(max((a[iz,k]-asum)/C[iz,k,k], -5), 5))
                e[k] = phi3(min(max((b[iz,k]-asum)/C[iz,k,k], -5), 5))
                f = f * (e[k] - d[k])

            idW = rand() / float(RAND_MAX)
            z = d[k]+idW*(e[k]-d[k])
            y[k] = phiinv3(z)

            warning = 0
            for k in xrange(Nmodels):
                theta[k] = 0
                for kk in xrange(Nmodels):
                   theta[k] = theta[k] + C[iz,k,kk]*y[kk]
                theta[k] = theta[k] + mu[iz,k]
                if theta[k]<0: warning = 1

            muv_model_0_sum = 0
            for k in xrange(Nmodels):
                muv_model_0_sum = muv_model_0_sum + muv_model_0[k]*theta[k]
            MUV = -2.5*log(muv_model_0_sum)/LOG10 - 48.6 - 2.5*log(1+z_array[iz])/LOG10 - DM[iz]

            MUV = 0.7497566 * MUV -33.38391793

            log_FOII = -2.5*log(theta[2])/LOG10 - DM[iz]

            prior_OII = gaussian1d(MUV, log_FOII, 0.32682459)

            g = prior_OII if warning==0 else 0
            f = f * g
            delta = (f-value)/(N+1)
            value = value + delta

        post[iz] = value

    return post


@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def get_posterior_2lims_with_prior_OII_new_specalibration(np.ndarray[DTYPE_t, ndim=2] a, np.ndarray[DTYPE_t, ndim=2] b, np.ndarray[DTYPE_t, ndim=3] C,int Nmax, np.ndarray[DTYPE_t, ndim=2] mu, np.ndarray[DTYPE_t, ndim=1] muv_model_0 , np.ndarray[DTYPE_t, ndim=1] z_array, np.ndarray[DTYPE_t, ndim=1] DM):
    cdef DTYPE_int_t iz = 0
    cdef DTYPE_int_t k = 0
    cdef DTYPE_int_t kk = 0
    cdef DTYPE_int_t Nmodels = a.shape[1]
    cdef DTYPE_int_t nz = a.shape[0]
    cdef DTYPE_int_t N = 0
    cdef DTYPE_int_t sign = 0
    cdef DTYPE_t z = 0
    cdef DTYPE_t f = 1
    cdef DTYPE_t delta = 0
    cdef DTYPE_t value = 0
    cdef DTYPE_t asum = 0
    cdef DTYPE_t idW = 0
    cdef DTYPE_t g = 0
    cdef DTYPE_t muv_model_0_sum = 0
    cdef DTYPE_t MUV = 0
    cdef DTYPE_t model_FOII = 0
    cdef DTYPE_t prior_OII = 0
    cdef DTYPE_t log_FOII = 0
    cdef np.ndarray[DTYPE_t, ndim=1] d = np.empty(Nmodels, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] e = np.empty(Nmodels, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] y = np.empty(Nmodels, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] theta = np.empty(Nmodels, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] post = np.empty(nz, dtype=DTYPE)

    for iz in xrange(nz):
        value = 0
        d[0] = phi3(min(max(a[iz,0]/C[iz,0,0], -5), 5))
        e[0] = phi3(min(max(b[iz,0]/C[iz,0,0], -5), 5))
        for N in xrange(1,Nmax+1):
            f = e[0] - d[0]
            for k in xrange(1,Nmodels):
                idW = rand() / float(RAND_MAX)
                z = d[k-1]+idW*(e[k-1]-d[k-1])
                y[k-1] = phiinv3(z)
                asum = 0
                for kk in xrange(k):
                    asum = asum + C[iz,k,kk]*y[kk]
                d[k] = phi3(min(max((a[iz,k]-asum)/C[iz,k,k], -5), 5))
                e[k] = phi3(min(max((b[iz,k]-asum)/C[iz,k,k], -5), 5))
                f = f * (e[k] - d[k])

            idW = rand() / float(RAND_MAX)
            z = d[k]+idW*(e[k]-d[k])
            y[k] = phiinv3(z)

            warning = 0
            for k in xrange(Nmodels):
                theta[k] = 0
                for kk in xrange(Nmodels):
                   theta[k] = theta[k] + C[iz,k,kk]*y[kk]
                theta[k] = theta[k] + mu[iz,k]
                if theta[k]<0: warning = 1

            muv_model_0_sum = 0
            for k in xrange(Nmodels):
                muv_model_0_sum = muv_model_0_sum + muv_model_0[k]*theta[k]
            MUV = -2.5*log(muv_model_0_sum)/LOG10 - 48.6 - 2.5*log(1+z_array[iz])/LOG10 - DM[iz]

            MUV = 0.7497566 *MUV -33.38391793

            #######################################
            #### THIS IS CALIBRATION FUNCTION #####
            #######################################

            log_FOII = -2.5*log(theta[2])/LOG10 - DM[iz]

            prior_OII = gaussian1d(MUV, log_FOII, 0.32682459)

            g = prior_OII if warning==0 else 0
            f = f * g
            delta = (f-value)/(N+1)
            value = value + delta

        post[iz] = value

    return post
