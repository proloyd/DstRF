# optimized statistics functions
#cython: boundscheck=False, wraparound=False

cimport cython
from libc.math cimport sqrt
import numpy as np
cimport numpy as cnp

ctypedef cnp.int8_t INT8
ctypedef cnp.int64_t INT64
ctypedef cnp.float64_t FLOAT64

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cproxg_group(cnp.ndarray[FLOAT64, ndim=2] y,
                 double mu, int n_dims,
                 cnp.ndarray[FLOAT64, ndim=2] out):
    cdef unsigned long i, j  #, v
    cdef double norm, mul

    cdef unsigned long n_voxels = y.shape[0]
    cdef unsigned long n_times = y.shape[1]

    for j in range(n_times):
        for i in range(0, n_voxels, n_dims):
            norm = y[i , j] ** 2 + y[i + 1, j] ** 2 + y[i + 2, j] ** 2
            # for v in range(n_dims):
            #     norm += y[i + v, j] ** 2
            norm = sqrt(norm)

            if norm > mu:
                mul = 1 - mu / norm
            else:
                mul = 0

            # for v in range(n_dims):
            #     out[i + v, j] = mul * y[i + v, j]
            out[i, j] = mul * y[i, j]
            out[i + 1, j] = mul * y[i + 1, j]
            out[i + 2, j] = mul * y[i + 2, j]

    return out


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def mm(cnp.ndarray[FLOAT64, ndim=2] a, cnp.ndarray[FLOAT64, ndim=2] b,
       cnp.ndarray[FLOAT64, ndim=2] c):
    cdef unsigned long i, j, k
    cdef double sum

    cdef unsigned long m = a.shape[0]
    cdef unsigned long n = a.shape[1]
    cdef unsigned long p = b.shape[1]
    cdef unsigned long t = 100

    for i in range(m):
        for j in range(p):
            sum = 0
            for k in range(n):
                sum += a[i, k] * b[k, j]
            c[i, j] = sum

    return 0