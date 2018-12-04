# optimized functions
# Author: Proloy Das <proloy@umd.edu>
#cython: boundscheck=False, wraparound=False
# cython: profile=False

cimport cython
from libc.math cimport sqrt
import numpy as np
cimport numpy as cnp
from dsyevh3C import eig3

ctypedef cnp.int8_t INT8
ctypedef cnp.int64_t INT64
ctypedef cnp.float64_t FLOAT64


cdef Py_ssize_t I = 3
cdef Py_ssize_t J = 3

@cython.cdivision(True)
def cproxg_group(cnp.ndarray[FLOAT64, ndim=3] y,
                 double mu, cnp.ndarray[FLOAT64, ndim=3] out):
    cdef unsigned long i, j  #, v
    cdef double norm, mul

    # cdef Py_ssize_t n_dims_c = 1
    cdef Py_ssize_t n_voxels = y.shape[0]
    cdef Py_ssize_t n_times = y.shape[2]

    for j in range(n_times):
        for i in range(n_voxels):
            norm = y[i, 0, j] ** 2 + y[i, 1, j] ** 2 + y[i, 2, j] ** 2
            # for v in range(n_dims):
            #     norm += y[i + v, j] ** 2
            norm = sqrt(norm)

            mul = 1
            if norm > mu:
                mul -= mu / norm
            else:
                mul = 0

            # for v in range(n_dims):
            #     out[i + v, j] = mul * y[i + v, j]
            out[i, 0, j] = mul * y[i, 0, j]
            out[i, 1, j] = mul * y[i, 1, j]
            out[i, 2, j] = mul * y[i, 2, j]

    return out


cdef int mm(FLOAT64[:,:] a, FLOAT64[:,:] b,
            FLOAT64[:,:] c):
    cdef Py_ssize_t i, j, k
    cdef double sum

    cdef Py_ssize_t m = a.shape[0]
    cdef Py_ssize_t n = a.shape[1]
    cdef Py_ssize_t p = b.shape[1]
    cdef unsigned long t = 100

    for i in range(m):
        for j in range(p):
            sum = 0
            for k in range(n):
                sum += a[i, k] * b[k, j]
            c[i, j] = sum

    return 0


def _compute_gamma_i(FLOAT64[:, :]z, FLOAT64[:, :]x):
    """

    Computes Gamma_i = Z**(-1/2) * ( Z**(1/2) X X' Z**(1/2)) ** (1/2) * Z**(-1/2)
                   = V(E)**(-1/2)V' * ( V ((E)**(1/2)V' X X' V(E)**(1/2)) V')** (1/2) * V(E)**(-1/2)V'
                   = V(E)**(-1/2)V' * ( V (UDU') V')** (1/2) * V(E)**(-1/2)V'
                   = V (E)**(-1/2) U (D)**(1/2) U' (E)**(-1/2) V'

    parameters
    ----------
    z: ndarray
        array of shape (dc, dc)
        auxiliary variable,  z_i

    x: ndarray
        array of shape (dc, dc)
        auxiliary variable, x_i

    returns
    -------
    ndarray
    array of shape (dc, dc)

    """
    cdef int i, j

    e = np.empty((3,))
    v = np.empty((3, 3))
    d = np.empty((3,))
    u = np.empty((3, 3))
    eig3(z, v, e)
    e = e.real
    e[e < 0] = 0
    temp = np.empty((3,3))
    mm(x, v, temp)
    mm(v.T, temp, x)
    for i in range(3):
        e[i] = sqrt(e[i])
    for i in range(3):
        for j in range(3):
            temp[i, j] = e[i] * x[i, j] * e[j]
    eig3(temp, u, d)
    d = d.real
    d[d < 0] = 0
    for i in range(3):
        d[i] = sqrt(d[i])

    for j in range(3):
        for i in range(3):
            v[i, j] = v[i, j] / e[j]
    mm(v, u, temp)
    for j in range(3):
        for i in range(3):
            v[i, j] = temp[i, j] * d[j]
    mm(v, temp.T, u)
    return u
