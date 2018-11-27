# optimized statistics functions
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


def _compute_gamma_i_new(FLOAT64[:, :]z, FLOAT64[:, :]x):
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


cdef float det3(FLOAT64[:, :] a):
    # return (a[0] * (b[1] * c[2] - b[2] * c[1]) -
    #         b[0] * (a[1] * c[2] - a[2] * c[1]) +
    #         c[0] * (a[1] * b[2] - a[2] * b[1]))
    return (a[0, 0] * (a[1, 1] * a[2, 2] - a[2, 1] * a[1, 2]) -
            a[0, 1] * (a[1, 0] * a[2, 2] - a[2, 0] * a[1, 2]) +
            a[0, 2] * (a[1, 0] * a[2, 1] - a[2, 0] * a[1, 1]))


# def solvemat3(cnp.ndarray[FLOAT64, ndim=2] a, cnp.ndarray[FLOAT64, ndim=1] b,
#               cnp.ndarray[FLOAT64, ndim=1] c):
#     cdef float D
#     D = det3(a[:, 0], a[:, 1], a[:, 2])
#     c[0] = det3(b, a[:, 1], a[:, 2]) / D
#     c[1] = det3(a[:, 0], b, a[:, 2]) / D
#     c[2] = det3(a[:, 0], a[:, 1], b) / D
#
#     return


# @cython.cdivision(True)
cdef int solvemat33(FLOAT64[:, :] a, FLOAT64[:,:] b,
                    FLOAT64[:,:] c, FLOAT64[:,:] temp_c):
    cdef unsigned long i, j
    cdef float D

    D = det3(a)

    # for i in range(3):
    #     c[0, i] = det3(b[:, i], a[:, 1], a[:, 2]) / D
    #     c[1, i] = det3(a[:, 0], b[:, i], a[:, 2]) / D
    #     c[2, i] = det3(a[:, 0], a[:, 1], b[:, i]) / D
    # temp = np.empty((3, 3), dtype=float)
    # cdef FLOAT64[:,:] temp_c = temp

    # First column
    for i in range(3):
        for j in range(3):
            if j == 0:
                temp_c[i, j] = b[i, 0]
            else:
                temp_c[i, j] = a[i, j]
    c[0, 0] = det3(temp_c) / D
    for i in range(3):
        for j in range(3):
            if j == 1:
                temp_c[i, j] = b[i, 0]
            else:
                temp_c[i, j] = a[i, j]
    c[1, 0] = det3(temp_c) / D
    for i in range(3):
        for j in range(3):
            if j == 2:
                temp_c[i, j] = b[i, 0]
            else:
                temp_c[i, j] = a[i, j]
    c[2, 0] = det3(temp_c) / D
    c[0, 1] = c[1, 0]
    c[0, 2] = c[2, 0]

    # Second column
    for i in range(3):
        for j in range(3):
            if j == 0:
                temp_c[i, j] = b[i, 1]
            else:
                temp_c[i, j] = a[i, j]
    c[0, 1] = det3(temp_c) / D
    for i in range(3):
        for j in range(3):
            if j == 1:
                temp_c[i, j] = b[i, 1]
            else:
                temp_c[i, j] = a[i, j]
    c[1, 1] = det3(temp_c) / D
    for i in range(3):
        for j in range(3):
            if j == 2:
                temp_c[i, j] = b[i, 1]
            else:
                temp_c[i, j] = a[i, j]
    c[2, 1] = det3(temp_c) / D
    c[1, 2] = c[2, 1]

    # Third column
    for i in range(3):
        for j in range(3):
            if j == 0:
                temp_c[i, j] = b[i, 2]
            else:
                temp_c[i, j] = a[i, j]
    c[0, 2] = det3(temp_c) / D
    for i in range(3):
        for j in range(3):
            if j == 1:
                temp_c[i, j] = b[i, 2]
            else:
                temp_c[i, j] = a[i, j]
    c[1, 2] = det3(temp_c) / D
    for i in range(3):
        for j in range(3):
            if j == 2:
                temp_c[i, j] = b[i, 2]
            else:
                temp_c[i, j] = a[i, j]
    c[2, 2] = det3(temp_c) / D
    return 0


def solvemat33py(FLOAT64[:, :] a, FLOAT64[:,:] b,
                 FLOAT64[:,:] c, FLOAT64[:,:] temp_c):
    solvemat33(a, b, c, temp_c)
    return


cdef void sqrtm33c(FLOAT64[:, :] M, FLOAT64[:, :] N,
                   FLOAT64[:, :] out1, FLOAT64[:, :] out2):
    cdef unsigned long k, i, j
    cdef unsigned long iter = 10000
    cdef unsigned long ndim = 3

    cdef double check, tol
    tol = 1e-30

    for k in range(iter):
        solvemat33(M, N, out1, out2)
        mm(N, out1, out2)
        check = 0
        for i in range(ndim):
            for j in range(ndim):
                N[i, j] = - out2[i, j]
                M[i, j] -= (out2[i, j] + out2[j, i])
                check += out2[i, j] ** 2
        if check < tol:
            break
    return


def sqrtm33(FLOAT64[:, :] M, FLOAT64[:, :] N,
            FLOAT64[:, :] out1, FLOAT64[:, :] out2):
    sqrtm33c(M, N, out1, out2)
    return


def compute_gamma_c(FLOAT64[:, :] a, FLOAT64[:, :] z, FLOAT64[:, :] M):
    carr1 = np.empty((I, J), dtype=np.float64)
    # cdef double carr1[I][J]
    cdef FLOAT64[:, :] N = carr1

    carr2 = np.empty((I, J), dtype=np.float64)
    # cdef double carr2[I, J]
    cdef FLOAT64[:, :] out1 = carr2

    carr3 = np.empty((I, J), dtype=np.float64)
    # cdef double carr3[I, J]
    cdef FLOAT64[:, :] out2 = carr3

    carr4 = np.empty((I, J), dtype=np.float64)
    # cdef double carr3[I, J]
    cdef FLOAT64[:, :] N0 = carr4

    carr5 = np.empty((I, J), dtype=np.float64)
    # cdef double carr3[I, J]
    cdef FLOAT64[:, :] O = carr5

    carr6 = np.empty((I, J), dtype=np.float64)
    # cdef double carr3[I, J]
    cdef FLOAT64[:, :] out3 = carr6

    for i in range(I):
        for j in range(J):
            if i == j:
                M[i, j] = 1.0
            else:
                M[i, j] = 0.0
    solvemat33(z, M, out1, out2)
    for i in range(I):
        for j in range(J):
            N[i, j] = out1[i, j] - a[i, j]
            M[i, j] = 2 * (out1[i, j] + a[i, j])
            N0[i, j] = N[i, j]
            O[i, j] = M[i, j]

    are33c(M, N, O, out1, out2, out3)
    # solvemat33(O, N0, M, out3)
    #
    # for i in range(I):
    #     for j in range(J):
    #         if i == j:
    #             out1[i, j] = -M[i,j] + 1
    #             out2[i, j] = -M[i,j] - 1
    #         else:
    #             out1[i, j] = -M[i,j]
    #             out2[i, j] = -M[i,j]
    # solvemat33(out2.T, out1, N, out3)
    # solvemat33(z, N.T, M, out3)
    return carr5


cdef void are33c(FLOAT64[:, :] M, FLOAT64[:, :] N, FLOAT64[:, :] O,
                   FLOAT64[:, :] out1, FLOAT64[:, :] out2, FLOAT64[:, :] out3):
    cdef unsigned long k, i, j
    cdef unsigned long iter = 10000
    cdef unsigned long ndim = 3

    cdef double check, tol
    tol = 1e-30

    for k in range(iter):
        solvemat33(M, N, out1, out2)
        mm(N, out1, out2)
        mm(N.T, out1, out3)
        check = 0
        for i in range(ndim):
            for j in range(ndim):
                N[i, j] = - out2[i, j]
                M[i, j] -= (out3[i, j] + out3[j, i])
                O[i, j] -= out3[j, i]
                check += out2[i, j] ** 2
        if check < tol:
            break
    return