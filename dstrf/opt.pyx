# optimized statistics functions
#cython: boundscheck=False, wraparound=False
# cython: profile=True

cimport cython
from libc.math cimport sqrt
import numpy as np
cimport numpy as cnp

ctypedef cnp.int8_t INT8
ctypedef cnp.int64_t INT64
ctypedef cnp.float64_t FLOAT64

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


def mm(cnp.ndarray[FLOAT64, ndim=2] a, cnp.ndarray[FLOAT64, ndim=2] b,
       cnp.ndarray[FLOAT64, ndim=2] c):
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


cdef float det3(cnp.ndarray[FLOAT64, ndim=2] a):
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


def solvemat33(cnp.ndarray[FLOAT64, ndim=2] a, cnp.ndarray[FLOAT64, ndim=2] b,
              cnp.ndarray[FLOAT64, ndim=2] c):
    cdef unsigned long i, j
    cdef float D

    D = det3(a)

    # for i in range(3):
    #     c[0, i] = det3(b[:, i], a[:, 1], a[:, 2]) / D
    #     c[1, i] = det3(a[:, 0], b[:, i], a[:, 2]) / D
    #     c[2, i] = det3(a[:, 0], a[:, 1], b[:, i]) / D
    temp = np.empty((3, 3), dtype=float)
    cdef cnp.ndarray[FLOAT64, ndim=2] temp_c = temp

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
    return


def sqrtm33(cnp.ndarray[FLOAT64, ndim=2] M, cnp.ndarray[FLOAT64, ndim=2] N,
            cnp.ndarray[FLOAT64, ndim=2] out1, cnp.ndarray[FLOAT64, ndim=2] out2):
    cdef unsigned long k, i, j
    cdef unsigned long iter = 100
    cdef unsigned long ndim = 3

    cdef double check, tol
    tol = 1e-10

    for k in range(iter):
        solvemat33(M, N, out1)
        mm(N, out1, out2)
        check = 0
        for i in range(ndim):
            for j in range(ndim):
                N[i, j] = - out2[i, j]
                M[i, j] += 2 * N[i, j]
                check += out2[i, j] ** 2
        if check < tol:
            break
    return