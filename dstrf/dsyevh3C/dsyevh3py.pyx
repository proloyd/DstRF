#%# distutils: sources = 'dstrf/dsyevh3C/dsyevh3.c'
# distutils: include_dirs = dstrf/dsyevh3C/

cimport cython
# cimport cdsyevh3
cimport numpy as cnp
from libc.math cimport sqrt

ctypedef cnp.float64_t FLOAT64
cdef Py_ssize_t I = 3
cdef Py_ssize_t J = 3


cdef extern from "dsyevh3.c":
    int dsyevh3(double A[3][3], double Q[3][3], double w[3])

cdef extern from "dsyevq3.c":
    int dsyevq3(double A[3][3], double Q[3][3], double w[3])

cdef extern from "dsytrd3.c":
    dsytrd3(double A[3][3], double Q[3][3], double d[3], double e[2])

cdef extern from "dsyevc3.c":
    int dsyevc3(double A[3][3], double w[3])


def eig3(FLOAT64[:,:] A, FLOAT64[:,:] Q, FLOAT64[:] w):
    cdef double Ac[3][3]
    cdef double [:, :] Ac_view = Ac
    cdef double Qc[3][3]
    cdef double [:, :] Qc_view = Qc
    cdef double wc[3]
    cdef double [:] wc_view = wc
    cdef int out

    cdef long int i, j

    for i in range(3):
        for j in range(3):
            Ac_view[i, j] = A[i, j]

    out = dsyevh3(Ac, Qc, wc)

    for i in range(3):
        w[i] = wc_view[i]
        for j in range(J):
            Q[i, j] = Qc_view[i, j]

    return out


cdef int mm(double a[3][3], double b[3][3],
            double c[3][3]):
    cdef int i, j, k
    cdef double sum

    for i in range(3):
        for j in range(3):
            sum = 0
            for k in range(3):
                sum += a[i][k] * b[k][j]
            c[i][j] = sum

    return 0


cdef int mmt(double a[3][3], double b[3][3],
             double c[3][3]):
    cdef int i, j, k
    cdef double sum

    for i in range(3):
        for j in range(3):
            sum = 0
            for k in range(3):
                sum += a[i][k] * b[j][k]
            c[i][j] = sum

    return 0


cdef int mtm(double a[3][3], double b[3][3],
            double c[3][3]):
    cdef int i, j, k
    cdef double sum

    for i in range(3):
        for j in range(3):
            sum = 0
            for k in range(3):
                sum += a[k][i] * b[k][j]
            c[i][j] = sum

    return 0

@cython.cdivision(True)
def compute_gamma_c(FLOAT64[:, :] zpy, FLOAT64[:, :] xpy, FLOAT64[:, :] gamma):
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

    # Data copy to memory view
    cdef double z[3][3]
    cdef double [:, :] zc = z
    cdef double x[3][3]
    cdef double [:, :] xc = x

    for i in range(3):
        for j in range(3):
            xc[i, j] = xpy[i, j]
            zc[i, j] = zpy[i, j]

    cdef double e[3]
    cdef double v[3][3]
    cdef double d[3]
    cdef double u[3][3]
    cdef double temp[3][3]

    dsyevh3(z, v, e)

    for i in range(3):
        if e[i] < 0:
            e[i] = 0
        else:
            e[i] = sqrt(e[i])

    mm(x, v, temp)
    mtm(v, temp, x)

    for i in range(3):
        for j in range(3):
            temp[i][j] = e[i] * x[i][j] * e[j]

    dsyevh3(temp, u, d)

    for i in range(3):
        if d[i] < 0:
            d[i] = 0
        else:
            d[i] = sqrt(d[i])

    for j in range(3):
        for i in range(3):
            if e[j] <= 0:
                v[i][j] = 0
            else:
                v[i][j] = v[i][j] / e[j]

    mm(v, u, temp)

    for j in range(3):
        for i in range(3):
            v[i][j] = temp[i][j] * d[j]

    mmt(v, temp, x)

    for i in range(3):
        for j in range(3):
            gamma[i, j] = x[i][j]

    return