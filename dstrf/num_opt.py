import numpy as np
from numpy import linalg
import opt


def msqrt3(A):
    """A needs to be square"""
    N = np.eye(3) - A
    M = 2 * (np.eye(3) + A)
    out1 = np.empty((3, 3))
    out2 = np.empty((3, 3))

    for k in range(100):
        opt.solvemat33(M, N, out1)
        opt.mm(N.T, out1, out2)
        N = - out2
        # N *= -1
        M += 2 * N
        if (N ** 2).sum() < 1e-5:
            break

    return M/4


def msqrt3new(A):
    """A needs to be 3x3"""
    N = np.eye(3) - A
    M = 2 * (np.eye(3) + A)
    out1 = np.empty((3, 3))
    out2 = np.empty((3, 3))

    opt.sqrtm33(M, N, out1, out2)

    return M/4


def compute_gamma(x, z):
    out = np.empty((3, 3))
    x = np.dot(x, x.T, out)
    z_sqrt = msqrt3new(z)
    temp = np.dot(np.dot(z_sqrt, x), z_sqrt)
    temp_sqrt = msqrt3new(temp)
    opt.solvemat33(z_sqrt, temp_sqrt, out)
    opt.solvemat33(z_sqrt.T, out.T, temp)
    # temp = np.linalg.solve(z_sqrt, temp_sqrt).T
    # temp = np.linalg.solve(z_sqrt.T, temp)

    return temp.T


def _myinv(x):
    """

    Computes inverse

    parameters
    ----------
    x: ndarray
    array of shape (dc, dc)

    returns
    -------
    ndarray
    array of shape (dc, dc)
    """
    x = np.real(np.array(x))
    y = np.zeros(x.shape)
    y[x > 0] = 1 / x[x > 0]
    return y


def _compute_gamma_i(z, x):
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
    [e, v] = linalg.eig(z)
    e = e.real
    e[e < 0] = 0
    temp = np.dot(x.T, v)
    temp = np.real(np.dot(temp.conj().T, temp))
    e = np.sqrt(e)
    [d, u] = linalg.eig((temp * e) * e[:, np.newaxis])
    d = d.real
    d[d < 0] = 0
    d = np.sqrt(d)
    temp = np.dot(v * _myinv(np.real(e)), u)
    return np.array(np.real(np.dot(temp * d, temp.conj().T)))


