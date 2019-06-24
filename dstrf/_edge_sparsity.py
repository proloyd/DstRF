# Author: Proloy Das <proloy@umd.edu>
import time
import copy
from operator import attrgetter
import numpy as np

# Some specialized functions
from numpy.core.umath_tests import inner1d
from scipy import linalg
from scipy.signal import find_peaks
from math import sqrt, log10
from tqdm import tqdm
from multiprocessing import current_process

# eelbrain imports
from eelbrain import fmtxt, UTS, NDVar, SourceSpace, VolumeSourceSpace
from eelbrain._utils import LazyProperty

from ._fastac import Fasta
from ._crossvalidation import CVResult, crossvalidate
from . import opt
from .dsyevh3C import compute_gamma_c

import logging

_R_tol = np.finfo(np.float64).eps


def create_v(source):
    connectivity = source.connectivity()
    m = connectivity.shape[0]
    v = np.zeros((m, len(source)), dtype=np.float64)
    for i, (j, k) in zip(range(m), connectivity):
        v[i, j] = +1.0
        v[i, k] = -1.0
    return v


def f(lambdas, f, v, alpha):
    m = v.shape[0]
    lambda1 = lambdas[:m]
    lambda2 = lambdas[m:]
    x = alpha * lambda2
    x += v.T @ lambda1
    x -= f
    return 0.5 * (x ** 2).sum()


def gradf(lambdas, f, v, alpha):
    m = v.shape[0]
    lambda1 = lambdas[:m]
    lambda2 = lambdas[m:]
    x = alpha * lambda2
    x += v.T @ lambda1
    x -= f
    return np.concatenate((v @ x, alpha * x), axis=0)


def prune(lambdas, mu):
    np.maximum(lambdas, -mu, out=lambdas)
    np.minimum(lambdas, mu, out=lambdas)
    return lambdas


def g_es(x, mu, v, alpha):
    val = np.abs(x).sum()
    val *= alpha
    val += np.abs(v @ x).sum()
    return mu * val


def prox_g_es(x, mu, v, alpha):
    dual_solver = Fasta(lambda l:f(l, x, v, alpha), lambda l: 0, lambda l: gradf(l, x, v, alpha),
                        lambda l, t: prune(l, mu), n_iter=100)
    k, n = x.shape
    m = v.shape[0]
    dual_solver.learn(np.zeros((m + k, n), dtype=np.float64), tol=1e-4)

    lambdas = dual_solver.coefs_
    lambda1 = lambdas[:m]
    lambda2 = lambdas[m:]
    x -= alpha * lambda2
    x -= v.T @ lambda1
    return x


def test():
    x = np.arange(10, dtype=np.float64)
    x.shape = (5, 2)
    v = np.zeros((6, 5))
    v[0, 4] = -1
    v[0, 1] = 1
    v[1, 2] = 1
    v[1, 3] = -1
    v[2, 4] = 1
    v[2, 2] = -1
    v[3, 1] = 1
    v[3, 3] = -1
    v[4, 0] = 1
    v[4, 3] = -1
    v[5, 0] = 1
    v[5, 2] = -1
    print(g_es(x, 10.00, v, 0.005))
    print(prox_g_es(x, 10.00, v, 0.005))