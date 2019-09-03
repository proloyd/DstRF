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

from fastapy import Fasta
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
                        lambda l, t: prune(l, mu), n_iter=10)
    k, n = x.shape
    m = v.shape[0]
    # import ipdb
    # ipdb.set_trace()
    lambdas = np.zeros((m + k, n), dtype=np.float64)
    # lambdas[:m][:, :] = linalg.solve(v @ v.T, v @ x)
    lambdas[m:][:, :] = x / alpha
    dual_solver.learn(prune(lambdas, mu), tol=1e-4, linesearch=False)

    lambdas = dual_solver.coefs_
    lambda1 = lambdas[:m]
    lambda2 = lambdas[m:]
    x -= alpha * lambda2
    x -= v.T @ lambda1
    return x

def prox_newton(x, mu, v, alpha):
    h = v @ v.T

    k, n = x.shape
    m = v.shape[0]

    lambdas = np.zeros((m + k, n), dtype=np.float64)
    g = gradf(lambdas, x, v, alpha)
    I = np.logical_or(np.logical_and(lambdas == -mu, g > 0),
                      np.logical_and(lambdas == +mu, g < 0))

    h_ibar = h[~I][:, ~I]
    g_ibar = [~I]


def shrink(x, mu):
    u = np.sign(x)
    return u * np.maximum(np.abs(x)-mu, 0)

def prox_admm(x, mu, v, alpha):
    k, n = x.shape
    m = v.shape[0]

    max_iter = 1000
    u, s, vh = linalg.svd(v, full_matrices=False)
    tau1 = 1000
    tau2 = 1000
    gamma = tau1 * s**2
    gamma += (1 + tau2)
    gamma = 1 / gamma
    mul1 = vh.T @ (gamma[:, None] * vh)
    gamma *= s
    mul2 = vh.T @ (gamma[:, None] * u.T)

    lambda1 = np.zeros((m, n), dtype=x.dtype)
    y = np.empty_like(lambda1)
    lambda2 = np.zeros_like(x)
    z = np.zeros_like(lambda2)
    z[:, :] = x[:, :]
    x0 = x.copy()
    vx0 = np.zeros_like(lambda1)
    z0 = np.empty_like(z)
    res = []

    # import ipdb
    for i in range(max_iter):
        np.matmul(v, x0, vx0)
        y[:, :] = vx0[:,:]
        y += lambda1 / tau1
        y = shrink(y, mu / tau1)

        z = x0 - lambda2 / tau2
        z = shrink(z, (mu * alpha) / tau2)

        vx0 -= y
        z0[:, :] = z[:, :]
        z0 -= x0
        # z0[:, :] = x0[:, :]
        # z0 -= z
        lambda1 += tau1 * vx0
        lambda2 += tau2 * z0
        # ipdb.set_trace()
        res.append(tau1 **2 * ((vx0 ** 2).sum() + tau2 ** 2 * (z0 ** 2).sum()) ** 0.5)
        # print(res[-1] / res[0])
        if res[-1] / res[0] < 1e-4:
            break

        x0 = mul1 @ (x - lambda2 + tau2 * z) + mul2 @ (lambda1 - tau1 * y)
        # x0 -= lambda2
        # x0 -= tau2 * z
        # x0 = mul1 @ x0
        # x0 +=  mul2 @ (lambda1 - tau1 * y)

    print(f'last iter {i}')
    x[:, :] =  x0[:, :]
    return x0


class prox_ADMM:
    def __init__(self, alpha, max_iter=1000, tau1='auto', tau2=1, source=None, v=None):
        if source is not None:
            self.connectivity = source.connectivity()
        else:
            self.connectivity = None
        if v is None:
            v = create_v(source)
        self.v = v
        self.m = v.shape[0]
        u, s, vh = linalg.svd(v, full_matrices=False)
        if tau1 == 'auto':
            tau1 = 1 / (s ** 2).mean()
        self.tau1 = tau1
        self.tau2 = tau2
        gamma = tau1 * s ** 2
        gamma += (1 + tau2)
        gamma = 1 / gamma
        self.mul1 = vh.T @ (gamma[:, None] * vh)
        gamma *= s
        self.mul2 = vh.T @ (gamma[:, None] * u.T)
        self.alpha = alpha
        self.max_iter = max_iter

    def g_es(self, x, mu):
        val = np.abs(x).sum()
        val *= self.alpha
        if self.connectivity is not None:
            src, dst = self.connectivity.T
            vx = x[src] - x[dst]
        else:
            vx = self.v @ x
        val += np.abs(vx).sum()
        return mu * val

    def g_es_prox(self, x, mu):
        k, n = x.shape
        m = self.m
        tau1 = self.tau1
        tau2 = self.tau2
        alpha = self.alpha

        lambda1 = np.ones((m, n), dtype=x.dtype)
        y = np.empty_like(lambda1)
        lambda2 = np.ones_like(x)
        z = np.zeros_like(lambda2)
        z[:, :] = x[:, :]
        x0 = x.copy() * 0
        vx0 = np.empty_like(lambda1)
        z0 = np.empty_like(z)
        res = []
        if self.connectivity is not None:
            src, dst = self.connectivity.T

        for i in range(self.max_iter):
            if self.connectivity is not None:
                vx0[:, :] = x0[src]
                vx0 -= x0[dst]
            else:
                np.matmul(self.v, x0, vx0)
            y[:, :] = vx0[:, :]
            y += lambda1 / tau1
            y = shrink(y, mu / tau1)

            z = x0 - lambda2 / tau2
            z = shrink(z, (mu * alpha) / tau2)

            vx0 -= y
            z0[:, :] = z[:, :]
            z0 -= x0
            # z0[:, :] = x0[:, :]
            # z0 -= z
            lambda1 += tau1 * vx0
            lambda2 += tau2 * z0
            # ipdb.set_trace()
            res.append(tau1 ** 2 * ((vx0 ** 2).sum() + tau2 ** 2 * (z0 ** 2).sum()) ** 0.5)
            print(res[-1] / res[0])

            x0 = self.mul1 @ (x - lambda2 + tau2 * z) + self.mul2 @ (lambda1 - tau1 * y)
            # x0 -= lambda2
            # x0 -= tau2 * z
            # x0 = mul1 @ x0
            # x0 +=  mul2 @ (lambda1 - tau1 * y)
            if res[-1] / res[0] < 1e-4:
                break

        print(f'last iter {i}')
        x[:, :] = x0[:, :]
        return x0


def test():
    x = np.random.normal(size=(5, 2))
    # x.shape = (5, 2)
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
    p_admm = prox_ADMM(v=v, alpha=0.05, max_iter=1000, tau1=2000, tau2=2000, source=None)
    for j in range(10):
        x1 = x.copy()
        print(g_es(x1, 1.00, v, 0.05))
        print(prox_g_es(x1, 1.00, v, 0.05))
        print(g_es(x1, 1.00, v, 0.05) + 0.5 * ((x-x1) ** 2).sum())
        x2 = x.copy()
        print(prox_admm(x2, 1.00, v, 0.05))
        print(g_es(x2, 1.00, v, 0.05) + + 0.5 * ((x-x2) ** 2).sum() )
        x3 = x.copy()
        print(p_admm.g_es_prox(x3, 1.00))
        print(p_admm.g_es(x3, 1.00) + + 0.5 * ((x-x3) ** 2).sum() )
