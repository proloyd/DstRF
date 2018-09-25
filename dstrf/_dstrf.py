"""
Module implementing the direct trf estimation algorithm
"""

__author__ = 'Proloy DAS'
__licence__ = 'apache 2.0'

import numpy as np
from scipy import linalg
from eelbrain import *
from math import sqrt
from ._basis import gaussian_basis
from ._fastac import Fasta
import time

orientation = {'fixed':1, 'free':3}


def f(A, L, x, b, E):
    """
    Main Objective function corresponding to each trial

        f(x) = || L * x * E - b  ||_A ** 2 + < z, x - theta > + rho/2 * || x - theta || ** 2

    where ||x||_A = sqrt( trace( x' * A * x) ).

    :param A: Sigma_b ** (-1)
            (M, M) 2D array
    :param L: lead-field matrix
            (K, N) 2D array
            uses either fixed or free orientation lead-field vectors.
    :param x: argument of f
            (N, M) 2D array
    :param b: meg data
            (K, T) 2D array
    :param E: co-variate matrix from predictor variable
            (M, T) 2D arrays.

    :return: float
            f(x)
    """

    y = b - np.dot(np.dot(L, x), E.T)
    Cb = np.dot(y, y.T)

    return 0.5 * np.sum(A * Cb)


def gradf(A, L, x, b, E):
    """
    Gradient of Main Objective function corresponding to each trial

        \nabla f(x) = L' * A * (L * x * E - b) * E'

    :param A: Sigma_b ** (-1)
            (M, M) 2D array
    :param L: lead-field matrix
            (K, N) 2D array
            uses either fixed or free orientation lead-field vectors.
    :param x: argument of f
            (N, M) 2D array
    :param b: meg data
            (K, T) 2D array
    :param E: co-variate matrix from predictor variable
            (M, T) 2D arrays.

    :return: (N, M) 2D array
            \nabla f(x)

    """

    y = (b - np.dot(np.dot(L, x), E.T))

    return -np.dot(L.T, np.dot(np.dot(A, y), E))


def g(x, mu):
    """
    vector l1-norm penalty

                        g(x) = mu * |x|_1

    :param x : strf corresponding to one trial
            (N,M) 2D array
            Note: the norm is taken after flattening the matrix
    :param mu: regularizing parameter
            scalar float

    :return: g(x)
            scalar float

    """
    return mu * np.sum(np.abs(x))


def proxg(x, mu, tau):
    """
    proximal operator for g(x):

            prox_{tau g}(x) = min mu * |z|_1 + 1/ (2 * tau) ||x-z|| ** 2

    :param x : strf corresponding to one trial,
            (N,M) 2D array
            Note: the norm is taken after flattening the matrix
    :param mu: regularizing parameter
            scalar float
    :param tau: step size for fasta iterations
            scalar float

    :return: prox_{tau g}(x)
            (N,M) 2D array
    """

    return shrink(x, mu * tau)


def shrink(x, mu):
    """
    Soft theresholding function--
    proximal function for l1-norm:

                    S_{tau}(x) = min  |z|_1 + 1/ (2 * mu) ||x-z|| ** 2
                            x_i = sign(x_i) * max(|x_i| - mu, 0)

    :param x: generic vector
            (N,M) 2D array
    :param mu: backward step-size parameter
            scalar float

    :return: S_{tau}(x)
            (N,M) 2D array
    """
    return np.multiply(np.sign(x), np.maximum(np.abs(x) - mu, 0))


def g_group(x, mu):
    """
    group (l12) norm  penalty:

                        gg(x) = \sum ||x_s_{i,t}||

    where s_{i,t} = {x_{j,t}: j = 1*dc:(i+1)*dc}, i \in {1,2,...,#sources}, t \in {1,2,...,M}

    :param x : strf corresponding to one trial,
            (N,M) 2D array
    :param mu: regularizing parameter
            scalar float

    :return group norm gg(x)
            scalar float
    """
    l = x.shape[1]
    x.shape = (-1, 3, l)
    val = mu * np.sqrt((x ** 2).sum(axis=1)).sum()
    x.shape = (-1, l)
    return val


def proxg_group(z, mu):
    """
    proximal operator for gg(x):

            prox_{mu gg}(x) = min  gg(z) + 1/ (2 * mu) ||x-z|| ** 2
                    x_s = max(1 - mu/||z_s||, 0) z_s

    :param x : strf corresponding to one trial,
            (N,M) 2D array
    :param mu: regularizing parameter
            scalar float

    :return prox_{mu gg}(x)
            (N,M) 2D array
    """
    x = z.view()
    l = x.shape[1]
    x.shape = (-1, 3, l)
    mul = np.maximum(1 - mu / np.sqrt((x ** 2).sum(axis=1)), 0)
    x = np.swapaxes(np.swapaxes(x, 1, 0) * mul, 1, 0)
    x.shape = (-1, l)
    return x


def covariate_from_stim(stim, M, normalize=False):
    """
    From covariate matrix from stimulus

    parameters
    ----------
    stim: ndvar
        array of shape (1, T)
        predictor variables

    M: int
        order of filter

    normalize: bool, optional
        indicates if the stimulus to be normalized

    returns
    -------
    covariate matrix: ndarray

    """
    if stim.has_case:
        w = stim.get_data(('case', 'time'))
    else:
        w = stim.get_data('time')
        if w.ndim == 1:
            w = w[np.newaxis, :]

    if normalize:
        w -= w.mean(axis=0)
        w /= w.var(axis=0)

    length = w.shape[1]
    Y = []
    for j in range(w.shape[0]):
        X = []
        i = 0
        while i + M <= length:
            X.append(np.flipud(w[j, i:i + M]))
            i += 1
        Y.append(np.array(X))

    return np.array(Y)


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
    e[e < 0] = 0
    temp = np.dot(x.T, v)
    temp = np.real(np.dot(temp.T, temp))
    e = np.sqrt(e)
    [d, u] = linalg.eig((temp * e) * e[:, np.newaxis])
    d[d < 0] = 0
    d = np.sqrt(d)
    temp = np.dot(v * _myinv(np.real(e)), u)
    return np.array(np.real(np.dot(temp * d, np.matrix(temp).H)))


def _compute_objective(Cb, isigma_b):
    """

    Compute objective value at a given iteration

    parameters
    ----------
    Cb: ndarray
    array of shape (K, K)

    isigma_b: ndarray
    array of shape (K, K)

    returns
    -------
    float

    """
    return np.sum(isigma_b * Cb) - 2 * np.sum(np.log(np.diag(linalg.cholesky(isigma_b))))


class DstRF:
    """
    Champagne algorithm

    Reference--
    Wipf, David P., et al. "Robust Bayesian estimation of the location, orientation, and time
    course of multiple correlated neural sources using MEG." NeuroImage 49.1 (2010): 641-655.

    Parameters
    ----------
    lead_field: NDVar
        array of shape (K, N)
        lead-field matrix.
        both fixed or free orientation lead-field vectors can be used.

    orientation: 'fixed'|'free'
        'fixed': orientation-constrained lead-field matrix.
        'free': free orientation lead-field matrix.

    noise_covariance: ndarray
        array of shape (K, K)
        noise covariance matrix
        use empty-room recordings to generate noise covariance matrix at sensor space.

    n_iter: int, optionnal
        number of iterations
        default is 1000

    Attributes
    ----------
    Gamma: list
        list of length N
        individual source covariance matrices

    inverse_sigma_b: ndarray
        array of shape (K, K)
        inverse of data covariance under the model

    objective: list, optional
        list of objective values at each iteration
        returned only if verbose=1

    est_data_covariance: ndarray
        array of shape (K, K)
        estimated data covariance under the model
        returned only if verbose=1

    emp_data_covariance: ndarray
        array of shape (K, K)
        empirical data covariance
        returned only if verbose=1

    inverse_kernel: ndarray
        array of shape (K, K)
        inverse imaging kernel
        returned only if return_inverse_kernel=1

    """
    _n_predictor_variables = 1

    def __init__(self, lead_field, noise_covariance, n_trials, filter_length=200, n_iter=30, n_iterc=1000,
            n_iterf=1000):
        if lead_field.has_dim('space'):
            self.lead_field = lead_field.get_data(dims=('sensor', 'source', 'space')).astype('float64')
            self.sources_n = self.lead_field.shape[1]
            self.lead_field = self.lead_field.reshape(self.lead_field.shape[0], -1)
            self.orientation = 'free'
            self.space = lead_field.space
        else:
            self.lead_field = lead_field.get_data(dims=('sensor', 'source')).astype('float64')
            self.sources_n = self.lead_field.shape[1]
            self.orientation = 'fixed'
        self.source = lead_field.source
        self.sensor = lead_field.sensor
        self.noise_covariance = noise_covariance
        self.n_trials = n_trials
        self.filter_length = filter_length
        x = np.linspace(5, 1000, self.filter_length)
        self.basis = gaussian_basis(self.filter_length, x)
        self._covariates = []
        self._meg = []
        # self._ytilde = []
        self.n_iter = n_iter
        self.n_iterc = n_iterc
        self.n_iterf = n_iterf
        self.__init__vars()

    def __init__vars(self):
        wf = linalg.cholesky(self.noise_covariance, lower=True)
        Gtilde = linalg.solve(wf, self.lead_field)
        self.eta = (self.lead_field.shape[0] / np.trace(np.dot(Gtilde, Gtilde.T)))
        # model data covariance
        sigma_b = self.noise_covariance + self.eta * np.dot(self.lead_field, self.lead_field.T)
        self.init_sigma_b = sigma_b
        return self

    def __init__iter(self):
        self.Gamma = []
        self.Sigma_b = []
        dc = orientation[self.orientation]
        for _ in range(self.n_trials):
            self.Gamma.append([self.eta * np.eye(dc, dtype='float64') for _ in range(self.sources_n)])
            self.Sigma_b.append(self.init_sigma_b.copy())

        # initializing \Theta
        self.theta = np.zeros((self.sources_n * dc, self._n_predictor_variables * self.basis.shape[1]))
        return self

    def setup(self, meg, stim, normalize_regresor=True, verbose=0):
        """

        :param meg:
        :param verbose:
        :return:
        """
        y = meg.get_data(('sensor', 'time'))
        y = y[:, self.basis.shape[1]:]
        self._meg.append(y / sqrt(y.shape[1]))  # Mind the normalization

        # set up covariate matrix
        covariates = np.dot(covariate_from_stim(stim, self.filter_length, normalize=normalize_regresor),
                            self.basis) / sqrt(y.shape[1])
        if covariates.ndim > 2:
            self._n_predictor_variables = covariates.shape[0]
            covariates = covariates.swapaxes(1, 0)

        first_dim = covariates.shape[0]
        self._covariates.append(covariates.reshape(first_dim, -1))

        return self

    def set_mu(self, mu):
        self.mu = mu
        self.__init__iter()
        return self

    def __solve_old(self, theta, trial):
        """

        :param theta:
        :param trial:
        :param verbose:
        :return:
        """
        y = self._meg[trial] - np.dot(np.dot(self.lead_field, theta), self._covariates[trial].T)
        Cb = np.dot(y, y.T)  # empirical data covariance
        yhat = linalg.cholesky(Cb, lower=True)
        gamma = self.Gamma[trial]
        isigma_b = linalg.inv(self.Sigma_b[trial])

        # Choose dc
        if self.orientation == 'fixed':
            dc = 1
        elif self.orientation == 'free':
            dc = 3

        # champagne iterations
        for it in range(self.n_iterc):
            # pre-compute some useful matrices
            lhat = np.dot(isigma_b, self.lead_field)

            # compute sigma_b for the next iteration
            sigma_b_next = self.noise_covariance.copy()

            for i in range(self.sources_n):
                # update Xi
                x = np.dot(gamma[i], np.dot(yhat.T, lhat[:, i * dc:(i + 1) * dc]).T)

                # update Zi
                z = np.dot(self.lead_field[:, i * dc:(i + 1) * dc].T, lhat[:, i * dc:(i + 1) * dc])

                # update Ti
                if dc == 1:
                    gamma[i] = sqrt(np.dot(x, x.T)) / np.real(sqrt(z))
                else:
                    gamma[i] = _compute_gamma_i(z, x)

                # update sigma_b for next iteration
                sigma_b_next = sigma_b_next + np.dot(self.lead_field[:, i * dc:(i + 1) * dc],
                                                     np.dot(gamma[i], self.lead_field[:, i * dc:(i + 1) * dc].T))

            # update sigma_b
            isigma_b = linalg.inv(sigma_b_next)

        self.Gamma[trial] = gamma
        self.Sigma_b[trial] = sigma_b_next

        return self

    def __solve(self, theta, trial):
        """

        :param theta:
        :param trial:
        :param verbose:
        :return:
        """
        y = self._meg[trial] - np.dot(np.dot(self.lead_field, theta), self._covariates[trial].T)
        Cb = np.dot(y, y.T)  # empirical data covariance
        yhat = linalg.cholesky(Cb, lower=True)
        gamma = self.Gamma[trial]
        sigma_b = self.Sigma_b[trial].copy()
        # isigma_b = linalg.inv(self.Sigma_b[trial])

        # Choose dc
        if self.orientation == 'fixed':
            dc = 1
        elif self.orientation == 'free':
            dc = 3

        # champagne iterations
        for it in range(self.n_iterc):
            # pre-compute some useful matrices
            Lc = linalg.cholesky(sigma_b, lower=True)
            lhat = linalg.solve(Lc, self.lead_field)
            ytilde = linalg.solve(Lc, yhat)

            # compute sigma_b for the next iteration
            sigma_b = self.noise_covariance.copy()

            for i in range(self.sources_n):
                # update Xi
                # x = np.dot(gamma[i], np.dot(yhat.T, lhat[:, i * dc:(i + 1) * dc]).T)
                x = np.dot(gamma[i], np.dot(ytilde.T, lhat[:, i * dc:(i + 1) * dc]).T)

                # update Zi
                # z = np.dot(self.lead_field[:, i * dc:(i + 1) * dc].T, lhat[:, i * dc:(i + 1) * dc])
                z = np.dot(lhat[:, i * dc:(i + 1) * dc].T, lhat[:, i * dc:(i + 1) * dc])

                # update Ti
                if dc == 1:
                    gamma[i] = sqrt(np.dot(x, x.T)) / np.real(sqrt(z))
                else:
                    gamma[i] = _compute_gamma_i(z, x)

                # update sigma_b for next iteration
                sigma_b += np.dot(self.lead_field[:, i * dc:(i + 1) * dc],
                                  np.dot(gamma[i], self.lead_field[:, i * dc:(i + 1) * dc].T))

        self.Gamma[trial] = gamma
        self.Sigma_b[trial] = sigma_b

        return self

    # def __solve_1_step(self, theta, trial, verbose=0):
    #     """
    #
    #     :param theta:
    #     :param verbose:
    #     :return:
    #     """
    #     y = self._meg[trial] - np.dot(np.dot(self.lead_field, theta), self._covariates[trial].T)
    #     Cb = np.dot(y, y.T)     # empirical data covariance
    #     yhat = linalg.cholesky(Cb, lower=True)
    #     gamma = self.Gamma[trial]
    #     isigma_b = linalg.inv(self.Sigma_b[trial])
    #
    #     # Choose dc
    #     if self.orientation == 'fixed': dc = 1
    #     elif self.orientation == 'free': dc = 3
    #
    #     # pre-compute some useful matrices
    #     lhat = np.dot (isigma_b, self.lead_field)
    #
    #     # compute sigma_b for the next iteration
    #     sigma_b = np.copy (self.noise_covariance)
    #     # ytilde = np.copy(self._meg[trial])
    #     for i in (xrange (self.sources_n)):
    #         # update Xi
    #         x = np.dot (gamma[i], np.dot (yhat.T, lhat[:, i * dc:(i + 1) * dc]).T)
    #         # ytilde = ytilde - np.dot(self.lead_field[:, i * dc:(i + 1) * dc], x)
    #
    #         # update Zi
    #         z = np.dot (self.lead_field[:, i * dc:(i + 1) * dc].T, lhat[:, i * dc:(i + 1) * dc])
    #
    #         # update Ti
    #         if dc == 1:
    #             gamma[i] = sqrt (np.dot (x, x.T)) / np.real (sqrt (z))
    #         else:
    #             gamma[i] = _compute_gamma_i (z, x)
    #
    #         # update sigma_b for next iteration
    #         sigma_b = sigma_b + np.dot (self.lead_field[:, i * dc:(i + 1) * dc],
    #                                               np.dot (gamma[i], self.lead_field[:, i * dc:(i + 1) * dc].T))
    #
    #     # update sigma_b
    #     sigma_b = sigma_b
    #
    #     self.Gamma[trial] = gamma
    #     self.Sigma_b[trial] = sigma_b
    #     inverse_kernel = np.copy(self.lead_field.T).astype ('float64')
    #     for i in xrange (self.sources_n):
    #         inverse_kernel[i * dc:(i + 1) * dc, :] \
    #             = np.dot (gamma[i], inverse_kernel[i * dc:(i + 1) * dc, :])
    #
    #     inverse_kernel = np.dot(inverse_kernel, isigma_b)
    #
    #     # se = np.dot(np.dot(self.lead_field, inverse_kernel), y)
    #
    #     self._ytilde[trial] = self._meg[trial] - np.dot(np.dot(self.lead_field, inverse_kernel), y)

    def fit(self, tol=1e-3, verbose=0):
        """

        :return:

        """
        # check if the probelm is set up properly
        if len(self._meg) is not self.n_trials:
            raise IndexError("n_trials(={:}) does not match number of MEGs(={:}) provided".format(
                self.n_trials,
                len(self._meg)
            ))

        if self.orientation == 'fixed':
            dc = 1
            g_funct = lambda x:g(x, self.mu)
            prox_g = lambda x, t:shrink(x, self.mu * t)
        elif self.orientation == 'free':
            dc = 3
            g_funct = lambda x:g_group(x, self.mu)
            prox_g = lambda x, t:proxg_group(x, self.mu * t)

        # inverse_noise_covariance = linalg.inv(self.noise_covariance)

        theta = self.theta

        self.err = []
        if verbose:
            self.objective_vals = []
            start = time.time()

        # run iterations
        for i in (range(self.n_iter)):
            funct, grad_funct = self._construct_f_new()
            Theta = Fasta(funct, g_funct, grad_funct, prox_g, n_iter=self.n_iterf)
            Theta.learn(theta)

            self.err.append(linalg.norm(theta - Theta.coefs_, 'fro') ** 2)
            theta = Theta.coefs_

            for trial in range(self.n_trials):
                self.__solve(theta, trial)

            if self.err[-1] / self.err[0] < tol:
                break

            if verbose:
                self.objective_vals.append(self.eval_obj())
                print("Iteration: {:}, objective value:{:10f}".format(i, self.objective_vals[-1]))

        if verbose:
            end = time.time()
            print("Time elapsed: {:10f} s".format(end - start))

        self.theta = theta

        return self

    def _construct_f(self):
        inverse_sigma_b = [linalg.inv(self.Sigma_b[trial])
                           for trial in range(self.n_trials)]

        funct = lambda x: sum(
            [f(
                inverse_sigma_b[trial],
                self.lead_field,
                x,
                self._meg[trial],
                self._covariates[trial]
            )
                for trial in range(self.n_trials)]
        )
        grad_funct = lambda x: np.array(
            [
                gradf(
                    inverse_sigma_b[trial],
                    self.lead_field,
                    x,
                    self._meg[trial],
                    self._covariates[trial])
                for trial in range(self.n_trials)
            ]
        ).sum(axis=0)

        return funct, grad_funct

    def _construct_f_new(self):
        L = [linalg.cholesky(self.Sigma_b[trial], lower=True) for trial in range(self.n_trials)]
        leadfields = [linalg.solve(L[trial], self.lead_field) for trial in range(self.n_trials)]
        megs = [linalg.solve(L[trial], self._meg[trial]) for trial in range(self.n_trials)]

        def f(L, x, b, E):
            y = b - np.dot(np.dot(L, x), E.T)

            return 0.5 * (y ** 2).sum()

        def gradf(L, x, b, E):
            y = b - np.dot(np.dot(L, x), E.T).astype('float64')

            return -np.dot(L.T, np.dot(y, E))

        def funct(x):
            val = 0.0
            for trial in range(self.n_trials):
                val += f(leadfields[trial], x, megs[trial], self._covariates[trial])

            return val

        def grad_funct(x):
            grad = gradf(leadfields[0], x, megs[0], self._covariates[0])
            for trial in range(1, self.n_trials):
                grad += gradf(leadfields[trial], x, megs[trial], self._covariates[trial])

            return grad

        return funct, grad_funct


    def get_strf(self, fs):
        """

        Updates the spatio-temporal response function as NDVar

        parameters
        ----------
         fs: float
            sampling frquency, usually 200 Hz

        returns
        -------
            self
        """
        trf = self.theta.copy()
        if self._n_predictor_variables > 1:
            shape = (trf.shape[0], 3, -1)
            trf.shape = shape
            trf = trf.swapaxes(1, 0)

        # trf = np.dot(self.basis, self.theta.T).T
        trf = np.dot(trf, self.basis.T)

        time = UTS(0, 1.0 / fs, trf.shape[-1])

        if self.orientation == 'fixed':
            if self._n_predictor_variables > 1:
                dims = (Case, self.source, time)
            else:
                dims = (self.source, time)
            trf = NDVar(trf, dims)

        elif self.orientation == 'free':
            dims = (time, self.source, self.space)
            if self._n_predictor_variables > 1:
                trfs = []
                for i in range(self._n_predictor_variables):
                    trfs.append(NDVar(trf[i, :, :].T.reshape(-1, self.sources_n, 3), dims))
                trf = combine(trfs)
            else:
                trf = NDVar(trf.T.reshape(-1, self.sources_n, 3), dims)

        return trf

    def eval_obj(self):
        v = 0
        for trial in range(self.n_trials):
            y = self._meg[trial] - np.dot(np.dot(self.lead_field, self.theta), self._covariates[trial].T)

            L = linalg.cholesky(self.Sigma_b[trial], lower=True)
            y = linalg.solve(L, y)
            v = v + 0.5 * (y ** 2).sum() + np.log(np.diag(L)).sum()

            # Cb = np.dot(y, y.T)
            # v = v + 0.5 * np.trace(linalg.solve(self.Sigma_b[trial], Cb)) \
            #     + np.sum(np.log(np.diag(linalg.cholesky(self.Sigma_b[trial]))))

        return v / self.n_trials
