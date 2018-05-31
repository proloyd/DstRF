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
import tqdm
from multiprocessing import Value
import time


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

    y = b - np.dot (np.dot (L, x), E.T)
    Cb = np.dot (y, y.T)

    return 0.5 * np.sum (A * Cb)


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

    y = (b - np.dot (np.dot (L, x), E.T))

    return -np.dot (L.T, np.dot (np.dot (A, y), E))


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
    return mu * np.sum (np.abs (x))


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

    return shrink (x, mu * tau)


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
    return np.multiply (np.sign (x), np.maximum (np.abs (x) - mu, 0))


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
    v = np.reshape (x, (-1, 3 * l))
    w = np.array ([v[:, 0:l], v[:, l:2 * l], v[:, 2 * l:3 * l]])
    return mu * np.sum (np.sqrt (np.mean (w ** 2, axis=0)))


def proxg_group(x, mu):
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
    l = x.shape[1]
    v = np.reshape (x, (-1, 3 * l))
    w = np.array ([v[:, 0:l], v[:, l:2 * l], v[:, 2 * l:3 * l]])
    z = np.sqrt (np.mean (w ** 2, axis=0))
    multiplier = np.maximum (z - mu, 0)
    multiplier[z > 0] = np.divide (multiplier[z > 0], z[z > 0])
    x = np.multiply (x, np.kron (multiplier, np.ones ((3, 1))))

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
    w = stim.get_data (('time'))

    if normalize:
        w = w - np.mean (w)  # centering
        w = w / np.std (w)  # normalizing

    length = w.shape[0]
    X = []
    i = 0
    while i + M <= length:
        X.append (np.flipud (w[i:i + M]))
        i = i + 1

    return np.array (X)


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
    y[x>0] = 1/x[x > 0]
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
    temp = np.real( np.dot(temp.T, temp))
    e = np.sqrt(e)
    [d, u] = linalg.eig( (temp * e) * e[:, np.newaxis] )
    d[d < 0] = 0
    d = np.sqrt(d)
    temp = np.dot(v * _myinv(np.real(e)), u)
    return np.array( np.real(np.dot(temp * d, np.matrix(temp).H)) )


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
    return np.sum(isigma_b*Cb) - 2 * np.sum (np.log(np.diag(linalg.cholesky(isigma_b))))


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
    def __init__(self, lead_field, noise_covariance, n_trials, mu, filter_length=200, n_iter=30, n_iterc=1000, n_iterf=1000):
        if lead_field.has_dim('space'):
            self.lead_field = lead_field.get_data (dims=('sensor', 'source', 'space')).astype('float64')
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
        self.Gamma = []
        self.Sigma_b = []
        # self._ytilde = []
        self.n_iter = n_iter
        self.n_iterc = n_iterc
        self.n_iterf = n_iterf

    def setup(self, meg, stim, normalize_regresor=True, verbose=0):
        """

        :param meg:
        :param verbose:
        :return:
        """

        y = meg.get_data(('sensor', 'time'))
        y = y[:, self.basis.shape[1]:]
        self._meg.append(y/sqrt(y.shape[1]))    # Mind the normalization
        # self._ytilde.append(y/sqrt(y.shape[1]))

        # set up covariate matrix
        self._covariates.append(np.dot (covariate_from_stim(stim, self.filter_length, normalize=normalize_regresor),
                                  self.basis)/sqrt(y.shape[1]))


        # y = meg
        Cb = np.dot(y, y.T)    # empirical data covariance
        yhat = linalg.cholesky(Cb, lower=True)

        # noise_covariance = np.eye(self.noise_covariance.shape[0])  # since the data is pre whitened
        noise_covariance = self.noise_covariance

        # Choose dc
        if self.orientation == 'fixed': dc = 1
        elif self.orientation == 'free': dc = 3

        # initializing gamma
        wf = linalg.cholesky(self.noise_covariance, lower=True)
        ytilde = linalg.solve(wf, yhat)
        eta = 1e-5 * (ytilde.shape[0] / np.trace (np.dot(ytilde, ytilde.T)))
        self.Gamma.append([eta * np.eye(dc, dtype='float64') for _ in range (self.sources_n)])  # Initial gamma
        # print "Gamma = {:10f}".format(eta)

        # model data covariance
        sigma_b = noise_covariance
        for j in range(self.sources_n):
            sigma_b = sigma_b + np.dot(self.lead_field[:, j * dc:(j + 1) * dc],
                                        np.dot (self.Gamma[-1][j], self.lead_field[:, j * dc:(j + 1) * dc].T))
        self.Sigma_b.append(sigma_b)

        # initializing \Theta
        self.theta = np.zeros((self.sources_n * dc, self.basis.shape[1]))

    def set_mu(self, mu):
        self.mu = mu

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
        isigma_b = linalg.inv(self.Sigma_b[trial])

        # Choose dc
        if self.orientation == 'fixed':
            dc = 1
        elif self.orientation == 'free':
            dc = 3

        # champagne iterations
        for it in xrange(self.n_iterc):
            # pre-compute some useful matrices
            lhat = np.dot(isigma_b, self.lead_field)

            # compute sigma_b for the next iteration
            sigma_b_next = np.copy(self.noise_covariance)

            for i in xrange (self.sources_n):
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

    def fit(self, mp=False, tol=1e-4, verbose=0):
        """

        :return:

        """
        # check if the probelm is set up properly
        if len(self._meg) is not self.n_trials:
            print "Warning: " \
                  "n_iter(={:}) does not match number of MEGs(={:}) provided".format(
                                                                                     self.n_trials,
                                                                                     len(self._meg)
                                                                                    )

        if self.orientation == 'fixed':
            dc = 1
            g_funct = lambda (x): g(x, self.mu)
            prox_g = lambda x, t: shrink(x, self.mu * t)
        elif self.orientation == 'free':
            dc = 3
            g_funct = lambda (x): g_group(x, self.mu)
            prox_g = lambda x, t: proxg_group(x, self.mu * t)

        # inverse_noise_covariance = linalg.inv(self.noise_covariance)

        if mp == 0:
            theta = self.theta
        else:
            theta = np.array(self.theta).reshape((self.sources_n * dc, self.basis.shape[1]))

        self.err = []
        if verbose:
            self.objective_vals = []
            start = time.time()

        # run iterations
        for i in (xrange(self.n_iter)):
            time.sleep(0.001)
            for trial in xrange(self.n_trials):
                self.__solve(theta, trial)

            inverse_sigma_b = [linalg.inv(self.Sigma_b[trial])
                               for trial in xrange(self.n_trials)]

            funct = lambda (x): sum(
                [f(
                    inverse_sigma_b[trial],
                    self.lead_field,
                    x,
                    self._meg[trial],
                    self._covariates[trial]
                )
                    for trial in xrange(self.n_trials)]
            )
            grad_funct = lambda (x): np.array(
                [
                    gradf(
                        inverse_sigma_b[trial],
                        self.lead_field,
                        x,
                        self._meg[trial],
                        self._covariates[trial])
                    for trial in xrange(self.n_trials)
                ]
            ).sum(axis=0)
            Theta = Fasta(funct, g_funct, grad_funct, prox_g, n_iter=self.n_iterf)
            Theta.learn(theta)
            self.err.append(linalg.norm(theta - Theta.coefs_, 'fro')**2)
            if self.err[-1]/self.err[0] < tol:
                break
            theta = Theta.coefs_
            if verbose:
                self.objective_vals.append(self.eval_obj())
                print "Iteration: {:}, objective value:{:10f}"\
                    .format(i, self.objective_vals[-1])

        if verbose:
            end = time.time()
            print "Time elapsed: {:10f} s".format(end -start)
        if mp == 0:
            self.theta = theta
        else:
            theta = np.squeeze(theta.reshape(1, -1))
            for i, _ in enumerate(self.theta):
                self.theta[i] = theta[i]

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
        trf = np.dot (self.basis, self.theta.T).T
        time = UTS (0, 1.0 / fs, trf.shape[1])

        if self.orientation == 'fixed':

            dims = (self.source, time)
            trf = NDVar (trf, dims)

        elif self.orientation == 'free':

            dims = (time, self.source, self.space)
            trf = NDVar(trf.T.reshape(-1, self.sources_n, 3), dims)

        self.trf = trf

        return self

    # def return_inverse_operator(self):
    #     """
    #
    #     Returns inverse operator
    #
    #     returns:
    #     -------
    #     ndarray
    #     array of shape (N, K)
    #
    #     """
    #     if self.orientation == 'free': dc = 3
    #     else: dc = 1
    #
    #     inverse_kernel = np.copy(self.lead_field.T).astype ('float64')
    #     for i in xrange (self.sources_n):
    #         inverse_kernel[i * dc:(i + 1) * dc, :] \
    #             = np.dot (self.Gamma[i], inverse_kernel[i * dc:(i + 1) * dc, :])
    #
    #     inverse_kernel = np.dot (inverse_kernel, self.inverse_sigma_b)
    #
    #     return inverse_kernel
    #
    def apply_inverse_operator(self, trial):
        """

        parameters
        ----------
        meg: NDVar
            meg data

        returns
        -------
        se: ndarray
            source estimates

        """
        y = self._meg

        if self.orientation == 'free': dc = 3
        else: dc = 1

        gamma = self.Gamma[trial]
        isigma_b = linalg.inv(self.Sigma_b[trial])

        inverse_kernel = np.copy(self.lead_field.T).astype ('float64')
        for i in xrange (self.sources_n):
            inverse_kernel[i * dc:(i + 1) * dc, :] \
                = np.dot (gamma[i], inverse_kernel[i * dc:(i + 1) * dc, :])

        inverse_kernel = np.dot(inverse_kernel, isigma_b)

        se = np.dot(inverse_kernel, y)
        # if self.orientation == 'fixed':
        #     ndvar = NDVar(se, dims=(self.source, meg.time))
        # else:
        #     ndvar = NDVar(se.T.reshape(-1, self.sources_n, dc), dims=(meg.time, self.source, self.space))

        return se

    def eval_obj(self):
        v = 0
        for trial in xrange(self.n_trials):
            y = self._meg[trial] - np.dot(np.dot(self.lead_field, self.theta), self._covariates[trial].T)
            Cb = np.dot(y, y.T)
            v = v + 0.5 * np.trace(linalg.solve(self.Sigma_b[trial], Cb)) \
                + 2 * np.sum(np.log(np.diag(linalg.cholesky(self.Sigma_b[trial]))))

        return v

