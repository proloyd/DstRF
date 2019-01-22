# Author: Proloy Das <proloy@umd.edu>
import time
import copy
import numpy as np

# Some specialized functions
from numpy.core.umath_tests import inner1d
from scipy import linalg
from math import sqrt

# eelbrain imports
from eelbrain import UTS, NDVar, combine, Case

from ._fastac import Fasta
from ._crossvalidation import crossvalidate
from . import opt
from .dsyevh3C import compute_gamma_c

def gaussian_basis(nlevel, span):
    """Construct Gabor basis for the TRFs.

    Parameters
    ----------
    nlevel: int
        number of atoms
    span: ndarray
        the span to cover by the atoms

    Returns
    -------
        ndarray (Gabor atoms)
    """
    x = span
    means = np.linspace(x[-1] / nlevel, x[-1] * (1 - 1 / nlevel), num=nlevel - 1)
    stds = 8.5
    W = []

    for mean in means:
        W.append(np.exp(-(x - mean) ** 2 / (2 * stds ** 2)))

    W = np.array(W)

    return W.T / np.max(W)


def g(x, mu):
    """vector l1-norm penalty"""
    return mu * np.sum(np.abs(x))


def proxg(x, mu, tau):
    """proximal operator for l1-norm penalty"""
    return shrink(x, mu * tau)


def shrink(x, mu):
    """Soft theresholding function"""
    return np.multiply(np.sign(x), np.maximum(np.abs(x) - mu, 0))


def g_group(x, mu):
    """group (l12) norm  penalty:

            gg(x) = \sum ||x_s_{i,t}||

    where s_{i,t} = {x_{j,t}: j = 1*dc:(i+1)*dc}, i \in {1,2,...,#sources}, t \in {1,2,...,M}
    """
    l = x.shape[1]
    x.shape = (-1, 3, l)
    val = mu * np.sqrt((x ** 2).sum(axis=1)).sum()
    x.shape = (-1, l)
    return val


def proxg_group_opt(z, mu):
    """proximal operator for gg(x):

            prox_{mu gg}(x) = min  gg(z) + 1/ (2 * mu) ||x-z|| ** 2
                    x_s = max(1 - mu/||z_s||, 0) z_s

    Note: It does update the supplied z. It is a wrapper for distributed Cython code.
    """
    # x = z.view()
    l = z.shape[1]
    z.shape = (-1, 3, l)
    opt.cproxg_group(z, mu, z)
    z.shape = (-1, l)
    return z


def covariate_from_stim(stim, M):
    """Form covariate matrix from stimulus

    parameters
    ----------
    stim: NDVar
        predictor variables
    M: int
        order of filter

    returns
    -------
        ndarray (covariate matrix)
    """
    if stim.has_case:
        w = stim.get_data(('case', 'time'))
    else:
        w = stim.get_data('time')
        if w.ndim == 1:
            w = w[np.newaxis, :]

    length = w.shape[1]
    Y = []
    for j in range(w.shape[0]):
        X = []
        for i in range(length - M + 1):
            X.append(np.flipud(w[j, i:i + M]))
        Y.append(np.array(X))

    return np.array(Y)


def _myinv(x):
    """Computes inverse"""
    x = np.real(np.array(x))
    y = np.zeros(x.shape)
    y[x > 0] = 1 / x[x > 0]
    return y


def _compute_gamma_i(z, x):
    """ Comptes Gamma_i

    Gamma_i = Z**(-1/2) * ( Z**(1/2) X X' Z**(1/2)) ** (1/2) * Z**(-1/2)
           = V(E)**(-1/2)V' * ( V ((E)**(1/2)V' X X' V(E)**(1/2)) V')** (1/2) * V(E)**(-1/2)V'
           = V(E)**(-1/2)V' * ( V (UDU') V')** (1/2) * V(E)**(-1/2)V'
           = V (E)**(-1/2) U (D)**(1/2) U' (E)**(-1/2) V'

    Parameters
    ----------
    z: ndarray
        auxiliary variable,  z_i
    x: ndarray
        auxiliary variable, x_i

    Returns
    -------
        ndarray
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


def _compute_gamma_ip(z, x, gamma):
    """Wrapper function of Cython function 'compute_gamma_c'

    Computes Gamma_i = Z**(-1/2) * ( Z**(1/2) X X' Z**(1/2)) ** (1/2) * Z**(-1/2)
                   = V(E)**(-1/2)V' * ( V ((E)**(1/2)V' X X' V(E)**(1/2)) V')** (1/2) * V(E)**(-1/2)V'
                   = V(E)**(-1/2)V' * ( V (UDU') V')** (1/2) * V(E)**(-1/2)V'
                   = V (E)**(-1/2) U (D)**(1/2) U' (E)**(-1/2) V'

    Parameters
    ----------
    z : ndarray
        array of shape (dc, dc)
        auxiliary variable,  z_i
    x : ndarray
        auxiliary variable, x_i
    gamma : ndarray
        place where Gamma_i is updated
    """
    a = np.dot(x, x.T)
    compute_gamma_c(z, a, gamma)
    return


class REG_Data:
    """Data Container for regression problem

    Parameters
    ----------
    tstart : float
        Start of the TRF in seconds.
    tstop : float
        Stop of the TRF in seconds.
    nlevels : int
        Decides the density of Gabor atoms. Bigger nlevel -> less dense basis.
        By default it is set to `1`. `nlevesl > 2` should be used with caution.
    """
    _n_predictor_variables = 1
    _prewhitened = None

    def __init__(self, tstart, tstop, nlevel=1):
        if tstart != 0:
            raise NotImplementedError("tstart != 0 is not implemented")
        self.tstart = tstart
        self.tstop = tstop
        self.nlevel = nlevel
        self.meg = []
        self.covariates = []
        self.tstep = None
        self.filter_length = None
        self.basis = None
        self._norm_factor = None
        self._stim_sequence = None
        self._stim_dims = None

    def add_data(self, meg, stim):
        """Add sensor measurements and predictor variables for one trial

        Call this function repeatedly to add data for multiple trials/recordings

        Parameters
        ----------
        meg : NDVar  (sensor, UTS)
            MEG Measurements.
        stim : list of NDVar  ([...,] UTS)
            One or more predictor variable. The time axis needs to match ``y``.
        """
        meg_time = meg.get_dim('time')
        if self.tstep is None:
            # initialize time axis
            self.tstep = meg_time.tstep
            start = int(round(self.tstart / self.tstep))
            stop = int(round(self.tstop / self.tstep))
            self.filter_length = stop - start + 1
            # basis
            x = np.linspace(int(round(1000*self.tstart)), int(round(1000*self.tstop)), self.filter_length)
            self.basis = gaussian_basis(int(round((self.filter_length-1)/self.nlevel)), x)
            # stimuli
            if isinstance(stim, NDVar):
                self._stim_sequence = False
                stims = [stim]
            elif not isinstance(stim, (list, tuple)):
                raise TypeError(f"stim={stim!r}")
            else:
                stims = stim
                self._stim_sequence = True
            stim_dims = []
            for x in stims:
                if x.ndim == 1:
                    stim_dims.append(())
                elif x.ndim == 2:
                    dim, _ = x.get_dims((None, 'time'))
                    stim_dims.append(dim)
                else:
                    raise ValueError(f"stim={stim}: stimulus with more than 2 dimensions")
            self._stim_dims = tuple(stim_dims)
        elif meg_time.tstep != self.tstep:
            raise ValueError(f"meg={meg!r}: incompatible time-step with previously added data")
        else:
            stims = stim if self._stim_sequence else [stim]
            # check stimuli dimensions
            if len(stims) != len(self._stim_dims):
                raise ValueError(f"stim={stim!r}: different number of stimuli from previously added data")
            for dim, x in zip(self._stim_dims, stims):
                if dim is None:
                    assert x.dimnames == ('time',)
                else:
                    x_dim, _ = x.get_dims((None, 'time'))
                    if x_dim != dim:
                        raise ValueError(f"stim={stim!r}: dimension {dim} incompatible with previously added data")

        # check stimuli time axis
        for x in stims:
            if x.get_dim('time') != meg_time:
                raise ValueError(f"stim={stim!r}: time axis incompatible with meg")

        # add meg data
        y = meg.get_data(('sensor', 'time'))
        y = y[:, self.basis.shape[0]-1:].astype(np.float64)
        self.meg.append(y / sqrt(y.shape[1]))  # Mind the normalization

        if self._norm_factor is None:
            self._norm_factor = sqrt(y.shape[1])

        # add corresponding covariate matrix
        covariates = np.dot(covariate_from_stim(stim, self.filter_length),
                            self.basis) / sqrt(y.shape[1])  # Mind the normalization
        if covariates.ndim > 2:
            self._n_predictor_variables = covariates.shape[0]
            covariates = covariates.swapaxes(1, 0)

        first_dim = covariates.shape[0]
        x = covariates.reshape(first_dim, -1).astype(np.float64)
        self.covariates.append(x)

        return self

    def _prewhiten(self, whitening_filter):
        """Called by DstRF instance"""
        if self._prewhitened is None:
            for i, (meg, _) in enumerate(self):
                self.meg[i] = np.dot(whitening_filter, meg)
            self._prewhitened = True
        return self

    def _precompute(self):
        """Called by DstRF instance"""
        self._bbt = []
        self._bE = []
        self._EtE = []
        for b, E in self:
            self._bbt.append(np.dot(b, b.T))
            self._bE.append(np.dot(b, E))
            self._EtE.append(np.dot(E.T, E))

    def __iter__(self):
        return zip(self.meg, self.covariates)

    def __len__(self):
        return len(self.meg)

    def __repr__(self):
        return 'Regression data'

    def timeslice(self, idx):
        """gets a time slice (used for cross-validation)

        Parameters
        ----------
        idx : kfold splits
        Returns
        -------
            REG_Data instance
        """
        obj = type(self).__new__(self.__class__)
        # take care of the copied values from the old_obj
        copy_keys = ['_n_predictor_variables', 'basis', 'filter_length', 'tstart', 'tstep', 'tstop',
                     '_stim_dims', '_stim_sequence', '_prewhitened']
        for key in copy_keys:
            obj.__dict__.update({key: self.__dict__.get(key, None)})
        # keep track of the normalization
        obj._norm_factor = sqrt(len(idx))
        # add splitted data
        obj.meg = []
        obj.covariates = []
        # Dont forget to take care of the normalization here
        mul = self._norm_factor / obj._norm_factor  # multiplier to take care of the time normalization
        for meg, covariate in self:
            obj.meg.append(meg[:, idx] * mul)
            obj.covariates.append(covariate[idx, :] * mul)

        return obj


class DstRF:
    """The object-based API for cortical TRF localization

    Parameters
    ----------
    lead_field : NDVar
        forward solution a.k.a. lead_field matrix.
    noise_covariance : ndarray
        noise covariance matrix, use empty-room recordings to generate noise covariance
        matrix at sensor space.
    n_iter : int
        Number of out iterations of the algorithm, by default set to 10.
    n_iterc : int
        Number of Champagne iterations within each outer iteration, by default set to 30.
    n_iterf : int
        Number of FASTA iterations within each outer iteration, by default set to 100.

    Attributes
    ----------
    Gamma: list
        individual source covariance matrices
    sigma_b: list of ndarray
        data covariances under the model
    theta: ndarray
        trf coefficients over Gabor basis.

    Notes
    -----
    Usage:

        1. Initialize :class:`DstRF` instance with desired properties
        2. Initialize :class:`REG_Data` instance with desired properties
        2. Call :meth:`REG_Data.add_data` once for each contiguous segment of MEG
           data
        3. Call :meth:`DstRF.fit` with REG_Data instance to estimate the cortical TRFs.
        4. Call :meth:`get_strf` with REG_Data instance to retrieve the cortical TRFs.
    """
    _name = 'cTRFs estimator'
    _cv_info = None
    _crossvalidated = False

    def __init__(self, lead_field, noise_covariance, n_iter=30, n_iterc=10, n_iterf=100):
        if lead_field.has_dim('space'):
            g = lead_field.get_data(dims=('sensor', 'source', 'space')).astype(np.float64)
            self.lead_field = g.reshape(g.shape[0], -1)
            self.space = lead_field.get_dim('space')
        else:
            g = lead_field.get_data(dims=('sensor', 'source')).astype(np.float64)
            self.lead_field = g
            self.space = None

        self.source = lead_field.get_dim('source')
        self.sensor = lead_field.get_dim('sensor')
        self.noise_covariance = noise_covariance.astype(np.float64)
        self.n_iter = n_iter
        self.n_iterc = n_iterc
        self.n_iterf = n_iterf

        # self._init_vars()
        self._whitening_filter = None

    def _init_vars(self):
        wf = linalg.cholesky(self.noise_covariance, lower=True)
        Gtilde = linalg.solve(wf, self.lead_field)
        self.eta = (self.lead_field.shape[0] / np.trace(np.dot(Gtilde, Gtilde.T)))
        # model data covariance
        sigma_b = self.noise_covariance + self.eta * np.dot(self.lead_field, self.lead_field.T)
        self.init_sigma_b = sigma_b

    def __repr__(self):
        if self.space:
            orientation = 'free'
        else:
            orientation = 'fixed'
        out = "<[%s orientation] %s on %r>" % (orientation, self._name, self.source)
        return out

    def __copy__(self):
        obj = type(self).__new__(self.__class__)
        copy_keys = ['lead_field', 'lead_field_scaling', 'source', 'space', 'sensor', '_whitening_filter',
                     'noise_covariance', 'n_iter', 'n_iterc', 'n_iterf', 'eta', 'init_sigma_b']
        for key in copy_keys:
            obj.__dict__.update({key: self.__dict__.get(key, None)})
        return obj

    def _prewhiten(self):
        e, v = linalg.eigh(self.noise_covariance)
        wf = np.dot(v * _myinv(np.sqrt(e)), v.T.conj())
        self._whitening_filter = wf
        self.lead_field = np.dot(wf, self.lead_field)
        self.noise_covariance = np.eye(e.shape[0], dtype=np.float64)
        self.lead_field_scaling = linalg.norm(self.lead_field, 2)
        self.lead_field /= self.lead_field_scaling

        # pre compute some necessary initializations
        self.eta = (self.lead_field.shape[0] / np.sum(self.lead_field ** 2))
        # model data covariance
        sigma_b = self.noise_covariance + self.eta * np.dot(self.lead_field, self.lead_field.T)
        self.init_sigma_b = sigma_b
        return self

    def _init_iter(self, data):
        if self.space:
            dc = len(self.space)
        else:
            dc = 1

        self.Gamma = []
        self.Sigma_b = []
        for _ in range(len(data)):
            self.Gamma.append([self.eta * np.eye(dc, dtype=np.float64) for _ in range(len(self.source))])
            self.Sigma_b.append(self.init_sigma_b.copy())

        # initializing \Theta
        self.theta = np.zeros((len(self.source) * dc, data._n_predictor_variables *
                               data.basis.shape[1]),
                              dtype=np.float64)

        return self

    def _set_mu(self, mu, data):
        self.mu = mu
        self._init_iter(data)
        data._precompute()
        return self

    def _solve(self, data, theta, **kwargs):
        """Champagne steps implementation

        Implementation details can be found at:
        D. P. Wipf, J. P. Owen, H. T. Attias, K. Sekihara, and S. S. Nagarajan,
        “Robust Bayesian estimation of the location, orientation, and time course
        of multiple correlated neural sources using MEG,” NeuroImage, vol. 49,
        no. 1, pp. 641–655, 2010
        Parameters
        ----------
        data : REG_Data
            regression data to fit.
        theta : ndarray
            co-effecients of the TRFs over Gabor atoms.
        """
        # Choose dc
        if self.space:
            dc = len(self.space)
        else:
            dc = 1

        idx = kwargs.get('idx', slice(None, None))

        n_iterc = kwargs.get('n_iterc', self.n_iterc)

        for key, (meg, covariates) in enumerate(data):
            meg = meg[idx]
            covariates = covariates[idx]
            y = meg - np.dot(np.dot(self.lead_field, theta), covariates.T)
            Cb = np.dot(y, y.T)  # empirical data covariance
            yhat = linalg.cholesky(Cb, lower=True)
            gamma = self.Gamma[key].copy()
            sigma_b = self.Sigma_b[key].copy()

            # champagne iterations
            for it in range(n_iterc):
                # pre-compute some useful matrices
                Lc = linalg.cholesky(sigma_b, lower=True)
                lhat = linalg.solve(Lc, self.lead_field)
                ytilde = linalg.solve(Lc, yhat)

                # compute sigma_b for the next iteration
                sigma_b = self.noise_covariance.copy()

                for i in range(len(self.source)):
                    # update Xi
                    x = np.dot(gamma[i], np.dot(ytilde.T, lhat[:, i * dc:(i + 1) * dc]).T)

                    # update Zi
                    z = np.dot(lhat[:, i * dc:(i + 1) * dc].T, lhat[:, i * dc:(i + 1) * dc])

                    # update Ti
                    if dc == 1:
                        gamma[i] = sqrt(np.dot(x, x.T)) / np.real(sqrt(z))
                    elif dc == 3:
                            _compute_gamma_ip(z, x, gamma[i])
                    else:
                        gamma[i] = _compute_gamma_i(z, x)

                    # update sigma_b for next iteration
                    sigma_b += np.dot(self.lead_field[:, i * dc:(i + 1) * dc],
                                      np.dot(gamma[i], self.lead_field[:, i * dc:(i + 1) * dc].T))

            self.Gamma[key] = gamma
            self.Sigma_b[key] = sigma_b

        return self

    def fit(self, data, mu=None, do_crossvalidation=False, tol=1e-4, verbose=False, **kwargs):
        """cTRF estimator implementation

        Estimate both TRFs and source variance from the observed MEG data by solving
        the Bayesian optimization problem mentioned in the paper:
        P. Das, C. Brodbeck, J. Z. Simon, B. Babadi, Cortical Localization of the
        Auditory Temporal Response Function from MEG via Non-Convex Optimization;
        2018 Asilomar Conference on Signals, Systems, and Computers, Oct. 28–31,
        Pacific Grove, CA(invited).

        Parameters
        ----------
        data : REG_Data instance
            meg data and the corresponding stimulus variables
        mu : float
            regularization parameter,  promote temporal sparsity and provide guard against
            over-fitting
        do_crossvalidation : bool
            if True, from a wide range of regularizing parameters, the one resulting in
            the least generalization error in a k-fold cross-validation procedure is chosen.
            Unless specified the range and k is chosed from cofig.py. The user can also pass
            several keyword arguments to overwrite them.
        tol : float (1e-4 Default)
            tolerence parameter. Decides when to stop outer iterations.
        verbose : Boolean
            If set True prints intermediate values of the cost functions.
            by Default it is set to be False
        mus : list | ndarray
            range of mu to be considered for cross-validation
        n_splits : int
            k value used in k-fold cross-validation
        n_workers : int
            number of workers to be used for cross-validation
        """
        # pre-whiten the object itself
        if self._whitening_filter is None:
            self._prewhiten()
        # pre-whiten data
        if isinstance(data, REG_Data):
            data = data._prewhiten(self._whitening_filter)

        # take care of cross-validation
        if do_crossvalidation:
            mus = kwargs.get('mus', None)
            n_splits = kwargs.get('n_splits', None)
            n_workers = kwargs.get('n_workers', None)
            mu, cv_info = crossvalidate(self, data, mus, n_splits, n_workers)
            self._cv_info = cv_info
            self._crossvalidated = True
        else:
            # use the passed mu
            if mu is None:
                raise ValueError(f'Needs mu to be specified if do_crossvalidation is False!')

        self._set_mu(mu, data)

        if self.space:
            g_funct = lambda x: g(x, self.mu)
            prox_g = lambda x, t: shrink(x, self.mu * t)
        else:
            g_funct = lambda x: g_group(x, self.mu)
            prox_g = lambda x, t: proxg_group_opt(x, self.mu * t)

        theta = self.theta

        self.err = []
        if verbose:
            self.objective_vals = []
            start = time.time()

        # run iterations
        for i in (range(self.n_iter)):
            if verbose:
                print('iteration: %i:' % i)
            funct, grad_funct = self._construct_f(data)
            Theta = Fasta(funct, g_funct, grad_funct, prox_g, n_iter=self.n_iterf)
            Theta.learn(theta)

            self.err.append(self._residual(theta, Theta.coefs_))
            theta = Theta.coefs_
            self.theta = theta

            if verbose:
                print('objective after fasta: %10f' % self.eval_obj(data))

            if self.err[-1] < tol:
                break

            self._solve(data, theta, **kwargs)

            if verbose:
                self.objective_vals.append(self.eval_obj(data))
                print("objective value after champ:{:10f}\n "
                      "%% change:{:2f}".format(self.objective_vals[-1], self.err[-1]*100))

        if verbose:
            end = time.time()
            print("Time elapsed: {:10f} s".format(end - start))

        return self

    def _construct_f(self, data,):
        """creates instances of objective function and its gradient to be passes to the FASTA algorithm

        Parameters
        ---------
            data: REG_Data instance"""
        L = [linalg.cholesky(self.Sigma_b[i], lower=True) for i in range(len(data))]
        leadfields = [linalg.solve(L[i], self.lead_field) for i in range(len(data))]

        bEs = [linalg.solve(L[i], data._bE[i]) for i in range(len(data))]
        bbts = [np.trace(linalg.solve(L[i], linalg.solve(L[i], data._bbt[i]).T)) for i in range(len(data))]

        def f(L, x, bbt, bE, EtE):
            Lx = np.dot(L, x)
            y = bbt - 2 * np.sum(inner1d(bE, Lx)) + np.sum(inner1d(Lx, np.dot(Lx, EtE)))
            return 0.5 * y

        def gradf(L, x, bE, EtE):
            y = bE - np.dot(np.dot(L, x), EtE)
            return -np.dot(L.T, y)

        def funct(x):
            fval = 0.0
            for i in range(len((data))):
                fval += f(leadfields[i], x, bbts[i], bEs[i], data._EtE[i])
            return fval

        def grad_funct(x):
            grad = gradf(leadfields[0], x, bEs[0], data._EtE[0])
            # for trial, key in enumerate(self.keys[1:]):
            #     grad += gradf(leadfields[trial+1], x, bEs[trial+1], data._EtE[trial+1])
            for i in range(1, len(data)):
                grad += gradf(leadfields[i], x, bEs[i], data._EtE[i])
            return grad

        return funct, grad_funct

    def eval_obj(self, data):
        """evaluates objective function

        Parameters
        ---------
        data : REG_Data instance

        Returns
        -------
            float
        """
        v = 0
        for key, (meg, covariate) in enumerate(data):
            y = meg - np.dot(np.dot(self.lead_field, self.theta), covariate.T)
            L = linalg.cholesky(self.Sigma_b[key], lower=True)
            y = linalg.solve(L, y)
            v = v + 0.5 * (y ** 2).sum() + np.log(np.diag(L)).sum()

        return v / len(data)

    def eval_cv(self, data):
        """evaluates whole cross-validation metric (used by CV only)

        Parameters
        ---------
        data : REG_Data instance

        Returns
        -------
            float
        """
        v = 0
        for key, (meg, covariate) in enumerate(data):
            y = meg - np.dot(np.dot(self.lead_field, self.theta), covariate.T)
            L = linalg.cholesky(self.Sigma_b[key], lower=True)
            y = linalg.solve(L, y)
            v = v + 0.5 * (y ** 2).sum()  # + np.log(np.diag(L)).sum()

        return v / len(data)

    def eval_cv1(self, data):
        """evaluates Theta cross-validation metric (used by CV only)

        Parameters
        ---------
        data : REG_Data instance

        Returns
        -------
            float
        """
        v = 0
        for key, (meg, covariate) in enumerate(data):
            y = meg - np.dot(np.dot(self.lead_field, self.theta), covariate.T)
            # L = linalg.cholesky(self.Sigma_b[key], lower=True)
            # y = linalg.solve(L, y)
            v = v + 0.5 * (y ** 2).sum()  # + np.log(np.diag(L)).sum()

        return v / len(data)

    def get_strf(self, data):
        """Returns the learned spatio-temporal response function as NDVar

        Parameters
        ---------
        data : REG_Data instance

        Returns
        -------
            NDVar (TRFs)
        """
        trf = self.theta.copy()
        n_predictor_variables = len(data._stim_dims[0])
        if n_predictor_variables > 1:
            shape = (trf.shape[0], n_predictor_variables, -1)
            trf.shape = shape
            trf = trf.swapaxes(1, 0)

        # trf = np.tensordot(trf, data.basis.T, axes=1)
        trf = np.dot(trf, data.basis.T)

        time = UTS(data.tstart, data.tstep, trf.shape[-1])

        if self.space:
            dims = (self.source, self.space, time)
            dims = (data._stim_dims + dims)
            trf = trf.reshape(trf.shape[0], -1, len(self.space), trf.shape[-1])
        else:
            dims = (data._stim_dims + (self.source, time))

        trf = NDVar(trf, dims)
        return trf

    @staticmethod
    def _residual(theta0, theta1):
        diff = theta1 - theta0
        num = diff ** 2
        den = theta0 ** 2
        if den.sum() <= 0:
            return np.inf
        else:
            return sqrt(num.sum() / den.sum())

    @staticmethod
    def compute_ES_metric(models, data):
        """Computes estimation stability metric

        Details can be found at:
        Lim, Chinghway, and Bin Yu. "Estimation stability with cross-validation (ESCV)."
        Journal of Computational and Graphical Statistics 25.2 (2016): 464-492.

        Parameters
        ----------
        models : DstRf instances
        data : REG_Data instances

        Returns
        -------
            float (estimation stability metric)
        """
        Y = []
        for model in models:
            y = np.empty(0)
            for trial in range(len(data)):
                y = np.append(y, np.dot(np.dot(model.lead_field, model.theta), data.covariates[trial].T))
            Y.append(y)
        Y = np.array(Y)
        Y_bar = Y.mean(axis=0)
        VarY = (((Y - Y_bar) ** 2).sum(axis=1)).mean()
        if (Y_bar ** 2).sum() <= 0:
            return np.inf
        else:
            return VarY / (Y_bar ** 2).sum()

    def _get_cvfunc(self, data, n_splits):
        """Method for creating function for crossvalidation

        In the cross-validation phase the workers will call this function for
        for different regularizer parameters.

        Parameters
        ----------
        data : object
            the instance should be compatible for fitting the model. In addition to
            that it shall have a timeslice method compatible to kfold objects.

        n_splits : int
            number of folds for cross-validation, If None, it will use values
            specified in config.py.

        Returns
        -------
            callable, return the cross-validation metrics
        """
        models_ = [copy.copy(self) for _ in range(n_splits)]
        from sklearn.model_selection import KFold

        def cvfunc(mu):
            kf = KFold(n_splits=n_splits)
            ll = []
            ll1 = []
            ll2 = []
            thetas = []
            for model_, (train, test) in zip(models_, kf.split(data.meg[0][0])):
                traindata = data.timeslice(train)
                testdata = data.timeslice(test)
                model_.fit(traindata, mu, tol=1e-5, verbose=False)
                ll.append(model_.eval_cv(testdata))
                ll1.append(model_.eval_obj(testdata))
                ll2.append(model_.eval_cv1(testdata))
                thetas.append(model_.get_strf(data))

            time.sleep(0.001)
            # val1 = np.array(ll).mean()
            val1 = sum(ll) / len(ll)

            val2 = self.compute_ES_metric(models_, data)

            val3 = sum(ll1) / len(ll1)

            val4 = sum(ll2) / len(ll2)

            return {'cv': val1, 'es': val2, 'cv1': val3, 'cv2': val4}

        return cvfunc
