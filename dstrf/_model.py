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
from eelbrain import fmtxt, UTS, NDVar
from eelbrain._utils import LazyProperty

from ._fastac import Fasta
from ._crossvalidation import CVResult, crossvalidate
from . import opt
from .dsyevh3C import compute_gamma_c

import logging

_R_tol = np.finfo(np.float64).eps


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
    r"""group (l12) norm  penalty:

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


def covariate_from_stim(stims, M):
    """Form covariate matrix from stimulus

    parameters
    ----------
    stims : list of NDVar
        Predictor variables.
    M : int
        Order of filter.

    returns
    -------
    x : ndarray
        Covariate matrix.
    """
    ws = []
    for stim in stims:
        if stim.ndim == 1:
            w = stim.get_data((np.newaxis, 'time'))
        else:
            dimnames = stim.get_dimnames(last='time')
            w = stim.get_data(dimnames)
        ws.append(w)
    w = ws[0] if len(ws) == 1 else np.concatenate(ws, 0)

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
    tol = _R_tol * x.max()
    ind = (x > tol)
    y = np.zeros(x.shape)
    y[ind] = 1 / x[ind]
    return y


def _inv_sqrtm(m):
    e, v = linalg.eigh(m)
    e = e.real
    tol = _R_tol * e.max()
    ind = (e > tol)
    y = np.zeros(e.shape)
    y[ind] = 1 / e[ind]
    return np.matmul(v * np.sqrt(y), v.T.conj())


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
    assert x.shape[0] == 3
    a = np.matmul(x, x.T)
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
    nlevel : int
        Decides the density of Gabor atoms. Bigger nlevel -> less dense basis.
        By default it is set to 1. ``nlevel > 2`` should be used with caution.
    baseline: list | None
        Mean that will be subtracted from ``stim``.
    scaling: list | None
        Scale by which ``stim`` was divided.
    """
    _n_predictor_variables = 1
    _prewhitened = None

    def __init__(self, tstart, tstop, nlevel=1, baseline=None, scaling=None, stim_is_single=None):
        if tstart != 0:
            raise NotImplementedError("tstart != 0 is not implemented")
        self.tstart = tstart
        self.tstop = tstop
        self.nlevel = nlevel
        self.s_baseline = baseline
        self.s_scaling = scaling
        self.meg = []
        self.covariates = []
        self.tstep = None
        self.filter_length = None
        self.basis = None
        self._norm_factor = None
        self._stim_is_single = stim_is_single
        self._stim_dims = None
        self._stim_names = None
        self.sensor_dim = None

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
        if self.sensor_dim is None:
            self.sensor_dim = meg.get_dim('sensor')
        elif meg.get_dim('sensor') != self.sensor_dim:
            raise NotImplementedError('combining data segments with different sensors is not supported')

        # check stim dimensions
        meg_time = meg.get_dim('time')
        stims = (stim,) if isinstance(stim, NDVar) else stim
        stim_dims = []
        for x in stims:
            if x.get_dim('time') != meg_time:
                raise ValueError(f"stim={stim!r}: time axis incompatible with meg")
            elif x.ndim == 1:
                stim_dims.append(None)
            elif x.ndim == 2:
                dim, _ = x.get_dims((None, 'time'))
                stim_dims.append(dim)
            else:
                raise ValueError(f"stim={stim}: stimulus with more than 2 dimensions")

        # stim normalization
        if self.s_baseline is not None:
            if len(self.s_baseline) != len(stims):
                raise ValueError(f"stim={stim!r}: incompatible with baseline={self.s_baseline!r}")
            for s, m in zip(stims, self.s_baseline):
                s -= m
        if self.s_scaling is not None:
            if len(self.s_scaling) != len(stims):
                raise ValueError(f"stim={stim!r}: incompatible with scaling={self.s_scaling!r}")
            for s, scale in zip(stims, self.s_scaling):
                s /= scale

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
            self._stim_dims = stim_dims
            self._stim_names = [x.name for x in stims]
        elif meg_time.tstep != self.tstep:
            raise ValueError(f"meg={meg!r}: incompatible time-step with previously added data")
        else:
            # check stimuli dimensions
            if stim_dims != self._stim_dims:
                raise ValueError(f"stim={stim!r}: dimensions incompatible with previously added data")

        # add meg data
        y = meg.get_data(('sensor', 'time'))
        y = y[:, self.basis.shape[0]-1:].astype(np.float64)
        self.meg.append(y / sqrt(y.shape[1]))  # Mind the normalization

        if self._norm_factor is None:
            self._norm_factor = sqrt(y.shape[1])

        # add corresponding covariate matrix
        covariates = np.dot(covariate_from_stim(stims, self.filter_length),
                            self.basis) / sqrt(y.shape[1])  # Mind the normalization
        if covariates.ndim > 2:
            self._n_predictor_variables = covariates.shape[0]
            covariates = covariates.swapaxes(1, 0)

        first_dim = covariates.shape[0]
        x = covariates.reshape(first_dim, -1).astype(np.float64)
        self.covariates.append(x)

    def _prewhiten(self, whitening_filter):
        """Called by DstRF instance"""
        if self._prewhitened is None:
            for i, (meg, _) in enumerate(self):
                self.meg[i] = np.dot(whitening_filter, meg)
            self._prewhitened = True

    def _precompute(self):
        """Called by DstRF instance"""
        self._bbt = []
        self._bE = []
        self._EtE = []
        for b, E in self:
            self._bbt.append(np.matmul(b, b.T))
            self._bE.append(np.matmul(b, E))
            self._EtE.append(np.matmul(E.T, E))

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
        copy_keys = ['_n_predictor_variables', 'basis', 'filter_length', 'tstart', 'tstep', 'tstop', '_stim_is_single',
                     '_stim_dims', '_stim_names', 's_baseline', 's_scaling', '_prewhitened']
        obj.__dict__.update({key: self.__dict__[key] for key in copy_keys})
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
    h : NDVar | tuple of NDVar
        The temporal response function. Whether ``h`` is an NDVar or a tuple of
        NDVars depends on whether the ``x`` parameter to :func:`dstrf` was
        an NDVar or a sequence of NDVars.
    Gamma: list
        individual source covariance matrices
    sigma_b: list of ndarray
        data covariances under the model
    theta: ndarray
        trf coefficients over Gabor basis.
    residual : float | NDVar
        The fit error, i.e. the result of the ``eval_obj`` error function on the
        final fit.
    stim_baseline: NDVar| float| list | None
        Mean that was subtracted from ``stim``.
    stim_scaling: NDVar| float| list | None
        Scale by which ``stim`` was divided.

    Notes
    -----
    Usage:

        1. Initialize :class:`DstRF` instance with desired properties
        2. Initialize :class:`REG_Data` instance with desired properties
        2. Call :meth:`REG_Data.add_data` once for each contiguous segment of MEG
           data
        3. Call :meth:`DstRF.fit` with REG_Data instance to estimate the cortical TRFs.
        4. Access the cortical TRFs in :attr:`DstRF.h`.
    """
    _name = 'cTRFs estimator'
    _cv_results = None
    # Attributes to be assigned after fit:
    _stim_is_single = None
    _stim_dims = None
    _stim_names = None
    _stim_baseline = None
    _stim_scaling = None
    _basis = None
    tstart = None
    tstep = None
    tstop = None
    residual = None

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

    _PICKLE_ATTRS = ('_basis', '_cv_results', 'mu',  '_name', '_stim_is_single', '_stim_dims', '_stim_names',
                     'noise_covariance', 'n_iter', 'n_iterc', 'n_iterf', 'lead_field',
                     '_stim_baseline', '_stim_scaling', 'lead_field_scaling', 'residual', 'source', 'space', 'theta', 'tstart', 'tstep', 'tstop')

    def __getstate__(self):
        return {k: getattr(self, k) for k in self._PICKLE_ATTRS}

    def __setstate__(self, state):
        for k in self._PICKLE_ATTRS:
            setattr(self, k, state.get(k, None))
        # make compatible with the previous version
        if self._cv_results is None:
            info = state.get('_cv_info')
            if info is not None:
                _cv_results = []
                for items in info:
                    if isinstance(items, np.ndarray):
                        for columns in items.T:
                            _cv_results.append(CVResult(*columns[[0, 1, 4, 2, 3]]))
                setattr(self, '_cv_results', _cv_results)

    def _prewhiten(self):
        wf = _inv_sqrtm(self.noise_covariance)
        self._whitening_filter = wf
        self.lead_field = np.matmul(wf, self.lead_field)
        self.noise_covariance = np.eye(self.lead_field.shape[0], dtype=np.float64)
        self.lead_field_scaling = linalg.norm(self.lead_field, 2)
        self.lead_field /= self.lead_field_scaling

        # pre compute some necessary initializations
        self.eta = (self.lead_field.shape[0] / np.sum(self.lead_field ** 2)) * 1e-2
        # model data covariance
        sigma_b = self.noise_covariance + self.eta * np.matmul(self.lead_field, self.lead_field.T)
        self.init_sigma_b = sigma_b

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
        self.theta = np.zeros((len(self.source) * dc, data._n_predictor_variables * data.basis.shape[1]),
                              dtype=np.float64)

    def _set_mu(self, mu, data):
        self.mu = mu
        self._init_iter(data)
        data._precompute()
        if mu == 0.0:
            self._solve(data, self.theta, n_iterc=30)

    def _solve(self, data, theta, idx=slice(None, None), n_iterc=None):
        """Champagne steps implementation

        Parameters
        ----------
        data : REG_Data
            regression data to fit.
        theta : ndarray
            co-effecients of the TRFs over Gabor atoms.

        Notes
        -----
        Implementation details can be found at:
        D. P. Wipf, J. P. Owen, H. T. Attias, K. Sekihara, and S. S. Nagarajan,
        “Robust Bayesian estimation of the location, orientation, and time course
        of multiple correlated neural sources using MEG,” NeuroImage, vol. 49,
        no. 1, pp. 641–655, 2010
        """
        logger = logging.getLogger('Champagne')
        # Choose dc
        if self.space:
            dc = len(self.space)
        else:
            dc = 1

        if n_iterc is None:
            n_iterc = self.n_iterc

        logger.debug('Champagne Iterations start:')
        logger.debug(f'trial \t time taken')
        for key, (meg, covariates) in enumerate(data):
            start = time.time()
            meg = meg[idx]
            covariates = covariates[idx]
            y = meg - np.matmul(np.matmul(self.lead_field, theta), covariates.T)
            Cb = np.matmul(y, y.T)  # empirical data covariance

            try:
                yhat = linalg.cholesky(Cb, lower=True)
            except np.linalg.LinAlgError:
                hi = y.shape[0] - 1
                lo = max(y.shape[0] - y.shape[1], 0)
                e, v = linalg.eigh(Cb, eigvals=(lo, hi))
                tol = e[-1] * _R_tol
                indices = e > tol
                yhat = v[:, indices] * np.sqrt(e[indices])

            gamma = self.Gamma[key].copy()
            sigma_b = self.Sigma_b[key].copy()

            # champagne iterations
            for it in range(n_iterc):
                # pre-compute some useful matrices
                try:
                    Lc = linalg.cholesky(sigma_b, lower=True)
                    lhat = linalg.solve(Lc, self.lead_field)
                    ytilde = linalg.solve(Lc, yhat)
                except np.linalg.LinAlgError:
                    Lc = _inv_sqrtm(sigma_b)
                    lhat = np.matmul(Lc, self.lead_field)
                    ytilde = np.matmul(Lc, yhat)

                # compute sigma_b for the next iteration
                sigma_b = self.noise_covariance.copy()

                for i in range(len(self.source)):
                    if dc > 1:
                        # update Xi
                        x = np.matmul(gamma[i], np.matmul(ytilde.T, lhat[:, i * dc:(i + 1) * dc]).T)
                        # update Zi
                        z = np.matmul(lhat[:, i * dc:(i + 1) * dc].T, lhat[:, i * dc:(i + 1) * dc])
                    else:
                        # update Xi
                        x = gamma[i] * np.matmul(ytilde.T, lhat[:, i]).T
                        # update Zi
                        z = inner1d(lhat[:, i], lhat[:, i])

                    # update Ti
                    if dc == 1:
                        gamma[i] = sqrt(inner1d(x, x)) / np.real(sqrt(z))
                    elif dc == 3:
                            _compute_gamma_ip(z, x, gamma[i])
                    else:
                        gamma[i] = _compute_gamma_i(z, x)

                    # update sigma_b for next iteration
                    sigma_b += np.dot(self.lead_field[:, i * dc:(i + 1) * dc],
                                      np.dot(gamma[i], self.lead_field[:, i * dc:(i + 1) * dc].T))

            self.Gamma[key] = gamma
            self.Sigma_b[key] = sigma_b
            end = time.time()
            logger.debug(f'{key} \t {end-start}')

    def fit(self, data, mu='auto', do_crossvalidation=False, tol=1e-5, verbose=False, use_ES=False, mus=None, n_splits=None, n_workers=None):
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
        use_ES : Boolean
            use estimation stability criterion _[1] to choose the best ``mu``. (False, by default)
        mus : list | ndarray | 'auto' (default)
            range of mu to be considered for cross-validation
        n_splits : int
            k value used in k-fold cross-validation
        n_workers : int
            number of workers to be used for cross-validation

        ..[1] Lim, Chinghway, and Bin Yu. "Estimation stability with cross-validation (ESCV)."
        Journal of Computational and Graphical Statistics 25.2 (2016): 464-492.
        """
        # if use_ES:
        #     raise NotImplementedError
        # pre-whiten the object itself
        logger = logging.getLogger(__name__)
        if self._whitening_filter is None:
            self._prewhiten()
        # pre-whiten data
        if isinstance(data, REG_Data):
            data._prewhiten(self._whitening_filter)

        # take care of cross-validation
        if do_crossvalidation:
            if mus == 'auto':
                mus = self._auto_mu(data)
            logger.info('Crossvalidation initiated!')
            cv_results = crossvalidate(self, data, mus, tol, n_splits, n_workers)
            best_cv = min(cv_results, key=attrgetter('cross_fit'))
            if best_cv.mu == min(mus):
                logger.info(f'CVmu is {best_cv.mu}: extending range of mu towards left')
                new_mus = np.logspace(np.log10(best_cv.mu) - 1, np.log10(best_cv.mu), 4)[:-1]
            elif best_cv.mu == max(mus):
                logger.info(f'CVmu is {best_cv.mu}: extending range of mu towards right')
                new_mus = np.logspace(np.log10(best_cv.mu), np.log10(best_cv.mu) + 1, 4)[1:]
            else:
                new_mus = None

            if new_mus is not None:
                cv_results.extend(crossvalidate(self, data, new_mus, tol, n_splits, n_workers))

            self._cv_results = cv_results
            best_cv = min(cv_results, key=attrgetter('cross_fit'))
            mu = best_cv.mu
            if use_ES:
                # FIXME
                cv_results_ = sorted(self._cv_results, key=attrgetter('mu'))
                if mu == cv_results[-1].mu:
                    logger.info(f'\nCVmu is {best_cv.mu}: could not find mu based on estimation' \
                                f' stability criterion'
                                f'\nContinuing with cross-validation only.')
                else:
                    best_es = None
                    for i, res in enumerate(cv_results_):
                        if res.mu < mu:
                            continue
                        else:
                            try:
                                if res.estimation_stability < cv_results_[i+1].estimation_stability:
                                    best_es = res
                                    break
                            except IndexError:
                                best_es = None
                    if best_es is None:
                        logger.warning(f'\nNo ES minima found: could not find mu based on estimation' 
                                       f' stability criterion. '
                                       f'\nContinuing with cross-validation only.')
                    else:
                        mu = best_es.mu

        else:
            # use the passed mu
            if mu is None:
                raise ValueError(f'mu needs mu to be specified if not \'auto\'')

        self._set_mu(mu, data)

        if self.space:
            g_funct = lambda x: g_group(x, self.mu)
            prox_g = lambda x, t: proxg_group_opt(x, self.mu * t)
        else:
            g_funct = lambda x: g(x, self.mu)
            prox_g = lambda x, t: shrink(x, self.mu * t)

        theta = self.theta

        myname = current_process().name

        self.err = []
        self.objective_vals = []
        if verbose:
            iter_o = tqdm(range(self.n_iter))
        else:
            iter_o = range(self.n_iter)

        logger.debug('process:iteration \t objective value \t %% change')
        # run iterations
        for i in iter_o:
            funct, grad_funct = self._construct_f(data)
            Theta = Fasta(funct, g_funct, grad_funct, prox_g, n_iter=self.n_iterf)
            Theta.learn(theta)

            self.err.append(self._residual(theta, Theta.coefs_))
            theta = Theta.coefs_
            self.theta = theta

            if self.err[-1] < tol:
                break

            self._solve(data, theta)

            self.objective_vals.append(self.eval_obj(data))

            logger.debug(f'{myname}:{i} \t {self.objective_vals[-1]} \t {self.err[-1]*100}')

        self.residual = self.eval_obj(data)
        self._stim_is_single = data._stim_is_single
        self._stim_dims = data._stim_dims
        self._stim_names = data._stim_names
        self._stim_baseline = data.s_baseline
        self._stim_scaling = data.s_scaling
        self._basis = data.basis
        self.tstart = data.tstart
        self.tstep = data.tstep
        self.tstop = data.tstop

    def _construct_f(self, data):
        """creates instances of objective function and its gradient to be passes to the FASTA algorithm

        Parameters
        ---------
        data : REG_Data
            Data.
        """
        leadfields = []
        bEs = []
        bbts = []
        for i in range(len(data)):
            try:
                L = linalg.cholesky(self.Sigma_b[i], lower=True)
                leadfields.append(linalg.solve(L, self.lead_field))
                bEs.append(linalg.solve(L, data._bE[i]))
                bbts.append(np.trace(linalg.solve(L, linalg.solve(L, data._bbt[i]).T)))
            except np.linalg.LinAlgError:
                Linv = _inv_sqrtm(self.Sigma_b[i])
                leadfields.append(np.matmul(Linv, self.lead_field))
                bEs.append(np.matmul(Linv, data._bE[i]))
                bbts.append(np.trace(np.matmul(Linv, np.matmul(Linv, data._bbt[i]).T)))

        def f(L, x, bbt, bE, EtE):
            Lx = np.matmul(L, x)
            y = bbt - 2 * np.sum(inner1d(bE, Lx)) + np.sum(inner1d(Lx, np.matmul(Lx, EtE)))
            return 0.5 * y

        def gradf(L, x, bE, EtE):
            y = bE - np.matmul(np.matmul(L, x), EtE)
            return -np.matmul(L.T, y)

        def funct(x):
            fval = 0.0
            for i in range(len(data)):
                fval += f(leadfields[i], x, bbts[i], bEs[i], data._EtE[i])
            return fval

        def grad_funct(x):
            grad = gradf(leadfields[0], x, bEs[0], data._EtE[0])
            for i in range(1, len(data)):
                grad += gradf(leadfields[i], x, bEs[i], data._EtE[i])
            return grad

        return funct, grad_funct

    def eval_obj(self, data):
        """evaluates objective function

        Parameters
        ---------
        data : REG_Data
            Data.

        Returns
        -------
            float
        """
        residual = 0
        for key, (meg, covariate) in enumerate(data):
            y = meg - np.matmul(np.matmul(self.lead_field, self.theta), covariate.T)
            L = linalg.cholesky(self.Sigma_b[key], lower=True)
            Cb = np.matmul(y, y.T)  # empirical data covariance
            try:
                yhat = linalg.cholesky(Cb, lower=True)
            except np.linalg.LinAlgError:
                hi = y.shape[0] - 1
                lo = max(y.shape[0] - y.shape[1], 0)
                e, v = linalg.eigh(Cb, eigvals=(lo, hi))
                tol = e[-1] * _R_tol
                indices = e > tol
                yhat = v[:, indices] * np.sqrt(e[indices])

            y = linalg.solve(L, yhat)
            residual += 0.5 * (y ** 2).sum() + np.log(np.diag(L)).sum()

        return residual / len(data)

    def eval_wl2(self, data):
        """evaluates whole cross-validation metric (used by CV only)

        Parameters
        ---------
        data : REG_Data instance

        Returns
        -------
            float
        """
        ll = 0
        for key, (meg, covariate) in enumerate(data):
            y = meg - np.matmul(np.matmul(self.lead_field, self.theta), covariate.T)
            L = linalg.cholesky(self.Sigma_b[key], lower=True)
            Cb = np.matmul(y, y.T)  # empirical data covariance
            try:
                yhat = linalg.cholesky(Cb, lower=True)
            except np.linalg.LinAlgError:
                hi = y.shape[0] - 1
                lo = max(y.shape[0] - y.shape[1], 0)
                e, v = linalg.eigh(Cb, eigvals=(lo, hi))
                tol = e[-1] * _R_tol
                indices = e > tol
                yhat = v[:, indices] * np.sqrt(e[indices])

            y = linalg.solve(L, yhat)
            ll += 0.5 * (y ** 2).sum()  # + np.log(np.diag(L)).sum()

        return ll / len(data)

    def eval_l2(self, data):
        """evaluates Theta cross-validation metric (used by CV only)

        Parameters
        ---------
        data : REG_Data instance

        Returns
        -------
            float
        """
        l2 = 0
        for key, (meg, covariate) in enumerate(data):
            y = meg - np.matmul(np.matmul(self.lead_field, self.theta), covariate.T)
            l2 = l2 + 0.5 * (y ** 2).sum()  # + np.log(np.diag(L)).sum()

        return l2 / len(data)

    @LazyProperty
    def h_scaled(self):
        """h with original stimulus scale restored"""
        if self._stim_scaling is None:
            return self.h
        elif self._stim_is_single:
            return self.h * self._stim_scaling[0]
        else:
            return [h * s for h, s in zip(self.h, self._stim_scaling)]

    @LazyProperty
    def h(self):
        """The spatio-temporal response function as (list of) NDVar"""
        n_vars = sum(len(dim) if dim else 1 for dim in self._stim_dims)
        if n_vars > 1:
            shape = (self.theta.shape[0], n_vars, -1)
            trf = self.theta.reshape(shape)
            trf = trf.swapaxes(1, 0)
        else:
            trf = self.theta[np.newaxis, :]

        trf = np.dot(trf, self._basis.T) / self.lead_field_scaling

        time = UTS(self.tstart, self.tstep, trf.shape[-1])
        if self.space:
            shared_dims = (self.source, self.space, time)
        else:
            shared_dims = (self.source, time)
        trf = trf.reshape((-1, *(map(len, shared_dims))))

        h = []
        i = 0
        for dim, name in zip(self._stim_dims, self._stim_names):
            if dim:
                dims = (dim, *shared_dims)
                i1 = i + len(dim)
                x = trf[i: i1]
                i = i1
            else:
                dims = shared_dims
                x = trf[i]
                i += 1
            h.append(NDVar(x, dims, name=name))

        if self._stim_is_single:
            return h[0]
        else:
            return h

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
                y = np.append(y, np.matmul(np.matmul(model.lead_field, model.theta), data.covariates[trial].T))
            Y.append(y)
        Y = np.array(Y)
        Y_bar = Y.mean(axis=0)
        VarY = (((Y - Y_bar) ** 2).sum(axis=1)).mean()
        if (Y_bar ** 2).sum() <= 0:
            return np.inf
        else:
            return VarY / (Y_bar ** 2).sum()

    def cvfunc(self, data, n_splits, tol, mu):
        cvfun = self._get_cvfunc(data, n_splits, tol)
        return cvfun(mu)

    def _get_cvfunc(self, data, n_splits, tol):
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
        tol : float
            tolerence parameter. Decides when to stop outer iterations.
        Returns
        -------
            callable, return the cross-validation metrics
        """
        models_ = [copy.copy(self) for _ in range(n_splits)]
        # from sklearn.model_selection import KFold
        from ._crossvalidation import TimeSeriesSplit

        def cvfunc(mu: float) -> CVResult:
            # kf = KFold(n_splits=n_splits)
            kf = TimeSeriesSplit(r=0.05, p=n_splits, d=data.basis.shape[1])
            ll = []
            ll1 = []
            ll2 = []
            for model_, (train, test) in zip(models_, kf.split(data.meg[0][0])):
                traindata = data.timeslice(train)
                testdata = data.timeslice(test)
                model_.fit(traindata, mu, tol=tol, verbose=False)
                ll.append(model_.eval_wl2(testdata))
                ll1.append(model_.eval_obj(testdata))
                ll2.append(model_.eval_l2(testdata))

            time.sleep(0.001)
            return CVResult(
                mu,
                sum(ll) / len(ll),  # weighted_l2_error
                self.compute_ES_metric(models_, data),  # estimation_stability
                sum(ll1) / len(ll1),  # cross_fit
                sum(ll2) / len(ll2),  # l2_error
            )

        return cvfunc

    def _auto_mu(self, data, p=99.0):
        self._set_mu(0.0, data)
        _, grad_funct = self._construct_f(data)
        if self.space:
            x = grad_funct(self.theta)
            l = x.shape[1]
            x.shape = (-1, 3, l)
            norm = np.linalg.norm(x, axis=1)
        else:
            x = grad_funct(self.theta)
            norm = np.abs(x)

        hi = log10(np.percentile(norm, p))
        lo = hi - 2
        return np.logspace(lo, hi, 7)

    def cv_info(self):
        if self._cv_results is None:
            raise ValueError(f"CV: no cross-validation was performed. Use mu='auto' to perform cross-validation.")
        cv_results = sorted(self._cv_results, key=attrgetter('mu'))
        criteria = ('cross-fit', 'l2/mu')
        best_mu = {criterion: self.cv_mu(criterion) for criterion in criteria}

        table = fmtxt.Table('lllll')
        table.cells('mu', 'cross-fit', 'l2-error', 'weighted l2-error', 'ES metric')
        table.midrule()
        fmt = '%.5f'
        for result in cv_results:
            table.cell(fmtxt.stat(result.mu, fmt=fmt))
            star = 1 if result.mu is best_mu['cross-fit'] else 0
            table.cell(fmtxt.stat(result.cross_fit, fmt, star, 1))
            star = 1 if result.mu is best_mu['l2/mu'] else 0
            table.cell(fmtxt.stat(result.l2_error, fmt, star, 1))
            table.cell(fmtxt.stat(result.weighted_l2_error, fmt=fmt))
            table.cell(fmtxt.stat(result.estimation_stability, fmt=fmt))
        # warnings
        mus = [res.mu for res in self._cv_results]
        warnings = []
        if self.mu == min(mus):
            warnings.append(f"Best mu is smallest mu")
        if warnings:
            table.caption(f"Warnings: {'; '.join(warnings)}")
        return table

    def cv_mu(self, criterion='cross-fit'):
        """Retrieve best mu based on cross-validation

        Parameters
        ----------
        criterion : str
            Criterion for best fit. Possible values:

            - ``'cross-fit'``: The smallest cross-fit value (default)
            - ``'l2'``: The smallest l2 error
            - ``'l2/mu'``: The local minimum in the l2 error with smallest mu
        """
        if criterion == 'cross-fit':
            best_cv = min(self._cv_results, key=attrgetter('cross_fit'))
        elif criterion == 'l2':
            best_cv = min(self._cv_results, key=attrgetter('l2_error'))
        elif criterion == 'l2/mu':
            cv_results = sorted(self._cv_results, key=attrgetter('mu'))
            peaks, _ = find_peaks([-result.l2_error for result in cv_results])  # find local minima
            if len(peaks) > 0:
                # higher mu -> smaller trf
                best_cv = max([cv_results[peak] for peak in peaks], key=attrgetter('mu'))
            else:
                best_cv = min(cv_results, key=attrgetter('l2_error'))
        else:
            raise ValueError(f'criterion={criterion}')
        return best_cv.mu
