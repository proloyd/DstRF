import numpy as np
from scipy import linalg
from eelbrain import *
from math import sqrt
from ._basis import gaussian_basis
from ._fastac import Fasta
import time

orientation = {'fixed': 1, 'free': 3}


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


class REG_Data:
    """Data Container for regression problem

    """
    _n_predictor_variables = 1

    def __init__(self, filter_length=200):
        self.filter_length = filter_length
        x = np.linspace(5, 1000, self.filter_length)
        self.basis = gaussian_basis(self.filter_length, x)
        self.covariates = dict()
        self.meg = dict()
        self.datakeys = []
        self.tstep = None
        self._norm_factor = None

    def load(self, key, meg, stim, normalize_regresor=True, verbose=0):
        """

        :param meg:
        :param verbose:
        :return:
        """
        # check if time lengths are same or not
        # skip for now

        self.datakeys.append(key)

        if self.tstep is None:
            self.tstep = meg.time.tstep

        # add meg data
        y = meg.get_data(('sensor', 'time'))
        y = y[:, self.basis.shape[1]:].astype('float64')
        self.meg[key] = y / sqrt(y.shape[1])  # Mind the normalization

        if self._norm_factor is None:
            self._norm_factor = sqrt(y.shape[1])

        # add corresponding covariate matrix
        covariates = np.dot(covariate_from_stim(stim, self.filter_length, normalize=normalize_regresor),
                            self.basis) / sqrt(y.shape[1])
        if covariates.ndim > 2:
            self._n_predictor_variables = covariates.shape[0]
            covariates = covariates.swapaxes(1, 0)

        first_dim = covariates.shape[0]
        self.covariates[key] = covariates.reshape(first_dim, -1).astype('float64')

        return self

    def __iter__(self):
        return ((self.meg[key], self.covariates[key], key) for key in self.datakeys)

    def __len__(self):
        return len(self.datakeys)

    def timeslice(self, idx):
        """

        idx: kfold splits
        :return:
        """
        regdata_ = REG_Data(self.filter_length)
        regdata_.datakeys = self.datakeys
        regdata_._n_predictor_variables = self._n_predictor_variables
        regdata_._norm_factor = sqrt(len(idx))
        for key in regdata_.datakeys:
            regdata_.meg[key] = self.meg[key][:, idx] * self._norm_factor / regdata_._norm_factor
            regdata_.covariates[key] = self.covariates[key][idx, :]

        return regdata_


class DstRF_new:
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

    def __init__(self, lead_field, noise_covariance, n_iter=30, n_iterc=1000, n_iterf=1000):
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
        self.n_iter = n_iter
        self.n_iterc = n_iterc
        self.n_iterf = n_iterf

        self.__init__vars()
        self._init_Sigma_b = None
        self._init_Gamma = None

    def __init__vars(self):
        wf = linalg.cholesky(self.noise_covariance, lower=True)
        Gtilde = linalg.solve(wf, self.lead_field)
        self.eta = (self.lead_field.shape[0] / np.trace(np.dot(Gtilde, Gtilde.T)))
        # model data covariance
        sigma_b = self.noise_covariance + self.eta * np.dot(self.lead_field, self.lead_field.T)
        self.init_sigma_b = sigma_b
        return self

    def __init__iter(self, data):
        dc = orientation[self.orientation]
        if self._init_Gamma is None:
            self.Gamma = {}
            self.Sigma_b = {}
            for key in data.datakeys:
                self.Gamma[key] = [self.eta * np.eye(dc, dtype='float64') for _ in range(self.sources_n)]
                self.Sigma_b[key] = self.init_sigma_b.copy()

            self.keys = data.datakeys.copy()
            # initializing \Theta
            self.theta = np.zeros((self.sources_n * dc, data._n_predictor_variables *
                                   data.basis.shape[1]),
                                  dtype='float64')

            self._solve(data, self.theta, n_iterc=200)
            self._init_Gamma = self.Gamma.copy()
            self._init_Sigma_b = self.Sigma_b.copy()

        else:
            print('Initializations already exists!')
            self.Gamma = self._init_Gamma.copy()
            self.Sigma_b = self._init_Sigma_b.copy()
            # initializing \Theta
            self.theta = np.zeros((self.sources_n * dc, data._n_predictor_variables *
                                   data.basis.shape[1]),
                                  dtype='float64')
        return self

    def _set_mu(self, mu, data):
        self.mu = mu
        self.__init__iter(data)
        return self

    def _solve(self, data, theta, **kwargs):
        """

        :param theta:
        :param trial:
        :param verbose:
        :return:
        """
        # Choose dc
        dc = orientation[self.orientation]

        idx = kwargs.get('idx', slice(None, None))

        n_iterc = kwargs.get('n_iterc', self.n_iterc)

        for meg, covariates, key in data:
            meg = meg[idx]
            covariates = covariates[idx]
            y = meg - np.dot(np.dot(self.lead_field, theta), covariates.T)
            Cb = np.dot(y, y.T)  # empirical data covariance
            yhat = linalg.cholesky(Cb, lower=True)
            gamma = self.Gamma[key]
            sigma_b = self.Sigma_b[key].copy()

            # champagne iterations
            for it in range(n_iterc):
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

            self.Gamma[key] = gamma
            self.Sigma_b[key] = sigma_b

        return self

    def fit(self, data, mu, tol=1e-4, verbose=False, **kwargs):
        """

        :return:

        """
        idx = kwargs.get('idx', None)
        if idx is not None:
            data = data.timeslice(idx)

        self._set_mu(mu, data)

        if self.orientation == 'fixed':
            g_funct = lambda x: g(x, self.mu)
            prox_g = lambda x, t: shrink(x, self.mu * t)
        elif self.orientation == 'free':
            g_funct = lambda x: g_group(x, self.mu)
            prox_g = lambda x, t: proxg_group(x, self.mu * t)

        theta = self.theta

        self.err = []
        if verbose:
            self.objective_vals = []
            start = time.time()

        # run iterations
        for i in (range(self.n_iter)):
            funct, grad_funct = self._construct_f(data, **kwargs)
            Theta = Fasta(funct, g_funct, grad_funct, prox_g, n_iter=self.n_iterf)
            Theta.learn(theta)

            # import pdb
            # pdb.set_trace()

            self.err.append(self._residual(theta, Theta.coefs_))
            theta = Theta.coefs_
            self.theta = theta

            if self.err[-1] < tol:
                break

            self._solve(data, theta, **kwargs)

            if verbose:
                self.objective_vals.append(self.eval_obj(data))
                print("Iteration: {:}, objective value:{:10f}, "
                      "%% change:{:2f}".format(i, self.objective_vals[-1], self.err[-1]*100))

        if verbose:
            end = time.time()
            print("Time elapsed: {:10f} s".format(end - start))

        return self

    def _construct_f(self, data):
        L = [linalg.cholesky(self.Sigma_b[key], lower=True) for key in self.keys]
        leadfields = [linalg.solve(L[trial], self.lead_field) for trial in range(len(self.keys))]
        megs = [linalg.solve(L[trial], data.meg[key]) for trial, key in enumerate(data.datakeys)]

        def f(L, x, b, E):
            y = b - np.dot(np.dot(L, x), E.T)

            return 0.5 * (y ** 2).sum()

        def gradf(L, x, b, E):
            y = b - np.dot(np.dot(L, x), E.T)

            return -np.dot(L.T, np.dot(y, E))

        def funct(x):
            val = 0.0
            for trial, key in enumerate(self.keys):
                val += f(leadfields[trial], x, megs[trial], data.covariates[key])

            return val

        def grad_funct(x):
            grad = gradf(leadfields[0], x, megs[0], data.covariates[self.keys[0]])
            for trial, key in enumerate(self.keys[1:]):
                grad += gradf(leadfields[trial], x, megs[trial], data.covariates[key])

            return grad

        return funct, grad_funct

    def eval_obj(self, data):
        v = 0
        for meg, covariate, key in data:
            y = meg - np.dot(np.dot(self.lead_field, self.theta), covariate.T)
            L = linalg.cholesky(self.Sigma_b[key], lower=True)
            y = linalg.solve(L, y)
            v = v + 0.5 * (y ** 2).sum() + np.log(np.diag(L)).sum()

        return v / len(data)

    def eval_cv(self, data):
        v = 0
        for meg, covariate, key in data:
            y = meg - np.dot(np.dot(self.lead_field, self.theta), covariate.T)
            L = linalg.cholesky(self.Sigma_b[key], lower=True)
            y = linalg.solve(L, y)
            v = v + 0.5 * (y ** 2).sum()  # + np.log(np.diag(L)).sum()

        return v / len(data)

    def get_strf(self, data):
        """

        Updates the spatio-temporal response function as NDVar


        returns
        -------
            self
        """
        trf = self.theta.copy()
        if data._n_predictor_variables > 1:
            shape = (trf.shape[0], 3, -1)
            trf.shape = shape
            trf = trf.swapaxes(1, 0)

        # trf = np.dot(self.basis, self.theta.T).T
        trf = np.dot(trf, data.basis.T)

        time = UTS(0, data.tstep, trf.shape[-1])

        if self.orientation == 'fixed':
            if data._n_predictor_variables > 1:
                dims = (Case, self.source, time)
            else:
                dims = (self.source, time)
            trf = NDVar(trf, dims)

        elif self.orientation == 'free':
            dims = (time, self.source, self.space)
            if data._n_predictor_variables > 1:
                trfs = []
                for i in range(data._n_predictor_variables):
                    trfs.append(NDVar(trf[i, :, :].T.reshape(-1, self.sources_n, 3), dims))
                trf = combine(trfs)
            else:
                trf = NDVar(trf.T.reshape(-1, self.sources_n, 3), dims)

        return trf

    @staticmethod
    def _residual(theta0, theta1):
        # import pdb
        # pdb.set_trace()
        diff = theta1 - theta0
        num = diff ** 2
        den = theta0 ** 2
        return sqrt(num.sum() / den.sum())

    @staticmethod
    def compute_ES_metric(models, data):
        """
        Estimation Stability matric

        Ref: Lim, Chinghway, and Bin Yu. "Estimation stability with cross-validation (ESCV)."
        Journal of Computational and Graphical Statistics 25.2 (2016): 464-492.

        :param models: DstRfCv instances
        list containing pseudo estimates
        :return: ES_mu
        """
        Y = []
        for model in models:
            y = np.empty(0)
            for trial, key in enumerate(data.datakeys):
                y = np.append(y, np.dot(np.dot(model.lead_field, model.theta), data.covariates[key].T))
            Y.append(y)
        Y = np.array(Y)
        Y_bar = Y.mean(axis=0)
        VarY = (((Y - Y_bar) ** 2).sum(axis=1)).mean()

        return VarY / (Y_bar ** 2).sum()

if __name__ == '__main__':
    from scipy import io
    import pickle

    ROOTDIR = '/mnt/c/Users/proloy/csslz/'
    STIM = ['h', 'l']
    n_Trials = 3
    # FEATURE = 'audspec-1'
    # FEATURE = 'word-log10wf'
    # FEATURE = 'word-comb'
    FEATURES = [
        'audspec-1',
        'word-log10wf',
        'word-comb'
    ]

    subject_id = 'R1801'
    subject_dir = ROOTDIR + 'meg/' + subject_id

    fname = ROOTDIR + 'fwdsol/' + '+3h-%s-vol-7-fwd.fif' % subject_id
    lead_field = load_fwd_from_mne(fname, 'vol-7', ss)
    sensor_dim = lead_field.sensor

    # noise covariance
    print("estimating noise covariance")
    er_file = subject_dir + '/emptyroom.mat'
    mat = io.loadmat(er_file, squeeze_me=True)
    er = np.array(mat['meg'])
    time_index = UTS(0, .001, er.shape[1])
    ER = NDVar(er, (sensor_dim, time_index))
    ER = filter_data(ER, 0.5, 90, fir_design='firwin')
    ER = resample(ER, 200)
    er = ER.get_data(('sensor', 'time'))
    noise_cov = np.dot(er, er.T) / er.shape[1]

    R = DstRF(lead_field, noise_cov, n_Trials * len(STIM), n_iter=25, n_iterc=5, n_iterf=20)

    print('Packing data')
    for st in STIM:
        stims = []
        for FEATURE in FEATURES:
            stim_file = ROOTDIR + 'predictors/' + "quiet_%s%s 200Hz.pickled" % (st, FEATURE)
            stim = pickle.load(open(stim_file, 'rb'), encoding='latin1')
            stims.append(stim)
        stims = combine(stims)
        # %%% normalize regressor:
        stims -= stims.mean('time')
        stims /= stims.std('time')
        for trial in range(n_Trials):
            key = '%s%i' % (st, trial)
            meg_file = subject_dir + '/MEG quiet_' + st + '-' + '%s' % trial + '.mat'
            mat = io.loadmat(meg_file, squeeze_me=True)
            Y = np.array(mat['meg'])
            time_index = UTS(0, .001, Y.shape[1])
            y = NDVar(Y, (sensor_dim, time_index))
            y = filter_data(y, 0.5, 90, fir_design="firwin2")
            data = resample(y, 200)
            R.load(key, data, stims, False)
    print('Packing Done')
