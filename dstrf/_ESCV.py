from eelbrain import *
import numpy as np
from scipy import linalg
from ._dstrf import covariate_from_stim, DstRF
from math import sqrt


class DstRFcv(DstRF):
    def __init__(self, lead_field, noise_covariance, n_trials, filter_length=200, n_iter=50, n_iterc=5, n_iterf=20, ES=True):
        # DstRF.__init__(self, lead_field, noise_covariance, n_trials, filter_length=filter_length,
        #                n_iter=n_iter, n_iterf=n_iterf)
        DstRF.__init__(self, lead_field, noise_covariance, n_trials, filter_length=filter_length,
                       n_iter=n_iter, n_iterc=n_iterc, n_iterf=n_iterf)
        self._testmeg = []
        self._teststim = []
        if ES:
            self._es_stim = []

    def setup_cv(self, meg, stim, normalize_regresor=False):
        y = meg.get_data(('sensor', 'time'))
        y = y[:, self.basis.shape[1]:]
        self._testmeg.append(y/sqrt(y.shape[1]))
        self._teststim.append(np.dot(covariate_from_stim(stim, self.filter_length, normalize=normalize_regresor),
                              self.basis)/sqrt(y.shape[1]))

    def setup_es(self, stim):
        self._es_stim.append(np.dot(covariate_from_stim(stim, self.filter_length, normalize=False),
                             self.basis))

    def set_mu(self, mu):
        self.mu = mu
        return self

    def eval_model_fit(self):
        v = 0
        for trial in range(self.n_trials):
            y = self._testmeg[trial] - np.dot(np.dot(self.lead_field, self.theta), self._teststim[trial].T)
            Cb = np.dot(y, y.T)
            v = v + 0.5 * np.trace(linalg.solve(self.Sigma_b[trial], Cb)) \
                + 2 * np.sum(np.log(np.diag(linalg.cholesky(self.Sigma_b[trial]))))

        return v

    def setup_from_self(self, obj):
        import copy
        self._meg = obj._meg
        self._covariates = obj._covariates
        self.Gamma = copy.deepcopy(obj.Gamma)
        self.Sigma_b = copy.deepcopy(obj.Sigma_b)
        self.theta = copy.deepcopy(obj.theta)
        self._testmeg = obj._testmeg
        self._teststim = obj._teststim
        self._es_stim = obj._es_stim

    def get_lead_field(self):
        if self.orientation == 'free':
            dims = (self.sensor, self.source, self.space)
            lead_field = NDVar(self.lead_field.reshape(self.lead_field.shape[0], -1, 3), dims=dims)
        elif self.orientation == 'fixed':
            dims = (self.sensor, self.source)
            lead_field = NDVar(self.lead_field, dims=dims)
        else:
            print('Oops! Couldn\'t determine lead-field orientation')

        return lead_field


def compute_ES_metric(models):
    """
    Estimation Stability matric

    Ref: Lim, Chinghway, and Bin Yu. "Estimation stability with cross-validation (ESCV)."
    Journal of Computational and Graphical Statistics 25.2 (2016): 464-492.

    :param models: DstRfCv instances
    list containing pseudo estimates
    :return: ES_mu
    """
    Y = np.array ([np.dot(model.lead_field, model.theta).ravel() for model in models])
    Y_bar = np.mean (Y, axis=0)
    VarY = linalg.norm(Y - Y_bar, 'fro') ** 2

    return VarY / linalg.norm (Y_bar) ** 2


