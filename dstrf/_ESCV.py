import cPickle as pickle
# import pickle
from eelbrain import *
from eelbrain._data_obj import SourceSpace
import numpy as np
# from math import ceil
from scipy import io, linalg
# from process_predictor_variable import covariate_matrix_from_mat_file, gaussian_basis
# from strf import estimate_STRF
from matplotlib import pyplot
from multiprocessing import Process, Manager, Array
from ._dstrf import covariate_from_stim, DstRF
from math import sqrt
from sklearn.model_selection import TimeSeriesSplit
import tqdm
import time


class DstRFcv(DstRF):
    def __init__(self, lead_field, noise_covariance, n_trials, mu, filter_length=200, n_iter=50, n_iterc=40, n_iterf=15, ES=True):
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
        self._teststim.append(np.dot (covariate_from_stim(stim, self.filter_length, normalize=normalize_regresor),
                                  self.basis)/sqrt(y.shape[1]))

    def setup_es(self, stim):
        self._es_stim.append(np.dot(covariate_from_stim(stim, self.filter_length, normalize=False),
                                     self.basis))

    def eval_model_fit(self):
        v = 0
        for trial in xrange(self.n_trials):
            y = self._testmeg[trial] - np.dot(np.dot(self.lead_field, self.theta), self._teststim[trial].T)
            Cb = np.dot(y, y.T)
            v = v + 0.5 * np.trace(linalg.solve(self.Sigma_b[trial], Cb)) \
                + 2 * np.sum(np.log(np.diag(linalg.cholesky(self.Sigma_b[trial]))))

        return v

    def setup_mp(self, manager):
        self.Sigma_b = manager.list(self.Sigma_b)
        self.Gamma = manager.list(self.Gamma)
        # self._ytilde = manager.list(self._ytilde)

    def setup_mp_for_theta(self):
        self.theta = Array('d', np.squeeze(self.theta.reshape(1, -1)))

    def reverse_mp_for_theta(self):
        if self.orientation == 'fixed': dc = 1
        elif self.orientation == 'free': dc = 3

        self.theta = np.array(self.theta).reshape((self.sources_n * dc, self.basis.shape[1]))

