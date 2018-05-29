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


def compute_ES_metric(Y):
    """
    Estimation Stability matric

    Ref: Lim, Chinghway, and Bin Yu. "Estimation stability with cross-validation (ESCV)."
    Journal of Computational and Graphical Statistics 25.2 (2016): 464-492.

    :param Y: (V,M) 2D array
    list containing pseudo estimates
    :return: ES_mu
    """
    Y = np.array(Y)
    Y_bar = np.mean(Y, axis=0)
    VarY = np.mean( np.array([linalg.norm(Y[i] - Y_bar) ** 2 for i in xrange(Y.shape[0])]) )

    return VarY / linalg.norm(Y_bar) ** 2


class DstRFcv(DstRF):
    def __init__(self, lead_field, noise_covariance, n_trials, filter_length=200, n_iter=50, n_iterc=40, n_iterf=15, ES=True):
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


def main():
    for subject_id in ['R2185']:
        print(subject_id)

        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-Prepare MEG data and Covariates-#-#-#-#-#-#-#-#-#-#-#-#
        subject_dir = 'G:/My Drive/Proloy/meg/' + subject_id

        ss = SourceSpace.from_file('G:/My Drive/Proloy/mri/', 'fsaverage', 'ico-4')

        ss_file = subject_dir + '/FWD ico-4-fixed.mat'
        # ss_file = subject_dir + '/FWD ico-4-free.mat'
        mat = io.loadmat(ss_file, squeeze_me=True)
        L = mat['lead_field']

        sensor_loc = np.kron(np.linspace (1, L.shape[0], L.shape[0]), [[1], [1], [1]])
        sensor_dim = Sensor(sensor_loc.T)

        lead_field = NDVar(L, (sensor_dim, ss))

        # Estimate noise covariance from empty room data
        er_file = subject_dir + '/emptyroom.mat'
        mat = io.loadmat(er_file, squeeze_me=True)
        er = np.array(mat['meg'])
        time_index = UTS(0, .001, er.shape[1])
        ER = NDVar(er, (sensor_dim, time_index))
        ER = filter_data(ER, 0.5, 40)
        ER = resample(ER, 200)
        er = ER.get_data(('sensor', 'time'))
        noise_cov = np.dot(er, er.T) / er.shape[1]

        tones_file = subject_dir + '/MEG tones'
        mat = io.loadmat(tones_file, squeeze_me=True)
        meg = mat['meg']
        time_index = UTS(0, 0.001, meg.shape[1])
        stim = np.zeros(meg.shape[1] / 5)
        data = NDVar(meg, (sensor_dim, time_index))
        data = filter_data(data, 0.5, 40)
        data = resample(data, 200)  # downsample to 200 Hz
        meg = data.get_data(('sensor', 'time'))

        # TRF params
        M = 200  # TRF length (for 200Hz this is a second of TRF)

        tones = mat['tones']
        # stim = np.zeros(meg.shape[1] / 5)
        stim[(np.round(tones / 5)).astype(int)] = 1
        # stim = NDVar(
        #     stim,
        #     UTS(0, 0.005, stim.shape[0])
        # )

        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-3-fold cross-validation-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

        n_trials = 1
        CVmetric = []
        ESmetric = []
        mu_range = np.flipud(np.logspace(3, 6, num=9))
        # mu_range = np.flipud(np.linspace(1e3, 1e6, endpoint=True, num=12))

        print mu_range[:]

        n_splits = 3
        mu = 0
        R = []
        manager = Manager()
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for train_index, test_index in tscv.split(stim):
            # Split data
            # X_train, X_test = meg[train_index], stim[test_index]
            X_train, y_train = meg[:, train_index], stim[train_index]
            # y_train, y_test = stim[train_index], stim[test_index]
            X_test, y_test = meg[:, test_index], stim[test_index]

            r = DstRFcv(lead_field, noise_cov, n_trials=n_trials)

            # Set up problem
            time_index = UTS(0, 0.005, y_train.shape[0])
            r.setup(
                NDVar(X_train, (sensor_dim, time_index)),
                NDVar(y_train, time_index),
                False
            )

            # Set up CV
            time_index = UTS(0, 0.005, y_test.shape[0])
            r.setup_cv(
                NDVar(X_test, (sensor_dim, time_index)),
                NDVar(y_test, time_index),
            )

            # Set up ES
            time_index = UTS(0, 0.005, stim.shape[0])
            r.setup_es(
                NDVar(stim, time_index),
            )

            r.setup_mp(manager)
            print 'r.n_iter = {:}'.format(r.n_iter)
            print 'r.n_iterf = {:}'.format(r.n_iterf)

            R.append(r)

        for mu in tqdm.tqdm(mu_range):
            time.sleep(0.001)
            jobs = []
            for i in xrange(n_splits):
                R[i].setup_mp_for_theta()
                # print 'index = {:} \n'.format(r.index.value)
                p = Process(target=R[i].fit, args=(mu, True, 1e-4))
                jobs.append(p)
                p.start()

            for processes in jobs:
                processes.join()

            for i in xrange(n_splits):
                R[i].reverse_mp_for_theta()

            CVmetric.append(
                sum(
                    [R[i].eval_model_fit() for i in xrange(n_splits)]
                )
            )
            print 'mu = {:f}, CV = {:f}'.format(mu, CVmetric[-1])

            ESmetric.append(
                compute_ES_metric(
                    [np.array(
                        [np.dot(np.dot(R[i].lead_field, R[i].theta), R[i]._es_stim[trial].T)
                         for trial in xrange(n_trials)]
                    ).ravel()
                     for i in xrange(n_splits)]
                )
            )

        out = {
            "mu": mu_range,
            "CV": CVmetric,
            "ES": ESmetric
        }
        pickle.dump(out, open(subject_dir + '/tone_response_cross-validation info.pickle', "wb"))

        # plot curves
        pyplot.subplot(1, 2, 1)
        pyplot.semilogx(mu_range, CVmetric)
        pyplot.title("Cross-validation metric")
        pyplot.subplot(1, 2, 2)
        pyplot.semilogx (mu_range, ESmetric)
        pyplot.title ("Estimation Stability metric")
        # pyplot.show()
        pyplot.savefig(subject_dir + '/cross-validation.pdf')

    return


if __name__=="__main__":
      # , 'R2079', 'R2085', 'R2130', 'R2135', 'R2148', 'R2153', 'R2185',
        # 'R2196', 'R2201', 'R2217', 'R2223', 'R2230', 'R2244', 'R2246', 'R2281', 'R2256']:  #
        main()
