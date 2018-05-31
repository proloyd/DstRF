import cPickle as pickle
import sys
from eelbrain import *
from eelbrain._data_obj import SourceSpace
import numpy as np
# from math import ceil
from scipy import io, linalg
from matplotlib import pyplot
import multiprocessing
import Queue as queue
from multiprocessing.managers import SyncManager
import time
from dstrf import DstRFcv
from sklearn.model_selection import TimeSeriesSplit

IP = '128.8.76.2'
PORTNUM = 8080
AUTHKEY = b'shufflin'


def make_server_manager(port, authkey):
    job_q = queue.Queue()
    result_q = queue.Queue()

    class JobQueueManager(SyncManager):
        pass

    JobQueueManager.register('get_job_q', callable=lambda: job_q)
    JobQueueManager.register('get_result_q', callable=lambda: result_q)

    manager = JobQueueManager(address=('', port), authkey=authkey)
    manager.start()
    print('Server started at port %s' % port)
    return manager


def make_client_manager(ip, port, authkey):
    class ServerQueueManager(SyncManager):
        pass

    ServerQueueManager.register('get_job_q')
    ServerQueueManager.register('get_result_q')

    print('Client Trying to connect: %s:%s' % (ip, port))
    manager = ServerQueueManager(address=(ip, port), authkey=authkey)
    manager.connect()

    print('Client connected to %s:%s' % (ip, port))
    return manager


def fit_worker(job_q, result_q):
    myname = multiprocessing.current_process().name
    while True:
        try:
            job = job_q.get_nowait()
            print('%s got %s jobs...' % (myname, len(job)))
            outlist = [R.fit() for R in job]
            result_q.put(outlist)
            print('  %s done' % myname)
        except queue.Empty:
            return


def mp_fit(shared_job_q, shared_result_q, nprocs):
    procs = []
    for i in range(nprocs):
        p = multiprocessing.Process(
                target=fit_worker,
                args=(shared_job_q, shared_result_q))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()


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


def runserver(subject_id):
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

    # Set up the class objects
    print('Setting up the class objects')
    n_trials = 1
    mu_range = np.flipud(np.logspace(3, 6, num=9))
    # mu_range = np.flipud(np.linspace(1e3, 1e6, endpoint=True, num=12))

    print mu_range[:]

    n_splits = 3
    mu = 0
    R = []
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

    RR = [list(R) for _ in mu_range]
    RR = reduce (lambda x, y: x + y, RR)
    N = len(RR)

    for counter, mu in enumerate(mu_range):
        for i in xrange(n_splits):
            RR[counter * n_splits + i].set_mu(mu)

    manager = make_server_manager(PORTNUM, AUTHKEY)
    shared_job_q = manager.get_job_q()
    shared_result_q = manager.get_result_q()

    # The numbers are split into chunks. Each chunk is pushed into the job
    # queue.
    chunksize = 1
    for i in range(0, N, chunksize):
        shared_job_q.put(RR[i:i + chunksize])

    # Wait until all results are ready in shared_result_q
    numresults = 0
    resultlist = []
    while numresults < N:
        outlist = shared_result_q.get()
        resultlist = resultlist + outlist
        numresults += len(outlist)


    CVmetric = []
    ESmetric = []
    for mu in mu_range:
        R = []
        for r in RR:
            if r.mu == mu:
                R.append(r)
            if len(R) == 0:
                print 'no estimate with mu = {:10f} found'.format(mu)
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

    print('--- DONE ---')
    time.sleep(2)
    manager.shutdown()

    return


def runclient():
    manager = make_client_manager(IP, PORTNUM, AUTHKEY)
    job_q = manager.get_job_q()
    result_q = manager.get_result_q()

    mp_fit(job_q, result_q, 16)


if __name__=="__main__":
      # , 'R2079', 'R2085', 'R2130', 'R2135', 'R2148', 'R2153', 'R2185',
        # 'R2196', 'R2201', 'R2217', 'R2223', 'R2230', 'R2244', 'R2246', 'R2281', 'R2256']:  #
      if len (sys.argv) > 1 and sys.argv[1] == 'client':
          runclient()
      else:
          runserver()