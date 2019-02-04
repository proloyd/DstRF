import time
import copy
import warnings
import numpy as np
from scipy import signal
from multiprocessing import Process, Queue, cpu_count, current_process
import queue
from math import ceil


def naive_worker(fun, job_q, result_q):
    """Worker function"""
    myname = current_process().name
    while True:
        try:
            job = job_q.get_nowait()
            print('%s got %s mus...' % (myname, len(job)))
            for mu in job:
                outdict = {mu: fun(mu)}
                result_q.put(outdict)
            print('%s done' % myname)
        except queue.Empty:
            print('returning from %s process' % myname)
            return


def mp_worker(fun, shared_job_q, shared_result_q, nprocs):
    """sets up workers"""
    procs = []
    for i in range(nprocs):
        p = Process(
            target=naive_worker,
            args=(fun, shared_job_q, shared_result_q))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    time.sleep(0.1)
    shared_result_q.put(None)


def crossvalidate(model, data, mus, n_splits, n_workers=None, ):
    """used to perform cross-validation of cTRF model

    This function assumes `model` class has method _get_cvfunc(data, n_splits)
    which returns a callable. It calls that object with different
    regularizing weights (i.e. mus) to compute cross-validation metric and
    finally compares them to obtain the best weight.

    Parameters
    ----------
    model: model instance
        the model to be validated, here `DstRF`. In addition to that it needs to
        support `copy.copy` function.
    data: REGdata
        Data.
    mus: list | ndarray  (floats)
        The range of the regularizing weights.
    n_splits: int
        number of folds for cross-validation.
    n_workers: int
        number of workers to be used. If None, it will use ``cpu_count/2``.

    Returns
    -------
    esmu : float
        The best weight.
    cv_info : ndarray
        Contains evaluated cross-validation metrics for ``mus``.
    """
    if n_workers is None:
        n_workers = int(round(cpu_count()/2))

    print('Preparing job...')
    fun = model._get_cvfunc(data, n_splits)

    print('Preparation done!')

    job_q = Queue()
    result_q = Queue()

    print('Putting job into queue')
    for mu in mus:
        job_q.put([mu])  # put the job as a list.
        
    mp_worker(fun, job_q, result_q, n_workers)

    numresults = 0
    resultdict = {}
    n_attempts = 0
    while True:
        try:
            outdict = result_q.get(False)  # no wait
            if outdict is None:
                break
            resultdict.update(outdict)
            numresults = len(resultdict)
            print('Received %i objects, waiting for %s more' % (numresults, len(mus) - numresults))
        except queue.Empty:
            time.sleep(10)
            print('Sleeping for 10s')
        except EOFError:
            print('Opps! EOFError encountered.')
            n_attempts += 1
            print('Retrying %i th time' % n_attempts)
            if n_attempts > 100:
                break

    if numresults < len(mus):
        warnings.warn('%i objects are missing' % (len(mus) - numresults), )

    cvmu, esmu, cv_info = format_to_array(resultdict)
    
    print('Crossvalidation Done.')
    print('Building cross-validated model with mu %f' % cvmu)
    
    return esmu, cv_info


def format_to_array(resultdict):
    """format crossvalidation info dict to np.array"""
    mu = []
    cv = []
    cv1 = []
    cv2 = []
    es = []
    for keys, values in resultdict.items():
        mu.append(keys)
        cv.append(values['cv'])
        cv1.append(values['cv1'])
        cv2.append(values['cv2'])
        es.append(values['es'])

    mu = np.array(mu)
    cv = np.array(cv)
    cv1 = np.array(cv1)
    cv2 = np.array(cv2)
    es = np.array(es)

    idx = np.argsort(np.array(mu))

    mu = mu[idx]
    cv = cv[idx]
    cv1 = cv1[idx]
    cv2 = cv2[idx]
    es = es[idx]

    Warn = None
    # take care of nan values
    es[np.isnan(es)] = 10  # replace Nan values by some big number (say 10)
    # CVmu = mu[cv2.argmin()]
    CVmu = mu[cv1.argmin()]
    # ESmu = mu[cv2.argmin():][es[cv2.argmin():].argmin()]
    # ESmu = mu[cv2.argmin() + signal.find_peaks(-es[cv2.argmin():])[0][0]]
    if CVmu == mu[-1]:
        ESmu = CVmu
        Warn = f'CVmu is {CVmu}: extend range of mu towards right'
    else:
        try:
            ESmu = mu[cv1.argmin() + signal.find_peaks(-es[cv1.argmin():])[0][0]]
        except IndexError:
            ESmu = CVmu
            Warn = f'ESmu is {ESmu}: extend range of mu towards right'
        if ESmu == mu[-1]:
            Warn = f'ESmu is {ESmu}: extend range of mu towards right'
    if CVmu == mu[0]:
        if Warn is None:
            Warn = f'CVmu is {CVmu}: extend range of mu towards left'
        else:
            Warn = f'{Warn}; CVmu is {CVmu}: extend range of mu towards left'

    return CVmu, ESmu, (np.array([mu, cv, cv1, cv2, es]), Warn)


class TimeSeriesSplit:
    def __init__(self, r=0.05, p=5, d=100):
        self.ratio = r
        self.p = p
        self.d = d

    def _iter_part_masks(self, X):
        n_v = ceil(self.ratio / (1 + self.ratio) * len(X))
        # print(n_v)
        for i in range(self.p, 0, -1):
            test_mask = np.zeros(len(X), dtype=np.bool)
            train_mask = np.ones(len(X), dtype=np.bool)
            # print(i*n_v-self.d)
            train_mask[-(i*n_v+self.d):] = False
            if i == 1:
                test_mask[-i*n_v:] = True
            else:
                test_mask[-i * n_v:-(i - 1) * n_v] = True
            yield (train_mask, test_mask)

    def split(self, X):
        indices = np.arange(len(X))
        for (train_mask, test_mask) in self._iter_part_masks(X):
            train_index = indices[train_mask]
            test_index = indices[test_mask]
            yield train_index, test_index
