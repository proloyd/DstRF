import time
import warnings
import numpy as np
from scipy import signal
from multiprocessing import Process, Queue, current_process
import queue
from math import ceil
from tqdm import tqdm


def naive_worker(fun, job_q, result_q):
    """Worker function"""
    # myname = current_process().name
    while True:
        try:
            job = job_q.get_nowait()
            # print('%s got %s mus...' % (myname, len(job)))
            for mu in job:
                outdict = {mu: fun(mu)}
                result_q.put(outdict)
            # print('%s done' % myname)
        except queue.Empty:
            # print('returning from %s process' % myname)
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


class collect_output(Process):
    def __init__(self, N, result_q):
        Process.__init__(self)
        self.N = N
        self.result_q = result_q

    def run(self):
        prog = tqdm(total=self.N, desc="Crossvalidation", unit='mu', unit_scale=True)
        numresults = 0
        resultdict = {}
        n_attempts = 0
        while True:
            try:
                outdict = self.result_q.get(False)  # no wait
                if outdict is None:
                    prog.close()
                    break
                resultdict.update(outdict)
                prog.update(n=len(outdict))
                time.sleep(0.01)
                numresults = len(resultdict)
            except queue.Empty:
                time.sleep(10)
                prog.update(n=0)
            except EOFError:
                print('Opps! EOFError encountered.')
                n_attempts += 1
                print('Retrying %i th time' % n_attempts)
                if n_attempts > 100:
                    break

        if numresults < self.N:
            warnings.warn('%i objects are missing' % (self.N - numresults), )

        self.result_q.put(format_to_array(resultdict))
        time.sleep(0.1)
        return


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
        support the :func:`copy.copy` function.
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
        from eelbrain._config import CONFIG
        n = CONFIG['n_workers'] or 1  # by default this is cpu_count()
        n_workers = ceil(n / 8)

    fun = model._get_cvfunc(data, n_splits)

    job_q = Queue()
    result_q = Queue()

    for mu in mus:
        job_q.put([mu])  # put the job as a list.

    prog = collect_output(len(mus), result_q)
    prog.start()
    mp_worker(fun, job_q, result_q, n_workers)
    prog.join()

    cvmu, esmu, cv_info = result_q.get()

    if cv_info[-1] is not None:
        warnings.warn(cv_info[-1])
    print('Building cross-validated model with mu %f' % cvmu)
    return cvmu, esmu, cv_info


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
        for i in range(self.p, 0, -1):
            test_mask = np.zeros(len(X), dtype=np.bool)
            train_mask = np.ones(len(X), dtype=np.bool)
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
