import time
import copy
import warnings
import numpy as np
from multiprocessing import Process, Queue, cpu_count, current_process
import queue

from . import config as cfg


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


def crossvalidate(model, data, mus=None, n_splits=None, n_workers=None):
    """used to perform cross-validation of cTRF model

    This function assumes `model` class has method _get_cvfunc(data, n_splits)
    which returns a callable. It calls that object with different
    regularizing weights (i.e. mus) to compute cross-validation metric and
    finally compares them to obtain the best weight.

    Parameters
    ----------
    model: object
        the model to be validated, here DstRF. In addition to that it needs to
        support `copy.copy` function. 
    data: object
        the instance should be compatible for fitting the model. In addition to 
        that it shall have a timeslice method compatible to kfold objects.
    mus: list | ndarray  (floats)
        The range of the regularizing weights. If None, it will use values
        specified in config.py.
    n_splits: int
        number of folds for cross-validation, If None, it will use values
        specified in config.py.
    n_workers: int
        number of workers to be used. If None, it will use values specified
        in config.py.

    Results
    -------
        float, the best weight. 
    """
    if mus is None:
        mus = cfg.mus
    if n_splits is None:
        n_splits = cfg.n_splits
    if n_workers is None:
        n_workers = cfg.n_workers

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
    
    return cvmu, cv_info


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

    # take care of nan values
    es[np.isnan(es)] = 10  # replace Nan values by some big number (say 10)
    CVmu = mu[cv2.argmin()]
    # ESmu = mu[cv2.argmin():][es[cv2.argmin():].argmin()]
    # ESmu = mu[cv2.argmin() + signal.find_peaks(-es[cv2.argmin():])[0][0]]
    ESmu = None

    return CVmu, ESmu, np.array([mu, cv, cv1, cv2, es])
