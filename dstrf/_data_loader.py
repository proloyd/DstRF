# Author: Proloy Das <proloy@umd.edu>
from . import config as cfg

import pickle
import numpy as np
from scipy import io, linalg

from eelbrain import *

from ._model import DstRF, REG_Data


def _myinv(x):
    """Computes inverse

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


def load_subject(subject_id, n_splits=1, normalize=None):
    """Loads data for running DstRF

    Parameters
    ----------
        subject_id: str
            File names are identified from this variable
        n_splits: int (Default 1)
            Decides how many instances of DstRF object to return.
        normalize: 'l1' | None (default)
            Normalization method
    Returns
    -------
    a tuple (DstRF object, REGData object)
    """
    # LOAD LEAD_FIELD MATRIX
    with open(cfg.fwdsol_file % subject_id, 'rb') as f:
        lead_field = pickle.load(f)
    
    # LOAD NOISE COVARIANCE (VANILLA)
    with open(cfg.emptyroom_file % subject_id, 'rb') as f:
        ER = pickle.load(f)
    ER = filter_data(ER, cfg.l_freq, cfg.h_freq, method='fir', fir_design='firwin')
    ER = resample(ER, cfg.sampling_freq)
    er = ER.get_data(('sensor', 'time'))
    noise_cov = np.dot(er, er.T) / er.shape[1]

    # PRE_WHITENING STEP (MEGs will be pre-whitened later on)
    e, v = linalg.eigh(noise_cov)
    wf = np.dot(v * _myinv(np.sqrt(e)), v.T.conj())

    if lead_field.ndim == 3:
        for i in range(lead_field.shape[-1]):
            lead_field.x[:, :, i] = np.dot(wf, lead_field.x[:, :, i])
    else:
        lead_field.x = np.dot(wf, lead_field.x)
    noise_cov = np.eye(er.shape[0])

    # INITIALIZE DstRF object
    R = [
        DstRF(lead_field, noise_cov, n_iter=cfg.n_iter, n_iterc=cfg.n_iterc, n_iterf=cfg.n_iterf)  # 20 for
        # cross-validation
        for _ in range(n_splits)
    ]
    if len(R) == 1:
        R = R[0]

    # PACK DATA
    ds = REG_Data()
    for cond in cfg.COND:
        # READ PREDICTORS
        predictors = []
        predictor_file = cfg.predictor_file % (cond)
        with open(predictor_file, 'rb') as f:
            predictor = pickle.load(f)

        # NORMALIZE PREDICTORS:
        predictor -= predictor.mean('time')
        if normalize is None:
            predictor /= predictor.std('time')
        elif normalize == 'l1':
            norm = np.abs(predictor.x).mean(axis=1)
            predictor.x /= norm[:, np.newaxis]

        for trial in range(cfg.n_Trials):
            key = '%s%i' % (cond, trial)
            with open(cfg.meg_file % (cond, trial), 'rb') as f:
                y = pickle.load(f)
            y = filter_data(y, cfg.l_freq, cfg.h_freq, method='fir', fir_design="firwin")
            data = resample(y, cfg.sampling_freq)
            data.x = np.dot(wf, data)   # pre-whitening step
            ds.load(key, data, predictors, False)

    return R, ds


def learn_model_for_subject(subject_id, mu, normalize='l1'):
    """Loads the data and performs model fitting using given mu"""
    R, ds = load_subject(subject_id, n_splits=1, normalize=normalize)
    R.fit(ds, mu, tol=1e-5, verbose=True)
    trf = R.get_strf(ds)
    with open(cfg.trf_file % subject_id, 'wb') as f:
        pickle.dump(trf, f)

    return
