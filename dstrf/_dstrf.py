from collections import Sequence

from eelbrain import NDVar, combine
from mne import Covariance
import numpy as np

from ._model import DstRF, REG_Data


DEFAULT_MUs = np.logspace(-3, -1, 7)


def dstrf(meg, stim, lead_field, noise, tstart=0, tstop=0.5, nlevels=1,
          n_iter=10, n_iterc=10, n_iterf=100, normalize=None, in_place=None,
          mu='auto', tol=1e-3, verbose=False, n_splits=3, n_workers=None,
          use_ES=False, **kwargs):
    """One shot function for cortical TRF localization

    Estimate both TRFs and source variance from the observed MEG data by solving
    the Bayesian optimization problem mentioned in the paper _[1].
    .. [1] P. Das, C. Brodbeck, J. Z. Simon, B. Babadi, Cortical Localization of the
    Auditory Temporal Response Function from MEG via Non-Convex Optimization;
    2018 Asilomar Conference on Signals, Systems, and Computers, Oct. 28–31,
    Pacific Grove, CA(invited).

    Parameters
    ----------
    meg :  NDVar ([case,] sensor, time) or list of such NDVars
        If multiple trials are the same length they can be specified as
        :class:`NDVar` with case dimension, if they are different length they
        can be supplied as list.
    stim : NDVar ([case, dim,] time) or (nested) list of such NDVars
        One or multiple predictors corresponding to each item in ``meg``.
    lead_field : NDVar
        forward solution a.k.a. lead_field matrix.
    noise : mne.Covariance | NDVar | ndarray
        The empty room noise covariance, or data from which to compute it as
        :class:`NDVar`.
    tstart : float
        Start of the TRF in seconds.
    tstop : float
        Stop of the TRF in seconds.
    nlevels : int
        Decides the density of Gabor atoms. Bigger nlevel -> less dense basis.
        By default it is set to `1`. `nlevesl > 2` should be used with caution.
    n_iter : int
        Number of out iterations of the algorithm, by default set to 10.
    n_iterc : int
        Number of Champagne iterations within each outer iteration, by default set to 10.
    n_iterf : int
        Number of FASTA iterations within each outer iteration, by default set to 100.
    normalize : bool | 'l2' | 'l1'
        Scale ``stim`` before model fitting: subtract the mean and divide by
        the standard deviation (when ``nomrmalize='l2'`` or ``normalize=True``)
        or the mean absolute value (when ``normalize='l1'``). By default,
         ``normalize=None`` it leaves ``stim`` data untouched.
    in_place: bool
        With ``in_place=False`` (default) the original ``meg`` and ``stims`` are left untouched;
        use ``in_place=True`` to save memory by using the original ``meg`` and ``stim``.
    mu : 'auto' or ndarray or list or tuple
        Choice of regularizer parameter. By default ``mu='auto'`` performs crossvalidation
        to choose optimal one, from the range
                ``np.logspace(-3, -1, 7)``.
        Additionally, user can choose to pass a range over which the cross-validation will be done.
        If a single choice if passed, model corresponding to that value is returned.
    tol : float
        Tolerance factor deciding stopping criterion for the overall algorithm. The iterations
        are stooped when ``norm(trf_new - trf_old)/norm(trf_old) < tol`` condition is met.
        By default ``tol=1e-3``.
    verbose : boolean
        if True prints intermediate results, by default False.
    n_splits : int
        number of cross-validation folds. By default it uses 3-fold cross-validation.
    n_workers : int (optional)
        number of workers to spawn for cross-validation. If None, it will use ``cpu_count/2``.
    use_ES : Boolean (optional)
        use estimation stability criterion _[2] to choose the best ``mu``. (False, by default)
        ..[2] Lim, Chinghway, and Bin Yu. "Estimation stability with cross-validation (ESCV)."
        Journal of Computational and Graphical Statistics 25.2 (2016): 464-492.

    Returns
    -------
    trf : NDVar
        TRF estimate
    model : DstRF
        The full model

    Examples
    --------
    MEG data ``y`` with dimensions (case, sensor, time) and predictor ``x``
    with dimensions (case, time)::

        dstrf(y, x, fwd, cov)

    ``x`` can always also have an additional predictor dimension, for example,
    if ``x`` represents a spectrogram: (case, frequency, time). The case
    dimension is optional, i.e. a single contiguous data segment is also
    accepted, but the case dimension should always match between ``y`` and
    ``x``.

    Multiple distinct predictor variables can be supplied as list; e.g., when
    modeling simultaneous responses to an attended and an unattended stimulus
    with ``x_attended`` and ``x_unattended``::

        dstrf(y, [x_attended, x_unattended], fwd, cov)

    Multiple data segments can also be specified as list. E.g., if ``y1`` and
    ``y2`` are responses to stimuli ``x1`` and ``x2``, respoectively::

        dstrf([y1, y2], [x1, x2], fwd, cov)

    And with multiple predictors::

        dstrf([y1, y2], [[x1_attended, x1_unattended], [x2_attended, x2_unattended]], fwd, cov)

    """
    # Note this before turning stim into lists
    stim_is_single = isinstance(stim, NDVar) if isinstance(meg, NDVar) else isinstance(stim[0], NDVar)

    if isinstance(meg, NDVar):
        megs = (meg,)
        stims = (stim,)
    elif isinstance(meg, Sequence):
        megs = meg
        stims = stim
    else:
        raise TypeError(f"meg={meg!r}")

    # normalize=True defaults to 'l2'
    if normalize is True:
        normalize = 'l2'
    elif isinstance(normalize, str):
        if normalize not in ('l1', 'l2'):
            raise ValueError(f"normalize={normalize!r}, need bool or 'l1' or 'l2'")
    else:
        raise TypeError(f"normalize={normalize!r}, need bool or str")

    # data copy?
    if not isinstance(in_place, bool):
        raise TypeError(f"in_place={in_place!r}, need bool or None")

    if normalize:  # screens False, None
        if isinstance(normalize, bool):  # normalize=True defaults to 'l2'
            normalize = 'l2'
        elif isinstance(normalize, str):
            if normalize not in ('l1', 'l2'):
                raise ValueError(f"normalize={normalize!r}, need bool or \'l1\' or \'l2\'")
        else:
            raise TypeError(f"normalize={normalize!r}, need bool or str")

        s_baseline, s_scale = get_scaling(stims, normalize)
    else:
        s_baseline, s_scale = (None, None)

    # Call `REG_Data.add_data` once for each contiguous segment of MEG data
    ds = REG_Data(tstart, tstop, nlevels, s_baseline, s_scale, stim_is_single)
    for r, ss in iter_data(megs, stims):
        if normalize:
            if not in_place:
                ss = [s.copy() for s in ss]
            # for s, m, scale in zip(ss, s_baseline, s_scale):
            #     s -= m
            #     s /= scale
        ds.add_data(r, ss)

    # TODO: make this less hacky when fixing normalization (iter_data() always turns stim into lists)
    # ds._stim_is_single = isinstance(stim, NDVar) if isinstance(meg, NDVar) else isinstance(stim[0], NDVar)

    # noise covariance
    if isinstance(noise, NDVar):
        er = noise.get_data(('sensor', 'time'))
        noise_cov = np.dot(er, er.T) / er.shape[1]
    elif isinstance(noise, Covariance):
        # check for channel mismatch
        chs_noise = set(noise.ch_names)
        chs_data = set(ds.sensor_dim.names)
        chs_both = sorted(chs_noise.intersection(chs_data))
        if len(chs_both) < len(chs_data):
            missing = sorted(chs_data.difference(chs_noise))
            raise NotImplementedError(f"Noise covariance is missing data for sensors {', '.join(missing)}")
        else:
            assert np.all(ds.sensor_dim.names == chs_both)
        if len(chs_both) < len(chs_noise):
            index = np.array([noise.ch_names.index(ch) for ch in chs_both])
            noise_cov = noise.data[index[:, np.newaxis], index]
        else:
            assert noise.ch_names == chs_both
            noise_cov = noise.data
    elif isinstance(noise, np.ndarray):
        n = len(ds.sensor_dim)
        if noise.shape == (n, n):
            noise_cov = noise
        else:
            raise ValueError(f'noise = array of shape {noise.shape}; should be {(n, n)}')
    else:
        raise TypeError(f'noise={noise!r}')

    # Regularizer Choice
    if isinstance(mu, (tuple, list, np.ndarray)):
        if len(mu) > 1:
            mus = mu
            do_crossvalidation = True
        else:
            mus = None
            do_crossvalidation = False
    elif isinstance(mu, float):
        mus = None
        do_crossvalidation = False
    elif mu == 'auto':
        mus = 'auto'
        do_crossvalidation = True
    else:
        raise ValueError(f"invalid mu={mu!r}, supports tuple, list, np.ndarray or scalar float"
                         f"optionally, may be left 'auto' if not sure!")

    if lead_field.get_dim('sensor') != ds.sensor_dim:
        lead_field = lead_field.sub(sensor=ds.sensor_dim)

    model = DstRF(lead_field, noise_cov, n_iter=n_iter, n_iterc=n_iterc, n_iterf=n_iterf)
    model.fit(ds, mu, do_crossvalidation, tol, verbose, mus=mus, n_splits=n_splits,
              n_workers=n_workers, use_ES=use_ES, ** kwargs)
    return model


def iter_data(megs, stims):
    """Iterate over data as ``meg, (stim1, ...)`` tuples"""
    for meg, stim in zip(megs, stims):
        if meg.has_case:
            # assume stim has case too
            if isinstance(stim, NDVar):
                # single stim
                assert stim.has_case and len(stim) == len(meg)
                for meg_trial, stim_trial in zip(meg, stim):
                    yield meg_trial, (stim_trial,)
            else:
                # sequence stim
                assert all(s.has_case for s in stim)
                assert all(len(s) == len(meg) for s in stim)
                for meg_trial, *stims_trial in zip(meg, *stim):
                    yield meg_trial, stims_trial
        elif isinstance(stim, NDVar):
            yield meg, (stim,)
        else:
            yield meg, stim


def get_scaling(stims, normalize):
    temp_m = []
    temp_s = []
    for stim_ in stims:
        m = _get_baseline(stim_)
        temp_m.append(m)
    m = np.array(temp_m).mean(axis=0)

    for stim_ in stims:
        s = _get_scale(stim_, m, normalize)
        temp_s.append(s)
    temp_s = np.array(temp_s)

    if normalize == 'l1':
        scaling = temp_s.mean(axis=0)
    else:
        scaling = (temp_s ** 2).mean(axis=0) ** 0.5

    return m, scaling



def _get_baseline(stim):
    if isinstance(stim, NDVar):
        stim = [stim, ]
    m = np.zeros(len(stim))
    for i, stim_ in enumerate(stim):
        m[i] = stim_.mean()
    return m


def _get_scale(stim, baseline, normalize):
    if isinstance(stim, NDVar):
        stim = [stim, ]
    scaling = np.zeros(len(stim))
    for i, (stim_, m) in enumerate(zip(stim, baseline)):
        if normalize == 'l1':
            temp = (stim_ - m).abs().mean('time')
            if stim_.has_case:
                scaling[i] = temp.mean()
            else:
                scaling[i] = temp
        else:
            temp = ((stim_ - m) ** 2).mean('time')
            if stim_.has_case:
                scaling[i] = temp.mean() ** 0.5
            else:
                scaling[i] = temp ** 0.5
    return scaling
