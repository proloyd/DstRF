from eelbrain import NDVar, combine
import numpy as np
from ._model import DstRF, REG_Data


DEFAULT_MUs = np.logspace(-3, -1, 7)


def dstrf(meg, stim, lead_field, noise, tstart=0, tstop=0.5, nlevels=1,
          n_iter=10, n_iterc=10, n_iterf=100, normalize=None, in_place=None,
          mu='auto', tol=1e-3, verbose=False, n_splits=3, n_workers=None,
          use_ES=False):
    """One shot function for cortical TRF localization

    Estimate both TRFs and source variance from the observed MEG data by solving
    the Bayesian optimization problem mentioned in the paper _[1].
    .. [1] P. Das, C. Brodbeck, J. Z. Simon, B. Babadi, Cortical Localization of the
    Auditory Temporal Response Function from MEG via Non-Convex Optimization;
    2018 Asilomar Conference on Signals, Systems, and Computers, Oct. 28–31,
    Pacific Grove, CA(invited).

    Parameters
    ----------
    meg :  NDVar (case, sensor, time) or list of such NDVars
        where case reflects different trials, different list elements reflects
        different conditions (i.e. stimulus).
    stim : NDVar (case, time) or list of such NDVars
        where case reflects different trials;  different list elements reflects
        different feature variables(e.g. envelope, wordlog10wf etc).
        TODO: [ stim2  # NDVar  (case, scalar, time)  (e.g. spectrogram with
         multiple bands) to be implemented]
    lead_field : NDVar
        forward solution a.k.a. lead_field matrix.
    noise : NDVar or ndarray
        empty room data as NDVar or the covariance matrix as an ndarray.
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
    """
    # noise covariance
    if isinstance(noise, NDVar):
        er = noise.get_data(('sensor', 'time'))
        noise_cov = np.dot(er, er.T) / er.shape[1]
    elif isinstance(noise, np.ndarray):
        if (noise.ndim == 2) and (noise.shape[0] == noise.shape[0]):
            noise_cov = noise
        else:
            raise ValueError(f'For noise as ndarray, noise dim1={noise.shape[0]} should'
                             f'match dim2={noise.shape[0]}')
    else:
        raise NotImplementedError

    # Initialize `REG_Data` instance with desired properties
    ds = REG_Data(tstart, tstop, nlevels)

    # data copy?
    if in_place is None:
        in_place = False
    if not isinstance(in_place, bool):
        raise TypeError(f"in_place={in_place!r}, need bool or None")

    # Call `REG_Data.add_data` once for each contiguous segment of MEG data
    for r, s in iter_data(meg, stim):
        if not in_place:
            s = s.copy()
            r = r.copy()

        if normalize:  # screens False, None
            s -= s.mean('time')
            if isinstance(normalize, bool):  # normalize=True defaults to 'l2'
                normalize = 'l2'
            if isinstance(normalize, str):
                if normalize == 'l2':
                    s_scale = (s.x ** 2).mean(-1) ** 0.5
                elif normalize == 'l1':
                    s_scale = np.abs(s.x).mean(-1)
                else:
                    raise ValueError(f"normalize={normalize!r}, need bool or \'l1\' or \'l2\'")
            else:
                raise TypeError(f"normalize={normalize!r}, need bool or str")

            s.x /= s_scale[:, np.newaxis]

        if r.has_case:
            dim = r.get_dim('case')
            for i in range(len(dim)):
                ds.add_data(r[i], s)
        else:
            ds.add_data(r, s)

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
        mus = DEFAULT_MUs
        do_crossvalidation = True
    else:
        raise ValueError(f"invalid mu={mu!r}, supports tuple, list, np.ndarray or scalar float"
                         f"optionally, may be left 'auto' if not sure!")

    if lead_field.get_dim('sensor') != ds.sensor_dim:
        lead_field = lead_field.sub(sensor=ds.sensor_dim)

    model = DstRF(lead_field, noise_cov, n_iter=n_iter, n_iterc=n_iterc, n_iterf=n_iterf)
    model.fit(ds, mu, do_crossvalidation, tol, verbose, mus=mus, n_splits=n_splits,
              n_workers=n_workers, use_ES=use_ES)
    return model


def iter_data(meg, stim):
    if isinstance(meg, list):
        if isinstance(stim, list):
            if isinstance(stim[0], NDVar):
                for meg_, *stim_ in zip(meg, stim):
                    time_dim = meg_.get_dim('time')
                    if any(r_.get_dim('time') != time_dim for r_ in stim_):
                        raise ValueError("Not all NDVars have the same time dimension")
                    yield (meg_, combine(stim_))
            else:
                for meg_, *stim_ in zip(meg, *stim):
                    time_dim = meg_.get_dim('time')
                    if any(r_.get_dim('time') != time_dim for r_ in stim_):
                        raise ValueError("Not all NDVars have the same time dimension")
                    yield (meg_, combine(stim_))
        else:
            raise ValueError(f'Invalid data format {stim}: if meg is a list of NDVar'
                             ', stim must be a list of NDVars')
    elif isinstance(meg, NDVar):
        time_dim = meg.get_dim('time')
        if isinstance(stim, list):
            for meg_, *stim_ in zip(meg, *stim):
                if any(r_.get_dim('time') != time_dim for r_ in stim_):
                    raise ValueError("Not all NDVars have the same time dimension")
                yield (meg_, combine(stim_))
        elif isinstance(stim, NDVar):
            if stim.get_dim('time') != time_dim:
                raise ValueError("Not all NDVars have the same time dimension")
            for meg_, *stim_ in zip(meg, stim):
                yield (meg_, combine(stim_))
        else:
            raise ValueError(f'Invalid data format {stim}: if meg is a NDVar, stim must be a NDVar'
                             f'or list of NDVars')
    else:
        raise ValueError(f'Invalid data format {meg}: Expected NDVar or list of NDVars')
