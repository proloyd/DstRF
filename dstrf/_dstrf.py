from ._model import *


def dstrf(meg, stim, lead_field, noise, tstart=0, tstop=0.5, nlevels=1, downsample=False,
          n_iter=10, n_iterc=30, n_iterf=100, normalize=None, mu=0.05, do_crossvalidation=False,
          tol=1e-3, verbose=False, mus=None, n_splits=3, n_workers=4):
    """One shot function for cortical TRF localization

    Estimate both TRFs and source variance from the observed MEG data by solving
    the Bayesian optimization problem mentioned in the paper:
    P. Das, C. Brodbeck, J. Z. Simon, B. Babadi, Cortical Localization of the
    Auditory Temporal Response Function from MEG via Non-Convex Optimization;
    2018 Asilomar Conference on Signals, Systems, and Computers, Oct. 28–31,
    Pacific Grove, CA(invited).

    Parameters
    ----------
    lead_field : NDVar
        forward solution a.k.a. lead_field matrix.
    noise : NDVar or ndarray
        empty room data as NDVar or the covariance matrix as an ndarray.
    meg :  NDVar (case, sensor, time) or list of such NDVars
        where case reflects different trials, different list elements reflects
        different conditions (i.e. stimulus).
    stim : NDVar (case, time) or list of such NDVars
        where case reflects different predictor variables (e.g. envelope,
        wordlog10wf etc);  different list elements reflects
        different conditions (i.e. stimulus).
        [ stim2  # NDVar  (case, scalar, time)  (e.g. spectrogram with multiple bands)
         to be implemented]
    tstart : float
        Start of the TRF in seconds.
    tstop : float
        Stop of the TRF in seconds.
    nlevels : int
        Decides the density of Gabor atoms. Bigger nlevel -> less dense basis.
        By default it is set to `1`. `nlevesl > 2` should be used with caution.
    downsample : boolean
        False by default. If True down samples noise and meg signals to 200Hz.
    n_iter : int
        Number of out iterations of the algorithm, by default set to 10.
    n_iterc : int
        Number of Champagne iterations within each outer iteration, by default set to 30.
    n_iterf : int
        Number of FASTA iterations within each outer iteration, by default set to 100.
    normalize : `l1` or None (`l2`)
        Decides how to normalize stim variables. By default it is None, which does
        `l2`-normalization.
    mu : None or float
        Single regularizer parameter. Needs to be specified when not performing
        cross-validation. Ignored when do_crossvalidation is True.
    do_crossvalidation : boolean
        By default False, if True performs crossvalidation with the given `mus`.
    verbose : boolean
        if True prints intermidiate results, by default False.
    mus : list or ndarray
        if crossvalidation is True, performs crossvalidation with values specified
        in `mus` array.
    n_splits : int
        number of cross-validation folds
    n_workers : int (optional)
        number of workers to spawn for cross-validation.

    Returns
    -------
        tuple (NDVar, DstRF)
    """
    from eelbrain import filter_data, resample
    from .config import sampling_freq, l_freq, h_freq
    # noise covariance
    if isinstance(noise, NDVar):
        if downsample:
            noise = filter_data(noise, l_freq, h_freq, method='fir', fir_design='firwin')
            noise = resample(noise, sampling_freq)
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

    # Initialize `DstRF` instance with desired properties
    model = DstRF(lead_field, noise_cov, n_iter=n_iter, n_iterc=n_iterc, n_iterf=n_iterf)

    # Initialize `REG_Data` instance with desired properties
    ds = REG_Data(tstart, tstop, nlevels)
    if isinstance(meg, list) and isinstance(stim, list):
        if len(meg) != len(stim):
            raise ValueError(f'meg list length={len(meg)} is not equal to '
                             f'stim list length={len(stim)}')
    elif isinstance(meg, NDVar) and isinstance(stim, NDVar):
        meg = [meg]
        stim = [stim]
    else:
        raise NotImplementedError

    # Call `REG_Data.add_data` once for each contiguous segment of MEG data
    for r, s in zip(meg, stim):
        # NORMALIZE PREDICTORS:
        s -= s.mean('time')
        if normalize is None:
            s /= s.std('time')
        elif normalize == 'l1':
            norm = np.abs(s.x).mean(axis=1)
            s.x /= norm[:, np.newaxis]

        if r.has_case:
            dim = r.get_dim('case')
            for i in range(len(dim)):
                if downsample:
                    data = filter_data(r[i], l_freq, h_freq, method='fir', fir_design='firwin')
                    data = resample(data, sampling_freq)
                else:
                    data = r[i]
                ds.add_data(data, s, False)
        else:
            ds.add_data(r, s, False)

    model.fit(ds, mu, do_crossvalidation, tol, verbose, mus=mus, n_splits=n_splits, n_workers=n_workers)

    trf = model.get_strf(ds)

    return trf, model
