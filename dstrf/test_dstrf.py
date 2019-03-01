from warnings import catch_warnings, filterwarnings
filterwarnings('ignore', category=FutureWarning)

import os
import pickle
from mne.utils import _fetch_file
from dstrf import dstrf
import math


# web url to fetch the file
url = "https://ece.umd.edu/~proloy/.datasets/%s.pickled"

names = ['meg', 'stim', 'fwd_sol', 'emptyroom']

# manage local storage
dirname = os.path.join(os.getcwd(), "dstrf_data")
if os.path.isdir(dirname) is False:
    os.mkdir(dirname)


def _load(name):
    if (name in names):
        fname = os.path.join(dirname, f"{name}.pickled")
        if not os.path.isfile(fname):
            _fetch_file(url % name, fname)
        else:
            print(f"{name}.pickled already downloaded.")
    else:
        raise ValueError(f"{name}: not found")
    with open(fname, 'rb') as f:
        v = pickle.load(f)
    return v


def load(name=names):
    data_dict = {name_: _load(name_) for name_ in name}
    return  data_dict


def test_dstrf(cmdopt):
    data = load()
    args = (data['meg'], data['stim'], data['fwd_sol'], data['emptyroom'])
    kwargs = {'tstop':1, 'normalize': 'l1', 'in_place': False, 'mu': 0.0019444,
              'verbose': True, 'n_iter': 10, 'n_iterc': 10, 'n_iterf': 100}
    model = dstrf(*args, **kwargs)
    # check scaling
    stim_baseline = data['stim'].mean()
    assert model._stim_baseline == stim_baseline
    assert math.isclose(model._stim_scaling, (data['stim'] -  stim_baseline).abs().mean(), rel_tol=0.02)
    h = model.h
    # check output
    assert  math.isclose(h.norm('time').norm('source').norm('space'), 6.12923692708188, rel_tol=0.1)

    kwargs['normalize'] = 'l2'
    model = dstrf(*args, **kwargs)
    # check scaling
    stim_baseline = data['stim'].mean()
    assert model._stim_baseline == stim_baseline
    assert math.isclose(model._stim_scaling, (data['stim'] -  stim_baseline).std(), rel_tol=0.02)
    h = model.h
    # check output
    assert  math.isclose(h.norm('time').norm('source').norm('space'),  6.332227345, rel_tol=0.1)

    if cmdopt:
        kwargs['mu'] = 'auto'
        kwargs['normalize'] = 'l1'
        kwargs['n_workers'] = 1
        with catch_warnings():
            filterwarnings('ignore', category=UserWarning)
            model = dstrf(*args, **kwargs)
        assert math.isclose(model.mu,  0.0019444, rel_tol=0.1)



