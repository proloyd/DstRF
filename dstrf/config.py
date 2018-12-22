# Author: Proloy Das <proloy@umd.edu>
from os.path import join
from numpy import linspace

# Specify the root directory
ROOTDIR = '/home/trf_experiment'

# Specification of experiments/trials
COND = ['h', 'l']  # conditions
n_Trials = 3  # trials in each condition

# input file-name templates
meg_template = 'meg_XXXX/meg_%s-%i.mat'
predictor_template = 'predictors/stim_%s.pickled'
fwdsol_template = 'fwdsol/%s-vol-7-fwd.pickled'
emptyroom_template = 'meg_XXXX/emptyroom.pickled'


meg_file = join(ROOTDIR, meg_template)
predictor_file = join(ROOTDIR, predictor_template)
fwdsol_file = join(ROOTDIR, fwdsol_template)
emptyroom_file = join(ROOTDIR, emptyroom_template)


# Cut-off frequencies for band-pass filter
l_freq = 1
h_freq = 80
sampling_freq = 200

# max # of iterations
n_iter = 10
n_iterc = 30
n_iterf = 100

# Cross validation params
normalize = 'l1'
mus = linspace(10, 100, 10) * 1e-4
n_splits = 3
n_workers = 4





