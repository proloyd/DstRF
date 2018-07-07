import glob
from os.path import join

from eelbrain import *
from eelbrain.plot._base import TimeSlicer
import mne

from _nifti_utils import _save_stc_as_volume


class GlassBrain(TimeSlicer):

    def __init__(self, ndvar, src, dest='mri', mri_resolution=False, black_bg=False, display_mode='lyrz',
                 threshold='None', colorbar=False, alpha=0.7, vmin=None, vmax=None, plot_abs=True):
        from nilearn.plotting import plot_glass_brain
        from matplotlib.pyplot import figure

        self._glass_brain = plot_glass_brain
        self.figure = figure()

        # src
        # check if file name
        if isinstance (src, basestring):
            print('Reading src file %s...' % src)
            self.src = mne.read_source_spaces(src)
        else:
            self.src = src

        src_type = src[0]['type']
        if src_type != 'vol':
            raise ValueError('You need a volume source space. Got type: %s.'
                              % src_type)

        if ndvar.has_dim ('space'):
            ndvar = ndvar.norm ('space')
        self._ndvar = ndvar

        if ndvar.has_dim('time'):
            t_in = 0
            self.time = ndvar.get_dim('time')
            ndvar0 = ndvar.sub(time=self.time[t_in])
            title = 'time = %s ms' % round(t_in*1e3)
        else:
            self.time = None
            ndvar0 = ndvar

        self.kwargs0 = dict(dest=dest,
                            mri_resolution=mri_resolution)

        self.kwargs1 = dict(black_bg=black_bg,
                            display_mode=display_mode,
                            threshold=threshold,
                            colorbar=colorbar,
                            alpha=alpha,
                            vmin=vmin,
                            vmax=vmax,
                            plot_abs=plot_abs
                            )
        self.glassbrain = plot_glass_brain(_save_stc_as_volume(None, ndvar0, self.src, **self.kwargs0),
                                           title=title,
                                           figure=self.figure,
                                           **self.kwargs1
                                           )
        TimeSlicer.__init__(self, (ndvar,))

    def _update_time(self, t, fixate):
        ndvart = self._ndvar.sub(time=t)
        title = 'time = %s ms' % round (t * 1e3)
        self.figure.clf()
        self.glassbrain = self._glass_brain(_save_stc_as_volume(None, ndvart, self.src, **self.kwargs0),
                                            title=title,
                                            figure=self.figure,
                                            **self.kwargs1
                                            )

    def animate(self):
        for t in self.time:
            self.set_time(t)

    # this is only needed for Eelbrain < 0.28
    def set_time(self, time):
        """Set the time point to display

        Parameters
        ----------
        time : scalar
            Time to display.
        """
        self._set_time(time, True)


def butterfly(ndvar, src, dest='mri', mri_resolution=False, black_bg=False, display_mode='lyrz',
                 threshold='None', colorbar=False, alpha=0.7, vmin=None, vmax=None, plot_abs=True):

    if ndvar.has_dim('space'):
        p = plot.Butterfly(ndvar.norm('space'), vmin=vmin, vmax=vmax)
    else:
        p = plot.Butterfly(ndvar, vmin=vmin, vmax=vmax)

    gb = GlassBrain(ndvar, src, dest=dest, mri_resolution=mri_resolution, black_bg=black_bg, display_mode=display_mode,
                    threshold=threshold, colorbar=colorbar, alpha=alpha, vmin=vmin, vmax=vmax, plot_abs=True)

    p.link_time_axis(gb)

    return p, gb


# if __name__ == '__main__':
#     import cPickle as pickle
#
#     ROOTDIR = 'G:/My Drive/Proloy/'
#
#     fname = ROOTDIR + '/mri/fsaverage/bem/fsaverage-vol-10-src.fif'
#     src = mne.read_source_spaces(fname)
#
#     fname = ROOTDIR + 'Group analysis/Dataset wf-onset-u.pickled'
#     ds = pickle.load(open(fname, 'rb'))
#     h = ds['trf'].mean('case')
#     p = plot.Butterfly(h.norm('space'))
#     gb = GlassBrain(h, src, dest='surf', threshold=5e-13)
#     p.link_time_axis(gb)


