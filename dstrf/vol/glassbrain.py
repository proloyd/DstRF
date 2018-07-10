import glob
from os.path import join

from eelbrain import *
from eelbrain.plot._base import TimeSlicer
import mne

import numpy as np
from nilearn.plotting import plot_glass_brain
import matplotlib

from _nifti_utils import _save_stc_as_volume


class GlassBrain(TimeSlicer):

    def __init__(self, ndvar, src, dest='mri', mri_resolution=False, black_bg=False, display_mode='lyrz',
                 threshold='auto', colorbar=False, cmap=None,alpha=0.7, vmin=None, vmax=None, plot_abs=True):

        if not matplotlib.is_interactive():
            print('Turning interactive backend on.')
            matplotlib.interactive(True)


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
            # set vmax and vmin
            if vmax is None:
                vmax = ndvar.max ()
            if vmin is None:
                vmin = ndvar.min ()
        else:
            # set vmax and vmin
            if vmax is None:
                vmax = np.maximum (ndvar.max (), -ndvar.min ())
            if vmin is None:
                vmin = np.minimum (-ndvar.max (), ndvar.min ())

        self._ndvar = ndvar

        if ndvar.has_dim('time'):
            t_in = 0
            self.time = ndvar.get_dim('time')
            ndvar0 = ndvar.sub(time=self.time[t_in])
            title = 'time = %s ms' % round(t_in*1e3)
        else:
            self.time = None
            title = 'time = None'
            ndvar0 = ndvar

        self.kwargs0 = dict(dest=dest,
                            mri_resolution=mri_resolution)

        self.kwargs1 = dict(black_bg=black_bg,
                            display_mode=display_mode,
                            threshold=threshold,
                            cmap=cmap,
                            colorbar=colorbar,
                            alpha=alpha,
                            vmin=vmin,
                            vmax=vmax,
                            plot_abs=plot_abs,
                            )
        self.glassbrain = plot_glass_brain(_save_stc_as_volume(None, ndvar0, self.src, **self.kwargs0),
                                           title=title,
                                           **self.kwargs1
                                           )
        TimeSlicer.__init__(self, (ndvar,))

    def _update_time(self, t, fixate):
        ndvart = self._ndvar.sub(time=t)
        title = 'time = %s ms' % round (t * 1e3)

        # remove existing image
        for display_ax in self.glassbrain.axes.values():
            if len(display_ax.ax.images) > 1:
                display_ax.ax.images[-1].remove()

        # No need to take care of the colorbar anymore
        # Still thete is some bug!
        if self.kwargs1['colorbar']:
            self.glassbrain._colorbar_ax.redraw_in_frame()
            self.glassbrain._colorbar = False

        self.glassbrain.add_overlay(_save_stc_as_volume(None, ndvart, self.src, **self.kwargs0),
                                    threshold=self.kwargs1['threshold'],
                                    colorbar=self.kwargs1['colorbar'],
                                    **dict(cmap=self.kwargs1['cmap'],
                                           # norm=self.kwargs1['norm'],
                                           vmax=self.kwargs1['vmax'],
                                           vmin=self.kwargs1['vmin'],
                                           alpha=self.kwargs1['alpha'],))
        self.glassbrain.title(title)
        # update colorbar
        # if self.kwargs1['colorbar']:
        #     self._update_colorbar(None, None)

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

    # def _update_colorbar(self, cmap=None, norm=None):
    #     """
    #     Parameters
    #     ----------
    #     cmap: a matplotlib colormap
    #         The colormap used
    #     norm: a matplotlib.colors.Normalize object
    #         This object is typically found as the 'norm' attribute of an
    #         matplotlib.image.AxesImage
    #     threshold: float or None
    #         The absolute value at which the colorbar is thresholded
    #     """
    #
    #     display_ax = self.glassbrain.axes.values()[0]
    #     if norm is None:
    #         norm = display_ax.ax.images[0].norm
    #     if cmap is None:
    #         cmap = display_ax.ax.images[0].cmap
    #
    #     threshold = self.kwargs1['threshold']
    #     if threshold is None:
    #         offset = 0
    #     else:
    #         offset = threshold
    #     if offset > norm.vmax:
    #         offset = norm.vmax
    #
    #     # create new  axis for the colorbar
    #     # figure = self.frame_axes.figure
    #     # _, y0, x1, y1 = self.rect
    #     # height = y1 - y0
    #     # x_adjusted_width = self._colorbar_width / len (self.axes)
    #     # x_adjusted_margin = self._colorbar_margin['right'] / len (self.axes)
    #     # lt_wid_top_ht = [x1 - (x_adjusted_width + x_adjusted_margin),
    #     #                  y0 + self._colorbar_margin['top'],
    #     #                  x_adjusted_width,
    #     #                  height - (self._colorbar_margin['top'] +
    #     #                            self._colorbar_margin['bottom'])]
    #     # self._colorbar_ax = figure.add_axes (lt_wid_top_ht)
    #     # if LooseVersion (matplotlib.__version__) >= LooseVersion ("1.6"):
    #     #     self._colorbar_ax.set_facecolor ('w')
    #     # else:
    #     #     self._colorbar_ax.set_axis_bgcolor ('w')
    #
    #     our_cmap = mpl_cm.get_cmap (cmap)
    #     # edge case where the data has a single value
    #     # yields a cryptic matplotlib error message
    #     # when trying to plot the color bar
    #     nb_ticks = 5 if norm.vmin != norm.vmax else 1
    #     ticks = np.linspace (norm.vmin, norm.vmax, nb_ticks)
    #     bounds = np.linspace (norm.vmin, norm.vmax, our_cmap.N)
    #
    #     # some colormap hacking
    #     cmaplist = [our_cmap (i) for i in range (our_cmap.N)]
    #     istart = int (norm (-offset, clip=True) * (our_cmap.N - 1))
    #     istop = int (norm (offset, clip=True) * (our_cmap.N - 1))
    #     for i in range (istart, istop):
    #         cmaplist[i] = (0.5, 0.5, 0.5, 1.)  # just an average gray color
    #     if norm.vmin == norm.vmax:  # len(np.unique(data)) == 1 ?
    #         return
    #     else:
    #         our_cmap = colors.LinearSegmentedColormap.from_list (
    #             'Custom cmap', cmaplist, our_cmap.N)
    #
    #     # self.glassbrain._cbar.set_cmap(our_cmap)
    #     # self.glassbrain._cbar.set_norm(norm)
    #     # self.glassbrain._cbar.set_ticks(ticks)
    #     # self.glassbrain._cbar.update_ticks()
    #
    #     self.glassbrain._cbar = ColorbarBase (
    #         self.glassbrain._colorbar_ax, ticks=ticks, norm=norm,
    #         orientation='vertical', cmap=our_cmap, boundaries=bounds,
    #         spacing='proportional', format='%.2g')
    #
    #     self.glassbrain._colorbar_ax.yaxis.tick_left ()
    #     tick_color = 'w' if self._black_bg else 'k'
    #     for tick in self.glassbrain._colorbar_ax.yaxis.get_ticklabels ():
    #         tick.set_color (tick_color)
    #     self.glassbrain._colorbar_ax.yaxis.set_tick_params (width=0)


def butterfly(ndvar, src, dest='mri', mri_resolution=False, black_bg=False, display_mode='lyrz',
                 threshold='auto', cmap=None, colorbar=False, alpha=0.7, vmin=None, vmax=None, plot_abs=True):

    if ndvar.has_dim('space'):
        p = plot.Butterfly(ndvar.norm('space'), vmin=vmin, vmax=vmax)
    else:
        p = plot.Butterfly(ndvar, vmin=vmin, vmax=vmax)

    gb = GlassBrain(ndvar, src, dest=dest, mri_resolution=mri_resolution, black_bg=black_bg, display_mode=display_mode,
                    threshold=threshold, cmap=cmap, colorbar=colorbar, alpha=alpha, vmin=vmin, vmax=vmax, plot_abs=True)

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
#     gb = butterfly(h, src, dest='surf', threshold=10)
#     p = plot.Butterfly(h.norm('space'))
#     gb = GlassBrain(h, src, dest='surf', threshold=5e-13)
#     p.link_time_axis(gb)


