import numpy as np
from mne import read_source_spaces
import warnings


def _save_stc_as_volume(fname, ndvar, src, dest='mri', mri_resolution=False):
    """Save a volume source estimate in a NIfTI file.
    Parameters
    ----------
    fname : string | None
        The name of the generated nifti file. If None, the image is only
        returned and not saved.
    ndvar : NDVar
        The source estimate
    src : list | string
        The list of source spaces (should actually be of length 1). If
        string, it is the filepath.

    dest : 'mri' | 'surf'
        If 'mri' the volume is defined in the coordinate system of
        the original T1 image. If 'surf' the coordinate system
        of the FreeSurfer surface is used (Surface RAS).
    mri_resolution: bool
        It True the image is saved in MRI resolution.
        WARNING: if you have many time points the file produced can be
        huge.
    Returns
    -------
    img : instance Nifti1Image
        The image object.
    """
    # check if file name
    if isinstance(src, basestring):
        print('Reading src file %s...' %src)
        src = read_source_spaces(src)
    else:
        src = src

    src_type = src[0]['type']
    if src_type != 'vol':
        raise ValueError('You need a volume source space. Got type: %s.'
                         % src_type)

    if ndvar.has_dim('space'):
        ndvar = ndvar.norm('space')

    if ndvar.has_dim('time'):
        data = ndvar.get_data(('source', 'time'))
    else:
        data = ndvar.get_data('source')
        if data.ndim == 1:
            data = data[:, np.newaxis]
    n_times = data.shape[1]
    shape = src[0]['shape']
    shape3d = (shape[2], shape[1], shape[0])
    shape = (n_times, shape[2], shape[1], shape[0])
    vol = np.zeros(shape)

    if mri_resolution:
        mri_shape3d = (src[0]['mri_height'], src[0]['mri_depth'],
                       src[0]['mri_width'])
        mri_shape = (n_times, src[0]['mri_height'], src[0]['mri_depth'],
                     src[0]['mri_width'])
        mri_vol = np.zeros(mri_shape)
        interpolator = src[0]['interpolator']

    n_vertices_seen = 0
    for this_src in src: # loop over source instants, which is basically one element only!
        assert tuple(this_src['shape']) == tuple(src[0]['shape'])
        mask3d = this_src['inuse'].reshape(shape3d).astype(np.bool)
        n_vertices = np.sum(mask3d)

        for k, v in enumerate(vol):  # loop over time instants
            stc_slice = slice(n_vertices_seen, n_vertices_seen + n_vertices)
            v[mask3d] = data[stc_slice, k]

        n_vertices_seen += n_vertices

    if mri_resolution:
        for k, v in enumerate(vol):
            mri_vol[k] = (interpolator * v.ravel()).reshape(mri_shape3d)
        vol = mri_vol

    vol = vol.T

    if mri_resolution:
        affine = src[0]['vox_mri_t']['trans'].copy()
    else:
        affine = src[0]['src_mri_t']['trans'].copy()
    if dest == 'mri':
        affine = np.dot(src[0]['mri_ras_t']['trans'], affine)
    affine[:3] *= 1e3

    try:
        import nibabel as nib  # lazy import to avoid dependency
    except ImportError:
        raise ImportError("nibabel is required to save volume images.")

    header = nib.nifti1.Nifti1Header()
    header.set_xyzt_units('mm', 'msec')
    if ndvar.has_dim('time'):
        header['pixdim'][4] = 1e3 * ndvar.time.tstep
    else:
        header['pixdim'][4] = None
    with warnings.catch_warnings(record=True):  # nibabel<->numpy warning
        img = nib.Nifti1Image(vol, affine, header=header)
        if fname is not None:
            nib.save(img, fname)
    return img


# if __name__ == '__main__':
#     from nilearn.plotting import plot_glass_brain
#     from nilearn.image import index_img
#     import cPickle as pickle
#     ROOTDIR = 'G:/My Drive/Proloy/'
#     fname = ROOTDIR + 'Group analysis/Dataset audspec-u.pickled'
#     ds = pickle.load(open(fname, 'rb'))
#     h = ds['trf'].mean('case')
#     fname = ROOTDIR + '/mri/fsaverage/bem/fsaverage-vol-10-src.fif'
#     src = read_source_spaces(fname)
#     niftimg = _save_stc_as_volume(None, h, src, 'surf')
#     gb = plot_glass_brain (
#         index_img(niftimg, 20), title='plot_glass_brain',
#         black_bg=True, display_mode='lyrz', threshold=1e-13
#     )