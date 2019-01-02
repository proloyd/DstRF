import numpy as np
from math import sqrt
# from cython.view cimport array as cvarray

"""To find group_list use the conn matrix as below:
>>>conn = src.connectivity()
>>>group_list = [np.unique(conn[np.where(conn[:, 0] == i)]) for i in np.unique(conn[:, 0])]

To find group_array and group_size use the conn matrix as below:
>>>group_size = np.empty(0, dtype=int)
>>>group_array = np.empty(0, dtype=int)
>>>for i in np.unique(conn[:, 0]):
>>>    tmp = np.unique(conn[np.where(conn[:, 0] == i)])
>>>    group_size = np.append(group_size, tmp.size)
>>>    group_array = np.append(group_array, tmp)
>>>group_idx = np.cumsum(group_size)
>>>w = np.sqrt(group_size)
"""


def g_o_g(x, mu, group_array, group_idx, w):
    """Overlapping Group penalty

    Parameters
    ----------
    x : ndarray

    mu : float
        Amount of penalization to use
    group_list: list
        Each list element contains members of a group. Note that the
        groups could be overlapping.

    Return
    ------
        float
    """
    norm = 0
    n_groups = group_idx.size
    for i in range(n_groups):
        # cython changes
        if i == 0:
            group = group_array[0:group_idx[0]]
        else:
            group = group_array[group_idx[i - 1]:group_idx[i]]
        norm += w[i] * sqrt((x[group] ** 2).sum())

    return mu * norm


def prox_o_g(u, mu, group_array, group_idx, w, tol):
    """proximal operator for overlapping groups

    Solves problem of the form:
    .. math::
        (1/2)*||z - u||^2_2 + mu * sum_j (w_j * ||z_j||_2)
    where :math:`z_j` is the coefficients in the :math:`j`-th group.
    Note that the groups may be overlapping, meaning :math: `z_j`
    and :math: `z_k` can contain same coefficients.
    TODO to be wriiten in Cython

    Parameters
    ----------
    u : ndarray
        the 1d array, on which proximal operator will be applied.
    mu : float
        Amount of penalization to use
    group_array : ndarry
        contains connectivity info, look at the top to see how to
        get it from src.connectivity().
    group_idx : ndarray
        contains connectivity info, look at the top to see how to
        get it from src.connectivity().
    w : ndarray
        weights to individual groups. Usually set to sqrt(group.size).
        look at the top to see how to get it from src.connectivity()
    tol : float
        Relative tolerance. ensures ||(z - y) / y|| < tol,
        where y is the approximate solution and z is the
        true solution.

    Returns
    -------
        ndarray
    """
    z = u.copy()
    n_groups = group_idx.size
    # identify as many zero groups as possible
    nonzero_flags = np.ones(n_groups, dtype=bool)
    while True:
        count = 0
        for i in range(n_groups):
            # cython changes
            if i == 0:
                group = group_array[0:group_idx[0]]
            else:
                group = group_array[group_idx[i-1]:group_idx[i]]
            if sqrt((z[group] ** 2).sum()) <= mu * w[i]:
                z[group] = 0
                # upto here
                if nonzero_flags[i]:
                    count += 1
                nonzero_flags[i] = False
        # stopping condition
        if count == 0:
            break

    if not np.any(nonzero_flags):
        return z
    else:
        u = z.copy()    # cython changes
        count = 0
        q = np.empty((np.count_nonzero(nonzero_flags), ) + z.shape)
        # # Memoryview on a Cython array
        # cyarr = cvarray(shape=(3, 3, 3), itemsize=sizeof(int), format="i")
        # cdef int [:, :, :] cyarr_view = cyarr
        while True:
            if count == 0:
                y = np.zeros(z.shape)
                n = 0
            else:
                y = n * u   # cython changes
            j = 0
            for i in range(n_groups):
                if nonzero_flags[i]:
                    if count == 0:
                        temp = z  # cython changes
                    else:
                        temp = z - q[j]
                    if i == 0:
                        group = group_array[0:group_idx[0]]  # cython changes
                        if count == 0:
                            n += group_idx[0]
                    else:
                        group = group_array[group_idx[i - 1]:group_idx[i]]  # cython changes
                        if count == 0:
                            n += (group_idx[i] - group_idx[i-1])
                    temp[group] = prox(temp[group], mu)  # cython changes, needs to be clever
                    if count == 0:
                        q[j] = temp - z  # cython changes
                    else:
                        q[j] += (temp - z)  # cython changes, note full update is not required
                    y += w[i] * (temp - q[j])  # cython changes
                    j += 1

            if count == 0:
                y += n * u  # may put n * u in Cache??

            y /= (2 * n)  # cython changes
            count += 1
            if ((z - y) ** 2).sum() < tol:  # cython changes
                break
            else:
                z = y  # cython changes

    return z


def prox(x, mu):
    norm = sqrt((x ** 2).sum())
    if norm > 0:
        x_n = x / norm
    else:
        x_n = x
    return max(norm-mu, 0) * x_n
