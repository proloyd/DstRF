"""generates basis"""

import numpy as np


def gaussian_basis(nlevel, span):
    x = span
    means = np.linspace( x[-1]/nlevel, x[-1]*(1 - 1/nlevel), num = nlevel-1 )
    stds = 8.5
    W = []

    for count in list(range(nlevel-1)):
        W.append( np.exp( -(x - means[count])**2/(2*stds**2) ) )

    W = np.array(W)

    return W.T/np.max(W)