'''November 26th 2018
Functions for dimensionality reduction.'''

from __future__ import division
import numpy as np
import numpy.linalg as la
from sklearn import decomposition, manifold

def run_dim_red(inp_data, params, method='iso', stabilize=True):
    # Variance stabilization option included, since we're usually 
    # working with Poisson-like data
    if stabilize:
        data_to_use = np.sqrt(inp_data)
    else:
        data_to_use = inp_data.copy()
    if method == 'iso':
        iso_instance = manifold.Isomap(params['n_neighbors'], 
            params['target_dim'])
        proj_data = iso_instance.fit_transform(data_to_use)
    return proj_data