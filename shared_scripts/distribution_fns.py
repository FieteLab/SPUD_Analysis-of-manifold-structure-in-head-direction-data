# February 2nd 2017
# Functions to get distributions and histograms and such

from __future__ import division
import numpy as np
import numpy.linalg as la
from sklearn import metrics
import angle_fns as af


def get_angle_bins(nBins, bin_type='angle'):
    '''Convenience function, since we do this so often'''
    d_theta = 2 * np.pi / nBins
    if bin_type == 'angle':
        bin_edges = np.arange(0, 2 * np.pi + 1e-10, d_theta)
    elif bin_type == 'delta':
        bin_edges = np.arange(-np.pi, np.pi + 1e-10, d_theta)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.
    return bin_edges, bin_centers

def get_autocorrelation(x):
    mn = np.mean(x)
    mean_subtracted_x = x - mn
    ac = np.correlate(mean_subtracted_x, mean_subtracted_x, mode='full')
    norm_fac = np.max(ac)
    # print norm_fac - np.var(x)*len(x) # Check that the normalization is N * var
    normalized_ac = ac / norm_fac
    return normalized_ac[int(np.floor(normalized_ac.size / 2)):]

def get_autocorrelation_no_mean_sub(x):
    # mn = np.mean(x); mean_subtracted_x = x-mn
    ac = np.correlate(x, x, mode='full')
    norm_fac = np.max(ac)
    # print norm_fac - np.var(x)*len(x) # Check that the normalization is N * var
    normalized_ac = ac / norm_fac
    return normalized_ac[int(np.floor(normalized_ac.size / 2)):]

def fixed_ac(x, t=1):
    return np.corrcoef(np.array([x[0:len(x) - t], x[t:len(x)]]))[0, 1]

def match_angles_to_shift_and_flip(first_angles, second_angles, shift_dt=0.1):
    '''Shift and flip second_angle to match first_angles as closely as possible'''

    # Reverse orientation (we'll test both)
    flip_second = 2 * np.pi - second_angles

    shifts = np.arange(0, 2 * np.pi, shift_dt)

    reg_diff = np.zeros_like(shifts)
    flip_diff = np.zeros_like(shifts)

    # Now look for optimal shift
    for i, sh in enumerate(shifts):
        tmp = np.mod(second_angles + sh, 2 * np.pi)
        reg_diff[i] = la.norm(af.abs_angular_diff(first_angles, tmp))

        tmp2 = np.mod(flip_second + sh, 2 * np.pi)
        flip_diff[i] = la.norm(af.abs_angular_diff(first_angles, tmp2))

    min_reg = np.argmin(reg_diff)
    min_flip = np.argmin(flip_diff)

    if reg_diff[min_reg] < flip_diff[min_flip]:
        final_second = np.mod(second_angles + shifts[min_reg], 2 * np.pi)
    else:
        final_second = np.mod(flip_second + shifts[min_flip], 2 * np.pi)

    return final_second

def get_centers(inp_arr):
    return (inp_arr[:-1] + inp_arr[1:]) / 2.

def get_mean_in_bins(inp_array, bin_edges):
    # inp_array is N x T, and bin_edges is of the form [idx0, idx1, ...,idxN]
    # returns out_array of shape N x len(bin_edges)-1, where the 0-th column is
    # the average of inp_array[:,idx0:idx1], etc.
    nBins = len(bin_edges) - 1
    out_array = np.zeros((inp_array.shape[0], nBins))
    for i, start_idx, end_idx in zip(np.arange(nBins), bin_edges[:-1], bin_edges[1:]):
        # print i, start_idx, end_idx
        out_array[:, i] = np.mean(inp_array[:, start_idx:end_idx], 1)
    return out_array

