'''March 22nd 2019
Some general functions for computing angular differences and other useful things for 
working with circular variables
'''

from __future__ import division
import numpy as np
import numpy.linalg as la

def wrap_angles(ang_list, ang_type='angle'):
    ''' Wrap a list of angles to [0,2*np.pi] if type is angle, and to
    [-pi, pi] if type is delta. Don't use this with ang_type='delta' if 
    the data contains NaNs.'''
    x = np.mod(ang_list, 2*np.pi)
    if ang_type=='delta':
        x[x>np.pi] = x[x>np.pi] - 2*np.pi
    return x

def signed_angular_diff(x, y):
    '''Compute the signed angular difference between x and y, and 
    return a value between -pi and pi. This is a little messy,
    because the angles can sometimes contain NaN's (when they weren't
    measured). Statements of the form np.array([2,np.nan])>3 give
    a warning, but statemants like np.nan>3 are fine. To avoid a warning
    looping through rather than vectorizing.'''
    
    # Check for scalar
    if isinstance(x, float):
        diff = np.mod(x-y, 2*np.pi)
        if diff>np.pi:
            diff = diff - 2*np.pi
    else:
        diff = np.mod(np.array(x)-np.array(y), 2*np.pi)
        # We'd like to say diff[diff>np.pi] = diff[diff>np.pi] - 2*np.pi
        for i in range(len(diff)):
            if diff[i]>np.pi:
                diff[i] = diff[i] - 2*np.pi
    return diff

def abs_angular_diff(theta_1, theta_2):
    signed_diff = signed_angular_diff(theta_1, theta_2)
    return np.abs(signed_diff)

def shifted_angular_diffs(angle_list, bin_sep):
    '''Find the signed angular difference of elements in angle_list separated by bin_sep.'''
    angle_array = np.array(angle_list)
    
    if len(angle_array)<=bin_sep:
        return np.array([])
    else:
        return signed_angular_diff(angle_array[bin_sep:], angle_array[:(-bin_sep)])

def get_variance_curve(angle_list, t_sep, nan_safe=False):
    '''Gets the variance of changes in angle at each t_sep.
    '''
    vr_list = []
    for dt in t_sep:
        curr_delta = shifted_angular_diffs(angle_list, dt)
        if nan_safe:
            vr_list.append(np.nanvar(curr_delta))
        else:
            vr_list.append(np.var(curr_delta))

    return np.array(vr_list)

def get_diffusion_curve(angle_list, t_sep, nan_safe=False):
    '''Gets the squared magnitude of changes in angle at each t_sep.
    '''
    mag_list = []
    for dt in t_sep:
        curr_delta = shifted_angular_diffs(angle_list, dt)
        if nan_safe:
            mag_list.append(np.nanmean(curr_delta**2))
        else:
            mag_list.append(np.mean(curr_delta**2))
    return np.array(mag_list)

def shift_to_match_given_trace(dec_params, actual_angles, shift_dt=0.1):
    '''Match dec_params to actual angles up to a shift and flip (assuming both are
    between 0 and 2*np.pi)'''

    flip_dec = 2 * np.pi - dec_params

    no_nan = np.isfinite(actual_angles)
    shifts = np.arange(0, 2 * np.pi, shift_dt)

    reg_diff = np.zeros_like(shifts)
    flip_diff = np.zeros_like(shifts)

    # Now look for optimal shift
    for i, sh in enumerate(shifts):
        tmp = np.mod(dec_params + sh, 2 * np.pi)
        reg_diff[i] = la.norm(abs_angular_diff(actual_angles[no_nan], tmp[no_nan]))

        tmp2 = np.mod(flip_dec + sh, 2 * np.pi)
        flip_diff[i] = la.norm(abs_angular_diff(actual_angles[no_nan], tmp2[no_nan]))

    min_reg = np.argmin(reg_diff)
    min_flip = np.argmin(flip_diff)

    if reg_diff[min_reg] < flip_diff[min_flip]:
        # print 'Shift = ', shifts[min_reg]
        final_dec = np.mod(dec_params + shifts[min_reg], 2 * np.pi)
        final_shift = shifts[min_reg]; final_flip = False
    else:
        # print 'Shift = ', shifts[min_flip],' and flipped'
        final_dec = np.mod(flip_dec + shifts[min_flip], 2 * np.pi)
        final_shift = shifts[min_flip]; final_flip = True

    # Also compute the MSE on non nan angles
    mse = np.mean(abs_angular_diff(actual_angles[no_nan], final_dec[no_nan])**2)

    return final_dec, mse, final_shift, final_flip

def circmean(x, thresh=0.5, axis=None):
    ''' Computes a circular mean. Has a threshold for nans. If 50% of the angle list is a
    nan, then returns a nan. Otherwise, returns the circular mean of the remaining angles.'''
    x = np.asarray(x)
    y = x[np.isfinite(x)]
    if (float(len(y)) / len(x)) <= thresh:
        return np.nan
    else:
        mean_angle = np.arctan2(np.mean(np.sin(y)), np.mean(np.cos(y)))
    if mean_angle < 0:
        mean_angle += 2 * np.pi
    return mean_angle
