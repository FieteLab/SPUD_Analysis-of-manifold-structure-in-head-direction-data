'''March 22nd 2019
Various functions to help with the decoding.
'''

from __future__ import division
import numpy as np
from numpy import linalg as la
from pandas import cut
import sys, os
from sklearn.neighbors import NearestNeighbors

gen_fn_dir = os.path.abspath('../..') + '/shared_scripts'
sys.path.append(gen_fn_dir)

def get_closest_manifold_coords(mani_pts, mani_coords, input_pts, return_all=False):
    '''Use the nearest neighbors function to project input_pts to the 
    manifold. Manifold is represented by mani_pts, with coordinates mani_coords.
    Returns either the coordinates of the closest manifold point, or the coords along
    with the distances from the nearest manifold points and the manifold points in the
    higher-dimensional embedding space.'''
    
    neighbgraph = NearestNeighbors(n_neighbors=1).fit(mani_pts)
    tmp_dists, tmp_inds = neighbgraph.kneighbors(input_pts)
    # Squeeze is needed for one-d coordinates (and shouldn't affect higher-d 
    # coordinates but check if you use for higher d)
    dists_from_mani = np.squeeze(tmp_dists)
    inds_of_nearest_mani_pts = np.squeeze(tmp_inds)
    input_coords = mani_coords[inds_of_nearest_mani_pts]
    if return_all:
        return input_coords, dists_from_mani, inds_of_nearest_mani_pts
    else:
        return input_coords

'''A few functions to turn a list of knots into a more finely sampled curve,
along with coordinates along the curve.
'''
def loop_knots(x):
    '''The curve is described by a list of knots. For a closed curve, when generating
    the curve (by interpolating between the knots) we want to repeat the first knot at 
    the end. This function just adds on the first point of x at the end.
    '''

    return np.append(x, np.array([x[0]]), axis=0)

def get_coord_bins_from_knots(knot_list):
    '''Take the interval [0,1] and return a set of bins [0, b1, b2, ..., 1]
    such that [0, b1] is proportional to the distance between the first and second knots,
    [b1, b2] is proportional to the distance between second and third knot, and so on.
    This is used to determine the rate at which the coordinate increases along the 
    manifold.'''

    segment_lens = [la.norm(y - x) for x, y in zip(knot_list[:-1], knot_list[1:])]
    bin_right_edges = np.cumsum(segment_lens) / np.sum(segment_lens)
    if bin_right_edges[-1] < 1:
        print 'Last one is <1, ', bin_right_edges[-1]
        bin_right_edges[-1] = 1
    return np.concatenate(([0], bin_right_edges))

def get_linear_interp_general(knots, param_bins, param_vals):
    '''Return a piecewise linear curve that interpolates between the knots 
    in knots. 
    Knots should be looped if we want a closed curve.
    '''

    param_int_idx = cut(param_vals, bins=param_bins, labels=False, include_lowest=True)
    # Turns the coordinate along the spline into a set of local coordinates between
    # 0 and 1 that run along each line segment, so that we can generate the curve.
    param_rsc = (param_vals - param_bins[param_int_idx]) / (
        param_bins[param_int_idx + 1] - param_bins[param_int_idx])
    ret_curve = knots[param_int_idx] + (knots[param_int_idx + 1] -
                                        knots[param_int_idx]) * param_rsc[:, np.newaxis]
    return ret_curve

def get_curve_from_knots(knot_list, bin_type, dt=0.005):
    '''Turns knot_list into a curve in the embedding space along with 
    appropriate coordinates. knots should be looped if we want a closed 
    curve.
    eq_int option is there for historical reasons, but isn't used.
    '''

    tt = np.arange(0, 1 + 1e-10, dt)
    if bin_type == 'eq_int':
        # I.e., line segment between each pair of knots gets equal consideration
        t_bins = np.linspace(0, 1., len(knot_list))
    elif bin_type == 'eq_vel':
        # Line segment between each pair of knots receives weight according
        # to distance in embedding or neural space
        t_bins = get_coord_bins_from_knots(knot_list)
    else:
        print 'Unknown bin type'
        return np.nan
    curve = get_linear_interp_general(knot_list, t_bins, tt)
    return tt, curve

'''Set of functions to find distances and nearest neighbors between points and a set. 
Primarily used in ordering the knots, though I used one of these in an earlier version
of the decoding. But for multiple calls I think the scikit nearest neighbors setup
is faster and cleaner, so switched to that (can compare them at some point).
'''

def find_smallest_dist_idx(pt, A):
    '''Find the smallest distance between the point pt and the 
    list of points in the array A. '''
    return np.argmin(np.linalg.norm(pt - A, axis=1))

def dist_pt_to_line_segment(l0, l1, p):
    '''Distance from point p to the line segment between l0 and l1'''
    line_len_sq = la.norm(l1 - l0)**2
    if line_len_sq == 0:
        return la.norm(l0 - p)
    t = max(0, min(1, np.dot(p - l0, l1 - l0) / line_len_sq))
    projection = l0 + t * (l1 - l0)
    return la.norm(p - projection)

def get_distances_near_line(l0, l1, data, r):
    '''Given some data, select down to points that are within a radius r of both l0 and l1,
    and return their distances from the line segment between l0 and l1'''
    ret_list = [dist_pt_to_line_segment(l0, l1, x) for x in data if la.norm(x - l0) < r and
                la.norm(x - l1) < r]
    return np.array(ret_list)

# No longer used so delete later
# def find_smallest_dist(pt, A):
#     '''Find the smallest distance between the point pt and the 
#     list of points in the array A. '''
#     return np.min(np.linalg.norm(pt - A, axis=1))




