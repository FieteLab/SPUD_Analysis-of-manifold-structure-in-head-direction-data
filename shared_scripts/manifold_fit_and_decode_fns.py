'''
March 22nd 2019
Functions to fit a 1D piecewise linear spline to a pointcloud and use
it to do decoding.
'''

from __future__ import division
import numpy as np
import numpy.linalg as la
from pandas import cut
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors

import angle_fns as af
import fit_helper_fns as fhf

class PiecewiseLinearFit:
    '''Fits a piecewise linear curve to passed data. The curve runs through
    a series of knots.'''

    def __init__(self, data_to_fit, params):
        self.data_to_fit = data_to_fit
        self.nDims = data_to_fit.shape[1]
        self.nKnots = params['nKnots']
        self.saved_knots = []

        # This sets the resolution at which the curve is sampled
        # Note that we don't care about these coordinates too much, since
        # they won't be used for decoding: they're just a way to generate the
        # curve so that we can compute fit error
        self.tt = np.arange(0, 1 + params['dalpha'] / 2., params['dalpha'])
        self.t_bins, self.t_int_idx, self.t_rsc = self.global_to_local_coords()

    def get_new_initial_knots(self, method='kmeans'):
        '''Place the initial knots for the optimization to use.'''
        if method == 'kmeans':
            kmeans = KMeans(n_clusters=self.nKnots, max_iter=300).fit(self.data_to_fit)
            return kmeans.cluster_centers_
        else:
            print 'Unknown method'

    def order_knots(self, knots, method='nearest'):
        '''Order the initial knots so that we can draw a curve through them. 
        Start with a randomly chosen knot and then successively move to the
        "nearest" knot, where nearest can be determined by a specified method.'''

        ord_knots = np.zeros_like(knots)
        rem_knots = knots.copy()

        # Pick a random knot to start
        next_idx = np.random.choice(len(rem_knots))
        ord_knots[0] = rem_knots[next_idx]

        for i in range(1, len(knots)):
            rem_knots = np.delete(rem_knots, next_idx, axis=0)
            if method == 'nearest':
                # Nearest ambient distance
                next_idx = fhf.find_smallest_dist_idx(ord_knots[i - 1], rem_knots)
            elif method == 'wt_per_len':
                # Choose the closest as measured by density (wt_per_len)
                dists = np.linalg.norm(ord_knots[i - 1] - rem_knots, axis=1)
                r = np.min(dists)
                wts = np.array([np.sum(np.exp(-fhf.get_distances_near_line(ord_knots[i - 1],
                    k, self.data_to_fit, r) / r)) for k in rem_knots])
                # Used to be wts / (dists**alpha)
                wt_per_len = wts / (dists)
                next_idx = np.argmax(wt_per_len)
            ord_knots[i] = rem_knots[next_idx].copy()
        return ord_knots

    def fit_data(self, fit_params):
        '''Main function to fit the data. Starting from the initial knots
        move them to minimize the distance of points to the curve, along with
        some (optional) penalty.'''

        save_dict = {'fit_params': fit_params}

        def cost_fn(flat_knots, penalty_params = fit_params):
            knots = np.reshape(flat_knots.copy(), (self.nKnots, self.nDims))
            loop_knots = fhf.loop_knots(np.reshape(flat_knots.copy(), (self.nKnots, self.nDims)))
            fit_curve = loop_knots[self.t_int_idx] + (loop_knots[self.t_int_idx + 1] -
                                            loop_knots[self.t_int_idx]) * self.t_rsc[:, np.newaxis]
            neighbgraph = NearestNeighbors(n_neighbors=1).fit(fit_curve)
            dists, inds = neighbgraph.kneighbors(self.data_to_fit)
            
            if penalty_params['penalty_type'] == 'none':
                cost = np.sum(dists)
            elif penalty_params['penalty_type'] == 'mult_len':
                cost = np.sum(dists) * self.tot_len(loop_knots)
            elif penalty_params['penalty_type'] == 'add_len': 
                cost = np.mean(dists) + penalty_params['len_coeff'] * self.tot_len(loop_knots)
            return cost
        
        init_knots = fit_params['init_knots']
        flat_init_knots = init_knots.flatten()
        fit_result = minimize(cost_fn, flat_init_knots, method='Nelder-Mead',
                              options={'maxiter': 3000})
        # print fit_result.fun
        knots = np.reshape(fit_result.x.copy(), (self.nKnots, self.nDims))
        save_dict = {'knots' : knots, 'err' : fit_result.fun,
            'init_knots' : init_knots}
        self.saved_knots.append(save_dict)

    # Various utility functions
    def global_to_local_coords(self):
        '''tt is a global coordinate that runs from 0 to 1. But the curve is made
        up of a series of line segments which have local coordinates. So we want to break
        tt into equally spaced sets, each corresponding to one line segment. 
        Note that these coordinates aren't used for decoding, just to generate the curve,
        so that the rate at which they increase around the curve doesn't matter, as long 
        as we generate the curve at a decent resolution. '''

        # Equally spaced bins of tt, with t_bins[0] to t_bins[1] corresponding to
        # the first line segment and so on.
        t_bins = np.linspace(0, 1., self.nKnots + 1)
        # For each element in tt, figure out which bin it should lie in
        # Replace cut later if we want
        t_int_idx = cut(self.tt, bins=t_bins, labels=False, include_lowest=True)

        # Now get the local coordinates along each line segment
        # t_int_idx is the link between this and the global coordinate and will
        # tell us which linear function to apply.
        t_rsc = (self.tt - t_bins[t_int_idx]) / (t_bins[t_int_idx + 1] - t_bins[t_int_idx])

        return t_bins, t_int_idx, t_rsc

    def get_curve_from_knots_internal(self, inp_knots):
        '''Turn a list of knots into a curve, sampled at the pre-specified
        resolution.'''

        # Repeat the first knot at the end so we get a loop.
        loop_knots = fhf.loop_knots(inp_knots)
        return loop_knots[self.t_int_idx] + (loop_knots[self.t_int_idx + 1] -
            loop_knots[self.t_int_idx]) * self.t_rsc[:, np.newaxis]

    def distance_from_curve(self, inp_knots):
        '''Cost function to test a given set of knots.
        Assuming knots aren't looped around '''

        fit_curve = self.get_curve_from_knots_internal(inp_knots)
        neighbgraph = NearestNeighbors(n_neighbors=1).fit(fit_curve)
        dists, inds = neighbgraph.kneighbors(self.data_to_fit)
        cost = np.sum(dists)
        return cost

    def tot_len(self, loop_knot_list):
        ls_lens = la.norm(loop_knot_list[1:]-loop_knot_list[:-1],axis=1)
        return np.sum(ls_lens)


def fit_manifold(data_to_fit, fit_params):
    '''fit_params takes nKnots : number of knots, dalpha : resolution for
    sampled curve, knot_order : method to initially order knots, penalty_type : 
    penalty'''
    # fit_params is a superset of the initial params that PiecewiseLinearFit needs
    fitter =  PiecewiseLinearFit(data_to_fit, fit_params)
    unord_knots = fitter.get_new_initial_knots()
    init_knots = fitter.order_knots(unord_knots, method=fit_params['knot_order'])
    curr_fit_params = {'init_knots' : init_knots, 'penalty_type' : 
        fit_params['penalty_type']}
    fitter.fit_data(curr_fit_params)
    fit_results = dict(fit_params)
    fit_results['init_knots'] = init_knots
    # The fit class appends the results of each fit to a list called saved_knots
    # Here we're just using the class once, hence saved_knots[0]
    fit_results['final_knots'] = fitter.saved_knots[0]['knots']
    fit_results['fit_err'] = fitter.saved_knots[0]['err']
    fit_results['loop_final_knots'] = fhf.loop_knots(fitter.saved_knots[0]['knots'])
    fit_results['tt'], fit_results['curve'] = fhf.get_curve_from_knots(
        fit_results['loop_final_knots'], 'eq_vel')
    return fit_results

def decode_from_passed_fit(data_to_decode, fit_coords, fit_curve, ref_angles):
    #  When calling, trim fit_coords and fit_curve before passing so first and last entries 
    # aren't identical, though I suspect it won't matter (check this)
    # loop_tt, loop_curve = fit_results['tt'], fit_results['curve']

    # Multiply coords by 2*pi so that we can compare with angles
    unshft_coords = fhf.get_closest_manifold_coords(fit_curve, 
        2*np.pi*fit_coords, data_to_decode)
    dec_angle, mse, shift, flip = af.shift_to_match_given_trace(unshft_coords,
        ref_angles)
    return dec_angle, mse
