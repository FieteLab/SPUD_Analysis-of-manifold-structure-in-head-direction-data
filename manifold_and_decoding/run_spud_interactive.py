'''March 25th 2019
More interactive version of SPUD fitting.'''


from __future__ import division
import numpy as np
import numpy.linalg as la
import sys, os 
import time, datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from sklearn import decomposition, manifold

sd=int((time.time()%1)*(2**31))
np.random.seed(sd)
curr_date=datetime.datetime.now().strftime('%Y_%m_%d')+'_'

gen_fn_dir = os.path.abspath('..') + '/shared_scripts'
sys.path.append(gen_fn_dir)

import general_file_fns as gff
gen_params = gff.load_pickle_file('../general_params/general_params.p')

from binned_spikes_class import spike_counts
import manifold_fit_and_decode_fns as mff
import fit_helper_fns as fhf
from dim_red_fns import run_dim_red

dir_to_save = gff.return_dir(gen_params['results_dir'] + '2019_06_03_interactive_curve_fits/')

session = 'Mouse28-140313'
fit_dim = 3
nKnots = 15
knot_order = 'wt_per_len'
penalty_type = 'mult_len'
train_frac = 0.8

print('Session: %s, fit dim: %d, nKnots: %d, knot_order: %s, penalty: %s, train_frac: %.2f'%(
    session, fit_dim, nKnots, knot_order, penalty_type, train_frac))
area = 'ADn'
state = 'Wake'
dt_kernel = 0.1
sigma = 0.1 # Kernel width

method = 'iso'
n_neighbors = 5

run_dim_red_here = True
if run_dim_red_here:
    rate_params = {'dt' : dt_kernel, 'sigma' : sigma}
    dim_red_params = {'n_neighbors' : n_neighbors, 'target_dim' : fit_dim}
    desired_nSamples = 15000

    session_rates = spike_counts(session, rate_params, count_type='rate', 
        anat_region='ADn')
    counts, tmp_angles = session_rates.get_spike_matrix(state)
    sel_counts = counts[:desired_nSamples]
    proj = run_dim_red(sel_counts, params=dim_red_params, method=method)
    embed = {state : proj, 'meas_angles' : tmp_angles[:desired_nSamples]}
else:
    load_dir = gen_params['results_dir'] + '2019_03_22_dim_red/'
    file_pattern = '%s_%s_kern_%dms_sigma_%dms_binsep_%s_embed_%s_%ddims_%dneighbors_*.p'%(
        session, area, sigma * 1000, dt_kernel * 1000, state, method, fit_dim, n_neighbors)
    # file_pattern = '%s_%s_kern_%dms_sigma_%dms_binsep_%s_embed_%s_multi_dim_%dneighbors_*.p'%(
    #     session, area, sigma * 1000, dt_kernel * 1000, state, method, n_neighbors)
    embed, fname = gff.load_file_from_pattern(load_dir+file_pattern)

curr_mani = embed[state]
nPoints = len(curr_mani)
nTrain = np.round(train_frac * nPoints).astype(int)

# Use measured angles to set origin and direction of coordinate increase
ref_angles = embed['meas_angles']

fit_params = {'dalpha' : 0.005, 'knot_order' : knot_order,
    'penalty_type' : penalty_type, 'nKnots' : nKnots}
train_idx = np.random.choice(nPoints, size=nTrain, replace=False)
test_idx = np.array([idx for idx in range(nPoints) if idx not in train_idx])
data_to_fit = curr_mani[train_idx].copy()
data_to_decode = curr_mani[test_idx].copy()

# Automated fit does:
# fit_result = mff.fit_manifold(data_to_fit, fit_params)
# And then matches it to a reference angle using mff.decode_from_passed_fit.
# Here let's look at the knot placement and pick a good one.
# Set up the fit class and initial knots
fitter = mff.PiecewiseLinearFit(data_to_fit, fit_params)
unord_knots = fitter.get_new_initial_knots()

# Generate knots and look at them to make sure we're happy with them. Mess around 
# with number of knots etc. if fits look bad.
# t0 = time.time()
init_knots = fitter.order_knots(unord_knots, method=fit_params['knot_order'])
loop_init = fhf.loop_knots(init_knots)
init_tt, init_curve = fhf.get_curve_from_knots(loop_init, 'eq_vel')
# print 'Time ', time.time()-t0

fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_to_fit[::2,0], data_to_fit[::2,1], data_to_fit[::2,2], c ='r', s=5)
ax.plot(loop_init[:,0], loop_init[:,1], loop_init[:,2], c='k', lw=2)
plt.show()

# Now optimize to get the best fit
t0 = time.time()
curr_fit_params = {'init_knots' : init_knots, 'penalty_type' : 
    fit_params['penalty_type']}
fitter.fit_data(curr_fit_params)
fit_results = dict(fit_params)
fit_results['init_knots'] = init_knots
fit_results['loop_init_knots'] = loop_init
fit_results['init_tt'], fit_results['init_curve'] = init_tt, init_curve

fit_results['final_knots'] = fitter.saved_knots[0]['knots']
fit_results['loop_final_knots'] = fhf.loop_knots(fitter.saved_knots[0]['knots'])
fit_results['tt'], fit_results['curve'] = fhf.get_curve_from_knots(
    fit_results['loop_final_knots'], 'eq_vel')

fit_results['fit_err'] = fitter.saved_knots[0]['err']
print 'Time ', time.time()-t0

i0 = 0; i1 = 1; i2 = 2
fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_to_fit[::2,i0], data_to_fit[::2,i1], data_to_fit[::2,i2], c=(0.5,0.5,0.5), s=5)
ax.plot(fit_results['loop_init_knots'][:,i0], fit_results['loop_init_knots'][:,i1], 
    fit_results['loop_init_knots'][:,i2], c='b', lw=2)
ax.plot(fit_results['loop_final_knots'][:,i0], fit_results['loop_final_knots'][:,i1], 
    fit_results['loop_final_knots'][:,i2], c='r', lw=2)
plt.show()

dec_angle, mse = mff.decode_from_passed_fit(data_to_decode, fit_results['tt'][:-1], 
    fit_results['curve'][:-1], ref_angles[test_idx])

to_save = {'fit_results' : fit_results, 'session' : session, 'area' : area, 'state' : state, 
            'embed_file' : fname if not run_dim_red_here else None} 
gff.save_pickle_file(to_save, dir_to_save + '%s_%s_dim%d_trainfrac%.2f_interactive_fits_sd%d.p'%(
                        session, state, fit_dim, train_frac, sd))

