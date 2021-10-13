'''March 21st 2019
Decode from saved fit 
'''

from __future__ import division
import numpy as np
import numpy.linalg as la
import sys, os, glob
import time, datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sd=int((time.time()%1)*(2**31))
np.random.seed(sd)
curr_date=datetime.datetime.now().strftime('%Y_%m_%d')+'_'

gen_fn_dir = os.path.abspath('..') + '/shared_scripts'
sys.path.append(gen_fn_dir)

import general_file_fns as gff
gen_params = gff.load_pickle_file('../general_params/general_params.p')

from binned_spikes_class import spike_counts
import manifold_fit_and_decode_fns as mfd
import fit_helper_fns as fhf
import distribution_fns as df
import angle_fns as af
# import tc_and_tc_decoding_fns as tcf

cols = gen_params['cols']
session = 'Mouse28-140313'
area = 'ADn'
state = 'Wake'
dt_kernel = 0.1
sigma = 0.1 # Kernel width

method = 'iso'
n_neighbors = 5
embed_dim = 3
train_frac = 0.8

# Load the dimensionality reduced data
dd = gen_params['results_dir'] + '2019_03_22_dim_red/'
file_pattern = '%s_%s_kern_%dms_sigma_%dms_binsep_%s_embed_%s_%ddims_%dneighbors_*.p'%(
    session, area, sigma * 1000, dt_kernel * 1000, state, method, embed_dim, n_neighbors)
embed, embed_fname = gff.load_file_from_pattern(dd+file_pattern)

curr_mani = embed[state]
nPoints = len(curr_mani)

# Load the fits
fit_source = 'single_interactive'
if fit_source == 'multiple':
    # This version from multiple automated fits
    dd = gen_params['results_dir'] + '2019_03_22_curve_fits/'
    file_pattern = '%s_%s_dim%d_trainfrac%.2f_decode_errors_sd*.p'%(
        session, state, embed_dim, train_frac)
    fit_data, fit_fname = gff.load_file_from_pattern(dd + file_pattern)
    # Select fit
    k = ('Mouse28-140313', 3, 15, 'wt_per_len', 'mult_len', 0.8)
    # Or do k = fit_data['fit_results'].keys()[0] or just pick one
    sel_knots = fit_data['fit_results'][k][0][-1]
elif fit_source == 'single_interactive':
    # This version from single interactive fit
    dd = gen_params['results_dir'] + '2019_06_03_interactive_curve_fits/'
    file_pattern = '%s_%s_dim%d_trainfrac%.2f_interactive_fits_sd*.p'%(
        session, state, embed_dim, train_frac)
    fit_data, fit_fname = gff.load_file_from_pattern(dd + file_pattern)
    sel_knots = fit_data['fit_results']['final_knots']

assert fit_data['embed_file'] == embed_fname, 'Fit seems to be to different data'
meas_angles = embed['meas_angles']

data_to_decode = curr_mani
loop_sel_knots = fhf.loop_knots(sel_knots)
tt, curve = fhf.get_curve_from_knots(loop_sel_knots, 'eq_vel')
dec_angle, mse = mfd.decode_from_passed_fit(data_to_decode, tt[:-1], 
    curve[:-1], meas_angles)

si = 2000; ei = 6000
fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(111)
ax.plot(meas_angles[si:ei], color='k', lw=2, label='Measured')
ax.plot(dec_angle[si:ei], color=cols['Wake'], lw=2, label='Decoded')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Angle (rad)')
ax.legend()
plt.show()

# Plot diffusion curve for decoded angle
delta_ang_meas = af.signed_angular_diff(meas_angles[1:],
    meas_angles[:-1])
delta_ang_mani = af.signed_angular_diff(dec_angle[1:], dec_angle[:(-1)])

# Diffusion
time_seps = np.arange(1,6)
dec_diffusion = af.get_diffusion_curve(dec_angle, time_seps)
meas_diffusion = af.get_diffusion_curve(meas_angles, time_seps, nan_safe=True)

def add_0(x):
    return np.concatenate(([0],x))

difftimes = add_0(time_seps * dt_kernel)
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax.plot(difftimes, add_0(dec_diffusion), lw=4, color=cols['Wake'])
ax.plot(difftimes, add_0(meas_diffusion), lw=4,color='k')
ax.set_xlim([0, 0.5])
ax.set_ylim([0, 0.2])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Mean squared change (rad^2)')
plt.show()


