'''March 21st 2019
Run dimensionality reduction on spike counts.
'''

from __future__ import division
import numpy as np
import numpy.linalg as la
import sys, time, os, datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition, manifold

sd=int((time.time()%1)*(2**31))
np.random.seed(sd)
curr_date=datetime.datetime.now().strftime('%Y_%m_%d')+'_'

gen_fn_dir = os.path.abspath('..') + '/shared_scripts'
sys.path.append(gen_fn_dir)

import general_file_fns as gff

gen_params = gff.load_pickle_file('../general_params/general_params.p')

from binned_spikes_class import spike_counts
from dim_red_fns import run_dim_red

cols = gen_params['cols']
dir_to_save = gff.return_dir(gen_params['results_dir'] + '2019_06_03_dim_red/')

command_line = False
if command_line:
    session = sys.argv[1]
    state = sys.argv[2]
    # If condition is 'joint' should unpack state into first and second
    condition = sys.argv[3]
    target_dim = int(sys.argv[4])
    desired_nSamples = int(sys.argv[5])
else:
    session = 'Mouse28-140313'
    state = 'Wake'; #state2 = 'REM'
    condition = 'solo'
    target_dim = 3
    desired_nSamples = 15000

print('Session %s, condition %s, target_dim %d, desired_nSamples %d'%(session, condition,
    target_dim, desired_nSamples))
area = 'ADn'
dt_kernel = 0.1
sigma = 0.1 # Kernel width
rate_params = {'dt' : dt_kernel, 'sigma' : sigma}
method = 'iso'
n_neighbors = 5
dim_red_params = {'n_neighbors' : n_neighbors, 'target_dim' : target_dim}
to_plot = True

session_rates = spike_counts(session, rate_params, count_type='rate', 
    anat_region='ADn')

t0 = time.time()
if condition == 'solo':
    counts, tmp_angles = session_rates.get_spike_matrix(state)
    sel_counts = counts[:desired_nSamples]
    proj = run_dim_red(sel_counts, params=dim_red_params, method=method)
    to_save = {'seed' : sd, state : proj, 'meas_angles' : tmp_angles[:desired_nSamples]}
    fname = '%s_%s_kern_%dms_sigma_%dms_binsep_%s_embed_%s_%ddims_%dneighbors_%d.p'%(
        session, area, sigma * 1000, dt_kernel * 1000, state, method, target_dim, n_neighbors, 
       sd)
elif condition == 'joint':
    counts1, _ = session_rates.get_spike_matrix(state)
    counts2, _ = session_rates.get_spike_matrix(state2)
    print('Counts for each ', len(counts1), len(counts2))
    nSamples = min(len(counts1), len(counts2), desired_nSamples)
    print('nSamples = ', nSamples)
    sel1 = counts1[:nSamples]; sel2 = counts2[:nSamples]
    concat_counts = np.concatenate((sel1,sel2),0)
    proj = run_dim_red(concat_counts, params=dim_red_params, method=method)
    to_save = {'seed':sd, state : proj[:nSamples].copy(), 
        state2 : proj[nSamples:].copy()}
    fname = '%s_%s_kern_%dms_sigma_%dms_binsep_%s_%s_embed_%s_%ddims_%dneighbors_%d.p'%(
        session, area, sigma * 1000, dt_kernel * 1000, state, state2, method, target_dim,
        n_neighbors, sd)
gff.save_pickle_file(to_save, dir_to_save + fname)
print('Time ', time.time() - t0)

if to_plot:
    fig = plt.figure(figsize=(8,8))
    if target_dim == 2:
        ax = fig.add_subplot(111)
        ax.scatter(to_save[state][:,0], to_save[state][:,1], s=10, alpha=0.4, 
            color=cols[state])
        if condition == 'joint':
            ax.scatter(to_save[state2][:,0], to_save[state2][:,1], s=10, alpha=0.4, 
                color=cols[state2])
    if target_dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(to_save[state][:,0], to_save[state][:,1], to_save[state][:,2], 
            s=5, alpha=0.4, edgecolor='face', c=cols[state])
        if condition == 'joint':
            ax.scatter(to_save[state2][:,0], to_save[state2][:,1], to_save[state2][:,2], 
                s=5, alpha=0.4, edgecolor='face', c=cols[state2])
    ax.set_title('Session %s'%(session))
    plt.show()


