'''April 18th 2019
Use Ripser to get Betti bar codes from saved rates. If nCells > 10, dim reduce 
spike counts using Isomap. Threshold out low density points if thrsh is True.
'''

from __future__ import division
import sys, os
import time, datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ripser import ripser as tda

sd=int((time.time()%1)*(2**31))
np.random.seed(sd)
curr_date=datetime.datetime.now().strftime('%Y_%m_%d')+'_'

gen_fn_dir = os.path.abspath('..') + '/shared_scripts'
sys.path.append(gen_fn_dir)

import general_file_fns as gff

gen_params = gff.load_pickle_file('../general_params/general_params.p')

from binned_spikes_class import spike_counts
from dim_red_fns import run_dim_red
from scipy.spatial.distance import pdist
from sklearn import neighbors

save_dir = gff.return_dir(gen_params['results_dir'] + '2019_03_22_tda/')

plot_barcode = True
cmd_line = False
# if thrsh is True then we threshold out low density points (nt-TDA in the 
# paper)
if cmd_line:
    session = sys.argv[1]
    state = sys.argv[2]
    thrsh = sys.argv[3]  # threshold out low density pts
else:
    session = 'Mouse28-140313'
    state = 'Wake'
    thrsh = False

area = 'ADn'
dt_kernel = 0.1
sigma = 0.1
d_idx = 10
rate_params = {'dt': dt_kernel, 'sigma': sigma}

# load the kernel spikes and smooth
print('Session: %s, state: %s' % (session, state))
session_rates = spike_counts(session, rate_params, count_type='rate',
                             anat_region='ADn')
rates_all = session_rates.get_spike_matrix(state)[0]
nCells_tot = rates_all.shape[1]
n_smooth_samples = np.floor(len(rates_all) / d_idx).astype(int)
sm_rates = np.zeros((n_smooth_samples, nCells_tot))
for i in range(n_smooth_samples):
    si = i * d_idx
    ei = (i + 1) * d_idx
    sm_rates[i] = np.mean(rates_all[si:ei], axis=0)

results = {'session': session, 'h0': [], 'h1': [], 'h2': []}

# if greater than 10 cells, dim reduce to 10 dims using Isomap
fit_dim = 10
dr_method = 'iso'
n_neighbors = 5
dim_red_params = {'n_neighbors': n_neighbors, 'target_dim': fit_dim}
if nCells_tot > 10:
    rates = run_dim_red(sm_rates, params=dim_red_params, method=dr_method)
else:
    rates = sm_rates

# threshold out outlier points with low neighborhood density
if thrsh:
    # a) find number of neighbors of each point within radius of 1st percentile of all
    # pairwise dist.
    dist = pdist(rates, 'euclidean')
    rad = np.percentile(dist, 1)
    neigh = neighbors.NearestNeighbors()
    neigh.fit(rates)
    num_nbrs = np.array(map(len, neigh.radius_neighbors(X=rates, radius=rad,
                        return_distance=False)))

    # b) threshold out points with low density
    thrsh_prcnt = 20
    threshold = np.percentile(num_nbrs, thrsh_prcnt)
    thrsh_rates = rates[num_nbrs > threshold]
    rates = thrsh_rates

# H0 & H1
H1_rates = rates
barcodes = tda(H1_rates, maxdim=1, coeff=2)['dgms']
results['h0'] = barcodes[0]
results['h1'] = barcodes[1]

# H2. Need to subsample points for computational tractability if 
# number of points is large (can go higher but very slow)
if len(rates) > 1500:
    idx = np.random.choice(np.arange(len(rates)), 1500, replace=False)
    H2_rates = rates[idx]
else:
    H2_rates = rates
barcodes = tda(H2_rates, maxdim=2, coeff=2)['dgms']
results['h2'] = barcodes[2]

# save
gff.save_pickle_file(results, save_dir + '%s_%s%s_ph_barcodes.p' % (session, state, ('_thresholded' * thrsh)))

# If plotting from a saved file, uncomment this and replace with appropriate file.
# results = gff.load_pickle_file(gen_params['results_dir'] + '2019_03_22_tda/Mouse28-140313_Wake_ph_barcodes.p')

if plot_barcode:
    col_list = ['r', 'g', 'm', 'c']
    h0, h1, h2 = results['h0'], results['h1'], results['h2']
    # replace the infinity bar (-1) in H0 by a really large number
    h0[~np.isfinite(h0)] = 100
    # Plot the longest barcodes only
    plot_prcnt = [99, 98, 90] # order is h0, h1, h2
    to_plot = []
    for curr_h, cutoff in zip([h0, h1, h2], plot_prcnt):
         bar_lens = curr_h[:,1] - curr_h[:,0]
         plot_h = curr_h[bar_lens > np.percentile(bar_lens, cutoff)]
         to_plot.append(plot_h)

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(3, 4)
    for curr_betti, curr_bar in enumerate(to_plot):
        ax = fig.add_subplot(gs[curr_betti, :])
        for i, interval in enumerate(reversed(curr_bar)):
            ax.plot([interval[0], interval[1]], [i, i], color=col_list[curr_betti],
                lw=1.5)
        # ax.set_xlim([0, xlim])
        # ax.set_xticks([0, xlim])
        ax.set_ylim([-1, len(curr_bar)])
        # ax.set_yticks([])
    plt.show()
