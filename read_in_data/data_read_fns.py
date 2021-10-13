'''March 27th 2019
Functions to take raw data in the form of .ang, .pos, .clu, and .res files and output
python-friendly data structures that gather all information together.'''

from __future__ import division
import numpy as np
import sys
import time
import glob
import re
import os

gen_fn_dir = os.path.abspath('..') + '/shared_scripts'
sys.path.append(gen_fn_dir)

import general_file_fns as gff

def match_spikes_to_cells(cluster_file_path, timing_file_path, verbose=True):
    '''
    Uses .clu and .res files to associate spike times with putative cells.
    
    Parameters
    ----------
    cluster_file_path: str
        path to .clu file associated with a session and shank
    timing_file_path: str
        path to .res file associated with same session and shank
    
    Returns
    -------
    nCells: int or nan
        number of cells on a given shank,  nan if no cells present
    spike_times: list
        list of floats indicating times where a spike occurred for a cluster
    '''

    tmp_clusters = gff.read_numerical_file(cluster_file_path, 'int', 'single')
    tmp_spikes = gff.read_numerical_file(timing_file_path, 'float', 'single')
    if verbose:
        print 'Cluster file:', cluster_file_path, 'Timing file ', timing_file_path

    # First line in cluster file is number of cells (with 0 corresponding to
    # artifacts and 1 to noise)
    nClusters = tmp_clusters[0]
    cluster_ids = list(tmp_clusters[1:])
    if nClusters <= 2:  # ony clusters are 0 and 1; so no cells
        print 'No cells found'
        nCells = 0
        spike_times = []
        return nCells, spike_times
    if np.max(cluster_ids) != (nClusters - 1):  
        print 'Clusters listed at beginning of file do not agree'
        nCells = np.nan
        spike_times = []
        return nCells, spike_times

    # Now break this up in various cells
    spike_time_list = [[] for i in range(nClusters)]

    for i in range(len(cluster_ids)):
        spike_time_list[cluster_ids[i]].append(tmp_spikes[i])

    spike_times_incl_noise = [np.array(x) for x in spike_time_list]
    spike_times = spike_times_incl_noise[2:]
    nCells = nClusters - 2  # since 0/1 are noise; subtract from nCluster

    return nCells, spike_times

def gather_session_spike_info(params, verbose=True):
    '''Gather data from the downloaded files.
    For each session we have (a) State information (Wake, REM, SWS), 
    (b) Position and angle info
    and (c) Spike info.
    '''

    session = params['session']
    curr_data_path = params['data_path']
    file_tag = curr_data_path + session
    if verbose:
        print 'Session: ', session
        print curr_data_path
    # First store the times the animal was in each state in state_times
    state_file_base = file_tag + '.states.'
    state_names = ['Wake', 'REM', 'SWS']

    state_times = {st: gff.read_numerical_file(state_file_base + st, 'float', 'multiple') 
        for st in state_names}

    # Store head direction in angle_list along with the corresponding times recorded.
    angle_list_orig = gff.read_numerical_file(file_tag + '.ang', 'float', 'single')

    # When angle couldn't be sampled, these files have -1. But this could mess up 
    # averaging if we're careless, so replace it with NaNs.
    angle_list = np.array(angle_list_orig)
    angle_list[angle_list < -0.5] = np.nan

    # Tag angles with the times at which they were sampled.
    pos_sampling_rate = params['eeg_sampling_rate'] / 32.  # Hz
    angle_times = np.arange(len(angle_list)) / pos_sampling_rate

    # Spike times.
    # There is a .res and .clu file for each shank.
    # .res file contains spike times. .clu contains putative cell identities.
    # Files are of the form session.clu.dd and session.res.dd, where dd is the shank number.
    # The first line in each clu file is the number of clusters for that shank, with clusters
    # 0 and 1 indicating artefacts and noise. So ignore those clusters and start with cluster number
    # 2 as cell number 0.
    # The length of the cluster file should be 1 entry more than the length of the spike 
    # timing files (because the first line is the number of clusters).
    # Note that there are occasionally extra .clu files, from previous rounds of sorting
    # so we want to exclude them. 
    nShanks = len(
        [fname for fname in glob.glob(curr_data_path + session + '.clu.*') if re.match(
        file_tag + '.clu.\\d+$', fname)])
    if verbose:
        print 'Number of shanks =', nShanks
    nCells_per_shank = np.zeros(nShanks)
    
    # Store spike times as dict where keys are (shank, cell) and the values
    # are spike times for that cell. We index shanks starting at 0 but these files 
    # are stored starting from 1, so make sure to subtract 1 where relevant.
    spike_times = {}

    for pyth_shank_idx in range(nShanks):
        data_shank_idx = pyth_shank_idx + 1
        print '\nAnalyzing shank', data_shank_idx
        cluster_file = file_tag + '.clu.' + str(data_shank_idx)
        timing_file = file_tag + '.res.' + str(data_shank_idx)

        nCells_per_shank[pyth_shank_idx], tmp_spike_list = match_spikes_to_cells(
            cluster_file, timing_file)
        
        nCells_current = nCells_per_shank[pyth_shank_idx]
        if nCells_current > 0:  # if the shank has actual cells
            # loops over each cell
            for curr_cell in range(int(nCells_current)):
                # Multiply by spike sampling interval to get into units of time
                # For reference this is 1.0/(20e3)
                spike_times[(pyth_shank_idx, curr_cell)] = params[
                    'spike_sampling_interval'] * tmp_spike_list[curr_cell]

    # Check for shanks where the number of clusters doesn't equal number of listed cells
    wrong_count_shanks = np.sum(np.isnan(nCells_per_shank))
    if wrong_count_shanks and verbose:
        print '\nThe number of shanks with wrong number of cells listed is', wrong_count_shanks

    # Gather up stuff and return it
    data_to_return = {'session' : session, 'state_times' : state_times, 'angle_list' : 
        np.array(angle_list), 'pos_sampling_rate': pos_sampling_rate, 'angle_times' : 
        np.array(angle_times), 'nShanks': nShanks, 'nCells': nCells_per_shank, 
        'spike_times': spike_times, 'cells_with_weird_clustering': wrong_count_shanks}

    return data_to_return
