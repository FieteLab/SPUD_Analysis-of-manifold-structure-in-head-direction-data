# April 18th 2018
# Collection of functions to pull out sweeps vs. staying in place during
# nREM

from __future__ import division
import numpy as np
import numpy.linalg as la

def get_epochs(start_idx, rlen):
	'''Our functions typically determine the start points of sweeps
	or staying in places epochs using a given run length. This function
	just turns that into a list of indices when we have a sweep/staying in place'''
	tmp = np.array([np.arange(x,x+rlen) for x in start_idx])
	tmp2 = tmp.flatten()
	unq = sorted(list(set(tmp2)))
	return np.array(unq)

def get_norm_dp(v1,v2):
	'''Get dot product between normalized vectors v1 and v2'''
	n1 = v1/la.norm(v1)
	n2 = v2/la.norm(v2)
	return np.dot(n1,n2)

def partition_low_high(inp_vec, run_len, low_thresh, high_thresh):
	'''Get start indices of consecutive epoches where below low_thresh
	or above high_thresh'''
	tr_len = len(inp_vec)-run_len
	start_idx_low = [idx for idx in range(tr_len) if 
		np.all(inp_vec[idx:(idx+run_len)]<=low_thresh)]
	start_idx_high = [idx for idx in range(tr_len) if 
		np.all(inp_vec[idx:(idx+run_len)]>=high_thresh)]
	return start_idx_low, start_idx_high
