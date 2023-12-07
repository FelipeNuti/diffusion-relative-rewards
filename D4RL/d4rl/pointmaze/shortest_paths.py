import numpy as np
from scipy.special import logsumexp as sp_lse

def softq_iteration(env, transition_matrix=None, reward_matrix=None, num_itrs=50, discount=0.99, ent_wt=0.1, warmstart_q=None, policy=None):