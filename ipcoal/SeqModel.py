#!/usr/bin/env python

import ipcoal
import toytree
import numpy as np
from scipy.linalg import expm

from numba import njit, objmode
from numba import config


# GLOBALS
BASES = np.array([0, 1, 2, 3])
RATES = np.array([0.25, 0.25, 0.25, 0.25])



class SeqModel():
    """
    Simulate seq-gen like non-infinite sites mutational process.

    Parameters:
    ------------
    tree: (toytree)
        A tree topology with edge lengths (node .dist attributes) in units
        of expected number of substitutions per site. To convert from 
        units of generations you simply multiply by the per-site mut rate.
        Example: gens=1e4 * mut=1e-8 yields brlen = 0.0001
    state_frequencies: (None, list)
        The relative frequencies of bases A,C,G,T respectively, entered
        as a list. The default is [0.25, 0.25, 0.25, 0.25].
    kappa: (float)
        The transition/traversio ratio entered as a decimal value >0. 
        Implemented in HKY or F84. 
    alpha: (int)
        Shape for the gamma rate heterogeneity. Default is no site-specific
        rate heterogeneity. (Not Yet Implemented)
    gamma: (float)
        Discrete number of rate categories for gamma rate heterogeneity.
        (Not Yet Implemented)
    invariable: (float)
        Proportion of invariable sites (Not Yet Implemented)

    """
    def __init__(
        self,
        state_frequencies=None,
        rate_matrix=None,
        kappa=None,
        alpha=None,
        gamma=None,
        seed=None,
        Ne=None,
        ):

        # save the tree object if one is provided with init
        self.kappa = (kappa if kappa else 1.)
        self.state_frequencies = (
            state_frequencies if state_frequencies else RATES
        )
        self.rate_matrix = (
            rate_matrix if rate_matrix else RATES
        )

        # get Q matrix from model params
        self.Q = None
        self.get_model_Q()

        # set the threading layer before any parallel target compilation
        if ipcoal.__forksafe__:
            config.THREADING_LAYER = 'forksafe'



    def get_model_Q(self):
        """
        Get transition probability matrix.
        """
        # shorthand reference to params
        k = self.kappa
        sd = self.state_frequencies

        # make non-normalized matrix (not in substitutions / unit of time)
        nonnormal_Q = np.array([
            [-(sd[1] + sd[2] * k + sd[3]), sd[1], sd[2] * k, sd[3]],
            [sd[0], -(sd[0] + sd[2] + k * sd[3]), sd[2], k * sd[3]],
            [sd[0] * k, sd[1], -(sd[0] * k + sd[1] + sd[3]), sd[3]],
            [sd[0], sd[1] * k, sd[2], -(sd[0] + sd[1] * k + sd[2])],
        ])

        # full matrix scaling factor
        mu = -1 / np.sum(np.array([nonnormal_Q[i][i] for i in range(4)]) * sd)

        # scale by Q to get adjusted rate matrix
        self.Q = nonnormal_Q * mu



    def feed_tree(self, newick, nsites=1, mut=1e-8, seed=None):
        """
        Simulate markov mutation process on tree and return sequences.        
        The returned seq array is ordered with taxa on rows by their idx
        number from 0-ntips. It is not ordered by tip 'name' order.
        """
        # get all as arrays
        np.random.seed(seed)
        tree = toytree._rawtree(newick)
        seqs = np.zeros((tree.nnodes + 1, nsites), dtype=np.int8)
        idxs = np.zeros(tree.nnodes, dtype=int)
        brlens = np.zeros(tree.nnodes)
        relate = np.zeros((tree.nnodes, 2), dtype=int)

        # prefill info needed to do jit funcs
        for idx, node in enumerate(tree.treenode.traverse()):
            idxs[idx] = node.idx
            if not node.is_root():
                brlens[node.idx] = node.dist * mut
                relate[node.idx] = node.idx, node.up.idx
            else:
                relate[node.idx] = node.idx, node.idx + 1

        # fill starting value to root
        start = np.random.choice(range(4), nsites, p=self.state_frequencies)
        seqs[-1] = start

        # run jitted funcs on arrays
        seqs = jevolve(self.Q, seqs, idxs, brlens, relate, seed)
        return seqs[:tree.ntips]


    def close(self):
        pass



######################################################
# JIT functions
######################################################


@njit
def jevolve_branch_probs(brlenQ):
    """
    jitted wrapper to allow scipy call within numba loop
    """
    with objmode(probs='float64[:,:]'):
        probs = expm(brlenQ)
    return probs


@njit
def jevolve(Q, seqs, idxs, brlens, relate, seed):
    """
    jitted function to sample substitutions on edges
    """
    np.random.seed(seed)
    for idx in idxs:
        bl = brlens[idx]
        pidx = relate[idx, 1]
        probmat = jevolve_branch_probs(bl * Q)
        seqs[idx] = jsubstitute(seqs[pidx], probmat)
    return seqs


@njit
def jsubstitute(pseq, probmat):
    """
    jitted function to probabilistically transition state
    """
    ns = len(pseq)
    narr = np.zeros(ns, dtype=np.int8)
    for i in range(ns):
        nbase = np.argmax(np.random.multinomial(1, probmat[pseq[i]]))
        narr[i] = nbase
    return narr




# def evolve_branch_probs(brlen, Q):
#     """
#     Exponentiate the matrix*br_len to get probability matrix
#     The longer the branch the more probabilities will converge 
#     towards the stationary distribution.
#     """
#     return expm(Q * brlen)


# @njit
# def substitute(parent_seq, prob_mat):
#     """
#     Start with a sequence and probability matix, and make substitutions
#     across the sequence.
#     """
#     # make an array to hold the new sequence
#     new_arr = np.zeros((len(parent_seq)), dtype=np.int8)

#     # for each base index...
#     for i in range(len(parent_seq)):
#         # store a random choice of base in each index, 
#         # based on probabilities associated with starting base
#         nbase = np.random.choice(BASES, p=prob_mat[parent_seq[i]])
#         new_arr[i] = nbase

#     # return new sequence
#     return new_arr





# def old_run(self, ttree, seq_length=50, return_leaves=True):
#     """
#     Simulate markov mutation process on tree and return sequences.
#     """

#     # in case we specified a tree in the init
#     if not ttree:
#         ttree = self.ttree

#     # make a dict to hold the alignment
#     alignment = {}

#     # start a traversal
#     for node in ttree.treenode.traverse():

#         # if not the root
#         if not node.is_root():
#             # get branch length
#             br_len = node.dist / (2 * node.Ne)

#             # get index of parent node
#             parent = node.up.idx

#             # get probability matrix for this branch length
#             prob_mat = self._evolve_branch_probs(br_len, self.Q)

#             # sim substitutions and save new seq to node index key
#             subs = self._substitute(alignment[parent], prob_mat)
#             alignment[node.idx] = subs
#         else:
#             # if root, pull starting sequence from stationary distribution
#             starting = np.random.choice(
#                 range(4), 
#                 p=self.stationary_distribution,
#                 size=seq_length,
#             )
#             alignment[node.idx] = starting

#     # if we just want the leaf sequences
#     if return_leaves:
#         # pull leaf indices from the tree
#         leaves = ttree.treenode.get_leaves()

#         # return the dictionary
#         return({k.name: alignment[k.idx] for k in leaves})

#     # or...
#     else:
#         # return full alignment
#         return(alignment)
