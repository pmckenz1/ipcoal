#!/usr/bin/env python

import ipcoal
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
        kappa=None,
        alpha=None,
        gamma=None,
        seed=None,
        Ne=None,
        ):

        # save the tree object if one is provided with init
        self.kappa = (kappa if kappa else 1.)
        self.state_frequencies = (
            state_frequencies if np.any(state_frequencies) else RATES
        )

        # this is reported in seqmodel summary but not used in computation
        # since it is redundant with kappa
        freqR = self.state_frequencies[0] + self.state_frequencies[2]
        freqY = self.state_frequencies[1] + self.state_frequencies[3]
        self.tstv = (
            self.kappa * sum([
                self.state_frequencies[0] * self.state_frequencies[2],
                self.state_frequencies[1] * self.state_frequencies[3]
            ])) / (freqR * freqY)

        # get Q matrix from model params
        self.Q = None
        self.mu = None
        self.get_model_Q()

        # store the ancestral starting sequence
        self.ancestral_seq = None

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
        self.mu = -1 / np.sum(np.array([nonnormal_Q[i][i] for i in range(4)]) * sd)

        # scale by Q to get adjusted rate matrix
        self.Q = nonnormal_Q * self.mu



    def feed_tree(self, tree, nsites=1, mut=1e-8, seed=None):
        """
        This takes a _rawtree that has already been renamed from msprime 
        numeric labels to the original alphanumeric tipnames. It uses the 
        idx labels to correctly traverse the tree and returns seqs in the
        proper alphanumeric roworder. 
        """
        # set seed
        np.random.seed(seed)    

        # set starting sequence to the root node idx
        seqs = np.zeros((tree.nnodes, nsites), dtype=np.int8)
        seqs[-1] = np.random.choice(range(4), nsites, p=self.state_frequencies)
        self.ancestral_seq = seqs[-1]

        # mutate sequence along edges of the tree
        for node in tree.treenode.traverse():
            if not node.is_root():
                probmat = jevolve_branch_probs(node.dist * mut * self.Q)
                seqs[node.idx] = jsubstitute(seqs[node.up.idx], probmat)

        # return seqs in alphanumeric order
        order = np.argsort(tree.treenode.get_leaf_names()[::-1])
        return seqs[:tree.ntips][order]


    def old_feed_tree(self, tree, nsites=1, mut=1e-8, seed=None):
        """
        Simulate markov mutation process on a gene tree and return a 
        sequence array. The returned array is ordered by...

        rows by their idx number from 0-ntips. It is not ordered by tip 'name' order.
        # 1. input tree has idx labels 0-nnodes
        # 2. seqs is (nnodes + 1, nsites)
        # 3. 

        """
        np.random.seed(seed)

        # empty array to store seqs, size determined by tree.
        seqs = np.zeros((tree.nnodes + 1, nsites), dtype=np.int8)

        # empty array to store traversal order (ints between 1 - nnodes+1)
        traversal = np.zeros(tree.nnodes, dtype=int)

        # store edge lengths
        brlens = np.zeros(tree.nnodes + 1)

        # store (offspring, parent) pairs for msprime numeric names
        relate = np.zeros((tree.nnodes + 1, 2), dtype=int)

        # prefill info needed to do jit funcs
        for idx, node in enumerate(tree.treenode.traverse()):

            # store the bls and parent name of each node
            if node.is_root():
                traversal[idx] = int(node.name) + 1
                relate[int(node.name) + 1] = int(node.name) + 1, 0

            # leaf parent has internal parent name + 1
            elif node.is_leaf():
                traversal[idx] = int(node.name)
                brlens[int(node.name)] = node.dist * mut
                relate[int(node.name)] = node.name, int(node.up.name) + 1

            # internal node names are currently idx labels b/c msprime 
            # internal names are not preserved (written), so they need
            # to be pushed +1 to be 1-indexed.
            else:
                traversal[idx] = int(node.name) + 1
                brlens[int(node.name) + 1] = node.dist * mut
                relate[int(node.name) + 1] = (
                    int(node.name) + 1, int(node.up.name) + 1)               

        # fill starting value to root
        start = np.random.choice(range(4), nsites, p=self.state_frequencies)
        seqs[0] = start

        # run jitted funcs on arrays. Returns an array in traversal order.
        seqs = jevolve(self.Q, seqs, traversal, brlens, relate, seed)

        # reorder seqs array alphanumeric tip-name order
        return seqs[1:tree.ntips + 1]


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
def jevolve(Q, seqs, traversal, brlens, relate, seed):
    """
    jitted function to sample substitutions on edges
    """
    np.random.seed(seed)

    # traverse tree from root to tips [44, 43, 24, 22, 42, ... 2, 1]
    # there is no zero value in traversal b/c they are msprime node names.
    # Thus seqs[0] is used to store the starting sequence.
    for idx in traversal:

        # get length of edge (e.g., 0.00123)
        bl = brlens[idx]

        # get who the parent is (e.g., 43)
        pidx = relate[idx, 1]

        # get transition probabilities for edge this length ([[x x],[x x]])
        probmat = jevolve_branch_probs(bl * Q)

        # apply evolution to parent sequence to get child sequence
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
