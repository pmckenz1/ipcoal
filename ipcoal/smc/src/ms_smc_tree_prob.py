#!/usr/bin/env python

"""Tree-unchanged and tree-changed probabilities from Embeddings.

This module contains functions optimized for calculating tree-change
and tree-unchanged probabilities and waiting distances from data
that has been pre-organized into arrays using a TreeEmbedding class
object. The functions are jit-compiled and so run much much faster
than the didactic functions that take ToyTrees as input.
"""

from typing import Optional
import numpy as np
from loguru import logger
from numba import njit, prange
from ipcoal.smc.src.ms_smc_interval_prob import _get_fij_set_sum


logger = logger.bind(name="ipcoal")

__all__ = [
    "get_prob_tree_unchanged_given_b_and_tr_from_arrays",
    "get_prob_tree_unchanged_given_b_from_arrays",
    "get_prob_tree_unchanged_from_arrays",
    "get_tree_unchanged_lambdas",
    "get_tree_changed_lambdas",
]


@njit
def get_prob_tree_unchanged_given_b_and_tr_from_arrays(
    gemb: np.ndarray,
    genc: np.ndarray,
    bidx: int,
    time: float,
) -> float:
    """Return prob tree-unchanged given recomb on branch b at time t.

    This function is mainly for didactic purposes.

    Parameters
    ----------
    gemb: np.ndarray
        A genealogy embedding from get_genealogy_embedding_arrays().
    genc: np.narray
        A branch encoding array from get_genealogy_embedding_arrays().
    bidx: int
        Index of a genealogy branch on which recombination occurs.
    time: float
        A time at which recombination occurs on branch index bidx.
    """
    # subselect array intervals for this genealogy branch
    bmask = genc[:, bidx]

    # subselect array intervals that end farther back in time than tr
    in_or_above_t = (time < gemb[:, 1]) # & (time >= gemb[:, 0])  # tr occurs before interval end

    # get intervals on b above or include time tr
    bidxs = (bmask & in_or_above_t).nonzero()[0]

    # assert intervals must exist on b at tr
    if not bidxs.size:
        raise ValueError("No interval exists on branch bidx at time tr.")
    tidx = bidxs.min()

    # (nedges / 2neff) * time
    inner = (gemb[tidx, 4] / (2 * gemb[tidx, 3])) * time
    inner = np.exp(inner)  # if inner < 100 else 1e15

    # (1 / nedges)
    term1 = (1 / gemb[tidx, 4])

    # pij * inner for all intervals on branch above or including tr
    term2 = 0
    for i in bidxs:
        term2 += _get_fij_set_sum(gemb, bidxs, [i]) * inner
    return term1 + term2


#####################################################################


@njit
def get_prob_tree_unchanged_given_b_from_arrays(
    gemb: np.ndarray,
    genc: np.ndarray,
    bidx: int,
) -> float:
    """Return prob tree-unchanged given recomb on branch b.

    Parameters
    ----------
    gemb: np.ndarray
        A genealogy embedding table for a single genealogy.
    genc: np.ndarray
        A branch encoding table for a single genealogy.
    bidx: int
        Index of a genealogy branch on which recombination occurs.
    ne_is_2Ne: bool
        (Users can ignore this.) Set to True if you modify the Ne
        values in the embedding array to be 2 * Ne instead of Ne.
    """
    # subselect array intervals for this genealogy branch
    bmask = genc[:, bidx]
    bidxs = np.nonzero(bmask)[0]

    # get top and bottom times on branch b
    tbl = gemb[bmask, 0].min()
    tbu = gemb[bmask, 1].max()

    # sum over the intervals on b where recomb could occur
    sumval = 0
    for idx in bidxs:
        neff2 = 2 * gemb[idx, 3]
        nedges = gemb[idx, 4]
        dist = gemb[idx, 5]
        term1 = (1 / nedges) * dist
        term2_outer = neff2 / nedges

        # Avoid overflow when inner value here is too large. Simply
        # setting it to a very large value seems asymptotically OK.
        estop = (nedges / neff2) * gemb[idx, 1]
        estart = (nedges / neff2) * gemb[idx, 0]
        # if estop > 100:
        #     term2_inner = 1e15
        # #     print("overflow", bidx, nedges, neff2, gemb[idx, 1], estop, np.exp(estop))
        # #     # logger.warning("overflow")  # no-jit
        # else:
        #     term2_inner = np.exp(estop) - np.exp(estart)
        term2_inner = np.exp(estop) - np.exp(estart)

        # pij component
        jidxs = bidxs[bidxs >= idx]
        term3 = _get_fij_set_sum(gemb, jidxs, jidxs)
        sumval += term1 + (term2_inner * term2_outer * term3)
        # print(bidx, sumval)
    brlen = max(1e-9, (tbu - tbl))
    return (1 / brlen) * sumval


#####################################################################


@njit(parallel=True)
def get_prob_tree_unchanged_from_arrays(
    gemb: np.ndarray,
    genc: np.ndarray,
    barr: np.ndarray,
    sumlen: float,
) -> float:
    """Return probability recombination does not cause tree-change.

    Returns the probability that recombination occurring on this
    genealogy embedded in this parameterized species tree causes a
    tree change, under the MS-SMC'. A tree-change is defined as the
    opposite of a no-change event, and includes any change to
    coalescent times (whether or not it changes the topology).

    Parameters
    ----------
    gemb: np.ndarray
        A genealogy embedding table for a single genealogy.
    genc: np.ndarray
        A branch encoding table for a single genealogy.
    barr: np.ndarray
        Branch lengths for each branch on the genealogy.
    sumlen: int
        Sum of branch lengths on the genealogy.
    """
    # traverse over all edges of the genealogy
    total_prob = 0
    for bidx in prange(barr.size):
        # for bidx, blen in enumerate(barr):
        blen = barr[bidx]
        # get P(tree-unchanged | S, G, b)
        prob = get_prob_tree_unchanged_given_b_from_arrays(gemb, genc, bidx=bidx)
        # contribute to total probability normalized by prop edge len
        total_prob += (blen / sumlen) * prob
    return total_prob


@njit
def get_prob_tree_changed_from_arrays(
    gemb: np.ndarray,
    genc: np.ndarray,
    barr: np.ndarray,
    sumlen: float,
) -> float:
    """Return probability recombination causes a tree-change.

    A tree-change is defined as the opposite of a tree-unchanged event
    (no-change event), and includes any change to coalescent times
    (whether or not it changes the topology).

    This probability is 1 - P(no-change | S,G), where S is the
    species tree and G is the genealogy.

    Parameters
    ----------
    gemb: np.ndarray
        A genealogy embedding table for a single genealogy.
    genc: np.ndarray
        A branch encoding table for a single genealogy.
    barr: np.ndarray
        Branch lengths for each branch on the genealogy.
    sumlen: int
        Sum of branch lengths on the genealogy.
    """
    return 1 - get_prob_tree_unchanged_from_arrays(gemb, genc, barr, sumlen)

#####################################################################


@njit  # (parallel=True)
def get_tree_unchanged_lambdas(
    emb: np.ndarray,
    enc: np.ndarray,
    barr: np.ndarray,
    sarr: np.ndarray,
    rarr: np.ndarray,  # placeholder
    recombination_rate: float,
    idxs: np.ndarray,
) -> np.ndarray:
    """Return LAMBDA rate parameters for waiting distance prob. density.

    Parameters
    ----------
    emb: np.ndarray
        Genealogy embedding array for multiple genealogies of shape
        (ngenealogies, nintervals, 6) with a unique genealogy index
        (gidx) in column 6.
    enc: np.ndarray
        Branch encoding array for multiple genealogies of shape
        (ngenealogies, nintervals, nnodes - 1).
    barr: np.ndarray
        Branch length array for multiple genealogies of shape
        (ngenealogies, nnodes - 1)
    sarr: np.ndarray
        Array of (ngenealogies,) w/ sum branch lengths of each tree.
    rarr: np.ndarray or None
        Placeholder here. Not used.
    recombination_rate: float
        The per-site per-generation recombination rate.
    idxs: np.ndarray
        An int array of the indices in the embedding array to use. This
        can be used to subselect topo-changes from an array with all
        events, or tree-change events in it. Default is usually
        idxs=np.arange(emb.shape[0])
    """
    lambdas = np.zeros(idxs.size, dtype=np.float64)

    # use numba parallel to iterate over genealogies
    # pylint-disable: not-an-iterable
    for idx, gidx in enumerate(idxs):
        gemb = emb[gidx]
        genc = enc[gidx]
        blens = barr[gidx]
        sumlen = sarr[gidx]
        # probability is a float in [0-1]
        prob_tree = get_prob_tree_unchanged_from_arrays(gemb, genc, blens, sumlen)
        # lambda is a rate > 0
        lambdas[gidx] = sumlen * prob_tree * recombination_rate
    return lambdas


@njit  # (parallel=True)
def get_tree_changed_lambdas(
    emb: np.ndarray,
    enc: np.ndarray,
    barr: np.ndarray,
    sarr: np.ndarray,
    rarr: np.ndarray,  # placeholder
    recombination_rate: float,
    idxs: np.ndarray,
) -> np.ndarray:
    """Return LAMBDA rate parameters for waiting distance prob. density.

    Parameters
    ----------
    emb: np.ndarray
        Genealogy embedding array for multiple genealogies of shape
        (ngenealogies, nintervals, 6) with a unique genealogy index
        (gidx) in column 6.
    enc: np.ndarray
        Branch encoding array for multiple genealogies of shape
        (ngenealogies, nintervals, nnodes - 1).
    barr: np.ndarray
        Branch length array for multiple genealogies of shape
        (ngenealogies, nnodes - 1)
    sarr: np.ndarray
        Array of (ngenealogies,) w/ sum branch lengths of each tree.
    rarr: np.ndarray or None
        Placeholder here. Not used.
    recombination_rate: float
        The per-site per-generation recombination rate.
    idxs: np.ndarray
        An int array of the indices in the embedding array to use. This
        can be used to subselect topo-changes from an array with all
        events, or tree-change events in it. Default is usually
        idxs=np.arange(emb.shape[0])
    """
    lambdas = np.zeros(idxs.size, dtype=np.float64)

    # use numba parallel to iterate over genealogies
    # pylint-disable: not-an-iterable
    for idx, gidx in enumerate(idxs):
        gemb = emb[gidx]
        genc = enc[gidx]
        blens = barr[gidx]
        sumlen = sarr[gidx]
        # probability is a float in [0-1]
        prob_tree = get_prob_tree_changed_from_arrays(gemb, genc, blens, sumlen)
        # lambda is a rate > 0
        lambdas[gidx] = sumlen * prob_tree * recombination_rate
    return lambdas

#####################################################################


if __name__ == "__main__":

    import ipcoal
    import pandas as pd
    import toytree
    from ipcoal.smc.src.embedding import TreeEmbedding

    ipcoal.set_log_level("WARNING")
    pd.options.display.max_columns = 14
    pd.options.display.width = 1000

    ##################################################################
    # example
    from ipcoal.msc.src.utils import get_test_data
    SPTREE, GTREE, IMAP = get_test_data()
    BRANCH = 8
    TIME = 150_000

    # get all embedding data
    emb, enc, barr, sarr, rarr = TreeEmbedding(SPTREE, GTREE, IMAP).get_data()

    p = get_prob_tree_unchanged_given_b_and_tr_from_arrays(emb[0], enc[0], bidx=BRANCH, time=TIME)
    print(f"Figure S6 Prob(tree-unchanged | S, G, b, tr) = {p:.4f}\n")

    p = get_prob_tree_unchanged_given_b_from_arrays(emb[0], enc[0], bidx=BRANCH)
    print(f"Figure S6 Prob(tree-unchanged | S, G, b) = {p:.4f}\n")

    p = get_prob_tree_unchanged_from_arrays(emb[0], enc[0], barr[0], sarr[0])
    print(f"Figure S6 Prob(tree-unchanged | S, G) = {p:.4f}\n")

    ###################################################################
    # test from Fig. S6 in paper
    from ipcoal.smc.src.utils import get_test_data
    SPTREE, GTREE, IMAP = get_test_data()
    BRANCH = 0
    TIME = 500

    # get all embedding data
    emb, enc, barr, sarr, rarr = TreeEmbedding(SPTREE, GTREE, IMAP).get_data()
    # emb, enc = get_genealogy_embedding_arrays(SPTREE, GTREE, IMAP)

    p = get_prob_tree_unchanged_given_b_and_tr_from_arrays(emb[0], enc[0], bidx=BRANCH, time=TIME)
    print(f"Figure S6 Prob(tree-unchanged | S, G, b, tr) = {p:.4f}\n")

    p = get_prob_tree_unchanged_given_b_from_arrays(emb[0], enc[0], bidx=BRANCH)
    print(f"Figure S6 Prob(tree-unchanged | S, G, b) = {p:.4f}\n")

    p = get_prob_tree_unchanged_from_arrays(emb[0], enc[0], barr[0], sarr[0])
    print(f"Figure S6 Prob(tree-unchanged | S, G) = {p:.4f}\n")

    lambdas = get_tree_unchanged_lambdas(emb, enc, barr, sarr, rarr, 2e-9, np.arange(1))
    print(lambdas)

    raise SystemExit(0)

    ###################################################################

    ###################################################################
    # Get genealogy embedding table
    SPTREE, GTREE, IMAP = get_test_data()
    get_probability_tree_unchanged_given_b()

    ## ...


    ETABLE = ipcoal.msc.get_genealogy_embedding_table(SPTREE, GTREE, IMAP, encode=False)
    EMB, ENC = ipcoal.msc.get_genealogy_embedding_arrays(SPTREE, GTREE, IMAP)
    print(f"Test Model Genealogy embedding table\n{ETABLE}\n")

    p_no = get_fast_prob_tree_unchanged(E.emb[0], E.enc[0], E.barr[0], E.sarr[0])
    # print(f"Probability of no-change\n{p_no:.3f}\n")


    t = (GTREE[2].up.height - GTREE[2].height) / 2
    print(ipcoal.smc.get_probability_tree_unchanged_given_b_and_tr(SPTREE, GTREE, IMAP, 2, t))

    print(enc[0])
    p = get_prob_tree_unchanged_given_b_and_tr(emb[0], enc[0], bidx=2, time=t)
    print(p)

    # p_tree = get_fast_prob_tree_changed(E.emb[0], E.enc[0], E.barr[0], E.sarr[0])
    # print(f"Probability of tree-change\n{p_tree:.3f}\n")

    # lambdas_ = get_fast_tree_changed_lambdas(E.emb, E.enc, E.barr, E.sarr, None, 2e-9)
    # print(lambdas_)

    # # p_tree = get_probability_tree_change(SPTREE, GTREE, IMAP)
    # # p_topo = get_probability_topology_change(SPTREE, GTREE, IMAP)

    # # print(f"Probability of tree-change\n{p_tree:.3f}\n")
    # # print(f"Probability of topology-change\n{p_topo:.3f}\n")


    # # setup species tree model
    # sptree = toytree.rtree.imbtree(ntips=4, treeheight=1e6)
    # model = ipcoal.Model(sptree, Ne=1e5, nsamples=2, seed_trees=123)
    # imap = model.get_imap_dict()

    # # simulate ARGs
    # model.sim_trees(100)
    # gtrees = model.df.genealogy

    # # get embedding tables
    # table = TreeEmbedding(model.tree, model.df.genealogy, imap, table=True)
    # # print(table.table.iloc[:, :8])
    # # print(table.barr.shape, table.genealogies[0].nnodes)

    # # get expected waiting distance parameters
    # print(get_fast_tree_changed_lambda(table.earr, table.barr, table.sarr, 2e-9))

    # get observed waiting distances from sims


    # print likelihood of genealogies




    # bidx = 0
    # blen = table.barr[0][0]
    # print(np.nonzero(table.earr[:, 7 + bidx])[0])

    # p = get_prob_tree_unchanged_given_b(table.earr)
    # print(p)
