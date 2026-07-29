#!/usr/bin/env python

"""Topology-unchanged (topo-unchanged) probability calculation.

This module contains functions optimized for calculating topo-change
and topo-unchanged probabilities and waiting distances from data
that has been preorganized into arrays using a TreeEmbedding class
object. The functions are jit-compiled for speed.
"""

from typing import Optional
import numpy as np
from numba import njit, prange
from ipcoal.smc.src.ms_smc_interval_prob import (
    _get_fast_sum_pb1,
    _get_fast_sum_pb2,
    _get_fij_set_sum,
    _get_pb1_set_sum,
    _get_pb2_set_sum,
)

__all__ = [
    "get_prob_topo_unchanged_given_b_and_tr_from_arrays",
    "get_prob_topo_unchanged_given_b_from_arrays",
    "get_prob_topo_unchanged_from_arrays",
    "get_topo_changed_lambdas",
]


@njit
def get_prob_topo_unchanged_given_b_and_tr_from_arrays(
    gemb: np.ndarray,
    genc: np.ndarray,
    bidx: int,
    sidx: int,
    pidx: int,
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
    sidx: int
        Index of the sister branch of bidx.
    pidx: int
        Index of the parent branch of bidx.
    time: float
        A time at which recombination occurs on branch index bidx.
    """
    # get all intervals on branch b including or above time t_r
    benc = genc[:, bidx] & (time < gemb[:, 1])
    bidxs = benc.nonzero()[0]

    # require time occurs on branch b (tr >= interval start)
    if not (benc & (time >= gemb[:, 0])).sum():
        raise ValueError("No interval exists on branch bidx at time tr.")

    # get intervals containing both b and sister above time t_r
    senc = genc[:, sidx]
    sidxs = (benc & senc).nonzero()[0]

    # get intervals containing parent
    penc = genc[:, pidx]
    pidxs = penc.nonzero()[0]

    # get intervals containing either b or its parent
    fidxs = (benc | penc).nonzero()[0]

    # get interval in which recomb event occurs
    idx = bidxs.min()

    # exp(nedges / 2neff) * tr)
    inner = (gemb[idx, 4] / (2 * gemb[idx, 3])) * time
    # inner = np.exp(inner) if inner < 100 else 1e15
    inner = np.exp(inner)

    # (1 / nedges)
    term1 = (1 / gemb[idx, 4])

    # if t_r < t_m then use three interval equation
    if time < gemb[sidxs, 0].min():

        # pij over j in all intervals between idx and top of parent
        # term2 = 0
        # print(ftab)
        #     for jdx in range(ftab.shape[0]):
        #     term2 += _get_fast_pij(ftab, idx, jdx)
        # print(f'f{idx}{jdx}', term2, ftab[jdx])
        # term2 *= inner
        sumfij_bc = _get_fij_set_sum(gemb, fidxs, fidxs)
        term2 = sumfij_bc * inner

        # pij over j in all intervals from m to end of b
        # term3 = 0
        # for jdx in range(mtab.shape[0]):
        #     print(term3)
        #     term3 += _get_fast_pij(mtab, 0, jdx)
        # term3 *= inner
        # return term1 + term2 + term3
        sumfij_m = _get_fij_set_sum(gemb, fidxs, sidxs)
        term3 = sumfij_m * inner
        return term1 + term2 + term3

    # pij over j all intervals on b
    # term2 = sum(_get_fast_pij(btab, idx, jdx) for jdx in range(btab.shape[0]))
    # term2 *= inner
    sumfij_b = _get_fij_set_sum(gemb, fidxs, bidxs)
    term2 = sumfij_b * inner

    # pij over j all intervals on c
    # term3 = sum(_get_fast_pij(ptab, idx, jdx) for jdx in range(ptab.shape[0]))
    # term3 *= inner
    sumfij_c = _get_fij_set_sum(gemb, fidxs, pidxs)
    term3 = sumfij_c * inner
    return 2 * (term1 + term2) + term3


@njit
def get_prob_topo_unchanged_given_b_from_arrays(
    gemb: np.ndarray,
    genc: np.ndarray,
    bidx: int,
    sidx: int,
    pidx: int,
) -> float:
    """Return probability of tree-change that does not change topology.

    Parameters
    ----------
    gemb: np.ndarray
        Genealogy embedding table for a single genealogy.
    genc: np.ndarray
        Genealogy branch encoding array.
    bidx: int
        A selected focal branch selected by Node index.
    sidx: int
        Node index of the sibling of 'branch'.
    pidx: int
        Node index of the parent of 'branch'.
    """
    # get all intervals on branch b
    benc = genc[:, bidx]
    bidxs = benc.nonzero()[0]
    btab = gemb[benc, :]

    # get intervals containing both b and b'
    senc = genc[:, sidx]
    midxs = (benc & senc).nonzero()[0]

    # get intervals containing either b or c
    penc = genc[:, pidx]
    fidxs = (benc | penc).nonzero()[0]

    # get lower and upper bounds of this gtree edge
    t_lb, t_ub = btab[:, 0].min(), btab[:, 1].max()

    # get sum pb1 from intervals 0 to m
    pb1 = _get_pb1_set_sum(gemb, bidxs, midxs, fidxs)
    # logger.info(f"branch {branch}, sum-pb1={pb1:.3f}")

    # get sum pb2 from m to end of b
    pb2 = _get_pb2_set_sum(gemb, bidxs, midxs, fidxs)
    # logger.info(f"branch {branch}, sum-pb2={pb2:.3f}")

    brlen = max(1e-9, (t_ub - t_lb))
    return (1 / brlen) * (pb1 + pb2)


@njit(parallel=True)
def get_prob_topo_unchanged_from_arrays(
    gemb: np.ndarray,
    genc: np.ndarray,
    barr: np.ndarray,
    sarr: float,
    rarr: np.ndarray,
) -> float:
    """Return probability that recombination causes a topology-change.

    Parallel prange works at this level without race conditions, but
    not at one level higher (get_topo_changed_lambdas).
    """
    # iter over all edges of genealogy
    total_prob = 0
    for bidx in prange(barr.shape[0]):
        blen = barr[bidx]
        # for bidx, blen in enumerate(barr):
        # get relationships
        # sidx = rarr[bidx, 1]
        # pidx = rarr[bidx, 2]
        _, sidx, pidx = rarr[bidx]

        # get P(tree-unchanged | S, G, b) for every genealogy
        prob = get_prob_topo_unchanged_given_b_from_arrays(
            gemb=gemb,
            genc=genc,
            bidx=bidx,
            sidx=sidx,
            pidx=pidx,
        )

        # get Prob scaled by the proportion of this branch on each tree.
        total_prob += (blen / sarr) * prob
    return total_prob


@njit
def get_topo_changed_lambdas(
    emb: np.ndarray,
    enc: np.ndarray,
    barr: np.ndarray,
    sarr: np.ndarray,
    rarr: np.ndarray,
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
        Array of relationships (node, sister, parent)
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
        relate = rarr[gidx]
        # probability is a float in [0-1]
        prob_topo = get_prob_topo_unchanged_from_arrays(gemb, genc, blens, sumlen, relate)
        # lambda is a rate > 0
        lambdas[idx] = sumlen * (1 - prob_topo) * recombination_rate
    return lambdas


######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################

@njit
def get_fast_prob_topo_unchanged_given_b(
    gemb: np.ndarray,
    genc: np.ndarray,
    bidx: int,
    sidx: int,
    pidx: int,
) -> float:
    """Return probability of tree-change that does not change topology.

    Parameters
    ----------
    arr: np.ndarray
        Genealogy embedding table for a single genealogy.
    branch: int
        A selected focal branch selected by Node index.
    sibling: int
        Node index of the sibling of 'branch'.
    parent: int
        Node index of the parent of 'branch'.
    """
    # get all intervals on branch b
    benc = genc[:, bidx]
    idxs = benc.nonzero()[0]
    btab = gemb[idxs, :]

    # get intervals containing both b and b' (above t_m)
    senc = genc[:, sidx]
    midxs = (benc & senc).nonzero()[0]
    mtab = gemb[midxs, :]

    # get intervals containing either b or c
    penc = genc[:, pidx]
    fidxs = (benc | penc).nonzero()[0]
    ftab = gemb[fidxs, :]

    # get lower and upper bounds of this gtree edge
    t_lb, t_ub = btab[:, 0].min(), btab[:, 1].max()

    # get sum pb1 from intervals 0 to m
    pb1 = _get_fast_sum_pb1(btab, ftab, mtab)
    # logger.info(f"branch {branch}, sum-pb1={pb1:.3f}")

    # get sum pb2 from m to end of b
    pb2 = _get_fast_sum_pb2(btab, ftab, mtab)
    # logger.info(f"branch {branch}, sum-pb2={pb2:.3f}")

    brlen = max(1e-9, (t_ub - t_lb))
    return (1 / brlen) * (pb1 + pb2)


@njit
def get_fast_prob_topo_changed_given_b(
    gemb: np.ndarray,
    genc: np.ndarray,
    bidx: int,
    sidx: int,
    pidx: int,
) -> float:
    """Return probability of tree-change that does not change topology.

    Parameters
    ----------
    arr: np.ndarray
        Genealogy embedding table for a single genealogy.
    branch: int
        A selected focal branch selected by Node index.
    sibling: int
        Node index of the sibling of 'branch'.
    parent: int
        Node index of the parent of 'branch'.
    """
    return 1 - get_fast_prob_topo_changed_given_b(gemb, genc, bidx, sidx, pidx)

####################################################################


@njit
def get_fast_prob_topo_unchanged(
    gemb: np.ndarray,
    genc: np.ndarray,
    barr: np.ndarray,
    sumlen: float,
    rarr: np.ndarray,
) -> float:
    """Return probability that recombination causes a topology-change.

    """
    total_prob = 0
    for bidx, blen in enumerate(barr):
        # get relationships
        sidx = rarr[bidx, 1]
        pidx = rarr[bidx, 2]

        # get P(tree-unchanged | S, G, b) for every genealogy
        prob = get_fast_prob_topo_unchanged_given_b(
            gemb=gemb,
            genc=genc,
            bidx=bidx,
            sidx=sidx,
            pidx=pidx,
        )

        # get Prob scaled by the proportion of this branch on each tree.
        total_prob += (blen / sumlen) * prob
    return total_prob


@njit
def get_fast_prob_topo_changed(
    gemb: np.ndarray,
    genc: np.ndarray,
    barr: np.ndarray,
    sumlen: float,
    rarr: np.ndarray,
) -> float:
    """Return probability that recombination causes a topology-change.

    """
    return 1 - get_fast_prob_topo_unchanged(gemb, genc, barr, sumlen, rarr)


####################################################################
####################################################################
####################################################################
####################################################################
####################################################################


@njit  # (parallel=True)
def get_fast_topo_unchanged_lambdas(
    emb: np.ndarray,
    enc: np.ndarray,
    barr: np.ndarray,
    sarr: np.ndarray,
    rarr: np.ndarray,  # placeholder
    recombination_rate: float,
) -> np.ndarray:
    """return LAMBDA rate parameters for waiting distance prob density.

    """
    lambdas = np.zeros(emb.shape[0])
    for gidx in range(emb.shape[0]):
        gemb = emb[gidx]
        genc = enc[gidx]
        blens = barr[gidx]
        sumlen = sarr[gidx]
        relate = rarr[gidx]
        # probability is a float in [0-1]
        prob_topo = get_fast_prob_topo_unchanged(gemb, genc, blens, sumlen, relate)
        # lambda is a rate > 0
        lambdas[gidx] = sumlen * prob_topo * recombination_rate
    return lambdas


@njit(parallel=True)
def get_fast_topo_changed_lambdas(
    emb: np.ndarray,
    enc: np.ndarray,
    barr: np.ndarray,
    sarr: np.ndarray,
    rarr: np.ndarray,  # placeholder
    recombination_rate: float,
) -> np.ndarray:
    """return LAMBDA rate parameters for waiting distance prob density.

    """
    lambdas = np.zeros(emb.shape[0])
    for gidx in prange(emb.shape[0]):
        gemb = emb[gidx]
        genc = enc[gidx]
        blens = barr[gidx]
        sumlen = sarr[gidx]
        relate = rarr[gidx]
        # probability is a float in [0-1]
        prob_topo = get_fast_prob_topo_changed(gemb, genc, blens, sumlen, relate)
        # lambda is a rate > 0
        lambdas[gidx] = sumlen * prob_topo * recombination_rate
    return lambdas


if __name__ == "__main__":

    import toytree
    import ipcoal
    from ipcoal.smc.src.embedding import TreeEmbedding
    from ipcoal.smc.src.utils import get_test_data
    from ipcoal.msc import get_genealogy_embedding_table

    ###################################################################
    # test from Fig. S7 in paper
    SPTREE, GTREE, IMAP = get_test_data()
    TIME = 500

    # show embedding table as df
    print(get_genealogy_embedding_table(SPTREE, GTREE, IMAP))

    # calculate probs from embedding arrays
    emb, enc, barr, sarr, rarr = TreeEmbedding(SPTREE, GTREE, IMAP, nproc=1).get_data()

    # p = get_prob_topo_unchanged_given_b_and_tr_from_arrays(emb[0], enc[0], bidx=0, sidx=4, pidx=5, time=TIME)
    # print(f"Figure S6 Prob(topo-unchanged | S, G, b, tr) = {p:.4f}\n")

    # p = get_prob_topo_unchanged_given_b_from_arrays(emb[0], enc[0], 0, 4, 5)
    # print(f"Figure S6 Prob(topo-unchanged | S, G, b) = {p:.4f}\n")

    # emb2 = emb[0].copy()
    # emb2[:, 3] *= 2
    # p = get_fast_prob_topo_unchanged_given_b(emb2, enc[0], bidx=0, sidx=4, pidx=5)
    # print(f"Figure S6 Prob(topo-unchanged | S, G, b) = {p:.4f}\n")

    # # p = get_fast_prob_topo_unchanged_given_b(emb[0], enc[0], 0, 4, 5)
    # # print(f"Figure S6 Prob(topo-unchanged | S, G, b) = {p:.4f}\n")

    # p = get_prob_topo_unchanged_from_arrays(emb[0], enc[0], barr[0], sarr[0], rarr[0])
    # print(f"Figure S6 Prob(topo-unchanged | S, G) = {p:.4f}\n")
    # p = get_fast_prob_topo_unchanged(emb[0], enc[0], barr[0], sarr[0], rarr[0])
    # print(f"Figure S6 Prob(topo-unchanged | S, G) = {p:.4f}\n")

    lambdas = get_topo_changed_lambdas(emb, enc, barr, sarr, rarr, 2e-9, np.arange(1))
    print(lambdas)

    # get_topo_changed_lambdas.parallel_diagnostics(level=4)

    raise SystemExit(0)
    ###################################################################


    # setup species tree model
    # sptree = toytree.rtree.imbtree(ntips=4, treeheight=1e6)
    # model = ipcoal.Model(sptree, Ne=1e5, nsamples=2, seed_trees=123)
    # imap = model.get_imap_dict()
    # model.sim_trees(1, 1e5)

    # Select a branch to plot and get its relations
    SPTREE, GTREE, IMAP = get_test_data()
    BIDX = 2
    BRANCH = GTREE[BIDX]
    SIDX = BRANCH.get_sisters()[0].idx
    PIDX = BRANCH.up.idx

    sptree, gtree, imap = get_test_data()
    E = TreeEmbedding(sptree, gtree, imap, nproc=1)

    # model = ipcoal.Model(sptree)
    # model = get_test_model()
    # model.sim_trees(1, 1e5)
    # imap = model.get_imap_dict()
    # topos = [i[0] for i in iter_topos_and_spans_from_model(model, False)]
    # E = TreeEmbedding(model.tree, topos, imap)
    # print(E.rarr[0])
    # p = get_fast_prob_topo_unchanged_given_b(
    #     E.emb[0], E.enc[0], 0, 1, 8,
    # )
    # print(p)

    args = (E.emb[0], E.enc[0], E.barr[0], E.sarr[0], E.rarr[0])
    p = get_fast_prob_topo_unchanged(*args)
    print(p)
    print(1 - p)
    # p = get_fast_prob_topo_changed_given_b(*args)

    # get expected waiting distance parameters
    # print(get_fast_topo_changed_lambdas(table.earr, table.barr, table.sarr, table.rarr, 2e-9))
