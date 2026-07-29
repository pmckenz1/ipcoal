#!/usr/bin/env python

"""Much faster tree-change probs.

This contains modified versions of the functions in ms_smc.py that
are written to be faster (using jit compilation) and to reduce some
redundancy that would arise when the same functions are run
repeatedly without changing the
"""

import numpy as np
from loguru import logger
from numba import njit

logger = logger.bind(name="ipcoal")

################################################################
################################################################
# Piece-wise constants function
################################################################
################################################################


@njit
def _get_fij_set_sum(emb: np.ndarray, idxs: np.ndarray, jdxs: np.ndarray) -> float:
    """Return summation of f(i,j) function over a set of intervals.

    Given an embedding table the idxs array indicates which intervals
    are associated with the branch on which recombination occurred,
    and the jdxs array indicates the intervals over which we are
    calculating its re-coalescence.

    Parameters
    ----------
    emb: ndarray
        A genealogy embedding table
    idxs: ndarray
        Array of intervals as ordered ints for the path of either all
        intervals on branch b, or the intervals above the time of
        recombination on branch b.
    jdxs: ndarray
        Array of intervals as ordered ints on which re-coalescence is
        being calculated.
    """
    sumfij = 0
    idx = idxs[0]
    for jdx in jdxs:

        # calculate f(i,i)
        if jdx == idx:
            # -(1/k)
            term1 = -(1 / emb[idx, 4])
            # exp(-k/2n * end)
            term2 = np.exp(-(emb[idx, 4] / (2 * emb[idx, 3])) * emb[idx, 1])
            fij = term1 * term2

        elif jdx < idx:
            fij = 0

        # calculate f(i,j)
        else:
            # (1/k)
            term1 = 1 / emb[jdx, 4]
            # (1 - exp(-k/2n * dist))
            term2 = (1 - np.exp(-(emb[jdx, 4] / (2 * emb[jdx, 3])) * emb[jdx, 5]))

            # involves connections to idx interval
            term3_inner_a = -(emb[idx, 4] / (2 * emb[idx, 3])) * emb[idx, 1]

            # involves connections to edges BETWEEN idx and jdx (not including idx or jdx)
            term3_inner_b = 0
            for qdx in idxs[1:]:
                if qdx == jdx:
                    break
                term3_inner_b += (emb[qdx, 4] / (2 * emb[qdx, 3])) * emb[qdx, 5]
            term3 = np.exp(term3_inner_a - term3_inner_b)
            fij = term1 * term2 * term3

        # sum fij across intervals
        sumfij += fij
        # print(f"* idx={idx}, jdx={jdx}, fij={fij:.4f}, sum(fij)={sumfij:.4f}")
    return sumfij


@njit
def _get_pb1_set_sum(emb: np.ndarray, bidxs: np.ndarray, midxs: np.ndarray, fidxs: np.ndarray):
    """Return summation over intervals from 0-M for prob topo-unchanged.

    emb: Embedding array of intervals
    bidxs: interval indices involving branch b
    midxs: interval indices involving branch b and b' (sister)
    fidxs: interval indices involving branch b or c (parent)
    """
    pbval = 0

    # get idxs on branch b below t_m
    lidxs = bidxs[bidxs < midxs.min()]

    # iterate over intervals on branch b
    for idx in lidxs:

        # get first term
        neff2 = (2 * emb[idx, 3])
        nedges = emb[idx, 4]
        estop = (nedges / neff2) * emb[idx, 1]
        estart = (nedges / neff2) * emb[idx, 0]
        # if estop > 100:
        #     term1 = 1e15
        # else:
        #     term1 = neff2 * (np.exp(estop) - np.exp(estart))
        term1 = neff2 * (np.exp(estop) - np.exp(estart))

        # for fidxs at and above idx
        term2a = _get_fij_set_sum(emb, fidxs[fidxs >= idx], fidxs)
        term2b = _get_fij_set_sum(emb, fidxs[fidxs >= idx], midxs)

        # ...
        inner = emb[idx, 5] + (term1 * (term2a + term2b))

        # ...
        pbval += (1 / nedges) * inner
    return pbval


@njit
def _get_pb2_set_sum(emb: np.ndarray, bidxs: np.ndarray, midxs: np.ndarray, fidxs: np.ndarray):
    """Return summation over intervals from 0-M for prob topo-unchanged.

    emb: Embedding array of intervals
    bidxs: interval indices involving branch b
    midxs: interval indices involving branch b and b' (sister)
    fidxs: interval indices involving branch b or c (parent)
    """
    pbval = 0

    # get idxs on branch b above t_m
    pidxs = fidxs[fidxs > bidxs.max()]

    # iterate over intervals on branch b
    for idx in midxs:

        # get first term
        neff2 = 2 * emb[idx, 3]
        estop = (emb[idx, 4] / neff2) * emb[idx, 1]
        estart = (emb[idx, 4] / neff2) * emb[idx, 0]
        # if estop > 100:
        #     term1 = 1e15
        # else:
        #     term1 = neff2 * (np.exp(estop) - np.exp(estart))
        term1 = neff2 * (np.exp(estop) - np.exp(estart))

        # for fidxs at and above idx
        term2a = _get_fij_set_sum(emb, fidxs[fidxs >= idx], bidxs)
        term2b = _get_fij_set_sum(emb, fidxs[fidxs >= idx], pidxs)

        # ...
        inner = (2 * emb[idx, 5]) + (term1 * (2 * term2a + term2b))

        # ...
        pbval += (1 / emb[idx, 4]) * inner
    return pbval


# OLDER and SLOWER CODE
#############################################################################
#############################################################################
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
def _get_fast_pij(itab: np.ndarray, idx: int, jdx: int) -> float:
    """Return pij value for two intervals.

    This returns a value associated with an integration over the
    possible intervals that a detached subtree could re-attach to
    if it was detached in interval idx and could reconnect in any
    intervals between idx and jdx. The idx interval is on branch
    b (intervals in itab), whereas the jdx interval can occur on
    branches b, b', or c (same, sibling or parent).

    Note
    ----
    This is not really intended to be called directly, since the
    table that is entered needs to be specifically constructed. See
    `get_probability_topology_unchanged_given_b_and_tr` for examples.

    - This assumes the Ne column is 2 * diploid Ne.

    Parameters
    ----------
    table
        Intervals on one or more branches betwen intervals idx and jdx.
        This table should include ONLY these intervals.
    idx:
        Index of an interval in itable.
    jdx:
        Index of an interval in jtable.
    """
    # pii
    if idx == jdx:
        term1 = -(1 / itab[idx, 4])
        term2 = np.exp(-(itab[idx, 4] / (itab[idx, 3])) * itab[idx, 1])
        return term1 * term2

    # ignore jdx < idx (speed hack so we don't need to trim tables below t_r)
    if jdx < idx:
        return 0

    # involves connections to jdx interval
    term1 = 1 / itab[jdx, 4]
    term2 = (1 - np.exp(-(itab[jdx, 4] / (itab[jdx, 3])) * itab[jdx, 5]))

    # involves connections to idx interval
    term3_inner_a = -(itab[idx, 4] / (itab[idx, 3])) * itab[idx, 1]

    # involves connections to edges BETWEEN idx and jdx (not including idx or jdx)
    term3_inner_b = 0
    for qdx in range(idx + 1, jdx):
        term3_inner_b += ((itab[qdx, 4] / (itab[qdx, 3])) * itab[qdx, 5])
    term3 = np.exp(term3_inner_a - term3_inner_b)
    return term1 * term2 * term3


@njit
def _get_fast_sum_pb1(btab: np.ndarray, ftab: np.ndarray, mtab: np.ndarray) -> float:
    """Return value for the $p_{b,1}$ variable.

    Parameters
    ----------
    btab: np.ndarray
        Array of all intervals on branch b.
    mtab: np.ndarray
        Array of a subset of btab, including intervals on branch b
        shared with b' (its sister lineage). This potentially excludes
        intervals on b below a species divergence separating b and b'.
    ftab: np.ndarray
        Array of a superset of btab, including all intervals on branch
        b or on its parent branch, c.
    """
    pbval = 0

    # iterate over all intervals from 0 to t_m
    t_m = mtab[:, 0].min()
    for idx in range(btab.shape[0]):
        row = btab[idx]

        # if idx interval start is >=tm it doesn't affect pb1 (its in pb2)
        if row[0] >= t_m:
            continue

        # get first term
        estop = (row[4] / row[3]) * row[1]
        estart = (row[4] / row[3]) * row[0]
        if estop > 100:
            first_term = 1e15
        else:
            first_term = row[3] * (np.exp(estop) - np.exp(estart))

        # pij across bc (from this i on b to each j on bc)
        sum1 = 0
        for jidx in range(ftab.shape[0]):
            sum1 += _get_fast_pij(ftab, idx, jidx)
        # logger.info(f"sum1={sum1}")

        # pij across b > tm (from this i on b to each j on b above tm)
        sum2 = 0
        for jidx in range(mtab.shape[0]):
            # which row in mtab corresponds to idx in btab
            midx = np.argmax(btab[:, 0] == mtab[jidx, 0])
            sum2 += _get_fast_pij(btab, idx, midx)
        # logger.info(f"sum2={sum2}")

        second_term = sum1 + sum2
        pbval += (1 / row[4]) * (row[5] + (first_term * second_term))
    return pbval


@njit
def _get_fast_sum_pb2(btab: np.ndarray, ftab: np.ndarray, mtab: np.ndarray) -> float:
    """Return value for the $p_{b,2}$ variable.

    Parameters
    ----------
    btab: np.ndarray
        Array of all intervals on branch b.
    mtab: np.ndarray
        Array of a subset of btab, including intervals on branch b
        shared with b' (its sister lineage). This potentially excludes
        intervals on b below a species divergence separating b and b'.
    ftab: np.ndarray
        Array of a superset of btab, including all intervals on branch
        b or on its parent branch, c.
    """
    pbval = 0

    # iterate over all intervals from m to bu
    for idx in range(mtab.shape[0]):
        row = mtab[idx]

        # get first term
        estop = (row[4] / row[3]) * row[1]
        estart = (row[4] / row[3]) * row[0]
        if estop > 100:
            first_term = 1e15
        else:
            first_term = row[3] * (np.exp(estop) - np.exp(estart))

        # pij across intervals on b
        sum1 = 0
        for jidx in range(idx, mtab.shape[0]):
            sum1 += _get_fast_pij(mtab, idx, jidx)

        # pij across intervals on c
        sum2 = 0
        for pidx in range(ftab.shape[0]):
            if pidx >= btab.shape[0]:
                midx = np.argmax(ftab[:, 0] == mtab[idx, 0])
                sum2 += _get_fast_pij(ftab, midx, pidx)

        second_term = (2 * sum1) + sum2
        pbval += (1 / row[4]) * ((2 * row[5]) + (first_term * second_term))
    return pbval


if __name__ == "__main__":

    import toytree
    import ipcoal
    from ipcoal.smc.src.utils import get_test_data
    from ipcoal.msc import get_genealogy_embedding_table

    ###################################################################
    # test from Fig. S7 in paper
    SPTREE, GTREE, IMAP = get_test_data()
    TIME = 500
    BIDX = 0
    SIDX = 4
    PIDX = 5

    # show embedding table as df
    print(get_genealogy_embedding_table(SPTREE, GTREE, IMAP))

    # get arrays
    gemb, genc = ipcoal.msc.get_genealogy_embedding_arrays(SPTREE, GTREE, IMAP)
    gemb = gemb[0]
    genc = genc[0]

    # multiply Ne x 2
    gemb[:, 3] *= 2

    # get all intervals on branch b if end is including or above time t_r
    benc = genc[:, BIDX] & (gemb[:, 0] >= TIME)
    bidxs = benc.nonzero()[0]

    # get intervals containing both b and sister above time tr
    senc = genc[:, SIDX]
    sidxs = (benc & senc).nonzero()[0]

    # get intervals containing parent
    penc = genc[:, PIDX]
    pidxs = penc.nonzero()[0]

    # get intervals containing either b or its parent
    fidxs = (benc | penc).nonzero()[0]

    # get interval in which recomb event occurs
    idx = bidxs.min()

    # exp(nedges / 2neff) * tr)
    inner = (gemb[idx, 4] / gemb[idx, 3]) * TIME
    inner = np.exp(inner) if inner < 100 else 1e15

    benc = genc[:, 0] #& (TIME < gemb[:, 1])
    bidxs = benc.nonzero()[0]
    btab = gemb[bidxs]

    # get intervals on branches b or c (0 and 5)
    fenc = genc[:, 0] | genc[:, 5]
    fidxs = fenc.nonzero()[0]
    ftab = gemb[fidxs]
    print(fidxs)

    # get intervals shared by branches b and b' (0 and 4)
    menc = genc[:, 0] & genc[:, 4]
    midxs = menc.nonzero()[0]
    mtab = gemb[midxs]
    print(midxs)

    lidxs = bidxs[bidxs < midxs.min()]

    # interval where recomb occurs
    idx = 0

    # get fij over I_bc
    val = 0
    for jdx in range(ftab.shape[0]):
        fij = _get_fast_pij(ftab, idx, jdx)
        print(f'f{idx},{jdx}', fij)
        val += fij
    print(val)

    print(_get_fij_set_sum(gemb, bidxs, fidxs))

    print(_get_fast_sum_pb1(btab, ftab, mtab))
    print(_get_pb1_set_sum(gemb, bidxs, midxs, fidxs))

    print(_get_fast_sum_pb2(btab, ftab, mtab))
    print(_get_pb2_set_sum(gemb, bidxs, midxs, fidxs))
