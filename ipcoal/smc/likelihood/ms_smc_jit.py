#!/usr/bin/env python

"""Much faster tree-change probs.

This contains modified versions of the functions in ms_smc.py that
are written to be faster (using jit compilation) and to reduce some
redundancy that would arise when the same functions are run
repeatedly without changing the

"""

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger
import toytree
from numba import njit, prange
from ipcoal.smc.likelihood.embedding import TreeEmbedding, TopologyEmbedding

logger = logger.bind(name="ipcoal")

################################################################
################################################################
## Piece-wise constants function
################################################################
################################################################


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
        term2 = np.exp(-(itab[idx, 4] / itab[idx, 3]) * itab[idx, 1])
        return term1 * term2

    # ignore jdx < idx (speed hack so we don't need to trim tables below t_r)
    if jdx < idx:
        return 0

    # involves connections to jdx interval
    term1 = 1 / itab[jdx, 4]
    term2 = (1 - np.exp(-(itab[jdx, 4] / itab[jdx, 3]) * itab[jdx, 5]))

    # involves connections to idx interval
    term3_inner_a = -(itab[idx, 4] / (itab[idx, 3])) * itab[idx, 1]

    # involves connections to edges BETWEEN idx and jdx (not including idx or jdx)
    term3_inner_b = 0
    for qdx in range(idx + 1, jdx):
        term3_inner_b += ((itab[qdx, 4] / itab[qdx, 3]) * itab[qdx, 5])
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


################################################################
################################################################
## TOPOLOGY CHANGE
################################################################
################################################################


@njit
def _get_fast_probability_of_topology_change(
    garr: np.ndarray,
    barr: np.ndarray,
    sumlen: float,
    rarr: np.ndarray,
    ) -> float:
    """Return probability that recombination causes a topology-change.

    """
    total_prob = 0

    # iterate over branch indices
    for bidx, blen in enumerate(barr[:-1]):

        # in contrast to the tree change probability function we do
        # not subselect the branch intervals here, but instead do it
        # the 'given_b' function, where it also selects based on
        # sibling and parent idxs.
        # idxs = np.nonzero(garr[:, 7 + bidx])[0]

        # get relationships
        sidx = rarr[bidx, 1]
        pidx = rarr[bidx, 2]

        # get P(tree-unchanged | S, G, b) for every genealogy
        prob = _get_fast_probability_topology_unchanged_given_b(
            arr=garr,
            branch=bidx,
            sibling=sidx,
            parent=pidx,
        )

        # get Prob scaled by the proportion of this branch on each tree.
        total_prob += (blen / sumlen) * prob
    return 1 - total_prob


@njit
def _get_fast_probability_topology_unchanged_given_b(
    arr: np.ndarray,
    branch: int,
    sibling: int,
    parent: int,
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
    btab = arr[arr[:, 7 + branch] == 1]

    # get intervals containing both b and b' (above t_m)
    mtab = arr[(arr[:, 7 + branch] == 1) & (arr[:, 7 + sibling] == 1)]

    # get intervals containing either b or c
    ftab = arr[(arr[:, 7 + branch] == 1) | (arr[:, 7 + parent] == 1)]

    # get lower and upper bounds of this gtree edge
    t_lb, t_ub = btab[:, 0].min(), btab[:, 1].max()

    # get sum pb1 from intervals 0 to m
    pb1 = _get_fast_sum_pb1(btab, ftab, mtab)
    # logger.info(f"branch {branch}, sum-pb1={pb1:.3f}")

    # get sum pb2 from m to end of b
    pb2 = _get_fast_sum_pb2(btab, ftab, mtab)
    # logger.info(f"branch {branch}, sum-pb2={pb2:.3f}")
    return (1 / (t_ub - t_lb)) * (pb1 + pb2)


@njit(parallel=True)
def get_fast_waiting_distance_to_topology_change_rates(
    earr: np.ndarray,
    barr: np.ndarray,
    sarr: np.ndarray,
    rarr: np.ndarray,
    recombination_rate: float,
    ) -> np.ndarray:
    """return LAMBDA rate parameters for waiting distance prob density.

    """
    lambdas = np.zeros(len(sarr))

    # use numba parallel to iterate over genealogies
    for gidx in prange(len(sarr)):
        garr = earr[earr[:, 6] == gidx]
        blens = barr[gidx]
        sumlen = sarr[gidx]
        relate = rarr[gidx]
        # probability is a float in [0-1]
        prob_topo = _get_fast_probability_of_topology_change(garr, blens, sumlen, relate)
        # lambda is a rate > 0
        lambdas[gidx] = sumlen * prob_topo * recombination_rate
    return lambdas


################################################################
################################################################
## TREE CHANGE
################################################################
################################################################

@njit
def _get_fast_probability_of_tree_change(
    garr: np.ndarray,
    barr: np.array,
    sumlen: float,
    ) -> float:
    """Return probability that recombination causes a tree-change.

    Returns the probability that recombination occurring on this
    genealogy embedded in this parameterized species tree causes a
    tree change, under the MS-SMC'. A tree-change is defined as the
    opposite of a no-change event, and includes any change to
    coalescent times (whether or not it changes the topology).

    Note
    ----
    This probability is 1 - P(no-change | S,G), where S is the
    species tree and G is the genealogy.

    Parameters
    ----------
    earr:
        Genealogy embedding intervals for this genealogy.
    iarr:
        Indexer to get intervals unique to each branch.
    barr:
        Branch lengths for each branch on the genealogy.
    sumlen:
        Sum of branch lengths on the genealogy.
    """
    # traverse over all edges of the genealogy
    total_prob = 0
    for bidx, blen in enumerate(barr[:-1]):
        # get P(tree-unchanged | S, G, b)
        idxs = np.nonzero(garr[:, 7 + bidx])[0]
        prob = _get_fast_probability_tree_unchanged_given_b(garr[idxs, :])
        # contribute to total probability normalized by prop edge len
        total_prob += (blen / sumlen) * prob
    return 1 - total_prob


@njit
def _get_fast_probability_tree_unchanged_given_b(arr: np.ndarray) -> float:
    """Return prob tree-change does not occur given recomb on branch b.

    Parameters
    ----------
    table:
        A table...
    """
    # get all intervals on branch b
    tbl = arr[:, 0].min()
    tbu = arr[:, 1].max()

    # sum over the intervals on b where recomb could occur
    sumval = 0
    for idx in range(arr.shape[0]):
        term1 = (1 / arr[idx, 4]) * arr[idx, 5]
        term2_outer = arr[idx, 3] / arr[idx, 4]

        # Avoid overflow when inner value here is too large. Simply
        # setting it to a very large value seems asymptotically OK.
        estop = (arr[idx, 4] / arr[idx, 3]) * arr[idx, 1]
        estart = (arr[idx, 4] / arr[idx, 3]) * arr[idx, 0]
        if estop > 100:
            term2_inner = 1e15
            # logger.warning("overflow")  # no-jit
        else:
            term2_inner = np.exp(estop) - np.exp(estart)

        # pij component
        term3 = 0
        for jdx in range(idx, arr.shape[0]):
            term3 += _get_fast_pij(arr, idx, jdx)
        sumval += term1 + (term2_inner * term2_outer * term3)
    return (1 / (tbu - tbl)) * sumval


@njit(parallel=True)
def get_fast_waiting_distance_to_tree_change_rates(
    earr: np.ndarray,
    barr: np.ndarray,
    sarr: np.ndarray,
    recombination_rate: float,
    ) -> np.ndarray:
    """Return LAMBDA rate parameters for waiting distance prob. density.

    Note
    ----
    etable here is 2X neff and no NaN (not default embedding table.)
    """
    lambdas = np.zeros(len(sarr))

    # use numba parallel to iterate over genealogies
    # pylint-disable: not-an-iterable
    for gidx in prange(len(sarr)):
        garr = earr[earr[:, 6] == gidx]
        blens = barr[gidx]
        sumlen = sarr[gidx]
        # probability is a float in [0-1]
        prob_tree = _get_fast_probability_of_tree_change(garr, blens, sumlen)
        # lambda is a rate > 0
        lambdas[gidx] = sumlen * prob_tree * recombination_rate
    return lambdas


################################################################
################################################################
# LIKELIHOOD FUNCTIONS
################################################################
################################################################


@njit
def _update_neffs(supertable: np.ndarray, popsizes: np.ndarray) -> None:
    """Updates 2X diploid Ne values in the concatenated embedding array.

    This is used during MCMC proposals to update Ne values.

    Note
    ----
    This function takes diploid Ne as input and stores it to the
    earr table as 2X the diploid Ne value!!!
    """
    for idx, popsize in enumerate(popsizes):
        supertable[supertable[:, 2] == idx, 3] = popsize * 2


def get_tree_distance_loglik(
    embedding: TreeEmbedding,
    params: np.ndarray,
    recomb: float,
    lengths: np.ndarray,
    ) -> float:
    """Return -loglik of tree-sequence waiting distances between
    tree change events given species tree parameters.

    Here we will assume a fixed known recombination rate.

    Parameters
    ----------
    params: np.ndarray
        An array of effective population sizes to apply to each linaege
        in the demographic model, ordered by their idx label in the
        species tree ToyTree object.
    recomb: float
        per site per generation recombination rate.
    lengths: np.ndarray
        An array of observed waiting distances until tree change
        events.
    embedding_arr: np.ndarray
        An array of genealogy embedding tables for all observed
        genealogies concatenated.
    blen_arr: np.ndarray
        An array of the branch lengths of each Node in each genealogy.
    sumlen_arr: np.ndarray
        An array of the sum of branch lengths in each genealogy.
    """
    # set test parameters on the species tree
    earr, barr, sarr = embedding.get_data()
    _update_neffs(earr, params)

    # get rates (lambdas) for waiting distances
    rates = get_fast_waiting_distance_to_tree_change_rates(
        earr, barr, sarr, recomb)

    # get logpdf of observed waiting distances given rates (lambdas)
    logliks = stats.expon.logpdf(scale=1/rates, x=lengths)
    return -np.sum(logliks)


def get_topology_distance_loglik(
    embedding: TopologyEmbedding,
    params: np.ndarray,
    recomb: float,
    lengths: np.ndarray,
    ) -> float:
    """Return -loglik of tree-sequence waiting distances between
    topology change events given species tree parameters.

    Parameters
    ----------
    params: np.ndarray
        An array of effective population sizes to apply to each linaege
        in the demographic model, ordered by their idx label in the
        species tree ToyTree object.
    recomb: float
        per site per generation recombination rate.
    lengths: np.ndarray
        An array of observed waiting distances until tree change
        events.
    embedding_arr: np.ndarray
        An array of genealogy embedding tables for all observed
        genealogies concatenated.
    blen_arr: np.ndarray
        An array of the branch lengths of each Node in each genealogy.
    sumlen_arr: np.ndarray
        An array of the sum of branch lengths in each genealogy.
    """
    # set test parameters on the species tree
    earr, barr, sarr, rarr = embedding.get_data()
    _update_neffs(earr, params)

    # get rates (lambdas) for waiting distances
    rates = get_fast_waiting_distance_to_topology_change_rates(
        earr, barr, sarr, rarr, recomb)
    # get logpdf of observed waiting distances given rates (lambdas)
    logliks = stats.expon.logpdf(scale=1/rates, x=lengths)
    return -np.sum(logliks)


################################################################
################################################################
# MAIN
################################################################
################################################################

if __name__ == "__main__":

    import ipcoal
    ipcoal.set_log_level("INFO")
    pd.options.display.max_columns = 20
    pd.options.display.width = 1000

    RECOMB = 2e-9
    SEED = 123
    NEFF = 150_000
    ROOT_HEIGHT = 1e6
    NSPECIES = 2
    NSAMPLES = 4
    NSITES = 1e5

    sptree = toytree.rtree.baltree(NSPECIES).mod.edges_scale_to_root_height(ROOT_HEIGHT, include_stem=True)

    sptree.set_node_data("Ne", {0: 2e5, 1: 3e5, 2: 4e5}, inplace=True)
    model = ipcoal.Model(sptree, nsamples=NSAMPLES, recomb=RECOMB, seed_trees=SEED)
    model.sim_trees(1, NSITES)
    imap = model.get_imap_dict()

    # genealogy topo change interval lengths
    topo_lengths = ipcoal.smc.likelihood.get_topology_interval_lengths(model)

    # N and avg tree change length
    print(f"{len(topo_lengths)} genealogy topologies w/ average length={np.mean(topo_lengths):.0f}")

    # load all genealogies as Toytrees
    genealogies = toytree.mtree(model.df.genealogy)

    tdata = TopologyEmbedding(model.tree, genealogies, imap, nproc=4)
    tdata.table[tdata.table.gindex == 10]

    g = tdata._get_genealogies()

    print([ipcoal.smc.get_probability_of_topology_change(model.tree, g[i], imap) for i in range(10)])
    print(
    [ipcoal.smc.likelihood.ms_smc_jit._get_fast_probability_of_topology_change(
        tdata.earr[tdata.earr[:, 6] == i],
        tdata.barr[i],
        tdata.sarr[i],
        tdata.rarr[i],
    ) for i in range(10)]
    )

    # # Setup a species tree with edge lengths in generations
    # SPTREE = toytree.tree("(((A,B),C),D);")
    # SPTREE.set_node_data("height", inplace=True, default=0, mapping={
    #     4: 200_000, 5: 400_000, 6: 600_000,
    # })

    # # Set constant or variable Ne on species tree edges
    # SPTREE.set_node_data("Ne", inplace=True, default=100_000)

    # # Setup a genealogy embedded in the species tree (see below to
    # # instead simulate one but here we just create it from newick.)
    # GTREE = toytree.tree("(((0,1),(2,(3,4))),(5,6));")
    # GTREE.set_node_data("height", inplace=True, default=0, mapping={
    #     7: 100_000, 8: 120_000, 9: 300_000, 10: 450_000, 11: 650_000, 12: 800_000,
    # })

    # # Setup a map of species names to a list sample names
    # IMAP = {
    #     "A": ['0', '1', '2'],
    #     "B": ['3', '4'],
    #     "C": ['5',],
    #     "D": ['6',],
    # }

    # # embedding data
    # tree_data = TreeEmbedding(SPTREE, [GTREE, GTREE], IMAP)
    # topo_data = TreeEmbedding(SPTREE, [GTREE, GTREE], IMAP)

    # # print
    # earr, barr, sarr = tree_data.get_data()

    # _get_fast_probability_of_tree_change()

    # p_topo = get_probability_of_topology_change(SPTREE, GTREE, IMAP)
    # print(f"Probability of no-change\n{p_no:.3f}\n")
    #print(f"Probability of tree-change\n{p_tree:.3f}\n")
    # print(f"Probability of topology-change\n{p_topo:.3f}\n")




    # # setup a species tree
    # SPTREE = toytree.tree("(A:100_000,B:100_000)AB;")
    # SPTREE.set_node_data("Ne", mapping={"A": 10_000, "B": 50_000, "AB": 100_000}, inplace=True)
    # print(SPTREE.get_node_data())

    # # simulate linked genealogies
    # RECOMB = 2e-9
    # MODEL = ipcoal.Model(SPTREE, recomb=RECOMB, nsamples={"A": 3, "B": 2})
    # MODEL.sim_trees(nloci=1, nsites=1e5)
    # print(MODEL.df.head())
    # IMAP = MODEL.get_imap_dict()

    # # get lengths and embedding tables
    # lengths = MODEL.df.nbps.values
    # etables = [
    #     ipcoal.smc.get_genealogy_embedding_table(SPTREE, g, IMAP)
    #     for g in toytree.mtree(MODEL.df.genealogy)
    # ]

    # for etable, length in zip(etables, lengths):

    #     get_waiting_distance_to_tree_change_rv(etable, )

