#!/usr/bin/env python

"""Deprecated: first attempt at faster implementation.

This contains modified versions of the functions in ms_smc.py that
are written to be faster (using jit compilation) and to reduce some
redundancy that would arise when the same functions are run 
repeatedly without changing the 

"""

from typing import TypeVar, Dict, Tuple, List
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger
import toytree
from numba import njit
import ipcoal
from ipcoal.smc import (
    get_probability_tree_unchanged_given_b,
    get_genealogy_embedding_table,
)

logger = logger.bind(name="ipcoal")
ToyTree = TypeVar("ToyTree")


def get_fast_probability_of_tree_change(edict: Dict[int, np.ndarray], sumlen: float) -> float:
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
    etable:
        Genealogy embedding table.
    """
    # traverse over all edges of the genealogy
    total_prob = 0
    for gidx in sorted(edict)[:-1]:

        # get P(tree-unchanged | S, G, b)
        arr, blen = edict[gidx]
        prob = _get_fast_probability_tree_unchanged_given_b(arr)

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


def get_fast_waiting_distance_to_tree_change_rv(
    edict: Dict[int, np.ndarray],
    recombination_rate: float,
    ) -> stats._distn_infrastructure.rv_frozen:
    """Return ...

    Note
    ----    
    etable here is 2X neff and no NaN (not default embedding table.)
    """    
    sumlen = sum(edict[i][1] for i in range(len(edict) - 1))
    prob_tree = get_fast_probability_of_tree_change(edict, sumlen)
    lambda_ = sumlen * prob_tree * recombination_rate
    return stats.expon.freeze(scale=1/lambda_)


def get_fast_embedding_dict(etable: pd.DataFrame) -> Dict[int, List[int]]:
    """..."""
    # check that etable is properly formatted
    nnodes = etable.values[-1, 6][0]

    # build dict mapping node to edge indices to be re-used
    edict = {}
    for gidx in range(nnodes + 1):
        btable = etable[etable.edges.apply(lambda x: gidx in x)]
        arr = btable.values[:, [0, 1, 2, 3, 4, 7]]
        blen = arr[:, 1].max() - arr[:, 0].min()
        arr[:, 3] *= 2
        if gidx == nnodes:
            arr[-1, [1, 5]] = 1e12
        arr = arr.astype(float)
        edict[gidx] = (arr, blen)

    # now convert embedding table to an array for faster operations
    return edict


def update_neffs(edicts: List[Dict[int, np.ndarray]], pop_to_ne: Dict[int, int]) -> None:
    """Updates the edict embedding tables with new Ne values."""
    for edict in edicts:
        for key, vals in edict.items():
            arr, _ = vals
            for row in range(arr.shape[0]):
                pop = int(vals[0][row][2])
                edict[key][0][row][3] = 2 * pop_to_ne[pop]


def get_loglik(params: List[int], recomb: float, lengths: np.ndarray, edicts: Dict) -> float:
    """Return -loglik of tree-sequence waiting distances given species tree.
    
    Here we will assume a fixed known recombination rate.

    This can be parallelized...
    """
    # set test parameters on the species tree
    update_neffs(edicts, dict(zip(range(len(params)), params)))
    
    # get probability distribution
    loglik = 0
    for edict, length in zip(edicts, lengths):
        dist = get_fast_waiting_distance_to_tree_change_rv(edict, recomb)
        loglik += dist.logpdf(length)
    return -loglik                


def get_loglik_parallel(
    params: List[int],
    recomb: float, 
    lengths: np.ndarray, 
    edicts: Dict,
    nworkers: int=4,
    ) -> float:
    """Return -loglik of tree-sequence waiting distances given species tree.
    """
    # set test parameters on the species tree
    update_neffs(edicts, dict(zip(range(len(params)), params)))
    
    # get probability distribution
    rasyncs = []
    with ProcessPoolExecutor(max_workers=nworkers) as pool:
        for edict in edicts:
            rasync = pool.submit(
                get_fast_waiting_distance_to_tree_change_rv, *(edict, recomb))            
            rasyncs.append(rasync)
    loglik = 0    
    for idx, rasync in enumerate(rasyncs):
        dist = rasync.result()
        loglik += dist.logpdf(lengths[idx])
        # print(idx, lengths[idx], dist.mean(), dist.pdf(lengths[idx]))
    return -loglik                    


if __name__ == "__main__":


    ipcoal.set_log_level("WARNING")
    pd.options.display.max_columns = 10
    pd.options.display.width = 1000

    # Setup a species tree with edge lengths in generations
    SPTREE = toytree.tree("(((A,B),C),D);")
    SPTREE.set_node_data("height", inplace=True, default=0, mapping={
        4: 200_000, 5: 400_000, 6: 600_000,
    })

    # Set constant or variable Ne on species tree edges
    SPTREE.set_node_data("Ne", inplace=True, default=100_000)

    # Setup a genealogy embedded in the species tree (see below to
    # instead simulate one but here we just create it from newick.)
    GTREE = toytree.tree("(((0,1),(2,(3,4))),(5,6));")
    GTREE.set_node_data("height", inplace=True, default=0, mapping={
        7: 100_000, 8: 120_000, 9: 300_000, 10: 450_000, 11: 650_000, 12: 800_000,
    })

    # Setup a map of species names to a list sample names
    IMAP = {
        "A": ['0', '1', '2'],
        "B": ['3', '4'],
        "C": ['5',],
        "D": ['6',],
    }

    # Select a branch to plot and get its relations
    BIDX = 7
    BRANCH = GTREE[BIDX]
    SIDX = BRANCH.get_sisters()[0].idx
    PIDX = BRANCH.up.idx

    # Get genealogy embedding table
    ETABLE = get_genealogy_embedding_table(SPTREE, GTREE, IMAP)
    print(f"Full genealogy embedding table\n{ETABLE}\n")

    EDICT = get_fast_embedding_dict(ETABLE)

    # Get subset of embedding table for a single genealogy branch path
    # BTABLE = get_genealogy_embedding_edge_path(ETABLE, BIDX)
    # print(f"Path of branch {BIDX} in genealogy embedding table\n{BTABLE}\n")

    # Get probabilities integrated over the genealogy
    # p_no = get_probability_of_no_change(SPTREE, GTREE, IMAP)

    SUMLEN = sum(EDICT[i][1] for i in range(len(EDICT) - 1))
    p_tree = get_fast_probability_of_tree_change(EDICT, SUMLEN)
    # p_topo = get_probability_of_topology_change(SPTREE, GTREE, IMAP)
    # print(f"Probability of no-change\n{p_no:.3f}\n")
    print(f"Probability of tree-change\n{p_tree:.3f}\n")
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

