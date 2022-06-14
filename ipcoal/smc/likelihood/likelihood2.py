#!/usr/bin/env python

"""Much faster tree-change probs.

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
from numba import njit, prange
import ipcoal
from ipcoal.smc import (
    get_probability_tree_unchanged_given_b,
    get_genealogy_embedding_table,
)

logger = logger.bind(name="ipcoal")
ToyTree = TypeVar("ToyTree")


################################################################
################################################################
# GET EMBEDDING DATA
################################################################
################################################################


def get_concat_embedding_data(etables: List[pd.DataFrame]) -> pd.DataFrame:
    """Return concatenated embedding table labeled by genealogy and branch.

    This function is provided primarily for didactic purposes. The 
    concatenated embedding table is 
    """
    # check that etable is properly formatted
    ninodes = etables[0].iloc[-1, 6][0]

    # iterate over each embedding table
    btables = []
    for gidx, etable in enumerate(etables):
        # make a column for the genealogy index
        etable["gindex"] = gidx
        
        # make presence/absence column for each branch
        bidxs = np.zeros((etable.shape[0], ninodes))
        for bidx in range(ninodes):
            btable = etable[etable.edges.apply(lambda x: bidx in x)]
            # record branches in this interval as 1.
            bidxs[btable.index, bidx] = 1
        bidxs = pd.DataFrame(bidxs, columns=range(ninodes))
        btable = pd.concat([etable, bidxs], axis=1)
        btable = btable.drop(columns=["coal", "edges"], index=[btable.index[-1]])
        btable.neff *= 2
        btables.append(btable)
    return pd.concat(btables, ignore_index=True)


def get_relationship_table(genealogies):
    """Return an array with relationships among nodes in each genealogy.

    The returned table is used in likelihood calculations for the 
    waiting distance to topology-change events.
    """
    ntrees = len(genealogies)
    nnodes = genealogies[0].nnodes
    farr = np.zeros((ntrees, nnodes, 3))
    for tidx, tree in enumerate(genealogies):
        for nidx, node in enumerate(tree):
            farr[tidx, nidx] = nidx, node.get_sisters()[0].idx, node.up.idx
    return farr


def get_data(etables: List[pd.DataFrame]) -> Tuple:
    """Returns a tuple of four numpy arrays for MS-SMC' likelihood func.

    This returns numpy arrays with all data needed for fast calculations
    of tree or topology waiting distances. The returned arrays are
    1. concatenated genealogy embedding table
    2. genealogy all edge lengths table
    3. summed genealogy edge lengths table
    4. genealogy edge relationships table 
    """
    earr = get_concat_embedding_data(etables)
    barr = get_super_lengths(earr)
    sarr = barr.sum(axis=1)
    return earr.values.astype(float), barr, sarr


def get_super_lengths(econcat: pd.DataFrame) -> np.ndarray:
    """Return array of shape=(ngenealogies, nnodes) with all edge lengths.

    Parameters
    ----------
    econcat: pd.DataFrame
        The concatenated genealogy embedding table.
    """
    gidxs = sorted(econcat.gindex.unique())
    nnodes = econcat.shape[1] - 7
    larr = np.zeros((len(gidxs), nnodes), dtype=float)

    for gidx in gidxs:
        # get etable for this genealogy
        garr = econcat[econcat.gindex == gidx].values
        ixs, iys = np.nonzero(garr[:, 7:])

        # iterate over nodes of the genealogy
        for bidx in range(nnodes):
            # get index of intervals with this branch
            idxs = ixs[iys == bidx]
            barr = garr[idxs, :]
            blow = barr[:, 0].min()
            btop = barr[:, 1].max()
            larr[gidx, bidx] = btop - blow
    return larr


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


def _get_fast_sum_pb1(btab: np.ndarray, ftab: np.ndarray, mtab: np.ndarray) -> float:
    """Return value for the $p_{b,1}$ variable. 

    This includes components on branch below below t_m.
    """
    pbval = 0

    # iterate over all intervals from 0 to t_m
    t_m = mtab[:, 0].min()
    for idx in range(btab.shape[0]):
        row = btab[idx]
        if row[0] >= t_m:
            continue

        estop = (row[4] / row[3]) * row[1]
        estart = (row[4] / row[3]) * row[0]
        if estop > 100:
            first_term = 1e15
        else:
            first_term = row[3] * (np.exp(estop) - np.exp(estart))

        # pij across bc
        sum1 = 0
        for jidx in range(ftab.shape[0]):
            sum1 += _get_fast_pij(ftab, idx, jidx)  # TODO CHECK IDXS

        # pij from m to end of b
        sum2 = 0
        for sidx in range(mtab.shape[0]):
            sum2 += _get_fast_pij(btab, idx, sidx)

        second_term = sum1 + sum2
        pbval += (1 / row[4]) * (row[5] + (first_term * second_term))
    return pbval


def _get_fast_sum_pb2(btab: np.ndarray, ftab: np.ndarray, mtab: np.ndarray) -> float:
    """Return value for the $p_{b,1}$ variable. 

    This includes components on branch below below t_m.
    """
    pbval = 0

    # iterate over all intervals from m to bu
    for idx in range(mtab.shape[0]):
        row = mtab[idx]

        estop = (row[4] / row[3]) * row[1]
        estart = (row[4] / row[3]) * row[0]
        if estop > 100:
            first_term = 1e15
        else:
            first_term = row[3] * (np.exp(estop) - np.exp(estart))

        # pij across intervals on b
        sum1 = 0
        for bidx in range(btab.shape[0]):
            sum1 += _get_fast_pij(btab, idx, bidx)  # TODO CHECK IDXS

        # pij across intervals on c
        sum2 = 0
        for pidx in range(ftab.shape[0]):
            # skip intervals ...
            if ftab[pidx, 0]:
                pass
            sum2 += _get_fast_pij(ftab, idx, pidx)

        second_term = (2 * sum1) + sum2
        pbval += (1 / row[4]) * ((2 * row[5]) + (first_term * second_term))
    return pbval


################################################################
################################################################
## TOPOLOGY CHANGE
################################################################
################################################################


def get_fast_probability_of_topology_change(
    garr: np.ndarray, 
    barr: np.array, 
    sumlen: float,
    ) -> float:
    """Return probability that recombination causes a topology-change.

    """
    total_prob = 0

    # iterate over branch indices 
    for bidx, blen in enumerate(barr):

        # get indices to select intervals of the genealogy embedding 
        # table that include this branch, for every genealogy.
        idxs = np.nonzero(garr[:, 7 + bidx])[0]

        # get P(tree-unchanged | S, G, b) for every genealogy
        prob = _get_fast_probability_topology_unchanged_given_b(garr[idxs, :])

        # get Prob scaled by the proportion of this branch on each tree.
        total_prob += (blen / sumlen) * prob
    return 1 - total_prob    


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
    mtab = arr[(arr[:, 7 + branch] == 1) and (arr[:, 7 + sibling] == 1)]

    # get intervals containing either b or c
    ftab = arr[(arr[:, 7 + branch] == 1) | (arr[:, 7 + parent] == 1)]

    # get lower and upper bounds of this gtree edge
    t_lb, t_ub = btab[:, 0].min(), btab[:, 1].max()

    # get sum pb1 from intervals 0 to m
    pb1 = _get_fast_sum_pb1(btab, ftab, mtab)

    # get sum pb2 from m to end of b
    pb2 = _get_fast_sum_pb2(btab, ftab, mtab)
    return (1 / (t_ub - t_lb)) * (pb1 + pb2)


def get_fast_waiting_distance_to_topology_change_rates():
    pass

################################################################
################################################################
## TREE CHANGE
################################################################
################################################################

@njit
def get_fast_probability_of_tree_change(
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
    for bidx, blen in enumerate(barr):
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
    embedding_arr: np.ndarray,
    blen_arr: np.ndarray,
    sumlen_arr: np.ndarray,
    recombination_rate: float,
    ) -> np.ndarray:
    """Return LAMBDA rate parameters for waiting distance prob. density.

    Note
    ----    
    etable here is 2X neff and no NaN (not default embedding table.)
    """
    lambdas = np.zeros(len(sumlen_arr))

    # use numba parallel to iterate over genealogies
    for gidx in prange(len(sumlen_arr)):
        sumlen = sumlen_arr[gidx]
        garr = embedding_arr[embedding_arr[:, 6] == gidx]
        barr = blen_arr[gidx]
        sumlen = sumlen_arr[gidx]
        prob_tree = get_fast_probability_of_tree_change(garr, barr, sumlen)
        lambdas[gidx] = sumlen * prob_tree * recombination_rate
    return lambdas


################################################################
################################################################
# LIKELIHOOD FUNCTIONS
################################################################
################################################################

@njit
def update_neffs(supertable: np.ndarray, popsizes: np.ndarray) -> None:
    """Updates the embedding table with new Ne values.
    
    Note: this function takes Ne as input and sets the ni variable
    in the embedding table to 2Ne.
    """
    for idx, popsize in enumerate(popsizes):
        supertable[supertable[:, 2] == idx, 3] = popsize * 2


def get_tree_distance_loglik(
    params: np.ndarray, 
    recomb: float, 
    lengths: np.ndarray, 
    embedding_arr: np.ndarray,
    blen_arr: np.ndarray,
    sumlen_arr: np.ndarray,
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
    update_neffs(embedding_arr, params)
    
    # get probability distribution
    args = embedding_arr, blen_arr, sumlen_arr, recomb
    # get rates (lambdas) for waiting distances
    rates = get_fast_waiting_distance_to_tree_change_rates(*args)
    # do not allow rates to be 0, minimum is 1bp b/c genome is discrete
    logger.debug(f"min rates={min(rates):.3f}")
    rates[rates == 0] = 1
    # get logpdf of observed waiting distances given rates (lambdas)
    logliks = stats.expon.logpdf(scale=1/rates, x=lengths)
    return -np.sum(logliks)


# def get_tree_distance_loglik_parallel(
#     params: List[int],
#     recomb: float, 
#     lengths: np.ndarray, 
#     edicts: Dict,
#     nworkers: int=4,
#     ) -> float:
#     """Return -loglik of tree-sequence waiting distances given species tree.
#     """
#     # set test parameters on the species tree
#     update_neffs(edicts, dict(zip(range(len(params)), params)))
    
#     # get probability distribution
#     rasyncs = []
#     with ProcessPoolExecutor(max_workers=nworkers) as pool:
#         for edict in edicts:
#             rasync = pool.submit(
#                 get_fast_waiting_distance_to_tree_change_rv, *(edict, recomb))            
#             rasyncs.append(rasync)
#     loglik = 0    
#     for idx, rasync in enumerate(rasyncs):
#         dist = rasync.result()
#         loglik += dist.logpdf(lengths[idx])
#         # print(idx, lengths[idx], dist.mean(), dist.pdf(lengths[idx]))
#     return -loglik                    


################################################################
################################################################
# MAIN
################################################################
################################################################

if __name__ == "__main__":


    ipcoal.set_log_level("WARNING")
    pd.options.display.max_columns = 20
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
    print(f"Single genealogy embedding table\n{ETABLE}\n")

    eee = get_concat_embedding_data([ETABLE, ETABLE, ETABLE])
    print(f"Fast multiple genealogy embedding table\n{eee}\n")

    EARR, BARR, SARR = get_data([ETABLE, ETABLE, ETABLE])

    # for gidx, sumlen in enumerate(SARR):
    #     garr = EARR[EARR[:, 6] == gidx]
    #     barr = BARR[gidx]
    #     sumlen = SARR[gidx]
    #     prob_tree = get_fast_probability_of_tree_change(garr, barr, sumlen)
    #     print(gidx, prob_tree)




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

