#!/usr/bin/env python

"""Calculate likelihood of a gene tree embedded in a species tree.

Fast calculation of ...

References
----------
- Rannala and Yang (...) "Bayesian"
- Degnan and Salter (...) "..."
- ... (...) "STELLS-mod..."

"""

from typing import Dict
import itertools
import numpy as np
import pandas as pd
from loguru import logger
import toytree
import ipcoal


logger = logger.bind(name="ipcoal")


def get_msc_embedded_gene_tree_table(
    species_tree: toytree.ToyTree,
    gene_tree: toytree.ToyTree,
    imap: Dict,
    ) -> pd.DataFrame:
    """Return a DataFrame with intervals of coal and non-coal events.

    Each row in the dataframe represents an interval of time along
    the selected gene tree edge (idx), where a recomb event could break
    the current gene tree. It returns the information needed for
    calculating the prob it reattaches to another gene tree nedges given
    the length of each interval (dist), the nedges that exist then
    (nedges), and the species tree params (tau and neff).

    Parameters
    ----------
    species_tree: toytree.ToyTree
        Species tree with a "Ne" feature assigned to every node, and
        edge lengths in units of generations. The tree can be non-
        ultrametric, representing differences in generation times.
    gene_tree: toytree.ToyTree
        Gene tree that could be embedded in the species tree. Edge
        lengths are in units of generations.
    imap: Dict
        A dict mapping species tree node idxs to gene tree node idxs.
    """
    # store temporary results in a dict
    data = {}

    # get table of gene tree node heights
    gt_node_heights = gene_tree.get_node_data("height")

    # iterate over species tree nodes from tips to root
    for nidx in range(species_tree.ntips)[::-1]: #.treenode.traverse("postorder"):
        st_node = species_tree[nidx]

        # get n nedges into the species tree interval, for tips it is
        # nsamples, for internal intervals get from child intervals.
        if st_node.is_leaf():
            gt_tips = set(imap[st_node.name])
            nedges_in = len(gt_tips)
        else:
            child_idxs = [i.idx for i in st_node.children]
            nedges_in = sum(data[idx]['nedges_out'] for idx in child_idxs)
            st_tips = st_node.get_leaf_names()
            gt_tips = set(itertools.chain(*[imap[i] for i in st_tips]))

        # get nodes that occur in the species tree interval (coalescences)
        mask_below = gt_node_heights > st_node.height + 0.0001
        if st_node.is_root():
            mask_above = gt_node_heights > 0
        else:
            mask_above = gt_node_heights < st_node.up.height
        nodes_in_time_slice = gt_node_heights[mask_below & mask_above]

        # get nodes in the appropriate species tree interval
        coal_events = []
        for gidx in nodes_in_time_slice.index:
            gt_node = gene_tree[gidx]
            tmp_tips = set(gt_node.get_leaf_names())
            if tmp_tips.issubset(gt_tips):
                coal_events.append(gt_node)

        # count nedges out of the interval
        nedges_out = nedges_in - len(coal_events)

        # sort coal events by height, and get height above last event
        # which is either st_node, or last gt_coal.
        coal_dists = []
        for node in sorted(coal_events, key=lambda x: x.height):
            if not coal_dists:
                coal_dists.append(node.height - st_node.height)
            else:
                coal_dists.append(node.height - st_node.height - sum(coal_dists))

        # store coalescent times in the interval
        data[st_node.idx] = {
            "dist": st_node.dist if st_node.up else np.inf,
            "neff": st_node.Ne,
            "nedges_in": nedges_in,
            "nedges_out": nedges_out,
            "coals": coal_dists,
        }
    data = pd.DataFrame(data).T.sort_index()
    return data


def get_gene_tree_log_prob_single_pop(neff: float, coal_times: np.ndarray):
    """Return log prob density of a gene tree in a single population.

    All labeled histories have equal probability in a single population
    model, and so the probability of a gene tree is calculated only 
    from the coalescent times.

    Time to the first coalescent event is geometric with parameter
    k_choose_2 * 1 / 2N. Kingman's coalescent makes a continous 
    approximation of this function, where waiting time is exponentially
    distributed.

    Modified from equation 5 of Rannala et al. (2020) to use edge 
    lens in units of gens, and population neffs, instead of thetas.

    Parameters
    ----------
    neff: float
        Effective population size
    coal_times: np.ndarray, shape=2, dtype=float
        An array of ordered coalescent times for all internal nodes
        in a tree except the root, where n = ntips in the tree the 
        order of times represents when n= [n-1, n-2, ..., 2].

    Example
    -------
    >>> import ipcoal, toytree, toyplot
    >>> neff = 1e6
    >>> model = ipcoal.Model(None, Ne=neff, nsamples=25)
    >>> model.sim_trees(1)
    >>> gtree = toytree.tree(model.df.genealogy[0])
    >>> coals = np.array(sorted(gtree.get_node_data("height")[gtree.ntips:]))
    >>> xs = np.logspace(np.log10(neff) - 1, np.log10(neff) + 1)
    >>> logliks = [get_gene_tree_log_prob_single_pop(i, coals) for i in xs]
    >>> canvas, axes, mark = toyplot.plot(
    >>>     xs, logliks,
    >>>     xscale="log", height=300, width=400,
    >>> )
    >>> axes.vlines([neff])
    """
    nlineages = len(coal_times) + 1
    rate = (1 / (2 * neff))
    prob = 1
    for idx, nlineages in enumerate(range(nlineages, 1, -1)):
        npairs = (nlineages * (nlineages - 1)) / 2
        time = (coal_times[idx] - coal_times[idx - 1] if idx else coal_times[idx])
        opportunity = npairs * time
        prob *= rate * np.exp(-rate * opportunity)
    if prob > 0:
        return np.log(prob)
    return np.inf


def optim_func(neff: float, coal_times: np.ndarray):
    """Return the log prob density of a set of gene trees in a single pop.

    Example
    -------
    >>> import ipcoal, toytree, toyplot
    >>> neff = 1e5
    >>> model = ipcoal.Model(None, Ne=neff, nsamples=20)
    >>> model.sim_trees(100)
    >>> coal_times = np.array([
    >>>     sorted(gtree.get_node_data("height")[gtree.ntips:])
    >>>     for gtree in toytree.mtree(model.df.genealogy)
    >>> ])
    >>> xs = np.logspace(np.log10(neff) - 1, np.log10(neff) + 1)
    >>> logliks = [optim_func(i, coal_times) for i in xs]
    >>> canvas, axes, mark = toyplot.plot(
    >>>     xs, logliks,
    >>>     xscale="log", height=300, width=400,
    >>> )
    >>> axes.vlines([neff]);
    """
    assert coal_times.ndim == 2, "coal_times must be shape: (ntrees, ncoals)"
    logprobs = [get_gene_tree_log_prob_single_pop(neff, i) for i in coal_times]
    loglik = np.sum(logprobs)
    if loglik == np.inf:
        return loglik
    return -loglik


def get_censored_interval_log_prob(
    neff: float,
    nedges_in: int,
    nedges_out: int,
    interval_dist: float,
    coal_dists: np.ndarray,
    ):
    """Return the log probability of a censored species tree interval.

    Parameters
    ----------
    neff: float
        The diploid effective population size (Ne).
    nedges_in: int
        Number of gene tree edges at beginning of interval.
    nedges_in: int
        Number of gene tree edges at end of interval.
    interval_dist: float
        Length of the species tree interval in units of generations.
    coal_dists: np.ndarray
        Array of ordered coal *interval lengths* within a censored 
        population interval, representing the dist from one event to 
        the next. Events are gene tree coalescent times ordered from 
        when the most edges existed to when the fewest existed.
    
    Notes
    -----
    Compared to the references below, we convert units to use the 
    population Ne values and time in units of generations, as opposed 
    popuation theta values and time in units of substitutions. This is
    because when working with genealogies alone we do not need the 
    mutation rate. This conversion represents the coalescent rate as
    (1 / 2Ne) instead (2 / theta), since theta=4Neu, and we will assume
    that u=1. Similarly, time in units of E(substitutions/site) 
    represents a measure of rate x time, where time is in generations, 
    and rate (u) is (mutations / site / generation). To get in 
    generations, we multiply by 1 / u, which has of effect when u=1.

    References
    ----------
    - Rannala and Yang (2003)
    - Rannala et al. (2020) book chapter.

    Example
    -------
    >>> ...
    """
    # coalescent rate in this interval
    rate = 1 / (2 * neff)

    # length of final subinterval, where no coalesce occurs
    remaining_time = interval_dist - np.sum(coal_dists)

    # The probability density of observing n-m coalescent events.
    # The opportunity for coalescence is described by the number of 
    # ways that m lineages can be joined, times the length of time
    # over which m lineages existed. This 'opportunity' is treated as
    # an *exponential waiting time* with coalescence rate lambda, so:
    #         prob_density = rate * np.exp(-lambda * rate)
    # this prob density is 1 if no lineages coalesce in the interval.
    prob_coal = 1.
    for idx, nedges in enumerate(range(nedges_in, nedges_out, -1)):
        npairs = (nedges * (nedges - 1)) / 2
        opportunity = npairs * coal_dists[idx]
        prob_coal *= rate * np.exp(-rate * opportunity)
        logger.warning(f"npairs={npairs}, coal_t={coal_dists[idx]}")

    p2 = 0
    for idx, nedges in enumerate(range(nedges_in, nedges_out, -1)):
        npairs = (nedges * (nedges - 1)) / 2
        time = coal_dists[idx]
        p2 += np.exp(-npairs * time)

    # The probability that no coalescent events occur from the last 
    # event until the end of the species tree interval. The more
    # edges that remain, and the longer the remaining distance, the 
    # higher this probability is. It is 1 if nedges_out=1, bc there 
    # is no one left to coalesce with in the interval. The species tree
    # root interval is always 1.
    prob_no_coal = 1.
    if nedges_out > 1:
        npairs_not_coalesced = (nedges_out * (nedges_out - 1)) / 2
        opportunity = npairs_not_coalesced * remaining_time
        prob_no_coal = np.exp(-rate * opportunity)

    # multiply to get joint prob dist of the gt in the pop
    prob = prob_coal * prob_no_coal
    logger.info(f"pcoal={prob_coal}, pno-coal={prob_no_coal}, prob={prob}")

    # return log positive results
    if prob > 0:
        return np.log(prob)
    return np.inf


def get_gene_tree_log_prob_msc(table: pd.DataFrame):
    """Return the log probability of a gene tree given a species tree.

    Example
    -------
    >>> 
    >>> 
    >>> 
    """
    # iterate over the species tree intervals to sum of logliks
    loglik = 0
    for interval in table.index:
        dat = table.loc[interval]

        # get log prob of censored coalescent
        prob = get_censored_interval_log_prob(
            dat.neff, dat.nedges_in, dat.nedges_out, dat.dist, dat.coals)
        loglik += prob     

    # species tree prob is the product of population probs
    if loglik == np.inf:
        return loglik
    return -loglik



if __name__ == "__main__":
    

    ipcoal.set_log_level("INFO")

    # setup species tree model
    SPTREE = toytree.rtree.unittree(ntips=3, treeheight=1e6, seed=123)
    SPTREE = SPTREE.set_node_data(
        "Ne", default=1e5, #mapping={i: 5e5 for i in (0, 1, 8, 9)}
    )

    # simulate genealogies
    RECOMB = 1e-9
    MUT = 1e-9
    NEFF = 5e5
    THETA = 4 * NEFF * MUT
    MODEL = ipcoal.Model(
        SPTREE,
        Ne=NEFF,
        seed_trees=123,
        nsamples=4,
        recomb=RECOMB,
        mut=MUT,
    )
    MODEL.sim_trees(100, 1)
    IMAP = MODEL.get_imap_dict()
    GTREES = toytree.mtree(MODEL.df.genealogy.tolist())

    # get embedding table
    DATA = get_msc_embedded_gene_tree_table(SPTREE, GTREES.treelist[0], IMAP)
    print(DATA)


    # # hello
    # LOGLIK = get_gene_tree_log_prob_msc(DATA)
    # print(LOGLIK)


    # # single
    # DATA = DATA.iloc[2]
    # LOGLIK = get_censored_interval_log_prob(
    #     DATA.neff, 
    #     DATA.nedges_in, 
    #     DATA.nedges_out, 
    #     DATA.dist, 
    #     DATA.coals,
    # )
    # print(LOGLIK)
