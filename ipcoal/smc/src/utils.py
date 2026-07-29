#!/usr/bin/env python

"""Utility functions for SMC subpackage.

Methods
-------
# Get one simple example dataset of S,G,I for testing MSC
- get_test_data()

# get interval lengths and trees from Model simulations
- iter_spans_and_trees_from_model(...)
- iter_spans_and_topos_from_model(...)

# get interval lengths and trees from a list of ToyTrees
- iter_spans_and_trees_from_model(...)
- iter_spans_and_topos_from_model(...)

# get topos
- iter_topos_from_trees

# parse ARGweaver results...
- ...

"""

from typing import Iterator, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor
from itertools import accumulate
from loguru import logger
import numpy as np
import pandas as pd
import toytree
from toytree import ToyTree
import ipcoal

logger = logger.bind(name="ipcoal")


def get_test_data():
    """Return (SPTREE, GTREE, IMAP) for manuscript example in Figs. S6-7
    """
    SPTREE = toytree.tree("((A,B),C);")
    SPTREE.set_node_data("height", inplace=True, default=0, data={3: 1000, 4: 3000})
    SPTREE.set_node_data("Ne", default=1e3, inplace=True)
    GTREE = toytree.tree("((0,(1,2)),3);")
    GTREE.set_node_data("height", inplace=True, default=0, data={4: 2000, 5: 4000, 6: 5000})
    IMAP = {"A": ['0'], "B": ['1', '2'], "C": ['3']}
    return SPTREE, GTREE, IMAP


def iter_spans_and_trees_from_model(
    model: ipcoal.Model,
    locus: Optional[int] = None,
) -> Iterator[Tuple[int, ToyTree]]:
    """Yield (int, ToyTree) tuples for each tree in a tree sequence.

    Note: This returns every tree in a tree sequence, it does not check
    whether the tree has changed or not. To treat these as tree-changes
    requires that data were simulated under the record_full_arg=False
    (default) option in ipcoal/msprime such that recombination events
    causing no-change are not recorded.
    """
    data = model.df[model.df.locus == locus] if locus is not None else model.df
    for span, newick in data[["nbps", "genealogy"]].values[:-1]:
        yield (span, toytree.tree(newick))


def iter_spans_and_topos_from_model(
    model: ipcoal.Model,
    locus: Optional[int] = None,
) -> Iterator[Tuple[int, ToyTree]]:
    """Yield (int, ToyTree) tuples for each topology in a tree sequence.

    This checks each tree from `iter_spans_and_trees_from_model` and
    returns a tree only when the rooted topology changes, and returns
    the sum of the intervals over which that topology spans.

    Parameters
    ----------
    model: ipcoal.Model
        An ipcoal.Model object that has called a simulate function
        such as `sim_trees`, `sim_loci`, etc. such that it has data
        in its `.df` attribute.
    """
    iterator = iter_spans_and_trees_from_model(model, locus=locus)
    span, gtree = next(iterator)
    current = gtree.get_topology_id(include_root=True)
    intervals = [span]

    for (span, gtree) in iterator:
        new = gtree.get_topology_id(include_root=True)
        if current != new:
            yield sum(intervals), gtree
            current = new
            intervals = [span]
        else:
            intervals.append(span)


def get_ms_smc_data_from_model(model: ipcoal.Model):
    """Return tree and topo change data

    # TODO: parallelize by 'locus'
    """
    trees = []
    tree_spans = []
    topo_idxs = []
    topo_spans = []

    gidx = 0
    tidx = 0
    iterator = iter_spans_and_trees_from_model(model)
    span, gtree = next(iterator)
    current = gtree.get_topology_id(include_root=True)
    intervals = [span]

    # store the first tree
    tree_spans.append(span)
    trees.append(gtree)

    # iterate over all subsequent trees
    for (span, gtree) in iterator:
        tree_spans.append(span)
        trees.append(gtree)
        new = gtree.get_topology_id(include_root=True)
        gidx += 1

        # if topo change occurred store last tree
        if current != new:
            topo_idxs.append(tidx)
            topo_spans.append(sum(intervals))
            current = new
            intervals = [span]
            tidx = gidx
        else:
            intervals.append(span)
    return np.array(tree_spans), np.array(topo_spans), np.array(topo_idxs), trees


# THIS IS THE MAIN FUNC USED IN MCMC2.py 
def get_waiting_distance_data_from_model(model: ipcoal.Model, skip_first: bool=True, skip_last: bool=True):
    """Return tree and topo change data.

    Parameters
    ----------
    skip_first: bool
        If True trees and intervals up until the first topology-change
        event are discarded.
    skip_last: bool
        If True trees and intervals following the last topology-change
        event are discarded.

    Returns
    -------
    tree_spans: 
        Interval lengths spanned by each tree (length=N)
    topo_spans:
        Interval lengths spanned by each tree (length=T)
    topo_idxs:
        Indices of trees in trees corresponding to the T intervals
    trees:
        List of ToyTree objects corresponding to the N intervals
    """
    trees = []
    tree_spans = []
    # tree_idxs = []
    topo_idxs = []
    topo_spans = []
    intervals = []
    gidx = 0
    tidx = 0

    # 
    df = model.df.copy()
    df["tree"] = [toytree.tree(i) for i in df.genealogy]
    df["tid"] = [i.get_topology_id(include_root=True) for i in df.tree]

    # iterate over independent loci (tree-sequences) one at a time.
    for lidx, ldf in df.groupby("locus"):
        # create an iterator over the locus dataframe
        iterator = ldf.iterrows()
        # sample the first row in the locus df as a Series
        ridx, row = next(iterator)
        # extract the first interval and tree and get the topology ID
        ospan, otree, otid = row.nbps, row.tree, row.tid  # toytree.tree(row.genealogy)        
        # ospan, otree = row.nbps, row.tree  # toytree.tree(row.genealogy)
        # otid = otree.get_topology_id(include_root=True)

        # advance until a new topology is found to treat as the true
        # starting interval. Leaving here otree and ospan contain the
        # first interval and tree that should be saved.
        if skip_first:
            for (ridx, row) in iterator:
                nspan, ntree, ntid = row.nbps, row.tree, row.tid
                # nspan, ntree = row.nbps, toytree.tree(row.genealogy)
                # ntid = ntree.get_topology_id(include_root=True)
                if ntid != otid:
                    logger.info(f"skipped to first topology change at index {ridx}")
                    otid = ntid
                    ospan = nspan
                    otree = ntree
                    break

        # iterate over all subsequent rows (interval, tree)
        for (nidx, row) in iterator:

            # parse new interval
            nspan, ntree, ntid = row.nbps, row.tree, row.tid            
            # nspan, ntree = row.nbps, toytree.tree(row.genealogy)

            # store the last tree and span
            # tree_idxs.append(ridx)
            tree_spans.append(ospan)
            trees.append(otree)            
            intervals.append(ospan)

            # counter of new interval
            gidx += 1

            # get id of new tree
            ntid = ntree.get_topology_id(include_root=True)

            # if topo change occurred store last tree
            if otid != ntid:
                # store index of otree and advance counter
                topo_idxs.append(tidx)
                tidx = gidx

                # store span since old tree
                topo_spans.append(sum(intervals))
                intervals = []

            # update variables representing the previous tree
            otid = ntid
            ospan = nspan
            otree = ntree
            ridx = nidx
    # drop the final topo dist and span
    # if skip_last:
    #     topo_spans = topo_spans[:-1]        
    #     topo_idxs = topo_idxs[:-1]
    return np.array(tree_spans), np.array(topo_spans), np.array(topo_idxs), trees


# NOT YET IPMLEMENTED: NEED TO CONCAT and validate against single-core
def new_get_waiting_distance_data_from_model(model: ipcoal.Model, nproc: int = None):
    """Return tree-dists, topo-dists, topo-change-indices, trees
    """
    rasyncs = {}
    with ProcessPoolExecutor(max_workers=nproc) as pool:
        # treat each locus at a time to skip last tree
        # in a locus that is cut-off at its ending.
        for lidx, ldf in model.df.groupby("locus"):
            rasyncs[lidx] = pool.submit(_remote_subfunc, ldf)
    gspans, tspans, tidxs, trees = zip(*[rasyncs[i].result() for i in rasyncs])
    gspans = np.concatenate(gspans, axis=0)
    tspans = np.concatenate(tspans, axis=0)
    return gspans, tspans, tidxs, trees


def _remote_subfunc(data):
    """Remote function called by get_waiting_distance_data_from_model when
    run in parallel.
    """
    trees = []
    tree_spans = []
    topo_idxs = []
    topo_spans = []
    intervals = []
    gidx = 0
    tidx = 0

    iterator = data.iterrows()
    _, row = next(iterator)
    ospan, otree = row.nbps, toytree.tree(row.genealogy)
    otid = otree.get_topology_id(include_root=True)

    # iterate over all subsequent trees
    for (_, row) in iterator:

        # parse new interval
        nspan, ntree = row.nbps, toytree.tree(row.genealogy)

        # counter of current interval
        gidx += 1

        # store the last tree and span
        tree_spans.append(ospan)
        trees.append(otree)
        intervals.append(ospan)

        # get id of new tree
        ntid = ntree.get_topology_id(include_root=True)

        # if topo change occurred store last tree
        if otid != ntid:
            # store index of otree and advance counter
            topo_idxs.append(tidx)
            tidx = gidx

            # store span since old tree
            topo_spans.append(sum(intervals))
            intervals = []

        otid = ntid
        ospan = nspan
        otree = ntree
    return np.array(tree_spans), np.array(topo_spans), np.array(topo_idxs), trees


def get_model_tree_and_topo_data(model: ipcoal.Model, skip_first: bool = True, skip_last: bool = True):
    """Returns a dict with tree and topo tree and length info.

    """
    gidx = 0
    trees = []
    tree_spans = []
    topo_spans = []
    topo_tlens = []
    topo_idxs = []

    # iterate over independent loci (tree-sequences) one at a time.
    for lidx, ldf in model.df.groupby("locus"):

        # operate on a copy of the ldf
        tidx = 0
        ldf["tree"] = [toytree.tree(i) for i in ldf.genealogy]
        ldf["tid"] = [i.get_topology_id(include_root=True) for i in ldf.tree]
        ldf["tidx"] = 0

        # iterate over rows in the locus df
        otid = ldf.iloc[0].tid
        for idx, row in ldf.iterrows():
            if row.tid != otid:
                tidx += 1
                otid = row.tid
            ldf.loc[idx, "tidx"] = tidx

        # remove first tree topology intervals
        if skip_first:
            ldf = ldf[ldf.tidx > 0]
        if skip_last:
            ldf = ldf[ldf.tidx < ldf.tidx.max()]
        
        # store trees and tree intervals
        trees.extend(ldf.tree.tolist())
        tree_spans.extend(ldf.nbps.tolist())
        # store topo intervals
        topo_spans.extend(j.nbps.sum() for i, j in ldf.groupby("tidx"))
        # store number of trees per topo
        lens = [j.nbps.size for i, j in ldf.groupby("tidx")]
        topo_tlens.extend(lens)
        # store indices of trees in trees used to represent the topos
        gidxs = [gidx] + [gidx + i for i in accumulate(lens[:-1])]
        topo_idxs.extend(gidxs)
        # advance counter
        gidx = len(trees)

    return {
        "trees": trees,
        "tree_spans": np.array(tree_spans),
        "topo_spans": np.array(topo_spans),
        "topo_tlens": np.array(topo_tlens),
        "topo_idxs": np.array(topo_idxs),
    }



if __name__ == "__main__":

    ipcoal.set_log_level("DEBUG")
    MODEL = ipcoal.Model(Ne=100_000, nsamples=10, seed_trees=333)
    MODEL.sim_trees(3, 50_000)
    # print(MODEL.df.head(15))
    # print("...")
    # print(MODEL.df.tail(15))
    # print("\n")    

    res = get_model_tree_and_topo_data(MODEL)
    gtrees = res["trees"]
    tree_spans = res["tree_spans"]
    topo_spans = res["topo_spans"]
    topo_tlens = res["topo_tlens"]
    topo_idxs = res["topo_idxs"]

    N=10
    print(len(gtrees), [i.get_topology_id(include_root=True) for i in gtrees[:]], "trees")
    print(len(tree_spans), tree_spans, tree_spans.dtype, "tree spans")
    print(len(topo_spans), topo_spans, topo_spans.dtype, "topo_spans")
    print(len(topo_tlens), topo_tlens, topo_tlens.dtype, "topo_tlens")
    print(len(topo_idxs), topo_idxs, "topo_idxs")

    print(gtrees[27].get_topology_id(include_root=True))
    print(gtrees[95].get_topology_id(include_root=True), topo_spans[list(topo_idxs).index(95)])
    print(gtrees[97].get_topology_id(include_root=True), topo_spans[list(topo_idxs).index(97)])
    print(gtrees[118].get_topology_id(include_root=True))
    print("-------")
    raise SystemExit(0)
    # print(get_model_dft(MODEL.df).head(20))
    # print('...')
    # print(get_model_dft(MODEL.df).tail(20))    
    # raise SystemExit(0)
    # i0 = iter_topos_and_spans_from_model(MODEL, True)
    # i1 = iter_topos_from_trees(toytree.mtree(MODEL.df.genealogy))

    # i1 = iter_spans_and_trees_from_model(MODEL)
    # i2 = iter_spans_and_topos_from_model(MODEL)

    # for i, j in zip(i1, i2):
    #     print(i[0], i[1])
    #     print(j[0], j[1])
    tree_spans, topo_spans, topo_idxs, gtrees = get_waiting_distance_data_from_model(MODEL)
    print(len(gtrees), [i.get_topology_id(include_root=True) for i in gtrees[:]], "trees")
    print(len(tree_spans), tree_spans, tree_spans.dtype, "tree spans")
    print(len(topo_spans), topo_spans, topo_spans.dtype, "topo_spans")
    print(len(topo_idxs), topo_idxs, "topo_idxs")
    I = 95
    print(gtrees[I].get_topology_id(include_root=True), topo_spans[list(topo_idxs).index(I)])
    raise SystemExit(0)


    print(len(gtrees), gtrees[:], "trees")
    print(len(tree_spans), tree_spans[:N], tree_spans.dtype, "tree spans")
    print(len(topo_spans), topo_spans[:N], '...', topo_spans[-N:], topo_spans.dtype, "topo_spans")
    # print(len(tree_idxs), tree_idxs[:N], '...', tree_idxs[-N:], "tree_idxs")
    print(len(topo_idxs), topo_idxs[:N], '...', topo_idxs[-N:], "topo_idxs")
    topo_sums = topo_idxs[1:] - topo_idxs[:-1]    
    print(len(topo_sums), topo_sums[:N], '...', topo_sums[-N:], "topo_ntrees")
    print(tree_spans[840:])


    # print(new_get_waiting_distance_data_from_model(MODEL))
    # c0, _, _ = gtrees[0].draw()
    # c1, _, _ = gtrees[1].draw()
    # c2, _, _ = gtrees[2].draw()

    # toytree.utils.show([c0, c1, c2])

    # for x in iter_topos_and_spans_from_model(MODEL, False):
    #     print(x)

    # for SPAN, GTREE in iter_spans_and_topologies_from_model(MODEL):
    #     print(SPAN)
    #     GTREE.treenode.draw_ascii()

    # ivals = get_topology_interval_lengths(MODEL)
    # print(ivals, ivals.sum())

    # GENEALOGIES = toytree.mtree(MODEL.df.genealogy)
    # for g in iter_unique_topologies_from_trees(GENEALOGIES):
    #     print(g.get_topology_id(include_root=True), g.get_node_data("height"))
