#!/usr/bin/env python

"""Utility functions for SMC subpackage.


>>> get_trees_and_spans_from_model(...)
>>> get_topos_and_spans_from_model(...)


"""

from typing import Iterator, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor
from loguru import logger
import numpy as np
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


def get_waiting_distance_data_from_model(model: ipcoal.Model):
    """Return tree and topo change data.
    """
    trees = []
    tree_spans = []
    topo_idxs = []
    topo_spans = []
    intervals = []
    gidx = 0
    tidx = 0

    # treat each locus at a time to skip last tree
    # in a locus that is cut-off at its ending.
    for _, ldf in model.df.groupby("locus"):
        iterator = ldf.iterrows()
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


# NOT YET IPMLEMENTED: NEED TO CONCAT
def new_get_waiting_distance_data_from_model(model: ipcoal.Model, nproc: int = None):
    """Return tree-dists, topo-dists, topo-change-indices, trees
    """
    rasyncs = {}
    with ProcessPoolExecutor(max_workers=nproc) as pool:
        # treat each locus at a time to skip last tree
        # in a locus that is cut-off at its ending.
        for lidx, ldf in model.df.groupby("locus"):
            rasyncs[lidx] = pool.submit(subfunc, ldf)
    gspans, tspans, tidxs, trees = zip(*[rasyncs[i].result() for i in rasyncs])
    gspans = np.concatenate(gspans, axis=0)
    tspans = np.concatenate(tspans, axis=0)
    return gspans, tspans, tidxs, trees


def subfunc(data):
    """Remote function called by get_waiting...
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


# def iter_topos_and_spans_from_model(
#     model: ipcoal.Model,
#     weighted_node_heights: bool = False,
# ) -> Iterator[ToyTree]:
#     """...

#     weighted_node_heights: bool
#         Not currently working, negative branch lengths can occur.
#     """
#     cnewick = None
#     ctree = None
#     ctopo = None
#     forest = {}

#     # iterate over trees
#     idata = zip(model.df.genealogy, model.df.nbps)
#     for tidx, (newick, length) in enumerate(idata):

#         # load first tree
#         if cnewick is None:
#             cnewick = newick
#             ctree = toytree.tree(newick)
#             ctopo = ctree.get_topology_id(include_root=True)
#             forest[ctree] = length
#             # logger.debug(f"first tree {length}")
#             continue

#         # newick str did not change (invisible recomb) just add length
#         # to the last stored tree.
#         if newick == cnewick:
#             forest[ctree] += length
#             # logger.debug(f"no-change {length}")
#             continue

#         # newick str changed, check if topology changed. If not, store
#         # the last tree and its length to forest, and update current.
#         else:
#             tree = toytree.tree(newick)
#             topo = tree.get_topology_id(include_root=True)
#             if topo != ctopo:
#                 # logger.debug(f"*topo-change -> {sum(forest[i] for i in forest)}")
#                 if not weighted_node_heights:
#                     yield ctree, sum(forest[i] for i in forest)
#                 else:
#                     heights = {}
#                     w = [forest[i] for i in forest]

#                     # cannot use idx labels here, must use anc names...
#                     for nidx in range(ctree.ntips, ctree.nnodes):
#                         hnodes = [t.get_mrca_node(*ctree[nidx].get_leaf_names()) for t in forest]
#                         h = [i.height for i in hnodes]
#                         height = np.average(h, weights=w)
#                         # do not allow averaging to cause negative height
#                         minh = max([heights.get(i._idx, 0) for i in ctree[nidx].children])
#                         if height <= minh:
#                             height = minh + 1
#                         heights[nidx] = height
#                     ctree = ctree.mod.edges_set_node_heights(heights)
#                     yield ctree, sum(forest[i] for i in forest)

#                 cnewick = newick
#                 ctree = tree
#                 ctopo = topo
#                 forest = {ctree: length}
#                 # logger.debug(f"first tree {length}")

#             else:
#                 # logger.debug(f"tree-change {length}")
#                 forest[tree] = length
#                 ctree = tree


# def iter_topos_from_trees(trees: Sequence[ToyTree]) -> Iterator[ToyTree]:
#     """Returns the genealogy at each topology change."""

#     # initial tree
#     current = trees[0]
#     cidx = current.get_topology_id(include_root=True)
#     tree_bunch = [current]

#     # iterate over genealogies
#     for gtree in trees[1:]:
#         nidx = gtree.get_topology_id(include_root=True)
#         if cidx != nidx:
#             yield current
#             current = gtree
#             cidx = nidx
#             tree_bunch = [gtree]
#         else:
#             tree_bunch.append(gtree)


# def get_topology_interval_lengths(
#     model: ipcoal.Model,
#     locus: Optional[int] = None,
# ) -> np.ndarray:
#     """Return an array of genealogical topology lengths.

#     Parameters
#     ----------
#     model: ipcoal.Model
#         A Model object that has simulated data in its `.df` attribute.
#     """
#     return np.array([i[1] for i in iter_topos_and_spans_from_model(model)])


if __name__ == "__main__":

    ipcoal.set_log_level("DEBUG")
    MODEL = ipcoal.Model(Ne=100_000, nsamples=5, seed_trees=333)
    MODEL.sim_trees(2, 50_000)
    print(MODEL.df.head())

    # i0 = iter_topos_and_spans_from_model(MODEL, True)
    # i1 = iter_topos_from_trees(toytree.mtree(MODEL.df.genealogy))

    # i1 = iter_spans_and_trees_from_model(MODEL)
    # i2 = iter_spans_and_topos_from_model(MODEL)

    # for i, j in zip(i1, i2):
    #     print(i[0], i[1])
    #     print(j[0], j[1])

    print(get_waiting_distance_data_from_model(MODEL))
    print(new_get_waiting_distance_data_from_model(MODEL))
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
