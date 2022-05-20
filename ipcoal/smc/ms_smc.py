#!/usr/bin/env python

"""Distribution of waiting distances under the SM-MSC'.

Measure genome distance until genealogical changes under the
Sequential Markovian Coalescent (SMC) given a parameterized species
tree model.

"""

from typing import Dict, TypeVar, Sequence
from loguru import logger
import numpy as np
import pandas as pd
import toytree
import toyplot
from scipy import stats
import ipcoal

# pylint: disable="cell-var-from-loop"

logger = logger.bind(name="ipcoal")
ToyTree = TypeVar("ToyTree")


def get_genealogy_embedding_table(
    species_tree: ToyTree,
    genealogy: ToyTree,
    imap: Dict[str, Sequence[str]],
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
        edge lengths in units of generations.
    genealogy: toytree.ToyTree
        Genealogy to be embedded in species tree w/ edge lengths in
        units of generations.
    imap: Dict
        A dict mapping species tree tip names to gene tree tip names.
    """
    # store temporary results in a dict
    split_data = []

    # dict to update tips to their ancestor if already coalesced.
    name_to_node = {i.name: i for i in genealogy.traverse() if i.is_leaf()}

    # get table of gene tree node heights
    gt_node_heights = genealogy.get_node_data("height")

    # iterate over stree from tips to root storing stree node data
    for st_node in species_tree.traverse("postorder"):
        # get all gtree names descended from this sptree interval
        gt_tips = set.union(*[set(imap[i]) for i in st_node.get_leaf_names()])

        # get gtree nodes descended from this interval.
        gt_nodes = set(name_to_node[i] for i in gt_tips)

        # get internal nodes in this TIME interval (coalescences)
        mask_below = gt_node_heights > st_node.height + 0.0001 # zero-align ipcoal bug
        mask_above = True if st_node.is_root() else gt_node_heights < st_node.up.height
        nodes_in_time_slice = gt_node_heights[mask_below & mask_above]

        # filter internal nodes must be ancestors of gt_nodes (in this interval)
        # get internal nodes in this interval by getting all nodes in this
        # time interval and requiring that all their desc are in gt_nodes.
        inodes = []
        for gidx in nodes_in_time_slice.sort_values().index:
            gt_node = genealogy[gidx]
            if not set(gt_node.get_leaf_names()).difference(gt_tips):
                inodes.append(gt_node)

        # vars to be updated at coal events
        start = st_node.height
        edges = {i.idx for i in gt_nodes}

        # iterate over internal nodes
        for gt_node in inodes:

            # add interval from start to first coal, or coal to next coal
            split_data.append([
                start,
                gt_node.height,
                st_node.idx,
                st_node.Ne,
                len(edges),
                gt_node.idx,
                sorted(edges),
            ])

            # update counters and indexers
            start = gt_node.height
            edges.add(gt_node.idx)
            for child in gt_node.children:
                edges.remove(child.idx)
            for tip in gt_node.get_leaves():
                name_to_node[tip.name] = gt_node

        # add non-coal interval
        split_data.append([
            start,
            st_node.up.height if st_node.up else pd.NA,
            st_node.idx,
            st_node.Ne,
            len(edges),
            pd.NA,
            sorted(edges),
        ])

    table = pd.DataFrame(
        data=split_data,
        columns=['start', 'stop', 'st_node', 'neff', 'nedges', 'coal', 'edges'],
    )
    table['dist'] = table.stop - table.start
    # table['rate'] = table.nedges / table.neff
    return table

def get_genealogy_embedding_edge_path(table: pd.DataFrame, branch: int) -> pd.DataFrame:
    """Return the gene tree embedding table intervals a gtree edge
    passes through.

    Parameters
    ----------
    table:
        A table returned by the `get_genealogy_embedding_table` func.
    branch:
        An integer index (idx) label to select a genealogy branch.
    """
    return table[table.edges.apply(lambda x: branch in x)]

def _get_pij(itab: pd.DataFrame, idx: int, jdx: int) -> float:
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
    # why would this be here, you cannot i==i in ij ???
    if idx == jdx:
        term1 = -(1 / itab.nedges[idx])
        term2 = np.exp(-(itab.nedges[idx] / itab.neff[idx]) * itab.stop[idx])
        return term1 * term2

    # ignore jdx < idx (speed hack so we don't need to trim tables below t_r)
    if jdx < idx:
        return 0

    # involves connections to jdx interval
    term1 = 1 / itab.nedges[jdx]
    term2 = (1 - np.exp(-(itab.nedges[jdx] / itab.neff[jdx]) * itab.dist[jdx]))

    # involves connections to idx interval
    term3_inner_a = -(itab.nedges[idx] / (itab.neff[idx])) * itab.stop[idx]

    # involves connections to edges BETWEEN idx and jdx (not including idx or jdx)
    term3_inner_b = sum(
        ((itab.nedges[qdx] / itab.neff[qdx]) * itab.dist[qdx])
        for qdx in itab.loc[idx + 1: jdx - 1].index
    )
    term3 = np.exp(term3_inner_a - term3_inner_b)
    return term1 * term2 * term3

def _get_sum_pb1(btab: pd.DataFrame, ptab: pd.DataFrame, mtab: pd.DataFrame) -> float:
    """Return value for the $p_{b,1}$ variable.

    $p_{b,1}$ includes components for the reconnection of branch b
    to (1) itself; (2) its parent; and (3) its sibling in intervals
    where they overlap. The first section corresponds to the `ntab`
    table intervals, the second section to the `mtab` table intervals,
    and the third section to the `ptab` table intervals. The `gtab`
    table contains all intervals on b.

    Parameters
    ----------
    gtab:
        All intervals on branch b.
    ptab:
        All intervals on branch c.
    mtab:
        All intervals shared by branches b and b' above t_m.
    """
    # value to return, summed over all intervals from 0 to m
    pbval = 0

    # get intervals spanning branch b and c
    ftab = pd.concat([btab, ptab.loc[ptab.index.difference(btab.index)]])
    ftab.sort_index(inplace=True) # maybe not necessary

    # iterate over all intervals from 0 to m (indices of utab)
    utab = btab[btab.index < mtab.index.min()]
    for idx in utab.index:

        # first term applies only to the interval in which recomb occurred
        estop = (btab.nedges[idx] / btab.neff[idx]) * btab.stop[idx]
        estart = (btab.nedges[idx] / btab.neff[idx]) * btab.start[idx]
        if estop > 100:
            first_term = 1e15
        else:
            first_term = btab.neff[idx] * (np.exp(estop) - np.exp(estart))

        # p_ij across bc
        # logger.warning(f"{[(ftab, idx, jidx) for jidx in ftab.index]}")
        sum1 = sum(_get_pij(ftab, idx, jidx) for jidx in ftab.index)
        # p_ij from m to end of b
        # logger.warning(f"{[(btab, idx, sidx) for sidx in mtab.index]}")
        sum2 = sum(_get_pij(btab, idx, sidx) for sidx in mtab.index)
        second_term = sum1 + sum2

        # and multiply them together.
        pbval += (1 / btab.nedges[idx]) * (btab.dist[idx] + (first_term * second_term))
    return pbval

def _get_sum_pb2(btab: pd.DataFrame, ptab: pd.DataFrame, mtab: pd.DataFrame) -> float:
    """Return value for the $p_{b,2}$ variable.

    $p_{b,2}$ includes components for the reconnection of branch b
    to (1) its parent; and (2) its sibling in intervals where they
    overlap.

    Parameters
    ----------
    gtab:
        All intervals on branch b.
    ptab:
        All intervals on branch c.
    mtab:
        All intervals shared by branches b and b' above t_m.
    """
    # get intervals spanning branch b and c
    ftab = pd.concat([btab, ptab.loc[ptab.index.difference(btab.index)]])
    ftab.sort_index(inplace=True) # maybe not necessary

    # value to return, summed over all intervals from m to b
    pbval = 0
    for idx in mtab.index:

        # first term applies to only to gtab (branch on which recomb occurs)
        estop = btab.nedges[idx] * btab.stop[idx] / btab.neff[idx]
        estart = btab.nedges[idx] * btab.start[idx] / btab.neff[idx]
        if estop > 100:
            first_term = 1e15
        else:
            first_term =  btab.neff[idx] * (np.exp(estop) - np.exp(estart))

        # p_ij across intervals on b
        sum1 = sum(_get_pij(btab, idx, bidx) for bidx in btab.index)
        # get pij across intervals on c
        sum2 = sum(_get_pij(ftab, idx, pidx) for pidx in ptab.index)
        second_term = (2 * sum1) + sum2

        # ...and now multiply them together.
        pbval += (1 / btab.nedges[idx]) * ((2 * btab.dist[idx]) + (first_term * second_term))
    return pbval

def get_probability_tree_unchanged_given_b_and_tr(table: pd.DataFrame, branch: int, time: float) -> float:
    """Return prob tree-change does not occur given recomb on branch b at time t.

    Parameters
    ----------
    table:
        A table returned by the `get_genealogy_embedding_table` func.
    branch:
        An integer index (idx) label to select a genealogy branch.
    time:
        A time at which recombination occurs.

    Example
    -------
    >>> # plot the distribution of probabilities across a branch.
    >>> ...
    """
    # get all intervals on branch b
    btab = table[table.edges.apply(lambda x: branch in x)]

    # get interval containing time tr
    idx = btab[(time >= btab.start) & (time <= btab.stop)].index[0]

    # get two terms and return the sum
    inner = (btab.nedges[idx] / btab.neff[idx]) * time
    inner = np.exp(inner) if inner < 100 else 1e15
    term1 = (1 / btab.nedges[idx]) + _get_pij(btab, idx, idx) * inner

    # iterate over all intervals from idx to end of b and get pij
    term2 = 0
    for jdx in btab.loc[idx + 1:].index:
        term2 += _get_pij(btab, idx, jdx) * inner
    return term1 + term2

def get_probability_tree_unchanged_given_b(table: pd.DataFrame, branch: int) -> float:
    """Return prob tree-change does not occur given recomb on branch b.

    Parameters
    ----------
    table:
        A table returned by the `get_genealogy_embedding_table` func.
    branch:
        An integer index (idx) label to select a genealogy branch.
    """
    # get all intervals on branch b
    btab = table[table.edges.apply(lambda x: branch in x)]
    tbl = btab.start.min()
    tbu = btab.stop.max()

    # sum over the intervals on b where recomb could occur
    prob = 0
    for idx in btab.index:
        term1 = (1 / btab.nedges[idx]) * btab.dist[idx]
        term2_outer = (btab.neff[idx] / btab.nedges[idx])

        # Avoid overflow when inner value here is too large. This happens
        # when neff is low. hack solution for now is to use float128,
        # but maybe use expit function in the future.
        estop = (btab.nedges[idx] / btab.neff[idx]) * btab.stop[idx]
        estart = (btab.nedges[idx] / btab.neff[idx]) * btab.start[idx]
        if estop > 100:
            term2_inner = 1e15
        else:
            term2_inner = np.exp(estop) - np.exp(estart)

        # pij component
        term3 = sum(_get_pij(btab, idx, jdx) for jdx in btab.loc[idx:].index)
        prob += term1 + (term2_inner * term2_outer * term3)
    return (1 / (tbu - tbl)) * prob

def get_probability_topology_unchanged_given_b_and_tr(
    table: pd.DataFrame,
    branch: int,
    sibling: int,
    parent: int,
    time: float,
    ) -> float:
    """Return prob topology-change does not occur given recomb on branch b at time t.

    Parameters
    ----------
    table:
        A table returned by the `get_genealogy_embedding_table` func.
    branch:
        An integer index (idx) label to select a genealogy branch.
    sibling:
        An integer index (idx) label of the sibling to 'branch'.
    parent:
        An integer index (idx) label of the parent to 'branch'.
    time:
        A time at which recombination occurs.

    Example
    -------
    >>> # plot the distribution of probabilities across a branch.
    >>> ...
    """
    # get all intervals on branches b, c, and b', respectively
    btab = table[table.edges.apply(lambda x: branch in x)]
    stab = table[table.edges.apply(lambda x: sibling in x)]
    ptab = table[table.edges.apply(lambda x: parent in x)]

    # get interval containing time tr
    idx = btab[(time >= btab.start) & (time < btab.stop)].index[0]

    # get intervals containing both b and b' (at and above t_r)
    mtab = btab.loc[btab.index.intersection(stab.index)]

    # get intervals spanning branch b and c
    ftab = pd.concat([btab, ptab.loc[ptab.index.difference(btab.index)]])
    ftab.sort_index(inplace=True) # maybe not necessary

    # rate
    inner = (btab.nedges[idx] / btab.neff[idx]) * time
    inner = np.exp(inner) if inner < 100 else 1e15

    # probability of recoalescing with self in interval idx
    term1 = (1 / btab.nedges[idx])

    logger.info(f"{time:.1f}, {idx}")
    # if t_r < t_m then use three interval equation
    if time < mtab.start.min():

        # pij over j in all intervals between idx and top of parent
        term2 = sum(_get_pij(ftab, idx, jidx) for jidx in ftab.index)
        term2 *= inner

        # pij over j in all intervals from m to end of b
        term3 = sum(_get_pij(btab, idx, jidx) for jidx in mtab.index)
        term3 *= inner
        return term1 + term2 + term3

    # pij over j all intervals on b
    term2 = sum(_get_pij(btab, idx, jidx) for jidx in btab.index)
    term2 *= inner

    # pij over j all intervals on c
    term3 = sum(_get_pij(ftab, idx, jidx) for jidx in ptab.index)
    term3 *= inner
    return 2 * (term1 + term2) + term3

def get_probability_topology_unchanged_given_b(
    table: pd.DataFrame,
    branch: int,
    sibling: int,
    parent: int,
    ) -> float:
    """Return prob topology-change does not occur given recomb on branch b at time t.

    Parameters
    ----------
    table:
        A table returned by the `get_genealogy_embedding_table` func.
    branch:
        An integer index (idx) label to select a genealogy branch.
    sibling:
        An integer index (idx) label of the sibling to 'branch'.
    parent:
        An integer index (idx) label of the parent to 'branch'.

    Example
    -------
    >>> # plot the distribution of probabilities across a branch.
    >>> ...
    """
    # get all intervals on branches b, c, and b', respectively
    btab = table[table.edges.apply(lambda x: branch in x)]
    ptab = table[table.edges.apply(lambda x: parent in x)]
    stab = table[table.edges.apply(lambda x: sibling in x)]

    # get intervals containing both b and b' (above t_m)
    mtab = btab.loc[btab.index.intersection(stab.index)]

    # get lower and upper bounds of this gtree edge
    t_lb, t_ub = btab.start.min(), btab.stop.max()

    # get sum pb1 from intervals 0 to m
    # logger.info(f"{idx},{jdx}, {list(itab.index)}")
    pb1 = _get_sum_pb1(btab, ptab, mtab)
    logger.info(f"sum-pb1={pb1:.3f}")

    # get sum pb2 from m to end of b
    pb2 = _get_sum_pb2(btab, ptab, mtab)
    logger.info(f"sum-pb2={pb2:.3f}")
    return (1 / (t_ub - t_lb)) * (pb1 + pb2)

def get_probability_of_no_change(
    species_tree: ToyTree,
    genealogy: ToyTree,
    imap: Dict[str, Sequence[str]],
    ) -> float:
    """Return probability that recombination causes no-change.

    Returns the probability under the MS-SMC' that recombination
    occurring on this genealogy embedded in this parameterized species
    tree causes no change to the genealogy. This occurs when the
    detached branch re-coalesces with the same lineage it was descended
    from before its previous MRCA, and is an "invisible recombination"
    event.

    Note
    ----
    This probability is 1 - P(tree-change | S,G), where S is the
    species tree and G is the genealogy.

    Parameters
    ----------
    species_tree: ToyTree
        A species tree as a ToyTree object with 'Ne' data on each Node.
    genealogy: ToyTree
        A genealogy as a ToyTree object.
    imap: Dict
        A dictionary mapping species tree tip names to a list of
        genealogy tip names to map samples to species.
    """
    return 1 - get_probability_of_tree_change(species_tree, genealogy, imap)

def get_probability_of_tree_change(
    species_tree: ToyTree,
    genealogy: ToyTree,
    imap: Dict[str, Sequence[str]],
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
    species_tree: ToyTree
        A species tree as a ToyTree object with 'Ne' data on each Node.
    genealogy: ToyTree
        A genealogy as a ToyTree object.
    imap: Dict
        A dictionary mapping species tree tip names to a list of
        genealogy tip names to map samples to species.
    """
    # get full genealogy information
    etable = get_genealogy_embedding_table(species_tree, genealogy, imap)
    etable.neff = etable.neff * 2
    sumlen = sum(i.dist for i in genealogy if not i.is_root())

    # set NA values for root node edge arbitrarily high
    etable.loc[etable.index[-1], 'stop'] = 1e15
    etable.loc[etable.index[-1], 'dist'] = 1e15

    # traverse over all edges of the genealogy
    total_prob = 0
    for gnode in genealogy.traverse(strategy='postorder'):
        if not gnode.is_root():

            # get P(tree-unchanged | S, G, b)
            prob = get_probability_tree_unchanged_given_b(etable, gnode.idx)

            # contribute to total probability normalized by prop edge len
            total_prob += (gnode.dist / sumlen) * prob
    return 1 - total_prob

def get_probability_of_topology_change(
    species_tree: ToyTree,
    genealogy: ToyTree,
    imap: Dict[str, Sequence[str]],
    ) -> float:
    """Return probability that recombination causes a tree-change.

    Returns the probability under the MS-SMC' that recombination
    occurring on this genealogy embedded in this parameterized species
    tree causes a topology change. A topology-change is defined
    as a subset of the possible tree-change events, where the detached
    branch re-coalesces with a branch on the genealogy that results
    in a different topology (different relationships, not just
    coalescent times). This occurs if it re-coalesces with a branch
    other than itself, its sibling, or its parent.

    Note
    ----
    This probability is a subset of the P(tree-change | S,G) where
    S is the species tree and G is the genealogy.

    Parameters
    ----------
    species_tree: ToyTree
        A species tree as a ToyTree object with 'Ne' data on each Node.
    genealogy: ToyTree
        A genealogy as a ToyTree object.
    imap: Dict
        A dictionary mapping species tree tip names to a list of
        genealogy tip names to map samples to species.
    """
    # get full genealogy information
    etable = get_genealogy_embedding_table(species_tree, genealogy, imap)
    etable.neff = etable.neff * 2
    sumlen = sum(i.dist for i in genealogy if not i.is_root())

    # set NA values for root node edge arbitrarily high
    etable.loc[etable.index[-1], 'stop'] = 1e15
    etable.loc[etable.index[-1], 'dist'] = 1e15

    # traverse over all edges of the genealogy
    total_prob = 0
    for gnode in genealogy.traverse(strategy='postorder'):
        if not gnode.is_root():

            # get sibling (b') and parent (c) nodes
            pnode = gnode.up
            snode = gnode.get_sisters()[0]

            topo_unchanged_prob = get_probability_topology_unchanged_given_b(
                etable, gnode.idx, snode.idx, pnode.idx)

            # contribute to total probability of unchanged topology as
            # the proportion of total edge len represented by this branch.
            total_prob += (gnode.dist / sumlen) * topo_unchanged_prob
    return 1 - total_prob

def plot_edge_probabilities(
    species_tree: ToyTree,
    genealogy: ToyTree,
    imap: Dict[str, Sequence[str]],
    branch: int,
    **kwargs,
    ) -> toyplot.canvas.Canvas:
    """Return a toyplot canvas with probabilities along an edge.

    """
    # setup the canvas and axes
    canvas = toyplot.Canvas(
        height=kwargs.get("height", 750),
        width=kwargs.get("width", 300),
    )
    axstyle = dict(ymin=0, ymax=1, margin=65)
    ax0 = canvas.cartesian(grid=(3, 1, 0), label="Prob(no-change)", **axstyle)
    ax1 = canvas.cartesian(grid=(3, 1, 1), label="Prob(tree-change)", **axstyle)
    ax2 = canvas.cartesian(grid=(3, 1, 2), label="Prob(topo-change)", **axstyle)

    # Select a branch to plot and get its relations
    branch = genealogy[branch]
    bidx = branch.idx
    sidx = branch.get_sisters()[0].idx
    pidx = branch.up.idx

    # Get genealogy embedding table
    etable = get_genealogy_embedding_table(species_tree, genealogy, imap)
    btable = get_genealogy_embedding_edge_path(etable, bidx)

    # Plot probabilities of change types over a single branch
    # Note these are 'unchange' probs and so we plot 1 - Prob here.
    times = np.linspace(branch.height, branch.up.height, 200, endpoint=False)
    pt_nochange_tree = [
        get_probability_tree_unchanged_given_b_and_tr(
        etable, bidx, itime) for itime in times
    ]
    pt_nochange_topo = [
        get_probability_topology_unchanged_given_b_and_tr(
        etable, bidx, sidx, pidx, itime) for itime in times
    ]

    # add line and fill for probabilities
    ax0.plot(times, pt_nochange_tree, stroke_width=5)
    ax0.fill(times, pt_nochange_tree, opacity=0.33)
    ax1.plot(times, 1 - np.array(pt_nochange_tree), stroke_width=5)
    ax1.fill(times, 1 - np.array(pt_nochange_tree), opacity=0.33)
    ax2.plot(times, 1 - np.array(pt_nochange_topo), stroke_width=5)
    ax2.fill(times, 1 - np.array(pt_nochange_topo), opacity=0.33)

    # add vertical lines at interval breaks
    style = {"stroke": "black", "stroke-width": 2, "stroke-dasharray": "4,2"}
    intervals = [btable.start.iloc[0]] + list(btable.stop - 0.001)
    for itime in intervals:
        iprob_tree = get_probability_tree_unchanged_given_b_and_tr(etable, bidx, itime)
        iprob_topo = get_probability_topology_unchanged_given_b_and_tr(etable, bidx, sidx, pidx, itime)
        ax0.plot([itime, itime], [0, iprob_tree], style=style)
        ax1.plot([itime, itime], [0, 1 - iprob_tree], style=style)
        ax2.plot([itime, itime], [0, 1 - iprob_topo], style=style)

    # style the axes
    for axis in (ax0, ax1, ax2):
        axis.x.ticks.show = axis.y.ticks.show = True
        axis.y.domain.show = False
        axis.x.ticks.near = axis.y.ticks.near = 7.5
        axis.x.ticks.far = axis.y.ticks.far = 0
        axis.x.ticks.labels.offset = axis.y.ticks.labels.offset = 15
        axis.x.label.text = f"Time of recombination on branch {bidx}"
        axis.y.label.text = "Probability"
        axis.x.label.offset = axis.y.label.offset = 35
        axis.x.spine.style['stroke-width'] = axis.y.spine.style['stroke-width'] = 1.5
        axis.x.ticks.style['stroke-width'] = axis.y.ticks.style['stroke-width'] = 1.5
        axis.label.offset = 20
        axis.x.ticks.locator = toyplot.locator.Explicit([btable.start.iloc[0]] + list(btable.stop))
    return canvas

def get_expected_waiting_distance_to_recombination_event(
    genealogy: ToyTree,
    recombination_rate: float,
    ) -> float:
    """..."""
    return get_waiting_distance_to_recombination_event_rv(
        genealogy, recombination_rate).mean()

def get_expected_waiting_distance_to_tree_change(
    species_tree: ToyTree,
    genealogy: ToyTree,
    imap: Dict[str, Sequence[str]],
    recombination_rate: float,
    ) -> float:
    """..."""
    return get_waiting_distance_to_tree_change_rv(
        species_tree,
        genealogy, 
        imap,
        recombination_rate).mean()

def get_expected_waiting_distance_to_topology_change(
    species_tree: ToyTree,
    genealogy: ToyTree,
    imap: Dict[str, Sequence[str]],
    recombination_rate: float,
    ) -> float:
    """..."""    
    return get_waiting_distance_to_topology_change_rv(
        species_tree,
        genealogy, 
        imap,
        recombination_rate).mean()

def get_waiting_distance_to_recombination_event_rv(
    genealogy: ToyTree,
    recombination_rate: float,
    ) -> stats._distn_infrastructure.rv_frozen:
    r"""Return ...

    Waiting distances between events are modeled as an exponentially
    distributed random variable (rv). This probability distribution
    is represented in scipy by an `rv_continous` class object. This
    function returns a "frozen" rv_continous object that has its 
    rate parameter fixed, where the rate of recombination on the 
    input genealogy is a product of its sum of edge lengths (L(G)) 
    and the per-site per-generation recombination rate (r). 

    $$ \lambda_r = L(G) * r $$

    The returned frozen `rv_continous` variable can be used to 
    calculate likelihoods using its `.pdf` method; to sample 
    random waiting distances using its `.rvs` method; to get the
    mean expected waiting distance from `.mean`; among other things.
    See scipy docs.
    
    Parameters
    -----------
    ...

    Examples
    --------
    >>> ...
    """
    sumlen = sum(i.dist for i in genealogy if not i.is_root())
    lambda_ = sumlen * recombination_rate
    return stats.expon.freeze(scale=1/lambda_)

def get_waiting_distance_to_tree_change_rv(
    species_tree: ToyTree,
    genealogy: ToyTree,
    imap: Dict[str, Sequence[str]],
    recombination_rate: float,
    ) -> stats._distn_infrastructure.rv_frozen:
    """Return ...
    """
    sumlen = sum(i.dist for i in genealogy if not i.is_root())
    prob_tree = get_probability_of_tree_change(species_tree, genealogy, imap)
    lambda_ = sumlen * prob_tree * recombination_rate
    return stats.expon.freeze(scale=1/lambda_)

def get_waiting_distance_to_topology_change_rv(
    species_tree: ToyTree,
    genealogy: ToyTree,
    imap: Dict[str, Sequence[str]],
    recombination_rate: float,
    ) -> stats._distn_infrastructure.rv_frozen:
    """..."""
    sumlen = sum(i.dist for i in genealogy if not i.is_root())
    prob_topo = get_probability_of_topology_change(species_tree, genealogy, imap)
    lambda_ = sumlen * prob_topo * recombination_rate
    return stats.expon.freeze(scale=1/lambda_)



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
    BIDX = 2
    BRANCH = GTREE[BIDX]
    SIDX = BRANCH.get_sisters()[0].idx
    PIDX = BRANCH.up.idx

    # Get genealogy embedding table
    ETABLE = get_genealogy_embedding_table(SPTREE, GTREE, IMAP)
    print(f"Full genealogy embedding table\n{ETABLE}\n")

    # Get subset of embedding table for a single genealogy branch path
    BTABLE = get_genealogy_embedding_edge_path(ETABLE, BIDX)
    print(f"Path of branch {BIDX} in genealogy embedding table\n{BTABLE}\n")

    # Get probabilities integrated over the genealogy
    p_no = get_probability_of_no_change(SPTREE, GTREE, IMAP)
    p_tree = get_probability_of_tree_change(SPTREE, GTREE, IMAP)
    p_topo = get_probability_of_topology_change(SPTREE, GTREE, IMAP)
    print(f"Probability of no-change\n{p_no:.3f}\n")
    print(f"Probability of tree-change\n{p_tree:.3f}\n")
    print(f"Probability of topology-change\n{p_topo:.3f}\n")

    CANVAS = plot_edge_probabilities(SPTREE, GTREE, IMAP, 2)
    toytree.utils.show(CANVAS)

    raise SystemExit(0)

    # Rather than use the fixed SPTREE and GTREE above, we can generate
    # genealogies using coalescent simulation.

    # Setup a species tree model (tree with Ne values on Nodes)
    SPTREE = toytree.rtree.imbtree(ntips=4, treeheight=2e6, seed=123)
    SPTREE = SPTREE.set_node_data("Ne", default=5e5)

    # Simulate a starting genealogy with diff n samples per species
    NSAMPS = dict(zip(SPTREE.get_tip_labels(), [3, 2, 1, 1]))
    MODEL = ipcoal.Model(SPTREE, seed_trees=123, nsamples=NSAMPS)
    MODEL.sim_trees(1)
    IMAP = MODEL.get_imap_dict()
    GTREE = toytree.tree(MODEL.df.genealogy[0])

    # =================================================================
    SPTREE = toytree.tree("(((A,B),C),D);")
    SPTREE.set_node_data("height", inplace=True, default=0, mapping={
        4: 200_000, 5: 400_000, 6: 600_000,
    })
    SPTREE.set_node_data("Ne", inplace=True, default=100_000);

    GTREE = toytree.tree("(((0,1),(2,(3,4))),(5,6));")
    GTREE.set_node_data("height", inplace=True, default=0, mapping={
        7: 100_000, 8: 120_000, 9: 300_000, 10: 450_000, 11: 650_000, 12: 800_000,
    })
    IMAP = {
        "A": ['0', '1', '2'],
        "B": ['3', '4'],
        "C": ['5',],
        "D": ['6',],
    }
