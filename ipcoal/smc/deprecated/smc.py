#!/usr/bin/env python

"""Distribution of waiting times in a species tree model.

Measure genome distance until genealogical changes under the 
Sequential Markovian Coalescent (SMC) given a parameterized species
tree model.
"""

from typing import Tuple
import toytree
import numpy as np
import pandas as pd
from loguru import logger

logger = logger.bind(name="ipcoal")


# def get_num_edges_at_time(tree: toytree.ToyTree, time: int) -> int:
#     """Return the number of edges in the tree at a specific time.

#     Note: If a node is at time=10 the edges subtending this node are
#     considered to exist at time < 10, but at 10 exactly.
#     """
#     return 1 + sum(tree.get_node_data("height") > time)

# def get_tree_clade_times(tree: toytree.ToyTree):
#     """Return a DataFrame with clade root node heights.

#     This is applied to gene trees to get the
#     """
#     nodes = []
#     heights = []
#     for node in tree.treenode.traverse():
#         if not node.is_leaf():
#             nodes.append(node.get_leaf_names())
#             heights.append(node.height)

#     # organize into a dataframe
#     data = pd.DataFrame({
#         "clades": nodes,
#         "heights": heights,
#     })
#     return data

def get_gene_tree_coal_intervals(
    species_tree: toytree.ToyTree,
    gene_tree: toytree.ToyTree,
    idx: int,
    ) -> pd.DataFrame:
    """Return a DataFrame with intervals of coal and non-coal events.

    Each interval (row) in the returned dataframe represents an edge
    given a species tree and gene tree over which two nodes on the
    gene tree either coalesced or didn't (ILS).

    Parameters
    ----------
    species_tree: toytree.ToyTree
        Species tree with a "Ne" feature assigned to every node, and
        edge lengths in units of generations. The tree can be non-
        ultrametric, representing differences in generation times.
    gene_tree: toytree.ToyTree
        Gene tree simulated from the species tree. Edge lengths are in
        units of generations.
    idx: int
        The node index below a focal edge in the gene_tree.
    """
    # get the gt node at bottom of the focal gt edge.
    gt_node = gene_tree.idx_dict[idx]                      # 0
    gt_top_node = gt_node.up                               # 6
    gt_tips = gt_node.get_leaf_names()                     # r4
    gt_clade_tips = gt_top_node.get_leaf_names()           # r3,r4

    # get the st node containing all gt_clade_tips
    st_idx = species_tree.get_mrca_idx_from_tip_labels(gt_clade_tips)  # s7
    st_top_node = species_tree.idx_dict[st_idx]                        # s7
    # st_node = [i for i in st_top_node.get_descendants()
    logger.info((gt_clade_tips, st_node.idx))

    # record each event of gene tree coal, or gene tree non-coal.
    end_time = gt_node.up.height
    events = []
    edges = 1
    while 1:
        logger.info(f"gt_node={gt_node.idx}; st_node={st_node.idx}")

        # record the start of the next event
        event = {
            "start_node": events[-1]['stop_node'] if events else ("g", gt_node),
            "start": events[-1]['stop_node'][1].height if events else gt_node.height,
            "edges": edges,
            "gt_node": gt_node.idx,
            "st_node": st_node.idx,
            "neff": st_node.Ne,
        }
        
        # record the edge up until a coalescent event occurred.
        # this is a record for an edge spanning a gt-gt or st-gt node.
        # if we are above the st root then gt is the only option.
        if (not st_node.up) or (gt_node.up.height < st_node.up.height):
            logger.info("1")
            event["stop_node"] = ("g", gt_node.up)
            event["stop"] = gt_node.up.height
            events.append(event)
            gt_node = gt_node.up
            if gt_node.height == end_time:
                break

        # record the edge up until a NON-coalescent (ILS) event occured.
        # this is a record for an edge spanning a gt-st or st-st nodes.
        else:
            logger.info("2")            
            event["stop_node"] = ("s", st_node.up)
            event["stop"] = st_node.up.height
            events.append(event)
            # add num edges incoming from the other sptree edge.
            logger.info((edges, len(st_node.up), len(st_node)))
            edges += (len(st_node.up) - len(st_node))
            st_node = st_node.up

    # build index labels for events
    index = []
    for event in events:
        snode = event['start_node']
        enode = event['stop_node']
        index.append(f"{snode[0]}{snode[1].idx}-{enode[0]}{enode[1].idx}")

    # build dataframe of events
    data = pd.DataFrame(
        index=index,
        columns=[
            "start", "stop", "dist", "edges",
            "neff", "st_node", "gt_node",
        ],
    )
    data.start = [i['start'] for i in events]
    data.stop = [i['stop'] for i in events]
    data.dist = data.stop - data.start
    data.edges = [i['edges'] for i in events]
    data.neff = [i['neff'] for i in events]
    data.st_node = [i['st_node'] for i in events]
    data.gt_node = [i['gt_node'] for i in events]
    # data.subtree = [
    #     gene_tree.prune(gene_tree.get_tip_labels(i['gt_node'])).write()
    #     for i in events
    # ]
    logger.info(f"\n{data}")
    return data


def get_prob_gene_tree_is_unchanged_by_recomb_event(
    species_tree: toytree.ToyTree,
    gene_tree: toytree.ToyTree,
    recombination_event: Tuple[int, float],
    ) -> float:
    """Return the prob a gene tree is unchanged given a recomb event.

    The returned value is a float representing the probability that 
    both the topology and branch lengths of the gene tree will remain
    the same given a recombination event occurred on a specific edge
    of the gene tree at a specific time, and within the context of a 
    parameterized species tree model.

    Latex of equation here.

    Parameters
    ----------
    species_tree: toytree.ToyTree
        Species tree with a "Ne" feature assigned to every node, and
        edge lengths in units of generations. The tree can be non-
        ultrametric, representing differences in generation times.
    gene_tree: toytree.ToyTree
        Gene tree simulated from the species tree. Edge lengths are in
        units of generations.
    recombination_event: Tuple[int, float]
        The first value indicates the int idx label of a node on the 
        gene tree for which a recombination event will occur on the
        edge above this node, at the specified time. If the entered time
        does not exist on this edge an error will be raised.
    """
    # get the coalescent intervals for the selected gene tree edge 
    interval_data = get_gene_tree_coal_intervals(
        species_tree, gene_tree, recombination_event[0]
    )

    # select only intervals with data for time >= the recombination event.
    # this must be >= 1 events else an error is raised below.
    idata = interval_data[interval_data.stop > recombination_event[1]]

    # raise an error if timed event cannot happen on the selected edge
    if not idata.size:
        node = gene_tree.idx_dict[recombination_event[0]]
        raise ValueError(
            f"The recombination time ({recombination_event[1]}) does not "
            "exist on the selected gene tree edge "
            f"({node.height:.1f} - {node.up.height:.1f})"
        )

    # First term is calculated only on the focal interval, i.e., the
    # one in which the recombination event occurs.
    foc = idata.iloc[0]

    # get first term of the equation
    term_a = np.exp(-1 * (foc.edges / foc.neff) * foc.stop)
    term_b = np.exp((foc.edges / foc.neff) * recombination_event[1])
    first_term = (1 / foc.edges) - (1 / foc.edges) * term_a * term_b

    # Second term loops through all remaining intervals in a nested way
    # such that if there werre 4 rows it would do (0, None), (1, 0),
    # (2, (0, 1)), (3, (0, 1, 2)), etc., as the (outer, inner) loops.
    sec = interval_data.iloc[1:]
    if not sec.size:
        return first_term

    # the outer loop
    second_term = 0
    for odx, _ in enumerate(sec.index):
        odat = sec.iloc[odx]

        # the inner loop
        inner_sum = 0
        for idx in range(odx):
            tmp = sec.iloc[idx]
            inner_sum += (tmp.edges / tmp.neff) * tmp.dist

        term_a = np.exp(-1 * (foc.edges / foc.neff) * foc.stop - inner_sum)
        term_b = np.exp((foc.edges / foc.neff) * recombination_event[1])
        term_c = (1 / odat.edges)
        term_d = (1 - np.exp(-1 * (odat.edges / odat.neff) * odat.dist))
        second_term += term_a * term_b * term_c * term_d
    return first_term + second_term


def get_prob_gene_tree_is_unchanged_by_recomb_on_edge(
    species_tree,
    gene_tree,
    idx: int
    ):
    """Return the prob a gene tree is unchanged given recomb on an edge.

    """
    # get all coalescent intervals spanning edge idx
    interval_data = get_gene_tree_coal_intervals(species_tree, gene_tree, idx)

    # all neff values should be X2 in this function
    interval_data.neff *= 2

    # get edge start and end positions
    interval = (interval_data.start.min(), interval_data.stop.max())

    # iterate over coal intervals (0, 1, 2, ...)
    full_branch_sum = 0
    first_expr = 0
    for odx, _ in enumerate(interval_data.index):
        
        # select a row interval
        odat = interval_data.iloc[odx]
        
        # get first term for interval
        first_term = (1 / odat.edges) * odat.dist
        second_expr = 0

        # terms used repeatedly below
        oterm_a = (odat.edges / odat.neff) * odat.stop
        oterm_b = (odat.edges / odat.neff) * odat.start

        # outer loop iterates over nested intervals:
        # (0, None), (1, 0), (2, (0, 1)), ...
        for ndx in range(odx + 1, interval_data.shape[0]):

            # select a row interval for inner
            logger.info(f"outer={odx}, inner={ndx}")
            idat = interval_data.iloc[ndx]

            # inner-inner loop iterates over nested nested intervals:
            # (0, None, None), (1, 0, None), (2, (0, 1), 0)
            inner_sum = 0
            for qdx in range(odx + 1, ndx):
                tmp = interval_data.iloc[qdx]
                logger.info(f"outer={odx}, inner={ndx}, inner-inner={qdx}")
                inner_sum += (tmp.edges / tmp.neff) * tmp.dist

            # outer values
            term_b = np.exp(-1 * oterm_a - inner_sum)

            # inner values
            term_c = (1 / idat.edges) * (1 - np.exp(-1 * (idat.edges / (idat.neff)) * idat.dist))

            # get the product
            second_expr += (term_b * term_c) * (odat.neff / odat.edges)

        # preventing overflow...??
        if (oterm_a < 709) and (oterm_b < 709):
            second_expr += -np.exp(-1 * oterm_a) * (odat.neff / (odat.edges ** 2))
            first_expr = np.exp(oterm_a) - np.exp(oterm_b)

        # if there is no internal sum then blah blah
        else:
            second_expr += 1
            first_expr = (1 - np.exp(oterm_b - oterm_a) * (odat.neff / odat.edges ** 2))
        full_branch_sum += first_term + first_expr * second_expr
    return full_branch_sum * (1 / (interval[1] - interval[0]))


def get_probability_gene_tree_is_unchanged(
    species_tree: toytree.ToyTree, 
    gene_tree: toytree.ToyTree,
    ) -> float:
    """Return the prob a gene tree is unchanged by recombination.
    """
    # get sum of all edge length (excluding the root stem) on gtree.
    sum_edge_lengths = sum(gene_tree.get_node_data("dist").iloc[:-1])

    # for each edge add the probability that recomb would change it.
    prob_tree_unchanged = 0
    for node in gene_tree.treenode.traverse():
        if not node.is_root():
            prob_unchanged = get_prob_gene_tree_is_unchanged_by_recomb_on_edge(
                species_tree, gene_tree, node.idx,
            )
            prob_tree_unchanged += (node.dist / sum_edge_lengths) * prob_unchanged
            logger.warning((node.idx, prob_unchanged))
    return prob_tree_unchanged




if __name__ == "__main__":

    import ipcoal
    ipcoal.set_log_level("INFO")

    SPTREE = toytree.rtree.unittree(ntips=6, treeheight=1e6, seed=123)
    SPTREE = SPTREE.set_node_data("Ne", {i: 5e4 for i in (0, 1, 8)}, default=1e5)
    MODEL = ipcoal.Model(SPTREE, seed_trees=123)
    MODEL.sim_trees(1, 1)
    GTREE = toytree.tree(MODEL.df.genealogy[0])
    # print(get_tree_total_length(SPTREE))
    # print(SPTREE.get_node_data("dist"))
    # print(get_tree_clade_times(SPTREE))
    # print(get_gene_tree_coal_intervals(SPTREE, GTREE, 7).T)
    # print(get_prob_gene_tree_is_unchanged_by_recomb_event(SPTREE, GTREE, (7, 700_000)))    
    # print(get_prob_gene_tree_is_unchanged_by_recomb_event(SPTREE, GTREE, (2, 400_000)))
    print(get_prob_gene_tree_is_unchanged_by_recomb_on_edge(SPTREE, GTREE, 0))
    # print(get_probability_gene_tree_is_unchanged(SPTREE, GTREE))
