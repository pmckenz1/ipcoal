#!/usr/bin/env python

"""Distribution of waiting times in a species tree model.

Measure genome distance until genealogical changes under the
Sequential Markovian Coalescent (SMC) given a parameterized species
tree model.
"""

from typing import Tuple, Optional, List
import toytree
import numpy as np
import pandas as pd
from loguru import logger

logger = logger.bind(name="ipcoal")


def get_num_edges_at_time(
    tree: toytree.ToyTree, 
    time: int,
    tips: Optional[List[str]]=None,
    ) -> int:
    """Return the number of edges in the tree at a specific time."""
    if tips:
        keep = [i for i in tree.get_tip_labels() if i not in tips]
        if keep:
            tree = tree.drop_tips(keep)
    return 1 + sum(tree.get_node_data("height") > time)


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
    # get all gt nodes descended from the top node of edge idx.
    gt_node = gene_tree.idx_dict[idx]         # 9
    gt_top_node = gt_node.up                  # 10
    gt_clade = gt_node.up.get_leaf_names()    # r0-r5
    gt_tips = gt_node.get_leaf_names()        # r0,r1
    gt_inodes = [gt_top_node]                 # [g10]

    # add internal gt nodes IF their coal w/ target edge is ILS
    for node in gt_top_node.get_descendants():                           # 9        8
        if node.height > gt_node.height:                                 # F        T
            tips = set(node.get_leaf_names() + gt_node.get_leaf_names()) # [r0,r1], r0-r5
            st_idx = species_tree.get_mrca_idx_from_tip_labels(tips)     # 8        10
            st_node = species_tree.idx_dict[st_idx]                      # 8        10
            if node.height > st_node.height:                             # T        F
                gt_inodes.append(node)
                logger.info(("!", node.idx, st_node.idx))                
    gt_inodes = [('g', i) for i in gt_inodes]                            # [g10]

    # get all st nodes on path from gt_node to gt_top_node
    st_idx = species_tree.get_mrca_idx_from_tip_labels(gt_clade)  # s10
    st_top_node = species_tree.idx_dict[st_idx]                   # s10
    st_clade = st_top_node.get_leaf_names()                       # r0-5
    st_idx = species_tree.get_mrca_idx_from_tip_labels(gt_tips)   # s8
    st_node = species_tree.idx_dict[st_idx]                       # s8
    st_inodes = []
    while 1:
        st_node = st_node.up
        if st_node and (st_node.height < gt_top_node.height):
            st_inodes.append(('s', st_node))
        else:
            break                                                 # [s10]

    # get all gt and st internal nodes sorted
    inodes = gt_inodes + st_inodes
    inodes = sorted(inodes, key=lambda x: x[1].height)

    # start w/ st_node reset to first event
    st_node = species_tree.idx_dict[st_idx]

    # record each event of gene tree coal, or gene tree non-coal.
    events = []
    coal_events = 0
    for node in inodes:
        logger.info(f'next={node[0]}{node[1].idx}, {node[1].height}')

        # record the start of the next event
        event = {
            "start": events[-1]['stop'] if events else gt_node.height,
            "stop": node[1].height,
            "gt_node": gt_node.idx,
            "st_node": st_node.idx,
            "neff": st_node.Ne,
        }
        logger.info((st_node.idx, gt_node.idx, gt_node.up.idx))
        # alt calculation
        logger.info(f"edges={len(st_node)}")

        # record the edge up until a st or gt node is passed
        if node[0] == 'g':
            event["gt_node"] = f"{gt_node.idx}->{node[1].idx}"
            event["st_node"] = f"{st_node.idx}->{st_node.idx}"
            event["edges"] = len(st_node) - coal_events
            gt_node = gt_node.up
            coal_events += 1            
        else:
            event["gt_node"] = f"{gt_node.idx}->{gt_node.idx}"
            event["st_node"] = f"{st_node.idx}->{node[1].idx}"
            event["edges"] = len(st_node)# - coal_events            
            st_node = st_node.up

        # get number of possible coalescences in the interval
        # st_edges = get_num_edges_at_time(species_tree, event['start'], st_clade)
        # gt_edges = get_num_edges_at_time(gene_tree, event['start'], st_clade)
        # edges_to_coal = 1 + gt_edges - st_edges
        # event["edges"] = edges_to_coal
        events.append(event)
        logger.info("-----")

    # build index labels for events
    data = pd.DataFrame(
        data=events,
        columns=['start', 'stop', 'gt_node', 'st_node', 'neff', 'edges'],
    )
    return data


if __name__ == "__main__":

    import ipcoal
    ipcoal.set_log_level("INFO")

    SPTREE = toytree.rtree.unittree(ntips=6, treeheight=1e6, seed=123)
    SPTREE = SPTREE.set_node_data("Ne", {i: 5e4 for i in (0, 1, 8)}, default=1e5)
    MODEL = ipcoal.Model(SPTREE, seed_trees=123)
    MODEL.sim_trees(1, 1)
    GTREE = toytree.tree(MODEL.df.genealogy[0])

    # 0, 1, 
    # 2 (needed to add internal node g6)
    # 3 (works w/ ILS only internal gs)
    # 4, 
    print(get_gene_tree_coal_intervals(SPTREE, GTREE, 9))

    # for nodex in GTREE.treenode.traverse():
        # if not nodex.is_root():
            # print(get_gene_tree_coal_intervals(SPTREE, GTREE, nodex.idx))

