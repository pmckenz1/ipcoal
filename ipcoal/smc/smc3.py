#!/usr/bin/env python

"""Distribution of waiting times in a species tree model.

Measure genome distance until genealogical changes under the
Sequential Markovian Coalescent (SMC) given a parameterized species
tree model.

TODO QUESTIONS
-------------- 
- can we compute separate probs for:
    - blens change but not topo   # genealogy bls always change, but not nec gtree
    - blens and topo change       # are some gene tree topos harder to 'get out of' than others?
- is non-ultrametric working?
    - how do expected waiting times vary among identical sptrees with gentime var?


- computing node statistics:
    - correlation among nodes, some will covary together along edges, relevance to introgression.
    - get a dist of simulated gene trees.
        - measure avg len and avg dist between midpoints of gtrees supporting node X.
            - meaning of shorter/longer and closer/farther for concat?
            - long and sparse can look like introgression


- SPTree based statistics:
    - On the same sptree but diff Ne-vs-g the gt len dist varies.
- If we can infer breakpoints:
    - Calculate prob of a sptree model given gtree lengths.
    - This could apply to viral genomes (few break points).


- The cited paper says:
    "We use these results to show that some of the recently proposed 
    methods for inferring sequences of trees along the genome provide 
    strongly biased distributions of waiting distances."
    -  What about this? It says the waiting times in msprime will be 
    wrong because it does not show cases where the genealogy does not
    change.


- Concatenation analysis:
    - known breakpoints:
        - prob gene trees in sp-tree * prob gene tree lengths
    - unknown breakpoints:
        - use 4-gamete test to find c-gene breakpoints
        - prob c-gene trees in sp-tree * prob c-gene tree lengths
    - Question in Felsenstein zone: 
        - Is true sptree lik > wrong-tree lik?

"""

from typing import Tuple
from concurrent.futures import ProcessPoolExecutor
from loguru import logger
import numpy as np
import pandas as pd
import toytree
import ipcoal

logger = logger.bind(name="ipcoal")


def get_coalescent_intervals(
    species_tree: toytree.ToyTree,
    gene_tree: toytree.ToyTree,
    ) -> pd.DataFrame:
    """Return a DataFrame with all coalescence intervals.

    This returns all intervals, and we can subset them later to 
    select only those relevant to a given gtree edge. Faster for later.

    start stop st_node st_neff 
    """
    # create a copy of the gene tree with 'age' feature on every node.
    # pruning does not retain 'height', which is relative, so we 
    # set 'age' as a copy on the full tree that is absolute. 
    # This is only relevant if trees are non-ultrametric (gentime var)
    gene_tree = gene_tree.set_node_data(
        feature="age",
        mapping=gene_tree.get_node_data("height").to_dict()
    )

    # iterate over st nodes getting the ids of gene tree edges present
    # at the lower border of each species tree interval.
    coal_ids = {}
    for node in species_tree.treenode.traverse("postorder"):
        if not node.is_root():
            st_tips = node.get_leaf_names()
            gt_subtree = gene_tree.prune(st_tips)
            gt_nodes = [
                i.idx for i in gt_subtree.treenode.traverse()
                if node.up.height > i.age > node.height
            ]
            coal_ids[node.idx] = gt_nodes
    return coal_ids


def get_gene_tree_coal_intervals(
    species_tree: toytree.ToyTree,
    gene_tree: toytree.ToyTree,
    idx: int,
    ) -> pd.DataFrame:
    """Return a DataFrame with intervals of coal and non-coal events.

    Each row in the dataframe represents an interval of time along
    the selected gene tree edge (idx), where a recomb event could break
    the current gene tree. It returns the information needed for 
    calculating the prob it reattaches to another gene tree edges given
    the length of each interval (dist), the edges that exist then 
    (edges), and the species tree params (tau and neff).

    Parameters
    ----------
    species_tree: toytree.ToyTree
        Species tree with a "Ne" feature assigned to every node, and
        edge lengths in units of generations. The tree can be non-
        ultrametric, representing differences in generation times.
    gene_tree: toytree.ToyTree
        Gene tree that could be embedded in the species tree. Edge 
        lengths are in units of generations.
    idx: int
        The node index below a focal edge in the gene_tree.
    """
    # create copy of the species tree with ncoals feature on every node
    # as the number of coal events that occurred below each st node
    species_tree = species_tree.set_node_data("ncoals", default=0)

    # create a copy of the gene tree with 'age' feature on every node.
    # pruning does not retain 'height', which is relative, so we 
    # set 'age' as a copy on the full tree that is absolute. 
    # This is only relevant if trees are non-ultrametric (gentime var)
    gene_tree = gene_tree.set_node_data(
        feature="age",
        mapping=gene_tree.get_node_data("height").to_dict()
    )

    # iterate over st nodes counting ncoals below each node. These 
    # values are used later to count gt edges remaining to coalesce.
    # TODO: there are faster ways to do this.
    for node in species_tree.treenode.traverse("postorder"):
        if not node.is_leaf():
            st_tips = node.get_leaf_names()
            gt_subtree = gene_tree.prune(st_tips)
            node.ncoals = len([
                i for i in gt_subtree.treenode.traverse()
                if (not i.is_leaf()) and i.age < node.height
            ])

    # get top and bottom gt nodes spanning the focal gt edge
    gt_node = gene_tree.idx_dict[idx]   # 7
    gt_top_node = gt_node.up            # 8
    gt_tips = gt_node.get_leaf_names()  # r2,3
    if gt_node.is_root():
        logger.error("Coalesce above the root gene tree node.")

    # get the current species tree edge containing gt_tips mrca
    st_idx = species_tree.get_mrca_idx_from_tip_labels(gt_tips)   # 6
    st_node = species_tree.idx_dict[st_idx]

    # get all st nodes on path from gt_node to gt_top_node # 7,9,10
    inodes = []
    while 1:
        st_node = st_node.up
        # logger.info((st_node.idx, st_node.height, gt_node.height, gt_top_node.height))
        # TODO: requiring > gt_node.height causes a rare problem in deep coal
        # where a relevant st node is skipped. Removing it fixed the problem, 
        # but need to check that all else is OK.
        if st_node and (gt_top_node.height > st_node.height):# > gt_node.height):
            inodes.append(('s', st_node))
        else:
            break

    ## TODO: idx=6 on gentime tree should start on st_node 9, not 6.
    # get the starting st_node (can be >1 st_nodes above gt_tips mrca)
    # if inodes:
        # st_node = inodes[0][1]
    # else:
    st_node = species_tree.idx_dict[st_idx]

    # get all gt nodes descended from gt_top_node that are not on path
    # if they are ILS'd w/ respect to their mrca st node w/ gt_node
    inodes.append(("g", gt_top_node))
    # for node in gt_top_node.get_descendants():

    # get all gt nodes that could be descended from the top species
    # tree node that is passed on the path from gt_node to gt_top_node
    if len(inodes) > 1:
        top_node = inodes[-2][1]
    else:
        top_node = gt_top_node
    for node in top_node.get_descendants():
        if (not node.is_leaf()) and (node.height > gt_node.height):
            gt_mrca = node.get_leaf_names() + gt_node.get_leaf_names()
            st_mrca = species_tree.get_mrca_idx_from_tip_labels(gt_mrca)
            st_tmp_node = species_tree.idx_dict[st_mrca]
            if node.height > st_tmp_node.height:
                inodes.append(("g", node))

    # internal nodes used for counting n_edges in intervals sorted by height.
    inodes = sorted(inodes, key=lambda x: x[1].height)
    logger.info(f"inodes={[(i[0], i[1].idx) for i in inodes]}")

    # if any gnodes occur on the edge between current st_node and its parent,
    # including the current gnode, if it is a coal event (i.e., internal node),
    # then these should be assigned to the current st_node.ncoals.
    if not gt_node.is_leaf():
        st_node.ncoals += 1 + len([
            i for i in gt_node.get_descendants()
            if i.height > st_node.height
        ])
    #logger.info(f"ncoals\n{species_tree.get_node_data('ncoals')}")

    # record each event of gene tree coal, or gene tree non-coal.
    events = []
    for node in inodes:
        logger.info(f'next={node[0]}{node[1].idx}, height={node[1].height:.0f}, g={gt_node.idx}, s={st_node.idx}, len.st={len(st_node)}, ncoals.st={st_node.ncoals}')

        # record the start of the next event
        event = {
            "start": events[-1]['stop'] if events else gt_node.height,
            "stop": node[1].height,
            "gt_node": gt_node.idx,
            "st_node": st_node.idx,
            "neff": st_node.Ne,
        }

        # record the st node that has been passed.
        if node[0] == 's':
            event['event'] = "ILS"
            event["st_node"] = f"{st_node.idx}->{node[1].idx}"
            event["edges_in"] = len(st_node) - st_node.ncoals
            if not st_node.is_root():
                st_node = st_node.up
        # the end 'g' node has been reached.
        else:
            event['event'] = f"COAL-g{node[1].idx}"
            event["st_node"] = f"{st_node.idx}->{st_node.idx}"
            event["edges_in"] = len(st_node) - st_node.ncoals
            st_node.ncoals += 1
        events.append(event)

    # build index labels for events
    data = pd.DataFrame(
        data=events,
        columns=['start', 'stop', 'st_node', 'neff', 'edges', 'edges_out', 'event'],
    )
    data['dist'] = data.stop - data.start
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
    idata = interval_data[interval_data.stop >= recombination_event[1]]

    # raise an error if timed event cannot happen on the selected edge
    if not idata.size:
        node = gene_tree.idx_dict[recombination_event[0]]
        raise ValueError(
            f"The recombination time ({recombination_event[1]}) does not "
            "exist on the selected gene tree edge "
            f"({node.height:.3f} - {node.up.height:.3f})"
        )

    # First term is calculated only on the focal interval, i.e., the
    # one in which the recombination event occurs.
    foc = idata.iloc[0]

    # get first term of the equation 
    # (TODO: should 'stop' be 'dist' or 'sumdist' for non-ultrametrics?
    # not sure yet whether this is a problem.
    # TODO: add verbal description here. Prob this happens, prob that happens...
    term_a = np.exp(-1 * (foc.edges / foc.neff) * foc.stop)
    term_b = np.exp((foc.edges / foc.neff) * recombination_event[1])
    first_term = (1 / foc.edges) - (1 / foc.edges) * term_a * term_b

    # Second term loops through all remaining intervals in a nested way
    # such that if there were 4 rows it would do (0, None), (1, 0),
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

        term_aa = np.exp(-1 * (foc.edges / foc.neff) * foc.stop - inner_sum)
        #term_b = np.exp((foc.edges / foc.neff) * recombination_event[1])
        term_c = (1 / odat.edges)
        term_d = (1 - np.exp(-1 * (odat.edges / odat.neff) * odat.dist))
        second_term += term_aa * term_b * term_c * term_d
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
    interval_data['neff'] *= 2

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

        # preventing overflow...?? exponent limit.
        if (oterm_a < 709) and (oterm_b < 709):
            second_expr += -np.exp(-1 * oterm_a) * (odat.neff / (odat.edges ** 2))
            first_expr = np.exp(oterm_a) - np.exp(oterm_b)

        # if there is no internal sum then blah blah
        else:
            second_expr += 1
            first_expr = (1 - np.exp(oterm_b - oterm_a) * (odat.neff / odat.edges ** 2))
        full_branch_sum += first_term + first_expr * second_expr
    return full_branch_sum * (1 / (interval[1] - interval[0]))


def get_prob_gene_tree_is_unchanged(
    species_tree: toytree.ToyTree,
    gene_tree: toytree.ToyTree,
    ) -> float:
    """Return the prob a gene tree is unchanged by recombination.

    This returns the cumulative probability that a recombination event
    on any edge in the gene tree will cause the gene tree topology
    to change given the length of each gene tree edge (the prob. that
    recomb. occurs on that edge) and the species tree model parameters.
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
            logger.info((node.idx, prob_unchanged))
    return prob_tree_unchanged


def get_expected_dist_until_gene_tree_changes(
    species_tree: toytree.ToyTree,
    gene_tree: toytree.ToyTree,
    recomb_rate: float,
    ) -> float:
    """Return the expected distance in bp until gene tree changes.

    This calculates the lambda of an exponential distributed waiting time
    given the total gt edge lengths, recomb_rate, and prob of change,
    and returns 1 / lambda as the expected distance until next event.

    Parameters
    ----------
    species_tree:
        ...
    gene_tree:
        ...
    recomb_rate: float
        Recombination rate in units of events per bp per generation.
    """
    gt_edge_lens = sum(gene_tree.get_node_data("dist").iloc[:-1])
    prob_change = 1 - get_prob_gene_tree_is_unchanged(species_tree, gene_tree)
    lambda_ = recomb_rate * prob_change * gt_edge_lens
    return 1 / lambda_


def get_distribution_of_gene_tree_distances(
    species_tree: toytree.ToyTree, 
    recomb_rate: float,
    genome_length: int,
    cores: int=4,
    ):
    """Return a distribution of gene tree distances.

    Given a species tree model a genome is simulated with a uniform
    recombination rate. The expected distance until the first gene tree
    changes is calculated, and this is repeated for all subsequent 
    genealogies on the genome.

    Parameters
    ----------
    species_tree: 
        A parameterized species tree...
    recomb_rate: float
        The per-site per-generation recombination rate.
    genome_length: int
        The length of the genome on which recombination can occur.
    cores: int
        Parallelize computation on N cores.

    Returns
    -------
    np.ndarray
        An array of float values as N expected genealogy distances for
        N genealogies existing on the genome.
    """
    model = ipcoal.Model(tree=species_tree, recomb=recomb_rate)
    model.sim_loci(nloci=1, nsites=genome_length)
    rasyncs = {}
    with ProcessPoolExecutor(max_workers=cores) as pool:
        for idx in model.df.index:
            gtree = toytree.tree(model.df.genealogy[idx])
            args = (species_tree, gtree, recomb_rate)
            rasyncs[idx] = pool.submit(get_expected_dist_until_gene_tree_changes, *args)

    # collect results
    expected_waiting_times = np.array([val.result() for _, val in rasyncs.items()])
    return expected_waiting_times


def get_distribution_of_gene_tree_node_distances(
    species_tree: toytree.ToyTree, 
    recomb_rate: float,
    genome_length: int,
    cores: int=4,
    ):
    """Return a distribution of gene tree distances.

    Given a species tree model a genome is simulated with a uniform
    recombination rate. The expected distance until the first gene tree
    changes is calculated, and this is repeated for all subsequent 
    genealogies on the genome.

    Parameters
    ----------
    species_tree: 
        A parameterized species tree...
    recomb_rate: float
        The per-site per-generation recombination rate.
    genome_length: int
        The length of the genome on which recombination can occur.
    cores: int
        Parallelize computation on N cores.

    Returns
    -------
    np.ndarray
        An array of float values as N expected genealogy distances for
        N genealogies existing on the genome.
    """
    model = ipcoal.Model(tree=species_tree, recomb=recomb_rate)
    model.sim_loci(nloci=1, nsites=genome_length)
    rasyncs = {}
    with ProcessPoolExecutor(max_workers=cores) as pool:
        for idx in model.df.index:
            gtree = toytree.tree(model.df.genealogy[idx])
            args = (species_tree, gtree, recomb_rate)
            rasyncs[idx] = pool.submit(get_expected_dist_until_gene_tree_changes, *args)

    # collect results
    expected_waiting_times = np.array([val.result() for _, val in rasyncs.items()])

    # TODO: put all into one parallelized loop.
    # organize results by node existence
    node_to_dists_map = {i: [] for i in species_tree.idx_dict if i >= species_tree.ntips}
    for idx in model.df.index:
        gtree = toytree.tree(model.df.genealogy[idx])
        for node in species_tree.treenode.traverse():
            if not node.is_leaf():
                # is st node monophyletic in gt?
                st_tips = node.get_leaf_names()
                gt_mrca_idx = gtree.get_mrca_idx_from_tip_labels(st_tips)
                if set(st_tips) == set(gtree.get_tip_labels(gt_mrca_idx)):
                    dist = expected_waiting_times[idx]
                    node_to_dists_map[node.idx].append(dist)
    return node_to_dists_map



if __name__ == "__main__":

    ipcoal.set_log_level("INFO")
    pd.set_option("max_columns", 20)
    pd.set_option("precision", 2)


    SPTREE = toytree.rtree.unittree(ntips=6, treeheight=1e6, seed=123)
    SPTREE = SPTREE.set_node_data("Ne", {i: 5e4 for i in (0, 1, 8)}, default=1e5)
    # SPTREE = SPTREE.set_node_data("Ne", default=2.5e5)
    MODEL = ipcoal.Model(SPTREE, seed_trees=123)
    MODEL.sim_trees(1, 1)
    GTREE = toytree.tree(MODEL.df.genealogy[0])


    GIDX = 0
    GNODE = GTREE.idx_dict[GIDX]

    print(get_gene_tree_coal_intervals(SPTREE, GTREE, 0))
    print(pd.get_option("display.max_colwidth"))
    # print(get_coalescent_intervals(SPTREE, GTREE))
    # for pos in np.linspace(GNODE.height, GNODE.up.height, 100):
        # args = (SPTREE, GTREE, (GIDX, pos))
        # p = get_prob_gene_tree_is_unchanged_by_recomb_event(*args)
        # print(p)
