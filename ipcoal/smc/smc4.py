#!/usr/bin/env python

"""Distribution of waiting times in a species tree model.

Measure genome distance until genealogical changes under the
Sequential Markovian Coalescent (SMC) given a parameterized species
tree model.
"""

from typing import Dict, List
import itertools
from concurrent.futures import ProcessPoolExecutor
from loguru import logger
import numpy as np
import pandas as pd
import toytree
from scipy import stats
import ipcoal

logger = logger.bind(name="ipcoal")


def get_embedded_gene_tree_table(
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
    for st_node in species_tree.treenode.traverse("postorder"):

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
        mask_below = gt_node_heights > st_node.height + 0.0001 # zero-align ipcoal bug
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

        # store coalescent times in the interval
        data[st_node.idx] = {
            "start": st_node.height,
            "stop": st_node.up.height if st_node.up else pd.NA,
            "st_node": st_node.idx,
            "neff": st_node.Ne,
            "nedges_in": nedges_in,
            "nedges_out": nedges_out,
            "coal_events": coal_events,
        }

    # split epochs on coalescent events
    split_data = []
    for nidx in range(species_tree.nnodes):

        # extract data from dict
        edict = data[nidx]
        start = edict['start']
        stop = edict['stop']
        nedges_in = edict['nedges_in']
        neff = edict['neff']
        nnedges = edict['nedges_in']

        # no events during epoch
        if not edict['coal_events']:
            split_data.append([start, stop, nidx, neff, nnedges, pd.NA])

        # split epoch between start - coals - end
        else:
            # add interval from start to first coal, or coal to next coal
            for coal in sorted(edict['coal_events'], key=lambda x: x.height):
                split_data.append([start, coal.height, nidx, neff, nnedges, coal.idx])
                start = coal.height
                nnedges -= 1
            # add final interval from coal to stop
            split_data.append([start, stop, nidx, neff, nnedges, pd.NA])

    table = pd.DataFrame(
        data=split_data,
        columns=['start', 'stop', 'st_node', 'neff', 'nedges', 'coal'],
    )
    table['dist'] = table.stop - table.start
    return table

def get_species_tree_intervals_on_gene_tree_path(
    species_tree: toytree.ToyTree,
    gene_tree: toytree.ToyTree,
    imap: Dict,
    idx: int,
    ):
    """Return list of species tree node idxs visited by gene tree edge.

    This function is used internally by other functions to subselect
    rows of the embedded gene tree table that are relevant to each
    gtree edge.
    """
    # get the st node containing the gt_node
    gt_node = gene_tree[idx]
    gt_tips = gt_node.get_leaf_names()
    st_tips = set()
    for st_tip in imap:
        for gtip in gt_tips:
            if gtip in imap[st_tip]:
                st_tips.add(st_tip)
    st_node = species_tree.get_mrca_node(*st_tips)

    # get the st_node path spanned by the gtree edge
    path = []
    while 1:
        # if at root then store and break
        if not st_node.up:
            path.append(st_node.idx)
            break

        # if gt_node top comes before next st_node then break
        if gt_node.up.height < st_node.up.height:
            path.append(st_node.idx)
            break

        # store current st_node as visited unless its above this st parent
        if not gt_node.height > st_node.up.height:
            path.append(st_node.idx)

        # advance towards st root
        st_node = st_node.up
    return path

def get_embedded_path_of_gene_tree_edge(
    table: pd.DataFrame,
    species_tree: toytree.ToyTree,
    gene_tree: toytree.ToyTree,
    imap: Dict,
    idx: int,
    ):
    """Return rows of an embedded path table for a selected gtree edge"""
    # get species tree intervals containing this gene tree edge
    sidxs = get_species_tree_intervals_on_gene_tree_path(
        species_tree, gene_tree, imap, idx)

    # select intervals
    gt_node = gene_tree[idx]
    mask0 = table.st_node.isin(sidxs)
    mask1 = table.start >= gt_node.height
    mask2 = table.stop <= gt_node.up.height
    subtable = table[mask0 & mask1 & mask2]
    return subtable

def get_prob_gene_tree_is_unchanged_by_recomb_event(
    table: pd.DataFrame,
    species_tree: toytree.ToyTree,
    gene_tree: toytree.ToyTree,
    imap: Dict,
    idx: int,
    time: float,
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
    table: pd.DataFrame
        Embedded gene tree table.
    species_tree: toytree.ToyTree
        Species tree with a "Ne" feature assigned to every node, and
        edge lengths in units of generations. The tree can be non-
        ultrametric, representing differences in generation times.
    gene_tree: toytree.ToyTree
        Gene tree simulated from the species tree. Edge lengths are in
        units of generations.
    imap: Dict
        ...
    idx: int
        The index of the gene tree edge where recomb event occurs.
    time:
        The time in generations at which recombination event occurs.
    """
    # get all coalescent intervals on this gt edge
    table = get_embedded_path_of_gene_tree_edge(
        table, species_tree, gene_tree, imap, idx)

    # get only the portion including the recomb event and above
    table = table[table.stop > time]

    # raise an error if timed event cannot happen on the selected edge
    if not table.size:
        gt_node = gene_tree[idx]
        raise ValueError(
            f"The time ({time}) does not exist on gene tree edge {idx} "
            f"({gt_node.height:.3f} - {gt_node.up.height:.3f})"
        )

    # select the interval containing the recomb event at time (t)
    dat = table.iloc[0]

    # calculate the first term
    term_a = np.exp(-1 * (dat.nedges / dat.neff * dat.stop))
    term_b = np.exp(dat.nedges / dat.neff * time)
    first_term = (1 / dat.nedges) - (1 / dat.nedges) * term_a * term_b

    # Second term loops through all remaining intervals in a nested way
    # such that if there were 4 rows it would do (0, None), (1, 0),
    # (2, (0, 1)), (3, (0, 1, 2)), etc., as the (outer, inner) loops.
    sec = table.iloc[1:]

    # the outer loop (intervals above the t-containing interval)
    second_term = 0
    for odx, _ in enumerate(sec.index):

        # the inner loop (the intervals between t-containing & current)
        inner_sum = 0
        for qdx in range(odx):
            qdat = sec.iloc[qdx]
            inner_sum += (qdat.nedges / qdat.neff) * qdat.dist

        # the current interval above the t-containing interval
        odat = sec.iloc[odx]
        second_term += (
            term_a - inner_sum *
            term_b * \
            (1 / odat.stop) * \
            (1 - np.exp(-1 * odat.nedges / odat.neff * odat.dist))
        )
    return first_term + second_term

def get_prob_gene_tree_is_unchanged_by_recomb_on_edge(
    table: pd.DataFrame,
    species_tree: toytree.ToyTree,
    gene_tree: toytree.ToyTree,
    imap: Dict[str,List[str]],
    idx: int
    ):
    """Return the prob a gene tree is unchanged given recomb on an edge.

    """
    # get all coalescent intervals on this gt edge
    table = get_embedded_path_of_gene_tree_edge(
        table, species_tree, gene_tree, imap, idx)
    logger.debug(f"path of edge {idx}:\n{table}\n")

    # iterate over all intervals on the gt edge
    full_branch_sum = 0
    for interval in table.index:

        # get data for the first interval
        odat = table.loc[interval]

        # coal rates become very high when neff <<< dist
        oterm_stop = (odat.nedges / (odat.neff * 2)) * odat.stop
        oterm_start = (odat.nedges / (odat.neff * 2)) * odat.start
        oterm_self = (odat.neff * 2) / (odat.nedges ** 2)
        oterm_other = (odat.neff * 2) / odat.nedges

        # prob sampling 1 avail edge this height above start
        first_term = (1 / odat.nedges) * odat.dist

        # iterate over each edge interval above this one to get second term
        second_term = 0
        for up_interval in table.loc[interval:].index[1:]:

            # get data for this interval
            jdat = table.loc[up_interval]

            # sum coal probabilities of edges between odat and jdat
            inner_sum = 0
            for inner in table.loc[interval:up_interval].index[1:-1]:
                qdat = table.loc[inner]
                inner_sum += (qdat.nedges / (qdat.neff * 2)) * qdat.dist

            # prob of coal NOT reconnecting in orig or inner intervals
            term_a = np.exp(-1 * oterm_stop - inner_sum)

            # prob of sampling an edge in up_interval
            term_b = (1 / jdat.nedges)

            # prob of coal reconnecting in up_interval
            term_c = 1 - np.exp(-1 * (jdat.nedges / (jdat.neff * 2)) * jdat.dist)

            # multiply all terms by prob of reconnecting in original interval
            second_term += term_a * term_b * term_c * oterm_other

        # check for overflow caused by dist >>> neff, handle w/ 128bit floats
        if (oterm_start > 709) or (oterm_stop > 709):
            oterm_start = np.float128(oterm_start)
            oterm_stop = np.float128(oterm_stop)
        first_expr = np.exp(oterm_stop) - np.exp(oterm_start)
        second_expr = second_term + -np.exp(-oterm_stop) * oterm_self
        expression = float(first_expr * second_expr)

        # debugging
        logger.debug(
            f"**** st_node={odat.st_node} interval={interval}\n"
            f"first_term={first_term}\n"
            f"second_term_before={second_term}\n"
            f"first_expr={first_expr}\n"
            f"second_expr={second_expr}\n"
            f"branch_sum={first_term + expression}\n"
            f"--------------\n"
        )

        # store the branch sum
        full_branch_sum += first_term + expression

    # return as branch sum weighted by its total length
    # FIXME: PROBLEM TO FIX HERE.
    if table.dist.sum() == 0:
        logger.error(f"error at idx={idx}:\n{gene_tree.get_node_data()}")

    return full_branch_sum / table.dist.sum()

def get_prob_gene_tree_is_unchanged(
    species_tree: toytree.ToyTree,
    gene_tree: toytree.ToyTree,
    imap: Dict,
    ) -> float:
    """Return the prob a gene tree is unchanged by recombination.

    This returns the cumulative probability that a recombination event
    on any edge in the gene tree will cause the gene tree topology
    to change given the length of each gene tree edge (the prob. that
    recomb. occurs on that edge) and the species tree model parameters.
    """
    # get coalescence table
    table = get_embedded_gene_tree_table(species_tree, gene_tree, imap)

    # get sum of all edge length (excluding the root stem) on gtree.
    sum_edge_lengths = sum(gene_tree.get_node_data("dist")[:-1])

    # for each edge add the probability that recomb would change it.
    prob_tree_unchanged = 0
    for node in gene_tree:
        if not node.is_root():
            prob_unchanged = get_prob_gene_tree_is_unchanged_by_recomb_on_edge(
                table, species_tree, gene_tree, imap, node.idx,
            )
            prob_tree_unchanged += (node.dist / sum_edge_lengths) * prob_unchanged
            # logger.info((node.idx, prob_unchanged))
    return prob_tree_unchanged

def get_expected_dist_until_gene_tree_changes(
    species_tree: toytree.ToyTree,
    gene_tree: toytree.ToyTree,
    imap: Dict,
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
    prob_change = 1 - get_prob_gene_tree_is_unchanged(species_tree, gene_tree, imap)
    lambda_ = recomb_rate * prob_change * gt_edge_lens
    return 1 / lambda_

####################################################################
## VALIDATION
####################################################################

def compare_to_ipcoal(
    ipcoal_model: ipcoal.Model,
    recombination_rate: float=1e-8,
    cores: int=4,
    ) -> pd.DataFrame:
    """Compute expected waiting distances between simulated genealogies.

    This takes an ipcoal Model object that contains simulated trees in
    its .df attribute table as input, and returns an array of distances
    in units of sites. Computation is parallelized using multiple cores.
    """
    stree = ipcoal_model.tree
    imap = ipcoal_model.get_imap_dict()
    func = get_expected_dist_until_gene_tree_changes

    # distribute computation of 'expected' in parallel
    with ProcessPoolExecutor(max_workers=cores) as pool:
        rasyncs = {}
        for idx in ipcoal_model.df.index:
            gtree = toytree.tree(ipcoal_model.df.genealogy[idx])
            args = (stree, gtree, imap, recombination_rate)
            rasyncs[idx] = pool.submit(func, *args)

    # collect results and return
    expected_wait = np.array([i.result() for i in rasyncs.values()])
    expected_dist = np.random.exponential(expected_wait)
    # expected_dist = stats.norm.pdf(expected_wait)

    # report results to logger info
    logger.info(
        "ipcoal comparison\n"
        "-----------------\n"
        f"ipcoal distances: "
        f"mean={ipcoal_model.df.nbps.mean():.2f}, "
        f"std={ipcoal_model.df.nbps.std():.2f}\n"
        f"sm-msc prediction: "
        f"mean={expected_dist.mean():.2f}, "
        f"std={expected_dist.std():.2f}\n"
        "-----------------"
    )

    # calculate likelihoods
    liks = stats.norm(
        loc=expected_wait, 
        scale=expected_wait**2
    ).pdf(ipcoal_model.df.nbps)
    logliks = -1 * np.log(liks)    

    # return modified ipcoal dataframe with results
    data = pd.DataFrame({
        "start": ipcoal_model.df.start,
        #"end": ipcoal_model.df.end,
        "obs_dist": ipcoal_model.df.nbps,
        "exp_wait": expected_wait,
        "lambda_": 1 / expected_wait,
        "likelihood": liks,
        "logliks": logliks,
        # "genealogy": ipcoal_model.df.genealogy,
    })
    return data

def compare_to_ipcoal_plot(data: pd.DataFrame):
    """Return a plot of the dataframe from compare_to_ipcoal func.

    Opens a toyplot drawing in the browser.
    """
    import toyplot
    import toyplot.browser

    # limit x-scale
    xmax = data.obs_dist.mean() * 10

    # create plot
    canvas = toyplot.Canvas(width=800, height=300)
    ax0 = canvas.cartesian(grid=(1, 3, 0), label="expected waiting")
    ax1 = canvas.cartesian(grid=(1, 3, 1), xmax=xmax, label="simulated dists")
    ax2 = canvas.cartesian(grid=(1, 3, 2), xmax=xmax, label="expected dists (sampled)")

    bins = np.linspace(0, xmax, 20)
    mark0 = ax0.bars(np.histogram(data.exp_wait, bins=20, density=True))
    mark1 = ax1.bars(np.histogram(data.obs_dist, bins=bins, density=True))
    mark2 = ax2.bars(np.histogram(np.random.exponential(data.exp_wait), bins=bins, density=True))

    for axes in (ax0, ax1, ax2):
        axes.x.ticks.show = True
        axes.y.ticks.show = True
    toyplot.browser.show(canvas)
    return canvas, (ax0, ax1, ax2), (mark0, mark1, mark2)

def get_distance_likelihood(
    species_tree: toytree.ToyTree,
    gene_tree: toytree.ToyTree,
    imap: Dict,
    recombination_rate: float,
    observed_distance: float,
    ):
    """Return the likelihood of a waiting distance given sptree and gene tree.

    Parameters
    ----------
    ...
    """
    # get the expected distance until next tree given model
    expected_dist = get_expected_dist_until_gene_tree_changes(
        species_tree, gene_tree, imap, recombination_rate)
    norm = stats.norm(
        loc=expected_dist, 
        scale=expected_dist ** 2,
    )
    lik = norm.pdf(observed_distance)
    return lik


if __name__ == "__main__":

    ipcoal.set_log_level("INFO")
    pd.options.display.max_columns = 20
    pd.options.display.width = 1000

    # setup species tree model
    # SPTREE = toytree.rtree.unittree(ntips=10, treeheight=2e6, seed=123)
    # SPTREE = SPTREE.set_node_data(
        # "Ne", default=1e6, mapping={i: 5e5 for i in (0, 1, 8, 9)})

    SPTREE = toytree.rtree.unittree(ntips=4, treeheight=2e6, seed=123)
    SPTREE = SPTREE.set_node_data("Ne", default=1e6)

    # simulate one genealogy
    RECOMB = 1e-9
    MODEL = ipcoal.Model(SPTREE, seed_trees=123, nsamples=2, recomb=RECOMB)
    MODEL.sim_trees(1, 1)
    IMAP = MODEL.get_imap_dict()
    GTREE = toytree.tree(MODEL.df.genealogy[0])

    logger.info("One embedded gene tree table:\n"
        f"{get_embedded_gene_tree_table(SPTREE, GTREE, IMAP)}")
    EDIST = get_expected_dist_until_gene_tree_changes(SPTREE, GTREE, IMAP, 1e-8)
    logger.warning(EDIST)

    # simulate a long chromosome with many genealogies
    NSITES = 1e4
    logger.info(f"simulating {NSITES} bp")
    MODEL.sim_loci(nloci=1, nsites=NSITES)
    logger.info(f"simulated {MODEL.df.shape[0]} genealogies")
    logger.info(f"simulated:\n{MODEL.df}\n")    

    # compare sim distances to predicted distances
    logger.info("computing expected waiting distances")
    CDATA = compare_to_ipcoal(ipcoal_model=MODEL, recombination_rate=RECOMB, cores=7)
    logger.info(
        "expected waiting distances\n"
        "--------------------------\n"
        f"{CDATA.head(15)}\n"
        "...\n"
        "--------------------------"
    )
    # compare_to_ipcoal_plot(CDATA)
    # logger.info(f"\n{CDATA.describe().T[['mean', 'std', 'min', 'max']]}")
