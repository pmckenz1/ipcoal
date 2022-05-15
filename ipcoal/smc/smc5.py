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

def p_ik(
    i: int, 
    k: int, 
    table: pd.DataFrame,
    ):
    '''
    Recycled in the math, for getting coal chances for intervals between i and k
    '''
    # Special case if i and k are equal:
    if i == k:
        # -1/a_i
        first=-1/table.iloc[i].nedges
        # e^{-a_i*T_i/n_i}
        second=np.exp((-table.iloc[i].nedges/(table.iloc[i].neff))*table.iloc[i].stop)
        return(first*second)
    
    # If i and k are not equal:
    else:
        # -(a_i/n_i)*sigma_{i+1}
        term1 = -(table.iloc[i].nedges/(table.iloc[i].neff))*table.iloc[i].stop
        
        # sum from q=i+1 to k of: (a_q/n_q)*T_q
        term2 = 0
        for q in range(i+1,k):
            term2 += (table.iloc[q].nedges/(table.iloc[q].neff))*table.iloc[q].dist
            
        # First half of the equation
        firsthalf = np.exp(term1 - term2)

        # Second half of the equation
        # 1/a_k * (1-e^{-(a_k/n_k)*T_k})
        secondhalf = (1/table.iloc[k].nedges) * (1-np.exp(-(table.iloc[k].nedges/table.iloc[k].neff)*table.iloc[k].dist))

        return(firsthalf*secondhalf)

def pb1(
    i: int, 
    table: pd.DataFrame, 
    m: int, 
    I_b: int, 
    I_bc: int,
    ):
    '''
    pb1(i), defined in the math
    '''
    ## Save i-related parameters
    # a_i
    curr_ai = table.iloc[i].nedges
    # T_i
    curr_Ti = table.iloc[i].stop-table.iloc[i].start
    # n_i
    curr_ni = table.iloc[i].neff
    # sigma_{i+1}
    curr_alpha_i1 = table.iloc[i].stop
    # sigma_i
    curr_alpha_i = table.iloc[i].start
    
    # Start with the second half of term 2...
    sum1 = 0
    for k in range(i,I_bc):
        sum1 += p_ik(i,k,table)
    sum2 = 0
    for k in range(m, I_b):
        sum2 += p_ik(i,k,table)
    
    # ...and now do the first half of term 2...
    firsthalf = (np.exp(curr_ai*curr_alpha_i1/curr_ni) - np.exp(curr_ai*curr_alpha_i/curr_ni))*curr_ni
    
    # ...and now multiply them together.
    second_term = firsthalf * (sum1+sum2)
    
    return((1/curr_ai) * (curr_Ti+second_term)) 

def pb2(
    i: int, 
    table: pd.DataFrame, 
    I_b: int, 
    I_bc: int,
    ):
    '''
    pb2(i), from the math
    '''
    ## Save i-related parameters
    # a_i
    curr_ai = table.iloc[i].nedges
    # T_i
    curr_Ti = table.iloc[i].stop-table.iloc[i].start
    # n_i
    curr_ni = table.iloc[i].neff
    # sigma_{i+1}
    curr_alpha_i1 = table.iloc[i].stop
    # sigma_i
    curr_alpha_i = table.iloc[i].start
    
    # Start with the second half of term 2...
    sum1 = 0
    for k in range(i,I_b):
        sum1 += p_ik(i,k,table)
    sum2 = 0
    for k in range(I_b,I_bc):
        sum2 += p_ik(i,k,table)
    
    # ...and now do the first half of term 2...
    firsthalf = (np.exp(curr_ai*curr_alpha_i1/curr_ni) - np.exp(curr_ai*curr_alpha_i/curr_ni))*curr_ni
    
    # ...and now multiply them together.
    second_term = firsthalf * (2*sum1+sum2)
    
    return((1/curr_ai) * (2*curr_Ti+second_term))

def topo_unch_prob_bt(
    gnode,
    t,
    species_tree,
    genealogy,
    imap,
    ):
    treetable = get_embedded_gene_tree_table(species_tree, genealogy, imap)
    treetable.neff = treetable.neff*2

    gnode_ints = get_embedded_path_of_gene_tree_edge(treetable,species_tree,genealogy,imap,gnode.idx).reset_index(drop=True)

    gnode_st_nodes = gnode_ints.st_node

    # get intervals for the parent branch
    parent = gnode.up
    if not parent.is_root():
        parent_ints = get_embedded_path_of_gene_tree_edge(treetable,species_tree,gt,imap,parent.idx).reset_index(drop=True)

    else:
        parent_ints = pd.DataFrame([gnode.up.height,
                             gnode.up.height + 1e9, # giant number here, infinite root branch length
                             gnode_ints.iloc[-1].st_node,
                             gnode_ints.iloc[-1].neff,
                             1,
                             np.nan,
                             1e9],index=['start','stop','st_node','neff','nedges','coal','dist']).T

    # get index of sibling, and its intervals
    sib = list(set(parent.children).difference(set([gnode])))[0]
    sib_ints = get_embedded_path_of_gene_tree_edge(treetable,species_tree,genealogy,imap,sib.idx).reset_index(drop=True)

    # get shared intervals of sibling
    # by asking which sib_ints are in the same species tree branch
    # (and later pruning to those which exist at the same time as gnode)
    sib_shared = sib_ints.loc[[i in np.array(gnode_st_nodes) for i in sib_ints.st_node]]

    # get important times
    # time at which sharing starts
    t_mb = sib_shared.iloc[0].start

    # time at which branch ends
    t_ub = gnode_ints.stop.iloc[-1]

    # time at which branch starts
    t_lb = gnode_ints.start.iloc[0]

    # in case the sibling branch starts earlier in sp tree branch than gnode branch
    if t_mb < t_lb:
        t_mb = t_lb
        sib_shared = gnode_ints.copy()

    # merging the current branch and parent branch dataframes
    merged_ints = pd.concat([gnode_ints,parent_ints],ignore_index=True)

    # get first interval index that is shared with sibling
    m = merged_ints.loc[merged_ints.stop > t_mb].index[0]

    # get number of intervals in current branch
    I_b = len(gnode_ints)

    # get number of intervals in combined branches
    I_bc = len(merged_ints)

    # get starting interval
    mask1 = merged_ints.start <= t
    mask2 = merged_ints.stop > t
    i = merged_ints[mask1 & mask2].index[0]

    # if first case
    if i < m:
        first_term = 1 / merged_ints.iloc[i].nedges
        second_term = 0
        for k in range(i,I_bc):
            second_term += p_ik(i, k, merged_ints)*np.exp((merged_ints.iloc[i].nedges / merged_ints.iloc[i].neff)*t)
        third_term = 0
        for k in range(m,I_b):
            third_term += p_ik(i, k, merged_ints)*np.exp((merged_ints.iloc[i].nedges / merged_ints.iloc[i].neff)*t)
        return(first_term + second_term + third_term)
    else:
        first_term = 1 / merged_ints.iloc[i].nedges
        second_term = 0
        for k in range(i,I_b):
            second_term += p_ik(i, k, merged_ints)*np.exp((merged_ints.iloc[i].nedges / merged_ints.iloc[i].neff)*t)
        third_term = 0
        for k in range(I_b,I_bc):
            third_term += p_ik(i, k, merged_ints)*np.exp((merged_ints.iloc[i].nedges / merged_ints.iloc[i].neff)*t)
        return(2*(first_term + second_term) + third_term)
    
    
def tree_unch_prob_bt(
    gnode,
    t,
    species_tree,
    genealogy,
    imap
    ):
    treetable = get_embedded_gene_tree_table(species_tree, genealogy, imap)
    treetable.neff = treetable.neff*2

    gnode_ints = get_embedded_path_of_gene_tree_edge(treetable,species_tree,genealogy,imap,gnode.idx).reset_index(drop=True)

    # time at which branch ends
    t_ub = gnode_ints.stop.iloc[-1]

    # time at which branch starts
    t_lb = gnode_ints.start.iloc[0]

    # get number of intervals in current branch
    I_b = len(gnode_ints)

    # get starting interval
    mask1 = gnode_ints.start <= t
    mask2 = gnode_ints.stop > t
    i = gnode_ints[mask1 & mask2].index[0] # if getting error here, probably t is on boundary of branch (eg 0 round error at tips)


    first_term = 1 / gnode_ints.iloc[i].nedges
    second_term = 0
    for k in range(i,I_b):
        second_term += p_ik(i, k, gnode_ints)*np.exp((gnode_ints.iloc[i].nedges / gnode_ints.iloc[i].neff)*t)

    return(first_term + second_term)

def get_topo_unchange_prob(
    species_tree: toytree.ToyTree, 
    genealogy: toytree.ToyTree, 
    imap: Dict,
    ):
    treetable = get_embedded_gene_tree_table(species_tree, genealogy, imap)
    # multiplier
    treetable.neff = treetable.neff*2

    gtree_total_length = np.sum(genealogy.get_node_data().dist.iloc[:-1])

    total_prob = 0
    for gnode in genealogy.treenode.traverse(strategy='postorder'):
        if not gnode.is_root():
            # get intervals for the current branch
            gnode_ints = get_embedded_path_of_gene_tree_edge(treetable,species_tree,genealogy,imap,gnode.idx).reset_index(drop=True)

            gnode_st_nodes = gnode_ints.st_node

            # get intervals for the parent branch
            parent = gnode.up
            if not parent.is_root():
                parent_ints = get_embedded_path_of_gene_tree_edge(treetable,species_tree,genealogy,imap,parent.idx).reset_index(drop=True)

            else:
                parent_ints = pd.DataFrame([gnode.up.height, 
                                     gnode.up.height + 1e9, # giant number here, infinite root branch length
                                     gnode_ints.iloc[-1].st_node,
                                     gnode_ints.iloc[-1].neff,
                                     1,
                                     np.nan,
                                     1e9],index=['start','stop','st_node','neff','nedges','coal','dist']).T

            # get index of sibling, and its intervals
            sib = list(set(parent.children).difference(set([gnode])))[0]
            sib_ints = get_embedded_path_of_gene_tree_edge(treetable,species_tree,genealogy,imap,sib.idx).reset_index(drop=True)
            
            # get shared intervals of sibling
            # by asking which sib_ints are in the same species tree branch
            # (and later pruning to those which exist at the same time as gnode)
            sib_shared = sib_ints.loc[[i in np.array(gnode_st_nodes) for i in sib_ints.st_node]]
            
            # get important times
            # time at which sharing starts
            t_mb = sib_shared.iloc[0].start

            # time at which branch ends
            t_ub = gnode_ints.stop.iloc[-1]
            
            # time at which branch starts
            t_lb = gnode_ints.start.iloc[0]

            # in case the sibling branch starts earlier in sp tree branch than gnode branch
            if t_mb < t_lb:
                t_mb = t_lb
                sib_shared = gnode_ints.copy()
            
            # merging the current branch and parent branch dataframes
            merged_ints = pd.concat([gnode_ints,parent_ints],ignore_index=True)

            # get first interval index that is shared with sibling
            m = merged_ints.loc[merged_ints.stop > t_mb].index[0]

            # get number of intervals in current branch
            I_b = len(gnode_ints)

            # get number of intervals in combined branches
            I_bc = len(merged_ints)
            
            # get first summation, using pb1
            firstsum = 0
            for i in range(0,m):
                firstsum += pb1(i,merged_ints,m,I_b,I_bc)

            # get second summation, using pb2
            secsum = 0
            for i in range(m,I_b):
                secsum += pb2(i,merged_ints,I_b,I_bc)

            # get normalizer using branch stop/start times
            normalize = 1/(t_ub-t_lb)

            #get probability of topology not changing if recomb event falls on this branch
            topo_unchanged_prob = normalize*(firstsum + secsum)
            
            #print(topo_unchanged_prob)

            # contribute to total probability of unchanged genealogical topology
            total_prob += ((t_ub-t_lb)/gtree_total_length)*topo_unchanged_prob
    return(total_prob)

def get_tree_unchange_prob(
    species_tree: toytree.ToyTree,
    genealogy: toytree.ToyTree, 
    imap: Dict,
    ):
    totaled_probs = 0
    
    gtree_total_length = np.sum(genealogy.get_node_data().dist.iloc[:-1])
    
    treetable = get_embedded_gene_tree_table(species_tree,genealogy,imap)
    treetable.neff = treetable.neff*2
    total_prob = 0
    all_int_tables = []
    for gnode in genealogy.treenode.traverse(strategy='postorder'):
        if not gnode.is_root():
            # get intervals for the current branch
            gnode_ints = get_embedded_path_of_gene_tree_edge(treetable,species_tree,genealogy,imap,gnode.idx).reset_index(drop=True)

            gnode_ints.neff = gnode_ints.neff
            all_int_tables.append(gnode_ints)
            # time at which branch ends
            t_ub = gnode_ints.stop.iloc[-1]
            # time at which branch starts
            t_lb = gnode_ints.start.iloc[0]

            # get number of intervals in current branch
            I_b = len(gnode_ints)

            sumval = 0
            for i in range(I_b):
                curr_ai = gnode_ints.iloc[i].nedges
                curr_Ti = gnode_ints.iloc[i].dist
                curr_ni = gnode_ints.iloc[i].neff
                curr_alpha_i1 = gnode_ints.iloc[i].stop
                curr_alpha_i = gnode_ints.iloc[i].start

                first=(1/curr_ai)*curr_Ti
                second=(curr_ni/curr_ai)
                third=np.exp((curr_ai/curr_ni)*curr_alpha_i1)-np.exp((curr_ai/curr_ni)*curr_alpha_i)
                fourth=0
                for k in range(i,I_b):
                    fourth += p_ik(i,k,gnode_ints)
                sumval += first + second*third*fourth

            brprob = sumval * (1/(t_ub-t_lb))

            totaled_probs += ((t_ub-t_lb) / gtree_total_length) * brprob
    return(totaled_probs)

