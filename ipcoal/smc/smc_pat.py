#!/usr/bin/env python

import toytree
import toyplot
import ipcoal
import numpy as np
import pandas as pd
import decimal

from loguru import logger
logger = logger.bind(name="ipcoal")


def get_tree_total_length(ttree):
    tot_len = 0
    for node_ in ttree.treenode.traverse():
        if not node_.is_root():
            tot_len += node_.dist
    return(tot_len)


def get_num_edges_at_time(tree, time):
    nodes_above = ([idx for idx, node in tree.idx_dict.items() if node.height > time])
    edges_above = len(nodes_above) + 1
    return edges_above


def get_tree_clade_times(tree):
    nodes_ = []
    heights_ = []
    for curr_node in tree.treenode.traverse():
        if not curr_node.is_leaf():
            nodes_.append(curr_node.get_leaf_names())
            heights_.append(curr_node.height)
    pddf = pd.DataFrame([nodes_,heights_],index=['clades','heights']).T
    return(pddf)


def get_branch_intervals(tr, gt, br):
    '''
    tr = species tree with Ne attribute
    gt = gene tree simulated on that species tree
    br = treenode representing a branch on the tree
    '''
    st_times = get_tree_clade_times(tr)
    gt_times = get_tree_clade_times(gt)
    coalclade = br.get_leaf_names()

    ###temp
    st_coal_node = tr.treenode.search_nodes(idx=tr.get_mrca_idx_from_tip_labels(br.get_leaf_names()))[0]
    nearest_st_node = st_coal_node
    while ((nearest_st_node.height + nearest_st_node.dist) < br.height):
        if nearest_st_node.is_root():
            break
        nearest_st_node = nearest_st_node.up

    coalclade = nearest_st_node.get_leaf_names()
    ###

    br_lower = br.height
    br_upper = br_lower + br.dist
    gt_clade_changes = (gt_times.heights < br_upper) & (gt_times.heights > br_lower)
    st_clade_changes = (st_times.heights < br_upper) & (st_times.heights > br_lower)
    st_time_diffed = st_times[st_clade_changes]
    #return(np.array([all(elem in clade for elem in coalclade) for clade in st_time_diffed.clades]))

    contains_clade = st_time_diffed[np.array([all(elem in clade for elem in coalclade) for clade in st_time_diffed.clades])]

    if not len(contains_clade.columns):
        contains_clade = pd.DataFrame(columns=['clades','heights'])
    contains_clade = pd.DataFrame([list(contains_clade.clades.append(pd.Series([coalclade]),ignore_index=True)),list(contains_clade.heights.append(pd.Series(br_lower)))],index=['clades','heights']).T
    contains_clade = contains_clade.sort_values('heights')

    all_members = []
    for i in contains_clade.clades:
        all_members.extend(i)
    all_members = np.unique(all_members)

    relevant_coals = pd.DataFrame(columns=["heights"])

    if np.sum(gt_clade_changes):
        potential_coals = gt_times[gt_clade_changes]
        relevant_coals = potential_coals[[set(i).issubset(all_members) for i in potential_coals.clades]]
        relevant_coals = relevant_coals.sort_values('heights')

    time_points = np.sort(list(contains_clade.heights) + list(relevant_coals.heights) + [br_upper])
    if int(time_points[-1]) == int(time_points[-2]):
        time_points = time_points[:-1]
    starts = time_points[:-1]
    stops = time_points[1:]
    lengths = stops-starts
    num_to_coal = np.repeat(1,len(starts))
    ne = np.repeat(1,len(starts))
    a_df = pd.DataFrame([starts,stops,lengths,num_to_coal,ne],index=['starts','stops','lengths','num_to_coal','ne']).T
    mids = (a_df.stops + a_df.starts)/2
    interval_reduced_trees=[]

    nes = []
    for mid in mids:
        clade = contains_clade.clades.iloc[np.sum(contains_clade.heights<mid)-1]

        cladeNe = tr.treenode.search_nodes(idx=tr.get_mrca_idx_from_tip_labels(clade))[0].Ne
        nes.append(cladeNe)
        reduced_tree = gt.prune(clade)
        interval_reduced_trees.append(reduced_tree.newick)

    a_df['reduced_trees'] = interval_reduced_trees
    a_df['mids'] = mids
    a_df['ne'] = nes
    a_df['num_to_coal'] = a_df.apply(lambda x: get_num_edges_at_time(toytree.tree(x['reduced_trees']), x['mids']), axis=1)

    return a_df

def calc_P_bT(df):
    last_index = len(df.starts)-1

    full_branch_summation = 0
    full_branch_start = df['starts'][0]
    full_branch_stop = df['stops'][last_index]

    for interval_index in range(len(df)):
        ai = df['num_to_coal'][interval_index]
        ni = df['ne'][interval_index]*2######################
        sigi = df['stops'][interval_index]
        sigb = df['starts'][interval_index]
        Ti = df['lengths'][interval_index]

        first_term = (1/ai)*Ti

        second_expr_second_term = 0
        for int_idx in range(interval_index+1,last_index+1): # for the *full* intervals above t
            # start with the summation
            internal_summation = 0
            if int_idx - interval_index > 1:
                for q_idx in range(interval_index+1,int_idx):
                    aq = df['num_to_coal'][q_idx]
                    nq = df['ne'][q_idx]*2############################
                    Tq = df['lengths'][q_idx]
                    internal_summation += ((aq/nq)*Tq)

            # define the properties of the current interval
            aint = df['num_to_coal'][int_idx]
            nint = df['ne'][int_idx]*2
            Tint = df['lengths'][int_idx]

            # calculate the expressions that are multiplied together
            #first_mult = np.exp((ai/ni)*t)
            second_mult = np.exp(-1*(ai/ni)*sigi - internal_summation)
            third_mult = (1/aint)*(1-np.exp(-1*(aint/nint)*Tint))

            #print(second_mult*third_mult)
            second_expr_second_term += (second_mult*third_mult)*(ni/ai)

        second_before = second_expr_second_term
        # preventing overflow
        # start by treating cases without overflow as usual:
        if ((ai/ni)*sigi < 709) and ((ai/ni)*sigb < 709): # prevent overflow...
            # print("NO OVERFLOW")
            second_expr_second_term += -np.exp(-1*(ai/ni)*sigi) * (ni/(ai*ai))
            first_expr_second_term = (np.exp((ai/ni)*sigi) - np.exp((ai/ni)*sigb))

        # if there is no internal summation, then the problem simplifies to (e^x-e^y)/e^x , which is 1-e^(y-x)
        elif second_expr_second_term == 0:
            logger.warning("NO SECOND TERM")
            second_expr_second_term += 1
            first_expr_second_term = (1-np.exp((ai/ni)*sigb-(ai/ni)*sigi))* (ni/(ai*ai))

        # if there IS internal summation, then we will just use very high precision
        elif ((ai/ni)*sigi > 709) or ((ai/ni)*sigb > 709): # overflow...
            print("OVERFLOW")
            # this is the nonzero value of the internal summation
            print("original value of second expression: " + str(second_expr_second_term))
            sigi_val = decimal.Decimal((ai/ni)*sigi)
            sigb_val = decimal.Decimal((ai/ni)*sigb)
            second_expr_second_term = decimal.Decimal(second_expr_second_term) + -np.exp(-1*sigi_val) * decimal.Decimal((ni/(ai*ai)))
            # this is the new value of the second expression, after adding the overflow term
            print("value of second expression after addition: " + str(second_expr_second_term))
            first_expr_second_term = (np.exp(sigi_val) - np.exp(sigb_val))
            # this is the value of the first expression
            print("value of first expression: " + str(first_expr_second_term))
            # these next two values, when divided first/second, are the
            # probability of recomb in this interval not changing tree
            print("value of interval sum: " + str(decimal.Decimal(first_term) + first_expr_second_term*second_expr_second_term))
            print("length of interval: " + str(Ti))

        # store the branch sum 
        logger.warning(
            f"**** {interval_index}\n"
            f"first_term={first_term}\n"
            f"second_term_before={second_before}\n"
            f"first_expr={first_expr_second_term}\n"
            f"second_expr={second_expr_second_term}\n"
            f"branch_sum={first_term + first_expr_second_term * second_expr_second_term}\n"
            f"--------------\n"
        )

        full_branch_summation += first_term + float(first_expr_second_term*second_expr_second_term)
    return(full_branch_summation * (1/(full_branch_stop-full_branch_start)))


# this is the function that just loops over the branches in the tree
def get_unchange_prob(tre, gtr):
    full_tree_length = 0
    for node in gtr.treenode.traverse():
        if not node.is_root():
            full_tree_length += node.dist
    prob_tree_unchanged = 0
    for node in gtr.treenode.traverse():
        if not node.is_root():
            df = get_branch_intervals(tre,gtr,node)
            unchanged_branch_prob = calc_P_bT(df) # this is the branch-specific function
            prob_tree_unchanged += (node.dist / full_tree_length) * unchanged_branch_prob
    return(prob_tree_unchanged)


if __name__ == "__main__":



    SPTREE = toytree.rtree.unittree(ntips=6, treeheight=1e6, seed=123)
    # SPTREE = SPTREE.set_node_data("Ne", {i: 5e4 for i in (0, 1, 8)}, default=1e5)
    SPTREE = SPTREE.set_node_data("Ne", default=2e5)
    MODEL = ipcoal.Model(SPTREE, seed_trees=123, nsamples=1)
    MODEL.sim_trees(1, 1)
    GTREE = toytree.tree(MODEL.df.genealogy[0])

    IDX = 8
    GIDX = GTREE.idx_dict[IDX]

    df = get_branch_intervals(SPTREE, GTREE, GIDX)
    print(df)

    print(calc_P_bT(df))
    # print(get_unchange_prob(SPTREE, GTREE))
