#!/usr/bin/env python

"""Development of MSC likelihoods"""


import numpy as np
import pandas as pd
import toytree


def get_probability_of_incongruence():
    """

    """
    # 1 / (j / 2)

    # coalescent rate during interval i when there are n lineages
    nlineages = 3
    coal_rate_i = (nlineages * (nlineages - 1)) / theta_i


def get_single_genealogy_multispecies_coalescent_likelihood(
    genealogy: toytree.ToyTree, 
    species_tree: toytree.ToyTree,
    ):
    """Return the likelihood of a genealogy given a species tree model.

    For each population the genealogy is traced backwards in time and 
    the number of lineages at the start (m) and end (n) recorded. The 
    waiting time until the next coalescence has an exponential density.



    Parameters
    ----------

    References
    ----------
    - Rannala B, Yang Z (August 2003). "Bayes estimation of species 
      divergence times and ancestral population sizes using DNA 
      sequences from multiple loci". Genetics. 164 (4): 1645–56. 
      doi:10.1093/genetics/164.4.1645. PMC 1462670. PMID 12930768.
    """
    # theta is the vector of species tree parameters (thetas and taus)
    # which we store as two node idx ordered arrays.
    thetas = np.zeros(species_tree.ntips)
    taus = np.zeros(species_tree.nnodes - species_tree.ntips)

    # 


def get_species_tree_table(species_tree: toytree.ToyTree):
    """Return a table with species tree intervals"""
    table = pd.DataFrame(
        index=range(species_tree.nnodes),
        columns=["theta", "dist", "edges_in", "edges_out"],
    )
    table["theta"] = species_tree.get_node_data("Ne") * 4 * 1
    return table


if __name__ == "__main__":

    import ipcoal

    SPTREE = toytree.rtree.unittree(ntips=5, treeheight=50000)
    SPTREE = SPTREE.set_node_data("Ne", default=10000)

    MODEL = ipcoal.Model(tree=SPTREE, nsamples=2, seed_trees=333)
    MODEL.sim_trees(1)
    GTREE = toytree.tree(MODEL.df.genealogy[0])

    print(get_species_tree_table(SPTREE))
    