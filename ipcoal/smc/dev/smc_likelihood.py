#!/usr/bin/env python

"""Species tree inference under the M-SMC.

Calculate the likelihood of a species tree under the M-SMC model, 
and compare this to the likelihood under ML concatenation. Show that 
with increasing locus lengths (increasing number of c-genes) the M-SMC
converges on the true topology whereas the ML-concatenation method does
not, when testing in the Felsenstein zone.

Likelihood calculations are performed by subprocess calls to IQTree.

TODO new
---------
1. Lik1 = Likelihood of a GT given an ST
2. Lik2 = Likelihood of length given the (last) GT.
3. A simple Felsenstein zone scenario...
    - Fixed topology, optimize parameters.
    - MCMC method

4. Validate ST inference on unlinked trees with standard Lik1
5. Validate ST inference on linked trees using Lik1 * Lik2


TODO
----
    - simulate species trees for 5-6 tips:
        - (1) simple scenario
        - (2) Felsenstein zone scenario
    - for each, simulate a large chromosome w/ thousands of linked genealogies.
    - get breakpoint positions:
        - simulated cgene breaks
        - simulated gene breaks
        - estimated cgene breaks
        - estimated gene breaks
    - calculate gene tree likelihoods using iqtree
        - concatenation  | ------------------------------------------|   TWrong > TTrue
        - cgenes         | ------------| ------------| --------------|   TTrue > TWrong
        - genes          | ---- | -----| ------| -------| -----------|   TTrue > TWrong  
                            
    - calculate MSC species tree likelihoods
        - cgenes known breakpoints
        - cgenes estimated breakpoints        
        - genes known breakpoints
        - genes estimated breakpoints
    - calculate composite M-SMC likelihoods
        - cgenes known breakpoints + length probs
        - cgenes estimated breakpoints + length probs
        - genes known breakpoints + length probs
        - genes estimated breakpoints + length probs

    Expectation
    -----------
    - ML concatenation converges on incorrect tree
    - SMC converges on incorrect tree for linked data
    - M-SMC converges on correct tree

"""

import numpy as np
import toytree


def get_probability_of_incongruence():
    """

    """


def get_single_genealogy_multispecies_coalescent_likelihood(
    genealogy, 
    species_tree,
    ):
    """Return the likelihood of a genealogy given a species tree model.

    Parameters
    ----------

    References
    ----------
    ...
    """
    # theta is the vector of species tree parameters (thetas and taus)
    # which we store as two node idx ordered arrays.
    thetas = np.zeros(species_tree.ntips)
    taus = np.zeros(species_tree.nnodes - species_tree.ntips)

    # 





def infer_cgenes_positions_from_sequence_alignment():
    """Returns the inferred c-gene breakpoint positions.
    """


def infer_cgene_tree_likelihood():
    """Calculate the likelihood a gene tree from an alignment using ML.
    
    """


def infer_species_tree_likelihood_under_M_SMC():
    """Return the likelihood of a species tree model given a set of 
    linked gene trees under the MSMC.

    """


def infer_species_tree_likelihood_under_MSC():
    """Return the likelihood of a species tree model given a set of 
    linked gene trees under the MSC.

    """


