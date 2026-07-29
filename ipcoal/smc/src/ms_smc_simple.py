#!/usr/bin/env python

"""General user-facing functions for accessing MS-SMC calculations.

This module provides user-friendly functions that accept either
ToyTrees as input, or a genealogy embedding table as a DataFrame.
Within the functions the faster jit-compiled numpy array-based methods
from the smc module are called, which take the more complicated
TreeEmbedding class inputs. The latter are much faster and should be
used by high-level users.
"""

from typing import Mapping, Sequence
import numpy as np
from toytree import ToyTree
from ipcoal.smc.src.embedding import get_relationships, get_genealogy_embedding_arrays
from ipcoal.smc.src.ms_smc_tree_prob import (
    get_prob_tree_unchanged_given_b_and_tr_from_arrays,
    get_prob_tree_unchanged_given_b_from_arrays,
    get_prob_tree_unchanged_from_arrays,
)
from ipcoal.smc.src.ms_smc_topo_prob import (
    get_prob_topo_unchanged_given_b_and_tr_from_arrays,
    get_prob_topo_unchanged_given_b_from_arrays,
    get_prob_topo_unchanged_from_arrays,
)

# from ipcoal.smc.src.embedding import TreeEmbedding, TopologyEmbedding
# from ipcoal.smc.src.ms_smc_tree_prob import (
#     get_fast_tree_changed_lambda,
# )
# from ipcoal.smc.src.ms_smc_topo_prob import (
#     get_fast_topo_changed_lambda,
# )
# from ipcoal.smc.src.likelihood import (
#     get_tree_distance_loglik,
#     get_topo_distance_loglik,
# )


__all__ = [
    "get_prob_tree_unchanged_given_b_and_tr",
    "get_prob_topo_unchanged_given_b_and_tr",
    "get_prob_tree_unchanged_given_b",
    "get_prob_topo_unchanged_given_b",
    "get_prob_tree_unchanged",
    "get_prob_topo_unchanged",
    #
    # "get_lambda_waiting_distance_to_recomb_event",
    # "get_lambda_waiting_distance_to_tree_change",
    # "get_lambda_waiting_distance_to_topo_change",
]


def get_prob_tree_unchanged_given_b_and_tr(
    species_tree: ToyTree,
    genealogy: ToyTree,
    imap: Mapping[str, Sequence[str]],
    branch: int,
    time: float,
) -> float:
    """Return probability of tree-unchanged given a gene tree embedded
    in a species and tree and the branch and timing of recombination.

    $P(tree-unchanged | S,G,B,tr)$

    Parameters
    ----------
    species_tree: ToyTree
        A species tree with 'Ne' feature set on each Node.
    genealogy: ToyTree
        A genealogy that can be embedded in the species tree.
    imap: Dict
        A dict mapping species tree tip names to a list of genealogy
        tip names to map samples to species.
    branch: int
        Index of branch in the genealogy on which recombination occurs.
    time: float
        Time at which recombination occurs on branch b.

    Examples
    --------
    >>> S, G, I = ipcoal.msc.get_test_data()
    >>> get_prob_tree_unchanged_given_b_and_tr(S, G, I, 0, 500)
    >>> # 0.2583966907988009
    """
    emb, enc = get_genealogy_embedding_arrays(species_tree, genealogy, imap)
    return get_prob_tree_unchanged_given_b_and_tr_from_arrays(emb[0], enc[0], branch, time)


def get_prob_topo_unchanged_given_b_and_tr(
    species_tree: ToyTree,
    genealogy: ToyTree,
    imap: Mapping[str, Sequence[str]],
    branch: int,
    time: float,
) -> float:
    """Return probability of tree-unchanged given a gene tree embedded
    in a species and tree and the branch and timing of recombination.

    $P(tree-unchanged | S,G,B,tr)$

    Parameters
    ----------
    species_tree: ToyTree
        A species tree with 'Ne' feature set on each Node.
    genealogy: ToyTree
        A genealogy that can be embedded in the species tree.
    imap: Dict
        A dict mapping species tree tip names to a list of genealogy
        tip names to map samples to species.
    branch: int
        Index of branch in the genealogy on which recombination occurs.
    time: float
        Time at which recombination occurs on branch b.

    Examples
    --------
    >>> S, G, I = ipcoal.msc.get_test_data()
    >>> get_prob_topo_unchanged_given_b_and_tr(S, G, I, 0, 500)
    >>> # 0.616289748348664
    """
    sister = genealogy[branch].get_sisters()[0].idx
    parent = genealogy[branch].up.idx
    emb, enc = get_genealogy_embedding_arrays(species_tree, genealogy, imap)
    return get_prob_topo_unchanged_given_b_and_tr_from_arrays(emb[0], enc[0], branch, sister, parent, time)


def get_prob_tree_unchanged_given_b(
    species_tree: ToyTree,
    genealogy: ToyTree,
    imap: Mapping[str, Sequence[str]],
    branch: int,
) -> float:
    """Return probability of tree-unchanged given a gene tree embedded
    in a species and tree integrated over all positions on a branch
    where recombination could occur.

    $P(tree-unchanged | S,G,B)$

    Parameters
    ----------
    species_tree: ToyTree
        A species tree with 'Ne' feature set on each Node.
    genealogy: ToyTree
        A genealogy that can be embedded in the species tree.
    imap: Dict
        A dict mapping species tree tip names to a list of genealogy
        tip names to map samples to species.
    branch: int
        Index of branch in the genealogy on which recombination occurs.

    Examples
    --------
    >>> S, G, I = ipcoal.msc.get_test_data()
    >>> get_prob_tree_unchanged_given_b(S, G, I, 0)
    >>> # 0.16069559114409554
    """
    emb, enc = get_genealogy_embedding_arrays(species_tree, genealogy, imap)
    return get_prob_tree_unchanged_given_b_from_arrays(emb[0], enc[0], branch)


def get_prob_topo_unchanged_given_b(
    species_tree: ToyTree,
    genealogy: ToyTree,
    imap: Mapping[str, Sequence[str]],
    branch: int,
) -> float:
    """Return probability of tree-unchanged given a gene tree embedded
    in a species and tree integrated over all positions on a branch
    where recombination could occur.

    $P(tree-unchanged | S,G,B)$

    Parameters
    ----------
    species_tree: ToyTree
        A species tree with 'Ne' feature set on each Node.
    genealogy: ToyTree
        A genealogy that can be embedded in the species tree.
    imap: Dict
        A dict mapping species tree tip names to a list of genealogy
        tip names to map samples to species.
    branch: int
        Index of branch in the genealogy on which recombination occurs.

    Examples
    --------
    >>> S, G, I = ipcoal.msc.get_test_data()
    >>> get_prob_topo_unchanged_given_b(S, G, I, 0)
    >>> # 0.5506091927430934
    """
    sister = genealogy[branch].get_sisters()[0].idx
    parent = genealogy[branch].up.idx
    emb, enc = get_genealogy_embedding_arrays(species_tree, genealogy, imap)
    return get_prob_topo_unchanged_given_b_from_arrays(emb[0], enc[0], branch, sister, parent)


def get_prob_tree_unchanged(
    species_tree: ToyTree,
    genealogy: ToyTree,
    imap: Mapping[str, Sequence[str]],
) -> float:
    """Return probability of tree-unchanged given a gene tree embedded
    in a species and tree integrated over all branches on which
    recombination could occur.

    $P(tree-unchanged | S,G)$

    Parameters
    ----------
    species_tree: ToyTree
        A species tree with 'Ne' feature set on each Node.
    genealogy: ToyTree
        A genealogy that can be embedded in the species tree.
    imap: Dict
        A dict mapping species tree tip names to a list of genealogy
        tip names to map samples to species.

    Examples
    --------
    >>> S, G, I = ipcoal.msc.get_test_data()
    >>> get_prob_tree_unchanged_given_b(S, G, I)
    >>> # 0.4139694696101504
    """
    emb, enc = get_genealogy_embedding_arrays(species_tree, genealogy, imap)
    barr = np.array([i._dist for i in genealogy[:-1]])
    sumlen = sum(barr)
    return get_prob_tree_unchanged_from_arrays(emb[0], enc[0], barr, sumlen)


def get_prob_topo_unchanged(
    species_tree: ToyTree,
    genealogy: ToyTree,
    imap: Mapping[str, Sequence[str]],
) -> float:
    """Return probability of topo-unchanged given a gene tree embedded
    in a species and tree integrated over all positions on a branch
    where recombination could occur.

    $P(topo-unchanged | S,G,B)$

    Parameters
    ----------
    species_tree: ToyTree
        A species tree with 'Ne' feature set on each Node.
    genealogy: ToyTree
        A genealogy that can be embedded in the species tree.
    imap: Dict
        A dict mapping species tree tip names to a list of genealogy
        tip names to map samples to species.
    branch: int
        Index of branch in the genealogy on which recombination occurs.

    Examples
    --------
    >>> S, G, I = ipcoal.msc.get_test_data()
    >>> get_prob_topo_unchanged(S, G, I, 0)
    >>> # 0.6921485857577693
    """
    emb, enc = get_genealogy_embedding_arrays(species_tree, genealogy, imap)
    barr = np.array([i._dist for i in genealogy[:-1]])
    sarr = sum(barr)
    rarr = get_relationships([genealogy])[0]
    return get_prob_topo_unchanged_from_arrays(emb[0], enc[0], barr, sarr, rarr)


if __name__ == "__main__":

    import toytree
    import ipcoal

    BRANCH = 0
    SISTER = 4
    PARENT = 5
    TIME = 500

    SPTREE, GTREE, IMAP = ipcoal.msc.get_test_data()
    SPTREE, GTREE, IMAP = ipcoal.smc.src.utils.get_test_data()
    print(IMAP)

    p = get_prob_tree_unchanged_given_b_and_tr(SPTREE, GTREE, IMAP, BRANCH, TIME)
    print(f"Prob(tree-unchanged | S, G, b=0, tr=500) = {p}\n")

    p = get_prob_topo_unchanged_given_b_and_tr(SPTREE, GTREE, IMAP, BRANCH, TIME)
    print(f"Prob(topo-unchanged | S, G, b=0, tr=500) = {p}\n")

    p = get_prob_tree_unchanged_given_b(SPTREE, GTREE, IMAP, BRANCH)
    print(f"Prob(tree-unchanged | S, G, b=0) = {p}\n")

    p = get_prob_topo_unchanged_given_b(SPTREE, GTREE, IMAP, BRANCH)
    print(f"Prob(topo-unchanged | S, G, b=0) = {p}\n")

    p = get_prob_tree_unchanged(SPTREE, GTREE, IMAP)
    print(f"Prob(tree-unchanged | S, G) = {p}\n")

    p = get_prob_topo_unchanged(SPTREE, GTREE, IMAP)
    print(f"Prob(topo-unchanged | S, G) = {p}\n")

    # SPTREE = toytree.tree("((A,B),C);")
    # SPTREE.set_node_data("height", inplace=True, default=0, data={3: 1000, 4: 3000})
    # SPTREE.set_node_data("Ne", default=1e3, inplace=True)
    # GTREE = toytree.tree("((0,(1,2)),3);")
    # GTREE.set_node_data("height", inplace=True, default=0, data={4: 2000, 5: 4000, 6: 5000})
    # IMAP = {"A": ['0'], "B": ['1', '2'], "C": ['3']}
    # TIME = 500
    # p = get_prob_tree_unchanged_given_b_and_tr(SPTREE, GTREE, IMAP, 0, TIME)
    # print(f"Figure S6 Prob(no-change | S, G, b, tr) = {p:.4f}\n")

