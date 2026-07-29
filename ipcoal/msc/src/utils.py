#!/usr/bin/env python

"""Convenience functions for coalescent simulations.

This is experimental, not yet used anywhere.

"""

from typing import TypeVar, Tuple, Dict, Sequence, Optional
import toytree
import ipcoal

ToyTree = TypeVar("ToyTree")

__all__ = ['get_embedded_genealogy', 'get_test_data', 'get_test_model']


def get_embedded_genealogy(tree: ToyTree, **kwargs) -> Tuple[ToyTree, Dict[str, Sequence[str]]]:
    """Return (genealogy, imap) for one simulated embedded genealogy.

    Arguments to ipcoal.Model can be modified using kwargs. This
    is a convenience function that performs the following steps:
    >>> model = ipcoal.Model(tree, **kwargs)
    >>> model.sim_trees(1)
    >>> gtree = toytree.tree(model.df.genealogy[0])
    >>> imap = model.get_imap()
    >>> return (gtree, imap)

    Parameters
    ----------
    tree: ToyTree
        A ToyTree object as a species tree. You can either set 'Ne'
        as a feature on Nodes or enter `Ne=...` as a kwarg.
    kwargs:
        All arguments to `ipcoal.Model` are supported, as well as
        the optional argument 'diploid=...` for `Model.get_imap'

    Example
    -------
    >>> stree = toytree.rtree.unittree(ntips=8, treeheight=1e6, seed=12345)
    >>> gtree = ipcoal.get_embedded_genealogy(stree, Ne=1000, nsamples=2)
    """
    model = ipcoal.Model(tree, **kwargs)
    model.sim_trees(1)
    gtree = toytree.tree(model.df.genealogy[0])
    imap = model.get_imap_dict(**{i: j for i, j in kwargs.items() if i == "diploid"})
    return (gtree, imap)


def get_test_model(seed: Optional[int] = None):
    """Return an ipcoal.Model for a simple test scenario used in the
    MS-SMC manuscript.

    Four tip species tree (((A,B),C),D); with divtimes at 2e5, 4e5,
    and 6e5, and a constant Ne of 1e5. The genealogy samples include
    3 in A, 2 in B, and 1 from C and D.
    """
    # Set up a species tree with edge lengths in generations
    SPTREE = toytree.tree("(((A,B),C),D);")
    SPTREE.set_node_data("height", inplace=True, default=0, data={
        4: 200_000, 5: 400_000, 6: 600_000,
    })

    # Set constant or variable Ne on species tree edges
    SPTREE.set_node_data("Ne", inplace=True, default=100_000)
    model = ipcoal.Model(
        SPTREE,
        nsamples={"A": 3, "B": 2, "C": 1, "D": 1},
        seed_trees=seed,
        seed_mutations=seed,
    )
    return model


def get_test_data(nloci: int = 0, nsites: int = 1, seed: Optional[int] = None):
    """Returns (sptree, gtree, imap) for one or more genealogies
    embedded in a species tree.

    Parameters
    ----------
    nloci: int
        Number of independent ARGs to simulate. If 0 then a single
        fixed example gene tree is returned, else nloci stochastic
        ARGs are simulated.
    nsites: int
        The length of simulated ARGs.
    seed: int, RNG, or None
        Random seed.
    """
    model = get_test_model(seed)

    # do not simulate trees, manually create the one example tree.
    if not nloci:
        # Setup a genealogy embedded in the species tree (see below to
        # instead simulate one but here we just create it from newick.)
        GTREE = toytree.tree("(((0,1),(2,(3,4))),(5,6));")
        GTREE.set_node_data("height", inplace=True, default=0, data={
            7: 100_000, 8: 120_000, 9: 300_000, 10: 450_000, 11: 650_000, 12: 800_000,
        })

        # Setup a map of species names to a list sample names
        IMAP = {
            "A": ['0', '1', '2'],
            "B": ['3', '4'],
            "C": ['5',],
            "D": ['6',],
        }
        return model.tree, GTREE, IMAP

    # simulate genealogies
    model.sim_trees(nloci, nsites)
    gtrees = toytree.mtree(model.df.genealogy).treelist
    return model.tree, gtrees, model.get_imap_dict()


if __name__ == "__main__":

    # get a single gtree embedded in a species tree
    _stree = toytree.rtree.unittree(ntips=8, treeheight=1e6, seed=12345)
    _gtree = ipcoal.get_embedded_genealogy(_stree, Ne=1000, nsamples=2)
    print(_gtree)

    # get the test data set
    SPTREE, GTREE, IMAP = get_test_data()
    t = ipcoal.msc.get_genealogy_embedding_table(SPTREE, GTREE, IMAP)
    print(t)

    #
    t = ipcoal.msc.get_genealogy_embedding_table(SPTREE, GTREE, IMAP)
    print(t)
