#!/usr/bin/env python

"""Convenience functions for coalescent simulations.

This is experimental, not yet used anywhere.

"""

from typing import TypeVar, Tuple, Dict, Sequence
import toytree
import ipcoal

ToyTree = TypeVar("ToyTree")

__all__ = ['get_embedded_genealogy']


def get_embedded_genealogy(tree: ToyTree, **kwargs) -> Tuple[ToyTree, Dict[str,Sequence[str]]]:
    """Return (genealogy, imap) for one simulated embedded genealogy.
    
    Arguments to ipcoal.Model can be modified using kwargs. This 
    is a convenience function that performs the following steps:
    >>> model = ipcoal.Model(species_tree, **kwargs)
    >>> model.sim_trees(1)
    >>> gtree = toytree.tree(model.df.genealogy[0])
    >>> imap = model.get_imap()
    >>> return (gtree, imap)

    Parameters
    ----------
    tree: 
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
    imap = model.get_imap_dict(**{i:j for i,j in kwargs.items() if i == "diploid"})
    return (gtree, imap)


if __name__ == "__main__":
    _stree = toytree.rtree.unittree(ntips=8, treeheight=1e6, seed=12345)
    _gtree = ipcoal.get_embedded_genealogy(_stree, Ne=1000, nsamples=2)
    print(_gtree)
