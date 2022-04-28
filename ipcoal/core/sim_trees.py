#!/usr/bin/env python

"""Simulate `nloci` unlinked genealogies.

"""

from typing import TypeVar
import numpy as np
import pandas as pd
import msprime as ms
from ipcoal.utils.utils import IpcoalError

Model = TypeVar("Model")


def sim_trees(model: Model, nloci: int, nsites: int, precision: int=14) -> None:
    """Simulate unlinked genealogies. 

    See `ipcoal.Model.sim_trees` docstring.
    """
    # check conflicting args
    if model._recomb_is_map:
        if nsites:
            raise IpcoalError(
                "Both nsites and recomb_map cannot be used together since"
                "the recomb_map also specifies nsites. To use a recomb_map"
                "specify nsites=None.")
        nsites = model.recomb.sequence_length

    datalist = []
    for lidx in range(nloci):
        msgen = model._get_tree_sequence_generator(nsites)
        tree_seq = next(msgen)
        breaks = [int(i) for i in tree_seq.breakpoints()]
        starts = breaks[0:len(breaks) - 1]
        ends = breaks[1:len(breaks)]
        lengths = [i - j for (i, j) in zip(ends, starts)]

        data = pd.DataFrame({
            "start": starts,
            "end": ends,
            "nbps": lengths,
            "nsnps": 0,
            "tidx": 0,
            "locus": lidx,
            "genealogy": "",
            },
            columns=[
                'locus', 'start', 'end', 'nbps',
                'nsnps', 'tidx', 'genealogy'
            ],
        )

        # iterate over the index of the dataframe to sim for each genealogy
        for mstree in tree_seq.trees():
            # convert nwk to original names
            nwk = mstree.newick(node_labels=model.tipdict, precision=precision)
            data.loc[mstree.index, "genealogy"] = nwk
            data.loc[mstree.index, "tidx"] = mstree.index
        datalist.append(data)

        # store the tree_sequence
        if model.store_tree_sequences:
            model.ts_dict[lidx] = tree_seq

    # concatenate all of the genetree dfs
    data = pd.concat(datalist)
    data = data.reset_index(drop=True)
    model.df = data
    model.seqs = np.array([])


if __name__ == "__main__":

    import toytree
    import ipcoal

    TREE = toytree.rtree.unittree(ntips=6, treeheight=1e6)
    MODEL = ipcoal.Model(TREE, Ne=1e4)
    sim_trees(MODEL, 10, 10)
    print(MODEL.df)
