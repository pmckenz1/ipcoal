#!/usr/bin/env python

"""Store genealogy embedding information in numpy arrays.

This allows for faster jit-compiled calculation of E[waiting distances].
"""


from typing import Mapping, Sequence, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import toytree

from ipcoal.smc import get_genealogy_embedding_table
from ipcoal.smc.likelihood.utils import iter_unique_topologies_from_genealogies


################################################################
################################################################
# GET EMBEDDING DATA
################################################################
################################################################

class Embedding:
    """Container class for storing genealogy embedding data as arrays.

    The embedding table is accessible from the `.table` attribute
    of the Embedding class. This object also stores additional data 
    as numpy arrays to use in fast jit-compiled computations. These 
    arrays include the following:
        1. concatenated genealogy embedding table
        2. all edge lengths across all genealogies
        3. summed edge lengths for each genealogy
        4. genealogy edge relationships

    The embedding class object is intended to be passed to one of
    the following functions:
        - `get_waiting_distance_to_tree_change_rates` 
        - `get_waiting_distance_to_topology_change_rates`
        - `get_tree_distance_loglik`
        - `get_topology_distance_loglik`        

    Parameters
    ----------
    species_tree: ToyTree
        A toytree object with diploid Ne values assigned as features 
        to Nodes.
    genealogies: Sequence[ToyTree]
        A collection of ToyTree objects representing genealogies that
        are embedded in the species tree.
    imap: Dict[str, Sequence[str]] 
        A dictionary mapping species tree tips names to genealogy
        tip names. 
    nproc: int
        Number of processors to use for parallelization. Default
        is to use all available.

    Examples
    --------
    >>> sptree = toytree.rtree.baltree(2)
    >>> sptree.mod.edges_scale_to_root_height(1e6, inplace=True)
    >>> model = ipcoal.Model(sptree, Ne=150_000, nsamples=4, recomb=2e-8)
    >>> model.sim_trees(1, 1e5)
    >>> imap = model.get_imap_dict()
    >>> genealogies = toytree.mtree(model.df.genealogy)
    >>> edata = Embedding(sptree, genealogies, imap)
    >>> loglik = get_tree_distance_loglik(...)
    """
    def __init__(
        self,
        species_tree: toytree.ToyTree,
        genealogies: Sequence[toytree.ToyTree],
        imap: Mapping[str, Sequence[str]],
        nproc: Optional[int]=None,
        ):
        # store inputs
        self.species_tree = species_tree
        self.genealogies = genealogies
        self.imap = imap
        self.nproc = nproc

        # require input species tree to have Ne information.
        assert "Ne" in species_tree.features, (
            "species_tree must have an 'Ne' feature assigned to all Nodes.\n"
            "e.g., sptree.set_node_data('Ne', default=10000, inplace=True).")

        self.table: pd.DataFrame = None
        self.earr: np.ndarray = None
        self.barr: np.ndarray = None
        self.sarr: np.ndarray = None 
        self.rarr: np.ndarray = None
        self.run()

    def _get_genealogies(self):
        return self.genealogies

    def run(self):
        """..."""
        # TopologyEmbedding subclass selects subset here.
        genealogies = self._get_genealogies()

        # store human-readable embedding table
        self.table = _parallel_get_multigenealogy_embedding_table(
            self.species_tree, genealogies, self.imap, self.nproc)

        # note earr array stores Ne as 2X, and also when updated !!!!
        self.earr = self.table.values.astype(float)
        self.earr[:, 3] *= 2

        # store edge lengths individually and summed
        self.barr = _get_super_lengths(self.table)
        self.sarr = self.barr[:, :-1].sum(axis=1)

        # store Node relationships
        self.rarr = _get_relationship_table(genealogies)

    def get_data(self) -> Tuple:
        """Return tuples of array data for waiting distance computations."""
        return (self.earr, self.barr, self.sarr)


class TreeEmbedding(Embedding):
    """Container for genealogy embedding data."""

class TopologyEmbedding(Embedding):
    """Container for genealogy topology embedding data."""
    def _get_genealogies(self):
        return list(iter_unique_topologies_from_genealogies(
            self.genealogies, 
            # average_branch_lengths=self.average_branch_lengths),
        ))

    def get_data(self):
        return (self.earr, self.barr, self.sarr, self.rarr)


def _parallel_get_multigenealogy_embedding_table(
    species_tree: toytree.ToyTree,
    genealogies: Sequence[toytree.ToyTree],
    imap: Mapping[str, Sequence[str]],
    nproc: Optional[int]=None,
    ) -> pd.DataFrame:
    """Return a concatenated DataFrame of genealogy embedding tables.

    Note
    ----
    This function is provided primarily for didactic reasons to make
    the general code and framework easier to understand. In practice,
    this function is called within the function `get_data()`, which
    returns a tuple of numpy arrays that are used for likelihood 
    calculations under the MS-SMC'. 
    
    Parameters
    ----------
    species_tree: ToyTree
        A toytree object with diploid Ne values assigned as features 
        to Nodes.
    genealogies: Sequence[ToyTree]
        A collection of ToyTree objects representing genealogies that
        are embedded in the species tree.
    imap: Dict[str, Sequence[str]] 
        A dictionary mapping species tree tips names to genealogy
        tip names. 
    nproc: int
        Number of processors to use for parallelization. Default
        is to use all available.
    """
    # parallelize calculation of etables
    rasyncs = {}
    with ProcessPoolExecutor(max_workers=nproc) as pool:
        for gidx, gtree in enumerate(genealogies):
            args = (species_tree, gtree, imap)
            rasyncs[gidx] = pool.submit(get_genealogy_embedding_table, *args)
    etables = [rasyncs[gidx].result() for gidx in sorted(rasyncs)]

    # concatenate etables
    etable = _concat_embedding_tables(etables)
    return etable


def _concat_embedding_tables(etables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Return concatenated embedding table labeled by genealogy and branch.

    This could be written faster with numpy...
    """
    # check that etable is properly formatted
    ninodes = etables[0].iloc[-1, 6][0] + 1

    # iterate over each embedding table
    # pylint-disable: cell-var-from-loop
    btables = []
    for gidx, etable in enumerate(etables):
        # make a column for the genealogy index
        etable["gindex"] = gidx
        
        # make presence/absence column for each branch
        bidxs = np.zeros((etable.shape[0], ninodes), dtype=int)
        for bidx in range(ninodes):
            btable = etable[etable.edges.apply(lambda x: bidx in x)]  
            # record branches in this interval as 1.
            bidxs[btable.index, bidx] = 1
        bidxs = pd.DataFrame(bidxs, columns=range(ninodes))
        btable = pd.concat([etable, bidxs], axis=1)
        btable = btable.drop(columns=["coal", "edges"])
        btable.iloc[-1, [1, 5]] = int(1e12)
        btables.append(btable)
    return pd.concat(btables, ignore_index=True)


def _get_relationship_table(genealogies: Sequence[toytree.ToyTree]) -> np.ndarray:
    """Return an array with relationships among nodes in each genealogy.

    The returned table is used in likelihood calculations for the 
    waiting distance to topology-change events.
    """
    ntrees = len(genealogies)
    nnodes = genealogies[0].nnodes
    farr = np.zeros((ntrees, nnodes - 1, 3), dtype=int)
    for tidx, tree in enumerate(genealogies):
        for nidx, node in enumerate(tree):
            if not node.is_root():
                farr[tidx, nidx] = nidx, node.get_sisters()[0].idx, node.up.idx
    return farr


def _get_super_lengths(econcat: pd.DataFrame) -> np.ndarray:
    """Return array of shape=(ngenealogies, nnodes) with all edge lengths.

    Parameters
    ----------
    econcat: pd.DataFrame
        The concatenated genealogy embedding table.
    """
    gidxs = sorted(econcat.gindex.unique())
    nnodes = econcat.shape[1] - 7
    larr = np.zeros((len(gidxs), nnodes), dtype=float)

    for gidx in gidxs:
        # get etable for this genealogy
        garr = econcat[econcat.gindex == gidx].values
        ixs, iys = np.nonzero(garr[:, 7:])

        # iterate over nodes of the genealogy
        for bidx in range(nnodes):
            # get index of intervals with this branch
            idxs = ixs[iys == bidx]
            barr = garr[idxs, :]
            blow = barr[:, 0].min()
            btop = barr[:, 1].max()
            larr[gidx, bidx] = btop - blow
    return larr


if __name__ == "__main__":

    import ipcoal
    SPTREE = toytree.rtree.baltree(2).mod.edges_scale_to_root_height(1e6, include_stem=True)
    MODEL = ipcoal.Model(SPTREE, Ne=200_000, nsamples=4, seed_trees=123)
    MODEL.sim_trees(1, 1e5)
    GENEALOGIES = toytree.mtree(MODEL.df.genealogy)
    IMAP = MODEL.get_imap_dict()
    data = TopologyEmbedding(MODEL.tree, GENEALOGIES, IMAP)
    print(data.table)
