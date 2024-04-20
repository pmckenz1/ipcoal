#!/usr/bin/env python

"""Special embedding tables for SMC.

This starts by generating the same gene tree embedding table as used
in MSC, but includes additional arrays with info on branch lengths and
node relationships. All of this is joined into a class object.

Important
---------
The TreeEmbedding class stores the normal genealogy embedding table
in the `.table` attribute, but stores 2 * diploid Ne in the `.earr`
attribute which is the .table stored as a ndarray for fast computing.
... maybe could make these arrays private?
"""


from typing import Mapping, Sequence, Optional, List
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from numba import njit, prange
import toytree
from toytree import ToyTree
from loguru import logger
from ipcoal.msc import get_genealogy_embedding_arrays

logger = logger.bind(name="ipcoal")


class TreeEmbedding:
    """Class object to parse and store gene tree embedding data.
    """
    def __init__(
        self,
        species_tree: ToyTree,
        genealogies: Sequence[ToyTree],
        imap: Mapping[str, Sequence[str]],
        # topo_idxs: Optional[np.ndarray] = None,
        nproc: Optional[int] = None,
    ):
        # store inputs
        self.species_tree = species_tree
        self.genealogies = genealogies
        self.imap = imap
        # self.topo_idxs = topo_idxs
        self._nproc = nproc
        self._check_sptree()

        # data store
        self.emb: np.ndarray = None
        """: array of embedding values w/ 2 * neff. (ntrees, nintervals, nnodes) float64"""
        self.enc: np.ndarray = None
        """: array of Node IDs in each interval. (ntrees, ninterval, nnodes) bool_"""
        self.barr: np.ndarray = None
        """: array of branch lengths. (ntrees, nnodes) float64"""
        self.sarr: np.ndarray = None
        """: summed branch lenths array. (ntrees) float64"""
        self.rarr: np.ndarray = None
        """: Node relationships array. (ntrees, nnodes, 3). uint8"""
        self.run()

    def _check_sptree(self) -> None:
        """Raise exception if trees are not proper types."""
        # require input species tree to have Ne information.
        assert "Ne" in self.species_tree.features, (
            "species_tree must have an 'Ne' feature assigned to all Nodes.\n"
            "e.g., sptree.set_node_data('Ne', default=10000, inplace=True).")

    def _get_genealogies(self) -> Sequence[ToyTree]:
        """Parse gene tree input to a list of ToyTrees."""
        if isinstance(self.genealogies, ToyTree):
            self.genealogies = [self.genealogies]

        if isinstance(self.genealogies, str):
            self.genealogies = [toytree.tree(self.genealogies)]

        # if given a list of newick strings then parallelize tree parsing.
        if isinstance(self.genealogies[0], str):
            chunksize = 1_000
            rasyncs = {}
            with ProcessPoolExecutor(max_workers=self._nproc) as pool:
                for chunk in range(0, len(self.genealogies), chunksize):
                    args = (self.genealogies[chunk: chunk + chunksize],)
                    rasyncs[chunk] = pool.submit(toytree.mtree, *args)

            mtrees = [rasyncs[i].result() for i in sorted(rasyncs)]
            genealogies = mtrees[0].treelist
            for i in mtrees[1:]:
                genealogies += i.treelist
            self.genealogies = genealogies

        if isinstance(self.genealogies[0], ToyTree):
            self.genealogies = self.genealogies
        else:
            raise TypeError(
                "genealogies input must be one or more ToyTree or str types.")

    def _get_embedding_table_parallel(self, chunksize: int = 500) -> np.ndarray:
        """Return embedding arrays."""
        # non-parallel return
        if self._nproc == 1:
            return get_genealogy_embedding_arrays(
                self.species_tree, self.genealogies, self.imap)

        # send N trees at a time to parallel engines
        rasyncs = {}
        with ProcessPoolExecutor(max_workers=self._nproc) as pool:
            for chunk in range(0, len(self.genealogies), chunksize):
                kwargs = dict(
                    species_tree=self.species_tree,
                    genealogies=self.genealogies[chunk: chunk + chunksize],
                    imap=self.imap,
                )
                rasyncs[chunk] = pool.submit(get_genealogy_embedding_arrays, **kwargs)

        # collect parallel results
        embs = []
        encs = []
        for key in sorted(rasyncs):
            emb, enc = rasyncs[key].result()
            embs.append(emb)
            encs.append(enc)
        emb = np.concatenate(embs)
        enc = np.concatenate(encs)
        return emb, enc

    def _get_relationship_table_parallel(self, chunksize: int = 500) -> np.ndarray:
        """Return an array with relationships among nodes in each genealogy.

        The returned table is used in likelihood calculations for the
        waiting distance to topology-change events.
        """
        # non parallel return
        if self._nproc == 1:
            return get_relationships(self.genealogies)

        # send N trees at a time to engines
        rasyncs = {}
        with ProcessPoolExecutor(max_workers=self._nproc) as pool:
            for chunk in range(0, len(self.genealogies), chunksize):
                trees = self.genealogies[chunk: chunk + chunksize]
                rasyncs[chunk] = pool.submit(get_relationships, trees)

        rtables = []
        for key in sorted(rasyncs):
            rtable = rasyncs[key].result()
            rtables.append(rtable)
        return np.vstack(rtables)

    def get_data(self) -> List[np.ndarray]:
        return [self.emb, self.enc, self.barr, self.sarr, self.rarr]

    def run(self):
        """Fill the data arrays using parallel processing."""
        # parse self.genealogies as a List[Toytree]
        self._get_genealogies()
        logger.debug('parsing gtree inputs done')

        # extract embeddings in parallel
        self.emb, self.enc = self._get_embedding_table_parallel()
        logger.debug('filling embedding table done')

        # get summed edge lengths
        self.barr = get_super_lengths_table_jit(self.emb, self.enc)
        self.sarr = self.barr.sum(axis=1)
        logger.debug('filling edge lengths table done')

        # get relationships table
        self.rarr = self._get_relationship_table_parallel()
        logger.debug('filling relationships table done')

    def _update_neffs(self, params: np.ndarray) -> None:
        """Update neff values in the embedding table.

        The length of the array must be the same length as the number
        of populations. values are assigned to populations based on
        the species tree interval labels in the embedding table. This
        is used during likelihood optimization.
        """
        self.emb = _jit_update_neffs(self.emb, params)

    def get_table(self, gidx: int = 0) -> pd.DataFrame:
        """Return a genealogy embedding table for a specific genealogy"""
        return pd.DataFrame(
            self.emb[gidx],
            columns=['start', 'stop', 'st_node', 'neff', 'nedges', 'dist', 'gidx'],
        )


@njit
def _jit_update_neffs(emb: np.ndarray, params: np.ndarray) -> None:
    """Update neff values in the embedding table.

    The length of the array must be the same length as the number
    of populations. values are assigned to populations based on
    the species tree interval labels in the embedding table.
    """
    for tidx in range(emb.shape[0]):
        arr = emb[tidx]
        for pidx in range(params.shape[0]):
            mask = arr[:, 2] == pidx
            arr[mask, 3] = params[pidx]
    return emb


@njit(parallel=True)
def _jit_update_neff(emb: np.ndarray, idx: int, neff: float) -> np.ndarray:
    """
    Update NEFF with no parallel race condition. Tested. This also does
    NOT change the original array, it returns a copy.
    """
    arr = np.zeros(emb.shape, dtype=np.float64)
    for lidx in prange(emb.shape[0]):
        a = emb[lidx].copy()
        a[a[:, 2] == idx, 3] = neff
        arr[lidx] = a
    return arr


@njit
def _old_jit_update_neff(emb: np.ndarray, idx: int, neff: float) -> None:
    """Update a single neff value in the embedding table.

    The length of the array must be the same length as the number
    of populations. values are assigned to populations based on
    the species tree interval labels in the embedding table.
    """
    for tidx in range(emb.shape[0]):
        arr = emb[tidx]
        arr[arr[:, 2] == idx, 3] = neff
    return emb


@njit
def _jit_update_2pop_tau(emb: np.ndarray, tau: float) -> None:
    """Update population assignments (st_node) in the embedding table.

    This function is only currently written to support a two-population
    model. Further developments are needed for more complex trees. This
    will update the 'st_node' value for every interval in a tree based
    on the 'tau' value provided, representing the divergence time of
    the two populations.
    """
    for tidx in range(emb.shape[0]):
        arr = emb[tidx]
        arr[arr[0] > tau] = 2
    return emb


def _embedding_move_tau_up(emb: np.ndarray, tau: float):
    """

    Any intervals in pop2 that ended below tau are coal events occuring
    in pop2. These are now coal events that must occur in either pop0
    or pop1 depending on the edge identities from the encoding table.
    The nedges values can also change since these are divided from
    edges that may occur in the other descendant lineage.
    """




    # def get_waiting_distance_likelihood(
    #     self,
    #     event_type: int,
    #     distances: np.ndarray,
    #     neffs: Mapping[int, float],
    #     recombination_rate: float,
    # ) -> float:
    #     """Return -loglik of ARG interval lengths under the MS-SMC.

    #     Parameters
    #     ----------
    #     embedding_arr: TreeEmbedding
    #         A TreeEmbedding object.
    #     event_type: int
    #         Indicator of the recombination event type that occurs
    #         between trees that are in the TreeEmbedding object.
    #         0 = any recombination event
    #         1 = tree-change event
    #         2 = topology-change event
    #     distances: np.ndarray
    #         An array of waiting distances between events (data).
    #     recombination_rate: float
    #         per site per generation recombination rate (param).
    #     neffs: np.ndarray
    #         Diploid Ne values for each species tree interval in idx
    #         order of the species tree. Note that using this arg changes
    #         the Embedding table Ne values in place.

    #     Examples
    #     --------
    #     >>> sptree = toytree.rtree.imbtree(3, treeheight=1e5)
    #     >>> model = ipcoal.Model(sptree, nsamples=3, Ne=2e4)
    #     >>> model.sim_trees(nsites=1e5)
    #     >>> embedding = TreeEmbedding(
    #     >>>     model.tree, model.df.genealogy, model.get_imap_dict())
    #     >>> embedding.get_waiting_distance_likelihood()
    #     """
    #     _update_neffs(self.emb, neffs)
    #     return get_waiting_distance_likelihood(
    #         self, recombination_rate, distances, event_type)

    # def get_tree_likelihood(self, neffs: np.ndarray) -> float:
    #     """

    #     """
    #     _update_neffs(self.emb, neffs)
    #     return _get_msc_likelihood_from_embedding()

    # def get_ms_smc_ARG_likelihood(
    #     self,
    #     distances: np.ndarray,
    #     neffs: Mapping[int, float],
    #     recombination_rate: float,
    # ) -> float:
    #     """Return -loglik of an ARG under the MS-SMC.

    #     This likelihood is calculated using both the coalescent times
    #     within each genealogy as well as the lengths of intervals
    #     between recombination events. For the latter it combines the
    #     likelihood of observed three distances of events: any recomb
    #     event, a tree-change event, and a topology-change event.

    #     The loglik of an ARG (G,D) given a parameterized species tree
    #     (θ) is the sum loglik of trees and the three distances metrics:
    #     recomb-event (Dr), tree-change (Dg) and topology-change (Dt),
    #     where Gg and Gt are the subsets of genealogies at interval
    #     breakpoints representing tree-change or topology-change events.

    #     >>> L(θ|G,D) = L(G|θ) + L(Dr|θ,G,r) + L(Dg|θ,Gg,r) + L(Dt|θ,Gt,r)

    #     Parameters
    #     ----------
    #     embedding_arr: TreeEmbedding
    #         A TreeEmbedding object.
    #     distances: np.ndarray
    #         An array of waiting distances between events (data).
    #     recombination_rate: float
    #         per site per generation recombination rate (param).
    #     neffs: np.ndarray
    #         Diploid Ne values for each species tree interval in idx
    #         order of the species tree. Note that using this arg changes
    #         the Embedding table Ne values in place.

    #     Examples
    #     --------
    #     >>> sptree = toytree.rtree.imbtree(3, treeheight=1e5)
    #     >>> model = ipcoal.Model(sptree, nsamples=3, Ne=2e4)
    #     >>> model.sim_trees(nsites=1e5)
    #     >>> embedding = TreeEmbedding(
    #     >>>     model.tree, model.df.genealogy, model.get_imap_dict())
    #     >>> embedding.get_waiting_distance_likelihood()
    #     """


@njit  # (parallel=True)
def get_super_lengths_table_jit(emb: np.ndarray, enc: np.ndarray) -> np.ndarray:
    """Return array of (ntrees, nnodes - 1) w/ all edge lengths.

    jit-compiled.
    """
    ntrees = emb.shape[0]
    nnodes = enc.shape[2]
    barr = np.zeros((ntrees, nnodes - 1), dtype=np.float64)

    # for gidx in prange(ntrees):
    for gidx in range(ntrees):
        garr = emb[gidx]
        genc = enc[gidx]
        # iterate over each node
        for nidx in range(nnodes - 1):
            masked = garr[genc[:, nidx], :]
            low = masked[:, 0].min()
            top = masked[:, 1].max()
            barr[gidx, nidx] = top - low
    return barr


def get_relationships(trees: Sequence[ToyTree]) -> np.ndarray:
    """Return an array with the rows of [node, sister, parent] IDs."""
    ntrees = len(trees)
    nnodes = trees[0].nnodes
    rarr = np.zeros((ntrees, nnodes - 1, 3), dtype=np.uint8)
    for tidx, tree in enumerate(trees):
        for nidx, node in enumerate(tree[:-1]):
            rarr[tidx, nidx] = (
                nidx, node.get_sisters()[0]._idx, node._up._idx
            )
    return rarr


# @njit
# def get_genealogy_embedding_edge_path(garr: np.ndarray, bidx: int) -> np.ndarray:
#     """Return intervals of an embedding table for one gtree branch.

#     The embedding table must include the Node IDs, e.g., it must come
#     from TreeEmbedding, TopologyEmbedding, or by using the arg
#     encode=True to `get_genealogy_embedding_table`.
#     """
#     idxs = np.nonzero(garr[:, 7 + bidx])[0]
#     return garr[idxs, :]


# EXPERIMENTAL IDEA OF USING JIT CLASS
# ------------------------------------
# from numba.experimental import jitclass
# from numba import float64, int64, int8

# @jitclass([
#     ("garr", float64[:,:]),
#     ("barr", float64[:,:]),
#     ("sarr", float64[:]),
#     ("rarr", int8[:,:]),
#     ("tidx", int64),
# ])
#
# class Embedding:
#     garr: np.ndarray
#     barr: np.ndarray
#     sarr: np.ndarray
#     rarr: np.ndarray
#     tidx: int64
#     def __init__(self, garr, barr, sarr, rarr, tidx):
#         self.garr = garr
#         self.barr = barr
#         self.sarr = sarr
#         self.rarr = rarr
#         self.tidx = tidx
#
#     def get_tree_data(self, tidx: int) -> 'Embedding':
#         """Return Embedding for a single genealogy."""
#         # mask =
#         return Embedding(self.garr[:, 10:20], self.barr[:, 10:20], self.sarr, self.rarr, tidx)


if __name__ == "__main__":

    import ipcoal
    ipcoal.set_log_level("DEBUG")
    from ipcoal.smc.src.utils import get_ms_smc_data_from_model

    sptree = toytree.rtree.imbtree(4, treeheight=1e6)
    model = ipcoal.Model(sptree, Ne=1e5, nsamples=2)
    model.sim_trees(1, 1e5)
    logger.debug('simulation done')
    imap = model.get_imap_dict()

    tree_dists, topo_dists, topo_idxs, trees = get_ms_smc_data_from_model(model)

    t1 = TreeEmbedding(model.tree, trees, imap, nproc=1)
    print('emb', t1.emb.shape, t1.emb.dtype)
    print('emc', t1.enc.shape, t1.enc.dtype)
    print('barr', t1.barr.shape, t1.barr.dtype)
    print('sarr', t1.sarr.shape, t1.sarr.dtype)
    print('rarr', t1.rarr.shape, t1.rarr.dtype)

    print("")
    t2 = TreeEmbedding(model.tree, [trees[i] for i in topo_idxs], imap)
    # t2.emb = _jit_update_neff(t2.emb, 0, 1.5e5)
    print('emb', t2.emb.shape, t2.emb.dtype)
    print('emc', t2.enc.shape, t2.enc.dtype)
    print('barr', t2.barr.shape, t2.barr.dtype)
    print('sarr', t2.sarr.shape, t2.sarr.dtype)
    print('rarr', t2.rarr.shape, t2.rarr.dtype)
    print(t2.get_table(0))
