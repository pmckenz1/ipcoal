#!/usr/bin/env python

"""Parse an embedding table for MSC calculations.

# ['start', 'stop', 'st_node', 'neff', 'nedges', 'dist', 'gidx', ...]

Methods
-------
>>> get_genealogy_embedding_table(sptree, gtree, imap)
"""

from typing import Mapping, Sequence, Tuple
from loguru import logger
import numpy as np
import pandas as pd
import toytree

__all__ = ["get_genealogy_embedding_table", "get_genealogy_embedding_arrays"]
logger = logger.bind(name="ipcoal")
COLUMNS = ['start', 'stop', 'st_node', 'neff', 'nedges', 'dist', 'gidx']


def _get_fast_genealogy_embedding_table(
    species_tree: toytree.ToyTree,
    genealogy: toytree.ToyTree,
    imap: Mapping[str, Sequence[str]],
    gidx: int,
) -> np.ndarray:
    """Internal function to embed a single genealogy.

    This function is used within `get_genealogy_embedding_table()`.
    See the latter function for usage.
    """
    # store temporary results in a dict
    split_data = []
    edge_encode = []

    # dict to update tips to their ancestor if already coalesced.
    name_to_node = {i._name: i for i in genealogy[:genealogy.ntips]}

    # get table of gene tree node heights
    gt_node_heights = np.array([i._height for i in genealogy])

    # iterate over stree from tips to root storing stree node data
    for st_node in species_tree:
        # get all gtree names descended from this sptree interval
        gt_tips = set.union(*(set(imap[i]) for i in st_node.get_leaf_names()))

        # get gtree nodes descended from this interval.
        gt_nodes = set(name_to_node[i] for i in gt_tips)

        # get internal nodes in this TIME interval (coalescences)
        lower = st_node._height + 0.0001  # zero-align ipcoal bug
        upper = np.inf if st_node.is_root() else st_node._up._height
        nodes_in_time_slice = (gt_node_heights > lower) & (gt_node_heights < upper)

        # filter internal nodes must be ancestors of gt_nodes (in this interval)
        # get internal nodes in this interval by getting all nodes in this
        # time interval and requiring that all their desc are in gt_nodes.
        inodes = []
        for iidx in nodes_in_time_slice.nonzero()[0]:
            gt_node = genealogy[iidx]
            if not (set(gt_node.get_leaf_names()) - gt_tips):
                inodes.append(gt_node)

        # vars to be updated at coal events
        start = st_node._height
        edges = {i._idx for i in gt_nodes}

        # iterate over internal nodes
        for gt_node in sorted(inodes, key=lambda x: x._height):
            # add interval from start to first coal, or coal to next coal
            split_data.append([
                start,
                gt_node._height,
                st_node._idx,
                st_node.Ne,
                len(edges),
                0.,
                gidx,
            ])
            edge_encode.append(sorted(edges))

            # update counters and indexers
            start = gt_node._height
            edges.add(gt_node._idx)
            try:
                for child in gt_node._children:
                    edges.remove(child._idx)
            except KeyError as err:
                raise ValueError(
                    "gene tree cannot be embedded in species tree. "
                    f"Error embedding Node {gt_node._idx} in Species Tree interval {st_node._idx}."
                ) from err
            for tip in gt_node.get_leaves():
                name_to_node[tip.name] = gt_node

        # add non-coal interval
        split_data.append([
            start,
            st_node._up._height if st_node._up else np.inf,
            st_node._idx,
            st_node.Ne,
            len(edges),
            0.,
            gidx,
        ])
        edge_encode.append(sorted(edges))

    # to array and calculate dists from start and stop, not allowing
    # zero-length dist values
    arr = np.vstack(split_data)
    arr[:, 5] = (arr[:, 1] - arr[:, 0])
    return arr, edge_encode
    # create one-hot-encoded node IDs
    # encoding = np.zeros((len(edge_encode), nnodes), dtype=np.uint8)
    # for interval, nodes in enumerate(edge_encode):
    #     encoding[interval, nodes] = 1

    # return encoding
    # # format as a sleek float array
    # if not df:
    #     if encode:
    #         return np.c_[arr, encoding]
    #     return arr

    # # format as a nice (but slow) human readable dataframe
    # table = pd.DataFrame(data=split_data, columns=COLUMNS)
    # table.dist = table.stop - table.start
    # if encode:
    #     encoding = pd.DataFrame(encoding, columns=list(range(nnodes)))
    #     return pd.concat([table, encoding], axis=1)
    # return table


def get_genealogy_embedding_table(
    species_tree: toytree.ToyTree,
    genealogies: Sequence[toytree.ToyTree],
    imap: Mapping[str, Sequence[str]],
    encode: bool = False,
) -> pd.DataFrame:
    """Return a genealogy embedding table as a human-readable dataframe.

    The embedding table represents intervals in a species tree that
    have piece-wise constant coalescent rates and which are divided
    by coalescent events in the gene tree, or speciation events in
    the species tree.

    Raises ipcoalError if genealogy cannot be embedded in the species
    tree (e.g., coalescence earlier than species divergence allows).

    Parameters
    ----------
    species_tree: ToyTree
        A ToyTree with Ne values mapped to Nodes and edge lengths in
        units of generations.
    genealogies: ToyTree or Sequence[ToyTree]
        One or more ToyTrees representing gene trees embedded in the
        species tree with edge lengths in units of generations.
    imap: Dict[str, List[str]]
        A dict mapping gene tree tip names to species tree tips names.
    encode: bool
        If True genealogy Node IDs are included as one-hot-encoded in
        place of the 'edges' column.

    Examples
    --------
    >>> sptree = toytree.rtree.imbtree(ntips=4, treeheight=1e6)
    >>> model = ipcoal.Model(sptree, Ne=1e5, nsamples=2)
    >>> model.sim_trees(10)
    >>> imap = model.get_imap_dict()
    >>> gtrees = model.df.genealogy
    >>> table = get_genealogy_embedding_table(model.tree, gtrees, imap)
    """
    # get genealogies as Sequence[ToyTree]
    if isinstance(genealogies, (toytree.ToyTree, str)):
        genealogies = [genealogies]
    if isinstance(genealogies[0], str):
        genealogies = toytree.mtree(genealogies).treelist

    # iterate over genealogies
    garrs = []
    earrs = []
    for gidx, gtree in enumerate(genealogies):
        args = (species_tree, gtree, imap, gidx)
        garr, earr = _get_fast_genealogy_embedding_table(*args)
        garrs.append(garr)
        earrs.append(earr)

    # concatenate
    embedding = pd.DataFrame(np.vstack(garrs), columns=COLUMNS)
    embedding[["st_node", "nedges", "gidx"]] = embedding[["st_node", "nedges", "gidx"]].astype(int)

    # return 'edges' column with lists of Node IDs
    if not encode:
        embedding["edges"] = np.array(earrs, dtype=object).flatten()
        return embedding

    # else return one-hot-encoded columns (kinda slow, for didactic)
    nnodes = genealogies[0].nnodes
    encoding = np.zeros((embedding.shape[0], nnodes), dtype=np.uint8)
    i = 0
    for gidx, earr in enumerate(earrs):
        for nodes in earr:
            encoding[i, nodes] = 1
            i += 1
    return pd.concat([embedding, pd.DataFrame(encoding)], axis=1)


def get_genealogy_embedding_arrays(
    species_tree: toytree.ToyTree,
    genealogies: Sequence[toytree.ToyTree],
    imap: Mapping[str, Sequence[str]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a genealogy embedding table as two numpy arrays.

    See `get_genealogy_embedding_table` to generate a genealogy
    embedding table as a more human-readable dataframe. This function
    instead returns the embedding table as two arrays with shape and
    dtypes optimized for much faster computing.

    The first array is shape=(ntrees, nintervals, 6) dtype=float64 and
    contains the tree embedding values (e.g., dist, neff, nedges), the
    second array is shape=(ntrees, nintervals, nnodes) dtype=np.uint8
    and contains the gtree embedding Node IDs.

    The embedding table represents intervals in a species tree that
    have piece-wise constant coalescent rates and which are divided
    by coalescent events in the gene tree, or speciation events in
    the species tree.

    Raises ipcoalError if genealogy cannot be embedded in the species
    tree (e.g., coalescence earlier than species divergence allows).

    Parameters
    ----------
    species_tree: ToyTree
        A ToyTree with Ne values mapped to Nodes and edge lengths in
        units of generations.
    genealogies: ToyTree or Sequence[ToyTree]
        One or more ToyTrees representing gene trees embedded in the
        species tree with edge lengths in units of generations.
    imap: Dict[str, List[str]]
        A dict mapping gene tree tip names to species tree tips names.

    Examples
    --------
    >>> sptree = toytree.rtree.imbtree(ntips=4, treeheight=1e6)
    >>> model = ipcoal.Model(sptree, Ne=1e5, nsamples=2)
    >>> model.sim_trees(10)
    >>> imap = model.get_imap_dict()
    >>> gtrees = model.df.genealogy
    >>> a0, a1 = get_genealogy_embedding_table(model.tree, gtrees, imap)
    """
    # get genealogies as Sequence[ToyTree]
    if isinstance(genealogies, (toytree.ToyTree, str)):
        genealogies = [genealogies]
    if isinstance(genealogies[0], str):
        genealogies = toytree.mtree(genealogies).treelist

    # iterate over genealogies
    garrs = []
    earrs = []
    for gidx, gtree in enumerate(genealogies):
        args = (species_tree, gtree, imap, gidx)
        garr, earr = _get_fast_genealogy_embedding_table(*args)
        garrs.append(garr)
        earrs.append(earr)

    # concatenate
    embedding = np.array(garrs)

    # one-hot encode the node IDs
    nnodes = genealogies[0].nnodes
    encoding = np.zeros((len(genealogies), len(earr), nnodes), dtype=np.bool_)
    for gidx, earr in enumerate(earrs):
        for interval, nodes in enumerate(earr):
            encoding[gidx, interval, nodes] = True
    return embedding, encoding


if __name__ == "__main__":

    import ipcoal
    SPTREE = toytree.rtree.baltree(2, treeheight=1e6)
    MODEL = ipcoal.Model(SPTREE, Ne=200_000, nsamples=4, seed_trees=123)
    MODEL.sim_trees(2)
    GENEALOGIES = toytree.mtree(MODEL.df.genealogy)
    IMAP = MODEL.get_imap_dict()

    # human readable embedding (2D) mostly for debugging and teaching.
    print(get_genealogy_embedding_table(MODEL.tree, GENEALOGIES, IMAP, encode=1).head(10))
    print('...')

    # fast array embedding (3D)
    emb, enc = get_genealogy_embedding_arrays(MODEL.tree, GENEALOGIES, IMAP)
    print(emb.shape, emb.dtype)
    print(enc.shape, enc.dtype)
