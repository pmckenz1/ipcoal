#!/usr/bin/env python

"""Simulate `nloci` unlinked genealogies.

Parallel code returns same result as non-parallel using rng_trees.
"""

from typing import TypeVar, Sequence, Tuple, Mapping
from concurrent.futures import ProcessPoolExecutor
from loguru import logger
import numpy as np
import pandas as pd
from ipcoal.utils.utils import IpcoalError
from tskit import TreeSequence

logger = logger.bind(name="ipcoal")
Model = TypeVar("Model")
COLUMNS = ['locus', 'start', 'end', 'nbps', 'nsnps', 'tidx', 'genealogy']


def sim_trees(
    model: Model,
    nloci: int,
    nsites: int,
    precision: int = 14,
    nproc: int = 1,
) -> None:
    """Simulate unlinked genealogies.

    Parallelization provides less than linear speed-up. Default is
    nproc=1. Only used and useful for very large simulations.
    """
    # clear any existing stored tree sequences
    nloci = int(nloci)
    model.ts_dict = {}
    model.seqs = np.array([])

    # check conflicting args
    if model._recomb_is_map:
        if nsites:
            raise IpcoalError(
                "Both nsites and recomb_map cannot be used together since"
                "the recomb_map also specifies nsites. To use a recomb_map"
                "specify nsites=None.")
        nsites = model.recomb.sequence_length

    # requires pickling TreeSeqs
    # if model.store_tree_sequences and (nproc > 1):
    #     nproc = 1
    #     logger.warning(
    #         "Cannot use nproc>1 with store_tree_sequences=True. Using nproc=1.")

    # run single core simulation. Stores results to Model and returns None
    if nproc == 1:
        data, ts_dict = _sim_trees(model, nloci, nsites, precision)
        model.df = data
        model.ts_dict = ts_dict
        model._reset_random_generators()
        return None

    # cache some model elements that are not pickle-able and not used.
    _old_substitution_model = model.subst_model
    rng_trees = model.rng_trees
    model.rng_trees = None
    model.rng_muts = None     # doesn't matter in sim_trees
    model.subst_model = None  # doesn't matter in sim_trees
    model.ts_dict = {}        # empty it

    # send simulate jobs in chunks
    chunksize = int(nloci / nproc) + (nloci % nproc)
    # chunksize = max(chunksize, 100)
    chunksize = max(chunksize, 1)
    # logger.info(f"simulating {chunksize} trees in parallel on {nproc} cpus")
    rasyncs = {}
    with ProcessPoolExecutor(max_workers=nproc) as pool:
        for chunk in range(0, nloci, chunksize):
            # get number of loci to sim on this engine
            low = chunk
            high = min(nloci, chunk + chunksize)
            chunkloci = high - low

            # generate seeds from rng_trees
            seeds = rng_trees.integers(2 ** 31, size=chunkloci)

            # send job to run
            args = (model, chunkloci, nsites, precision, seeds)
            rasyncs[low] = pool.submit(_sim_trees, *args)

    # store dataframe to Model and clear any existing sequences
    datalist = []
    for low, future in rasyncs.items():
        data, ts_dict = future.result()
        data.locus += low
        datalist.append(data)
        # advance key counters
        ts_dict = {i + low: j for (i, j ) in ts_dict.items()}
        model.ts_dict.update(ts_dict)
    model.df = pd.concat(datalist, axis=0)
    model.df = model.df.reset_index(drop=True)

    # re-set RNG generators and subst model on objects
    model._reset_random_generators()
    model.subst_model = _old_substitution_model
    return None


def _sim_trees(
    model: Model,
    nloci: int,
    nsites: int,
    precision: int = 14,
    seeds: Sequence[int] = None,
) -> Tuple[pd.DataFrame, Mapping[int, TreeSequence]]:
    """Simulate unlinked genealogies.

    This is the main loop that can be run on parallel engines.
    """
    # optionally store tree_sequences when not run remotely
    ts_dict = {}

    # optionally use pre-generated seeds when run remotely
    if seeds is None:
        seeds = model.rng_trees.integers(2**31, size=nloci)
    else:
        assert len(seeds) == nloci

    # generate a new ts for each independent locus
    datalist = []
    for lidx, seed in zip(range(nloci), seeds):
        # init ts generator and get first ts
        msgen = model._get_tree_sequence_generator(nsites, seed=seed)
        tree_seq = next(msgen)

        # iterate over linked trees in ts
        for tree in tree_seq.trees():
            # get interval for this tree
            ival = tree.get_interval()
            index = tree.get_index()
            # get newick for this tree w/ original names
            nwk = tree.newick(node_labels=model.tipdict, precision=precision)
            # store all to a list
            # row = [lidx, int(ival.left), int(ival.right), int(ival.span), 0, index, nwk]
            if model.discrete_genome:
                row = [lidx, int(ival.left), int(ival.right), int(ival.span), 0, index, nwk]
            else:
                row = [lidx, ival.left, ival.right, ival.span, 0, index, nwk]
            datalist.append(row)

        # store the tree_sequence
        if model.store_tree_sequences:
            ts_dict[lidx] = tree_seq

    # organize into a dataframe and return
    return pd.DataFrame(datalist, columns=COLUMNS), ts_dict


# DEPRECATED FOR NEW sim_trees and _sim_trees combo that allows for parallel.
# def sim_trees(model: Model, nloci: int, nsites: int, precision: int = 14) -> None:
#     """Simulate unlinked genealogies.

#     See `ipcoal.Model.sim_trees` docstring.
#     """
#     # check conflicting args
#     if model._recomb_is_map:
#         if nsites:
#             raise IpcoalError(
#                 "Both nsites and recomb_map cannot be used together since"
#                 "the recomb_map also specifies nsites. To use a recomb_map"
#                 "specify nsites=None.")
#         nsites = model.recomb.sequence_length

#     # clear any existing stored tree sequences
#     model.ts_dict = {}

#     datalist = []
#     for lidx in range(nloci):
#         msgen = model._get_tree_sequence_generator(nsites)
#         tree_seq = next(msgen)
#         breaks = [int(i) for i in tree_seq.breakpoints()]
#         starts = breaks[0:len(breaks) - 1]
#         ends = breaks[1:len(breaks)]
#         lengths = [i - j for (i, j) in zip(ends, starts)]

#         data = pd.DataFrame({
#             "start": starts,
#             "end": ends,
#             "nbps": lengths,
#             "nsnps": 0,
#             "tidx": 0,
#             "locus": lidx,
#             "genealogy": "",
#         },
#             columns=[
#                 'locus', 'start', 'end', 'nbps',
#                 'nsnps', 'tidx', 'genealogy'
#         ],
#         )

#         # iterate over the index of the dataframe to sim for each genealogy
#         for mstree in tree_seq.trees():
#             # convert nwk to original names
#             nwk = mstree.newick(node_labels=model.tipdict, precision=precision)
#             data.loc[mstree.index, "genealogy"] = nwk
#             data.loc[mstree.index, "tidx"] = mstree.index
#         datalist.append(data)

#         # store the tree_sequence
#         if model.store_tree_sequences:
#             model.ts_dict[lidx] = tree_seq

#     # concatenate all of the genetree dfs
#     data = pd.concat(datalist)
#     data = data.reset_index(drop=True)
#     model.df = data
#     model.seqs = np.array([])


if __name__ == "__main__":

    import toytree
    import ipcoal

    # TREE = toytree.rtree.unittree(ntips=6, treeheight=1e6)
    # MODEL = ipcoal.Model(TREE, Ne=1e4, seed_trees=123)
    # MODEL.sim_trees(40, 1e4, nproc=1)
    # print(MODEL.df)

    TREE = toytree.rtree.unittree(ntips=6, treeheight=1e6)
    MODEL = ipcoal.Model(TREE, Ne=1e4, seed_trees=123, store_tree_sequences=True)
    MODEL.sim_trees(10, 1e4, nproc=4)
    # print(_sim_trees(MODEL, 10, 10, 14))
    print(MODEL.df)
    print(MODEL.ts_dict)