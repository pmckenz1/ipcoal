#!/usr/bin/env python

"""Compute likelihood of interval lengths given gene tree embedded
in a species tree.

"""

from typing import Sequence, Union, Mapping, Optional
from scipy import stats
import numpy as np
from toytree import ToyTree, MultiTree
from numba import njit, prange
from loguru import logger
from ipcoal.smc.src.embedding import TreeEmbedding
from ipcoal.smc.src.ms_smc_tree_prob import get_tree_changed_lambdas
from ipcoal.smc.src.ms_smc_topo_prob import get_topo_changed_lambdas
from ipcoal.smc.src.ms_smc_tree_prob import get_prob_tree_unchanged_from_arrays
from ipcoal.smc.src.ms_smc_topo_prob import get_prob_topo_unchanged_from_arrays

logger = logger.bind(name="ipcoal")

__all__ = [
    "get_ms_smc_loglik_from_embedding",
    "get_ms_smc_loglik",
]


def _update_neffs(emb: np.ndarray, popsizes: np.ndarray) -> None:
    """Updates diploid Ne values in the concatenated embedding array.

    This is used during MCMC proposals to update Ne values. It takes
    Ne values as input, but stores to the array as 2Ne.

    TODO: faster method use stored masks
    """
    if len(set(popsizes)) == 1:
        emb[:, :, 3] = popsizes[0]
    else:
        for idx, popsize in enumerate(popsizes):
            mask = emb[:, :, 2] == idx
            emb[mask, 3] = popsize


def get_recomb_event_lambdas(sarr: np.ndarray, recombination_rate: float, *args, **kwargs) -> np.ndarray:
    """Return loglikelihood of observed waiting distances to ANY recomb
    events given summed gene tree branch lens and recomb rate."""
    return recombination_rate * sarr


def get_ms_smc_loglik_from_embedding(
    embedding: TreeEmbedding,
    recombination_rate: float,
    lengths: np.ndarray,
    event_type: int = 1,
    idxs: Optional[np.ndarray] = None,
    normalize: bool = False,
) -> float:
    """Return -loglik of observed waiting distances between specific
    recombination event-types given a species tree and genealogies.

    Parameters
    ----------
    embedding_arr: TreeEmbedding
        A TreeEmbedding object with genealogy embedding arrays.
    recombination_rate: float
        per site per generation recombination rate.
    lengths: np.ndarray
        An array of observed waiting distances between tree changes
    event_type: int
        0 = any recombination event.
        1 = tree-change event.
        2 = topology-change event.
    idxs: np.ndarray or None
        An optional int array to select a subset of trees from the
        Embedding. This allows using the same embedding table for tree
        and topology changes by subsetting the topology-change indices.
    normalize: bool
        If True the log-likelhood of observing each interval distance
        is weighted by the proportion of length of the ARG that it
        represents. This arg should be True when comparing different
        ARGs for the same model. It should be False when comparing
        different models to fit the same ARG.

    Examples
    --------
    >>> embedding = TreeEmbedding(model.tree, model.df.genealogy, imap)
    >>> intervals = model.df.nbps.values
    >>> params = np.array([1e5, 1e5, 1e5])
    >>> get_tree_distance_loglik(embedding, params, 2e-9, intervals)
    """
    # get rates (lambdas) for waiting distances
    if event_type == 0:
        rate_function = get_recomb_event_lambdas
    elif event_type == 1:
        rate_function = get_tree_changed_lambdas
    else:
        rate_function = get_topo_changed_lambdas

    # get mask
    if idxs is None:
        idxs = np.arange(embedding.emb.shape[0])

    # get lambdas from rate function
    rates = rate_function(
        emb=embedding.emb,
        enc=embedding.enc,
        barr=embedding.barr,
        sarr=embedding.sarr,
        rarr=embedding.rarr,
        recombination_rate=recombination_rate,
        idxs=idxs,
    )

    # get logpdf of observed waiting distances given rates (lambdas)
    logliks = stats.expon.logpdf(scale=1 / rates, x=lengths)
    # logger.warning(f"{1/rates[:5]}, {lengths[:5]}, {logliks[:5]}")
    # weight each loglik by its proportion of the length of the ARG
    # wlogliks = logliks / (lengths / lengths.sum())
    # logger.warning(f"{-np.sum(logliks):.7e} | {-np.sum(wlogliks) * lengths.size}")
    # logger.warning([logliks[:5], lengths[:5], logliks[:5] * lengths[:5]])
    # return as neg sum loglik

    # when normalized we are comparing ARGs of the same total length
    # but composed of different sub interval lengths, so we need to
    # ask how well the data fit our model per unit length, otherwise
    # models that split into more smaller intervals are always better.
    if normalize:
        logliks *= (lengths / lengths.sum())
        return -np.sum(logliks)  # do not use * lengths.size here

    # when not normalized we are comparing the exact same ARGs under
    # different species tree models. We do not want to normalize per
    # unit length, because that would always favor models with lower Ne,
    # since there are more small than large intervals in an expon dist
    # by definition.
    return -np.sum(logliks)


# @njit  # (parallel=True)
# def faster_likelihood(
#     emb: np.ndarray,
#     enc: np.ndarray,
#     barr: np.ndarray,
#     sarr: np.ndarray,
#     rarr: np.ndarray,
#     recombination_rate: float,
#     tree_lengths: np.ndarray,
#     topo_lengths: np.ndarray,
#     topo_idxs: np.ndarray = None,
# ) -> float:
#     """
#     """
#     # ...
#     sum_neg_loglik = 0.
#     tidx = 0
#     for gidx in range(emb.shape[0]):
#         gemb = emb[gidx]
#         genc = enc[gidx]
#         blens = barr[gidx]
#         sumlen = sarr[gidx]

#         prob_un_tree = get_prob_tree_unchanged_from_arrays(gemb, genc, blens, sumlen)
#         lambda_tree = sumlen * (1 - prob_un_tree) * recombination_rate
#         sum_neg_loglik += -np.log(lambda_tree * np.exp(-lambda_tree * tree_lengths[gidx]))

#     for tidx, gidx in enumerate(topo_idxs):
#         gemb = emb[gidx]
#         genc = enc[gidx]
#         blens = barr[gidx]
#         sumlen = sarr[gidx]
#         relate = rarr[gidx]
#         prob_un_topo = get_prob_topo_unchanged_from_arrays(gemb, genc, blens, sumlen, relate)
#         lambda_topo = sumlen * (1 - prob_un_topo) * recombination_rate
#         sum_neg_loglik += -np.log(lambda_topo * np.exp(-lambda_topo * topo_lengths[tidx]))
#     return sum_neg_loglik


# @njit(parallel=True)
# def faster_likelihood_parallel(
#     emb: np.ndarray,
#     enc: np.ndarray,
#     barr: np.ndarray,
#     sarr: np.ndarray,
#     rarr: np.ndarray,
#     recombination_rate: float,
#     tree_lengths: np.ndarray,
#     topo_lengths: np.ndarray,
#     topo_idxs: np.ndarray = None,
# ) -> float:
#     """
#     """
#     # ...
#     sum_neg_loglik = 0.
#     tidx = 0
#     for gidx in prange(emb.shape[0]):
#         gemb = emb[gidx]
#         genc = enc[gidx]
#         blens = barr[gidx]
#         sumlen = sarr[gidx]

#         prob_un_tree = get_prob_tree_unchanged_from_arrays(gemb, genc, blens, sumlen)
#         lambda_tree = sumlen * (1 - prob_un_tree) * recombination_rate
#         loglik = -np.log(lambda_tree * np.exp(-lambda_tree * tree_lengths[gidx]))

#         if gidx in topo_idxs:
#             relate = rarr[gidx]
#             prob_un_topo = get_prob_topo_unchanged_from_arrays(gemb, genc, blens, sumlen, relate)
#             lambda_topo = sumlen * (1 - prob_un_topo) * recombination_rate
#             loglik += -np.log(lambda_topo * np.exp(-lambda_topo * topo_lengths[tidx]))
#             tidx += 1
#         sum_neg_loglik += loglik
#     return sum_neg_loglik


# @njit # parallel=True)
# def faster_likelihood_not_parallel(
#     emb: np.ndarray,
#     enc: np.ndarray,
#     barr: np.ndarray,
#     sarr: np.ndarray,
#     rarr: np.ndarray,
#     recombination_rate: float,
#     tree_lengths: np.ndarray,
#     topo_lengths: np.ndarray,
#     topo_idxs: np.ndarray = None,
# ) -> float:
#     """
#     """
#     # ...
#     sum_neg_loglik = 0.
#     tidx = 0
#     for gidx in range(emb.shape[0]):
#         gemb = emb[gidx]
#         genc = enc[gidx]
#         blens = barr[gidx]
#         sumlen = sarr[gidx]

#         prob_un_tree = get_prob_tree_unchanged_from_arrays(gemb, genc, blens, sumlen)
#         lambda_tree = sumlen * (1 - prob_un_tree) * recombination_rate
#         loglik = -np.log(lambda_tree * np.exp(-lambda_tree * tree_lengths[gidx]))

#         if gidx in topo_idxs:
#             relate = rarr[gidx]
#             prob_un_topo = get_prob_topo_unchanged_from_arrays(gemb, genc, blens, sumlen, relate)
#             lambda_topo = sumlen * (1 - prob_un_topo) * recombination_rate
#             loglik += -np.log(lambda_topo * np.exp(-lambda_topo * topo_lengths[tidx]))
#             tidx += 1
#         sum_neg_loglik += loglik
#     return sum_neg_loglik


def get_ms_smc_loglik(
    species_tree: ToyTree,
    genealogies: Union[ToyTree, Sequence[ToyTree], MultiTree],
    imap: Mapping[str, Sequence[str]],
    recombination_rate: float,
    lengths: np.ndarray,
    event_type: int = 1,
    idxs: Optional[np.ndarray] = None,
    normalize: bool = False,
) -> float:
    """Return -loglik of tree-sequence waiting distances between
    tree change events given species tree parameters.

    This function returns the log likelihood of an observed waiting
    distance between a specific recombination event type. This func is
    primarily for didactic purposes, since it must infer the genealogy
    embeddings each time you run it. It is generally much faster to
    first get the embeddings and run `get_smc_loglik_from_embedding`.

    Parameters
    ----------
    embedding_arr: TreeEmbedding
        A TreeEmbedding object with genealogy embedding arrays.
    recombination_rate: float
        per site per generation recombination rate.
    lengths: np.ndarray
        An array of observed waiting distances between tree changes.
        This must be the same length as number of genealogies.
    event_type: int
        0 = any recombination event.
        1 = tree-change event.
        2 = topology-change event.
    normalize: bool
        If True the log-likelhood of observing each interval distance
        is weighted by the proportion of length of the ARG that it
        represents. This arg should be True when comparing different
        ARGs for the same model. It should be False when comparing
        different models to fit the same ARG.

    See Also
    ---------
    `get_smc_loglik_from_embedding()`

    Examples
    --------
    >>> S, G, I = ipcoal.msc.get_test_data()
    >>> L = 100
    >>> R = 1e-9
    >>> get_mssmc_loglik(S, G, I, L, R, 1)
    >>> # ...
    """
    # ensure genealogies is a sequence
    if isinstance(genealogies, ToyTree):
        genealogies = [genealogies]
    # ensure lengths is an array
    lengths = np.array(lengths)

    # ensure same size lengths and trees
    if idxs is not None:
        assert len(idxs) == len(lengths), "N trees must match N waiting distances"
    else:
        assert len(lengths) == len(genealogies), "N trees must match N waiting distances"

    # get embedding and calculate likelihood
    embedding = TreeEmbedding(species_tree, genealogies, imap)
    return get_ms_smc_loglik_from_embedding(
        embedding,
        recombination_rate,
        lengths,
        event_type,
        idxs,
        normalize,
    )


# def get_simple_waiting_distance_likelihood(
#     ts: TreeSequence,
#     recombination: float,
# ) -> float:
#     """Return the likelihood of waiting distances in an ARG.

#     This calculates the likelihood of interval lengths in a tree
#     sequence under the assumption that they are drawn from an
#     exponential probability density with rate parameter r x L, where
#     r is the recombination rate and L is the sum branch lengths of the
#     last genealogy.

#     This can only be run on TreeSequences simulated under the following
#     settings and will raise an exception if not.
#     >>> ancestry="hudson"
#     >>> discrete_genome=False
#     """
#     assert ts.discrete_genome == False
#     # assert ts.ancestry_model == "hudson"  # don't know how to check.


# def get_simple_arg_likelihood(
#     ts: TreeSequence,
#     recombination: float,
# ) -> float:
#     """Return the likelihood of an ARG.

#     This calculates (1) the likelihood of interval lengths in a tree
#     sequence under the assumption that they are drawn from an
#     exponential probability density with rate parameter r x L, where
#     r is the recombination rate and L is the sum branch lengths of the
#     last genealogy; and (2) the likelihood of the coalescent times
#     given the demographic model.

#     This can only be run on TreeSequences simulated under the following
#     settings and will raise an exception if not.
#     >>> ancestry="hudson"
#     >>> discrete_genome=False
#     """
#     assert ts.discrete_genome == False
#     # assert ts.ancestry_model == "hudson"  # don't know how to check.


if __name__ == "__main__":

    import toytree
    import ipcoal
    from ipcoal.msc import get_msc_loglik_from_embedding
    from ipcoal.smc.src.utils import get_waiting_distance_data_from_model

    ############################################################
    RECOMB = 2e-9
    SEED = 123
    NEFF = 2e5
    ROOT_HEIGHT = 5e5  # 1e6
    NSPECIES = 2
    NSAMPLES = 4
    NSITES = 2e6
    NLOCI = 50

    sptree = toytree.rtree.baltree(NSPECIES, treeheight=ROOT_HEIGHT)
    sptree.set_node_data("Ne", {0: NEFF, 1: 2e5, 2: 2e5}, inplace=True)
    model = ipcoal.Model(
        sptree, nsamples=NSAMPLES, recomb=RECOMB, seed_trees=SEED,
        discrete_genome=False, ancestry_model="smc_prime")
    model.sim_trees(NLOCI, NSITES)
    imap = model.get_imap_dict()

    tree_spans, topo_spans, topo_idxs, gtrees = get_waiting_distance_data_from_model(model)
    print(len(gtrees))

    # genealogies = toytree.mtree(model.df.genealogy)
    # glens = model.df.nbps.values
    # G = TreeEmbedding(model.tree, genealogies, imap, nproc=20)
    G = TreeEmbedding(model.tree, gtrees, imap, nproc=6)
    # print(len(genealogies), "gtrees")

    TOPO = True
    if TOPO:
        etype = 2
        spans = topo_spans
        tidxs = topo_idxs
    else:
        etype = 1
        spans = tree_spans
        tidxs = model.df.index.values

    values = np.linspace(10_000, 400_000, 32)
    test_values = np.logspace(np.log10(NEFF) - 1, np.log10(NEFF) + 1, 20)
    values = sorted(list(values) + [NEFF])
    for val in values:
        G._update_neffs(np.array([val, 2e5, 2e5]))
        mloglik1 = get_msc_loglik_from_embedding(G.emb, tree_spans)
        mloglik2 = get_msc_loglik_from_embedding(G.emb)
        gloglik = get_ms_smc_loglik_from_embedding(G, RECOMB, tree_spans, event_type=1)
        tloglik = get_ms_smc_loglik_from_embedding(G, RECOMB, topo_spans, event_type=2, idxs=topo_idxs)
        # wloglik = get_mssmc_loglik_from_embedding(G, RECOMB, glens, event_type=2)
        xloglik = gloglik + tloglik
        loglik1 = mloglik1 + gloglik + tloglik
        loglik2 = mloglik2 + gloglik + tloglik
        print(f"{val:.3e} | {loglik1:.4f} {loglik2:.4f} | {mloglik1:.4f} {mloglik2:.4f} | {gloglik:.4f} {tloglik:.4f} | {xloglik:.4f}")
    raise SystemExit(0)

    ############################################################

    # raise SystemExit(0)

    # sptree = toytree.rtree.imbtree(ntips=4, treeheight=1e6)
    # model = ipcoal.Model(sptree, Ne=1e5, nsamples=2, seed_trees=123)
    # model.sim_trees(2, 1e5)
    # gtrees = model.df.genealogy
    # imap = model.get_imap_dict()

    # G = TreeEmbedding(model.tree, model.df.genealogy, imap)
    # # tree_loglik = _get_msc_loglik_from_embedding_array(G.emb)
    # # wait_loglik = get_waiting_distance_loglik(G, 1e-9, model.df.nbps.values)

    # values = np.linspace(10_000, 300_000, 31)
    # for val in values:
    #     _update_neffs(G.emb, np.array([val] * sptree.nnodes))
    #     tree_loglik = 0#_get_msc_loglik_from_embedding_array(G.emb)
    #     wait_loglik = get_waiting_distance_loglik(G, 1e-8, model.df.nbps.values)
    #     loglik = tree_loglik + wait_loglik
    #     print(f"{val:.2e} {loglik:.2f} {tree_loglik:.2f} {wait_loglik:.2f}")
    # # print(table.table.iloc[:, :8])
    # # print(table.barr.shape, table.genealogies[0].nnodes)
    # # print(get_fast_tree_changed_lambda(table.earr, table.barr, table.sarr, 2e-9))

