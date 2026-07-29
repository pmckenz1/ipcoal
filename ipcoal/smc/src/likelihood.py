#!/usr/bin/env python

"""Compute log-likelihood of interval lengths given a gene tree 
embedded in a species tree.

The likelihood of tree change or topology-change intervals can be
calculated, and the joint likelihood can be calculated as the 
(likelihood of tree-change) * (conditional likelihood of topo-change | tree-change).

"""

from typing import Sequence, Union, Mapping, Optional
from scipy import stats
from scipy.special import gammaln  # Log of the Gamma function
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
    diploid Ne values as input, but stores to the array as 2 x Ne since
    all internal math uses this multiplied value. (Note: this comment
    says this, but it doesn't look like the func does it anymore...)

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
        3 = tree-change and topology-change (joint likelihood)
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
    elif event_type == 2:
        rate_function = get_topo_changed_lambdas
    else:
        raise ValueError("event_type must be 0, 1, or 2")

    # get mask
    if idxs is None:
        idxs = np.arange(embedding.emb.shape[0])

    # get expon rate parameters for event type
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


def get_ms_smc_joint_loglik_from_embedding(
    embedding: TreeEmbedding,
    recombination_rate: float,
    tree_spans: np.ndarray,
    topo_idxs: np.ndarray,
    topo_tlens: np.ndarray,
    # normalize: bool = False,
) -> float:
    """Return -loglik of observed waiting distances between specific
    recombination event-types given a species tree and genealogies.
    
    The joint likelihood calculates the log likelihood of both waiting
    distances given the model parameters, which is the log L of tree
    waiting distances + the conditional log likelihood of topo waiting
    distances given the N trees per topo interval. Then we can add to
    this the sum of the probabilities of each topo-change event, as
    event-specific probabilities in the joint likelihood.

    log L_joint = log L(TA,TB|S,G,r) + log P(B | S,G,r)

    Parameters
    ----------
    embedding_arr: TreeEmbedding
        A TreeEmbedding object with genealogy embedding arrays.
    recombination_rate: float
        per site per generation recombination rate.
    tree_spans: np.ndarray
        An array of observed waiting distances between tree changes
    topo_idxs: np.ndarray
        An int array to select a subset of trees from the Embedding
        that are the trees starting each topology-change interval.
    topo_tlens:
        ...
    """
    # get expon probability rate functions (lambdas) for each gene tree
    tree_rates = get_tree_changed_lambdas(
        emb=embedding.emb,
        enc=embedding.enc,
        barr=embedding.barr,
        sarr=embedding.sarr,
        rarr=embedding.rarr,
        recombination_rate=recombination_rate,
        idxs=np.arange(embedding.emb.shape[0]),
    )

    # get tree-change log-likelihoods
    tree_logliks = stats.expon.logpdf(scale=1 / tree_rates, x=tree_spans)

    # get topo-change conditional log-likelihood
    topo_spans = tree_spans[topo_idxs]
    first_tree_rates = tree_rates[topo_idxs]
    topo_conditional_logliks = get_conditional_log_likelihood(topo_spans, topo_tlens, first_tree_rates)

    # get event specific probabilities of topo-change events
    topo_probs = np.zeros(topo_idxs.size, np.float64)
    for idx, gidx in enumerate(topo_idxs):
        gemb = embedding.emb[gidx]
        genc = embedding.enc[gidx]
        blens = embedding.barr[gidx]
        sumlen = embedding.sarr[gidx]
        relate = embedding.rarr[gidx]
        topo_probs[idx] = get_prob_topo_unchanged_from_arrays(gemb, genc, blens, sumlen, relate)
    topo_probs = np.log(1 - topo_probs)

    # examine (print) values
    # tidx = 0
    # for idx in range(embedding.emb.shape[0]):
    #     tree_span = tree_spans[idx]
    #     print(f"{idx}\ttree\t{tree_span}\t{tree_logliks[idx]}")
    #     if idx in topo_idxs:
    #         print(f"{idx}\ttopo\t{topo_spans[tidx]}\t{topo_conditional_logliks[tidx]}\t{topo_probs[tidx]}\t{topo_tlens[tidx]}")
    #         tidx += 1

    # return joint log-likelihood
    # L(tree|lambda_t) * L(topo|lamda_t, lambda_conditional likelihood 
    # return -(np.sum(tree_logliks) + np.sum(topo_conditional_logliks) + np.sum(topo_probs))
    # return -(np.sum(tree_logliks))
    # return -(np.sum(topo_conditional_logliks))
    # return -(np.sum(topo_probs))
    # return -(np.sum(tree_logliks) + np.sum(topo_conditional_logliks))
    return (np.sum(tree_logliks), np.sum(topo_conditional_logliks), np.sum(topo_probs))


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
    idxs: Optional[Sequence[int]]
        An optional array or sequence of integers to use as an index
        to select a subset of trees and distances from the inputs to
        ...
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
    >>> get_ms_smc_loglik(S, G, I, L, R, 1)
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


def get_conditional_log_likelihood(
    topo_dists: np.ndarray,
    trees_per_topo_interval: np.ndarray,
    tree_lambdas: np.ndarray,
) -> np.ndarray:
    """
    Calculate the neg log-likelihood for type B waiting times (Gamma distribution).

    Parameters:
    -----------
    tb : array-like
        Observed waiting times for type B events.
    k : array-like
        Number of type A events that make up each type B event (same length as tb).
    r_A : float
        Rate parameter for type A events.

    Returns
    -------
    log_likelihood : float
        Total log-likelihood for all type B events.

    Example
    -------
    >>> tb = [1.2, 2.4, 3.1]  # Observed waiting times for type B events
    >>> k = [2, 3, 1]         # Number of type A events contributing to each type B event
    >>> r_A = 0.5             # Rate of type A events
    >>> 
    >>> # Calculate log-likelihood
    >>> total_log_likelihood = log_likelihood_B(tb, k, r_A)
    >>> print(f"Total log-likelihood for type B events: {total_log_likelihood}")
    """
    tb = np.array(topo_dists)                # Ensure tb is a numpy array
    k = np.array(trees_per_topo_interval)    # Ensure k is a numpy array
    r_A = np.array(tree_lambdas)

    # Validate inputs
    if len(tb) != len(k):
        raise ValueError("tb and k must have the same length.")

    # Log-likelihood for each t_B
    log_likelihoods = (k * np.log(r_A)) + ((k - 1) * np.log(tb)) - (r_A * tb) - gammaln(k)

    # Sum log-likelihoods over all observations
    return log_likelihoods



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
    from ipcoal.smc.src.utils import (
        get_waiting_distance_data_from_model, 
        get_model_tree_and_topo_data,
    )

    ############################################################
    RECOMB = 2e-9
    SEED = 123
    NEFF = 2e5
    ROOT_HEIGHT = 5e5  # 1e6
    NSPECIES = 2
    NSAMPLES = 4
    NSITES = 4e5 # 2e6
    NLOCI = 50  # 50

    # simulate NLOCI x NSITES using the SMC' model so that only tree
    # change recomb events are recorded, not invisible events.
    sptree = toytree.rtree.baltree(NSPECIES, treeheight=ROOT_HEIGHT)
    sptree.set_node_data("Ne", {0: NEFF, 1: 2e5, 2: 2e5}, inplace=True)
    model = ipcoal.Model(
        sptree, nsamples=NSAMPLES, recomb=RECOMB, seed_trees=SEED,
        discrete_genome=False, ancestry_model="hudson")#smc_prime")
    model.sim_trees(NLOCI, NSITES)
    imap = model.get_imap_dict()

    # decompose simulations into data
    # tree_spans, topo_spans, topo_idxs, gtrees = get_waiting_distance_data_from_model(model)

    # decompose simulations into data
    data = get_model_tree_and_topo_data(model)
    gtrees = data["trees"]
    tree_spans = data["tree_spans"]
    topo_spans = data["topo_spans"]    
    topo_tlens = data["topo_tlens"]
    topo_idxs = data["topo_idxs"]

    # ...
    print('trees', tree_spans.size, )
    print('topos', topo_idxs.size, topo_idxs[:10], topo_idxs[-10:])
    print('tlens', topo_tlens.size, topo_tlens[:10], topo_tlens[-10:])    

    # organize SMC data into embedding tables
    G = TreeEmbedding(model.tree, gtrees, imap, nproc=6)
    # cliks = get_ms_smc_joint_loglik_from_embedding(G, RECOMB, tree_spans, topo_idxs, topo_tlens)
    # print(cliks)

    # gloglik = get_ms_smc_loglik_from_embedding(G, RECOMB, tree_spans, event_type=1)
    # print(gloglik)
    # tloglik = get_ms_smc_loglik_from_embedding(G, RECOMB, topo_spans, event_type=2, idxs=topo_idxs)
    # print(tloglik)
    # cloglik = get_ms_smc_loglik_from_embedding(G, RECOMB, topo_spans, event_type=3, idxs=topo_idxs)
    # print(cloglik)    
    # raise SystemExit(0)

    # test over a range of Ne values where NEFF is the true value
    # test_values = np.logspace(np.log10(NEFF) - 1, np.log10(NEFF) + 1, 20)
    # values = np.linspace(10_000, 800_000, 32)
    values = np.linspace(10_000, 400_000, 32)
    values = sorted(list(values) + [NEFF])
    for val in values:
        # update the parameters in the embedding table
        G._update_neffs(np.array([val, 2e5, 2e5]))
        # calculate the MSC likelihood weighted and unweighted
        mloglik1 = get_msc_loglik_from_embedding(G.emb, tree_spans)
        mloglik2 = get_msc_loglik_from_embedding(G.emb)
        # calculate the SMC likelihood for tree and topo distances
        gloglik = get_ms_smc_loglik_from_embedding(G, RECOMB, tree_spans, event_type=1)
        tloglik = get_ms_smc_loglik_from_embedding(G, RECOMB, topo_spans, event_type=2, idxs=topo_idxs)
        # print(f"{val:.3e} | {gloglik:.4f} {tloglik:.4f} {gloglik + tloglik:.4f} | {mloglik1:.4f} | {mloglik2:.4f}")
        wloglik = get_ms_smc_joint_loglik_from_embedding(G, RECOMB, tree_spans, topo_idxs, topo_tlens)
        # print(f"{val:.3e} | {wloglik[0]:.4f} {wloglik[1]:.4f} {wloglik[2]:.4f} | {wloglik[1] + wloglik[2]:.4f} | {sum(wloglik):.4f}")
        print(f"{val:.3e} | {wloglik[0]:.4f} {wloglik[1]:.4f} {wloglik[2]:.4f} | {wloglik[0] + wloglik[2]:.4f} | {sum(wloglik):.4f}")        
        if val == NEFF:
            print("-----------------------------------------------------")
        # report likelihoods
        # xloglik = gloglik + tloglik
        # zloglik = mloglik1 + wloglik
        # loglik1 = mloglik1 + gloglik + tloglik
        # loglik2 = mloglik2 + gloglik + tloglik
        # print(f"{val:.3e} | {mloglik1:.4f} {mloglik2:.4f} | {gloglik:.4f} {tloglik:.4f} | {wloglik:.4f} {xloglik:.4f} | {zloglik:.4f}")        
        # print(f"{val:.3e} | {loglik1:.4f} {loglik2:.4f} | {mloglik1:.4f} {mloglik2:.4f} | {gloglik:.4f} {tloglik:.4f} | {xloglik:.4f}")
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

