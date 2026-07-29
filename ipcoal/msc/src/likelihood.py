#!/usr/bin/env python

"""Calculate likelihood of a gene tree embedded in a species tree.

Given a distribution of gene trees the likelihood of different species
tree models can be compared.

References
----------
- Rannala and Yang (...) "Bayes Estimation of Species Divergence
  Times and Ancestral Population Sizes Using DNA Sequences From Multiple Loci
- Degnan and Salter (...) "..."
- ... (...) "STELLS-mod..."
"""

from typing import Dict, Sequence, Optional
import numpy as np
from numba import njit, prange
from loguru import logger
import toytree
from ipcoal.msc import get_genealogy_embedding_arrays

logger = logger.bind(name="ipcoal")

__all__ = [
    "get_msc_loglik",
    "get_msc_loglik_from_embedding",
]


def get_msc_loglik(
    species_tree: toytree.ToyTree,
    gene_trees: Sequence[toytree.ToyTree],
    imap: Dict,
    dists: Optional[np.ndarray] = None,
) -> float:
    """Return sum -loglik of genealogies embedded in a species tree.

    This function returns the likelihood of observing a set of
    coalescence times in the set of gene trees given a parameterized
    species tree. This can be used to evaluate different species tree
    model parameters to find the optimum model to explain the observed
    data. A smaller returned value indicates the model is a better fit
    to the data. Note: If you are evaluating a dataset at many parameter
    settings it is faster to use the `get_msc_loglik_from_embedding`
    function which does not need to recompile the embedding table each
    time.

    Parameters
    ----------
    species_tree: ToyTree
        Species tree with a "Ne" feature assigned to every Node, and
        edge lengths in units of generations. The tree can be non-
        ultrametric, representing differences in generation times.
    gene_trees: ToyTree, MultiTree, or Sequence[ToyTree]
        One or more gene trees that can be embedded in the species
        tree. Edge lengths are in units of generations.
    imap: Dict
        A dict mapping species tree tip Node names to lists of gene
        tree tip Node names.
    dists: np.ndarray or None
        An int or float array with the length of chromosome that each
        gene tree represents to use as a weight. This option is for
        advanced users and not typically used in MSC analyses, but is
        relevant to SMC-based calculations. When set to None all trees
        are multipled by a weight of 1/ntrees. When an array of dists
        is entered each is multipled by its length divided by the
        sum length of dists.

    See Also
    --------
    get_msc_loglik_from_embedding

    Example
    -------
    >>> S, G, I = ipcoal.msc.get_test_data()
    >>> loglik = ipcoal.msc.get_msc_loglik(S, G, I)
    >>> print(loglik)  # 80.83643587318105
    """
    if isinstance(gene_trees, (toytree.ToyTree, str)):
        gene_trees = [gene_trees]
    if not isinstance(gene_trees[0], toytree.ToyTree):
        gene_trees = toytree.mtree(gene_trees).treelist

    emb, _ = get_genealogy_embedding_arrays(species_tree, gene_trees, imap)
    loglik = get_msc_loglik_from_embedding(emb, dists)
    return loglik


@njit(parallel=True)
def get_msc_loglik_from_embedding(
    embedding: np.ndarray,
    dists: Optional[np.ndarray] = None,
) -> float:
    """Return sum -loglik of genealogies embedded in a species tree.

    This function returns the likelihood of observing a set of
    coalescence times in the set of gene trees given a parameterized
    species tree. This can be used to evaluate different species tree
    model parameters to find the optimum model to explain the observed
    data. A smaller returned value indicates the model is a better fit
    to the data.

    Parameters
    ----------
    embedding: np.ndarray
        An embedding array with genealogy and species tree information.
        generated using `ipcoal.msc.get_genealogy_embedding_arrays`.
    dists: np.ndarray or None
        An int or float array with the length of chromosome that each
        gene tree represents to use as a weight on the log-likelihood.
    """
    ntrees = embedding.shape[0]
    nspecies = int(embedding[0, -1, 2])

    # get weights for trees based on the length of a locus that they
    # take up. By default we typically assume each locus is independent
    # and equally weighted. However, in whole-genome analyses we can
    # weight each by its proportion of sequence length.
    if dists is None:
        weights = np.ones(ntrees, dtype=np.float64) / ntrees
    else:
        weights = dists / dists.sum()

    # iterate over gtrees
    sum_neg_loglik = 0
    for gidx in prange(ntrees):
        garr = embedding[gidx]

        # iterate over species tree intervals
        loglik = 0.
        for sval in range(nspecies + 1):

            # get coal rate in this interval
            iarr = garr[garr[:, 2] == sval]
            rate = 1 / (2 * iarr[0, 3])  # 2 * diploid Ne

            # prob of all events in this sptree interval
            prob = 1.
            # get prob of each coal event in this sptree interval
            for ridx in range(iarr.shape[0] - 1):
                nedges = iarr[ridx, 4]
                npairs = (nedges * (nedges - 1)) / 2
                lambda_ = rate * npairs
                dist = iarr[ridx, 5]
                # prob *= (1 / npairs) * lambda_ * np.exp(-lambda_ * dist)
                prob *= rate * np.exp(-lambda_ * dist)

            # get prob no coal in remaining time of interval
            nedges = iarr[-1, 4]
            npairs = (nedges * (nedges - 1)) / 2
            lambda_ = rate * npairs
            dist = iarr[-1, 5]
            if not np.isinf(dist):
                prob *= np.exp(-lambda_ * dist)

            # store as loglik
            if prob > 0:
                loglik += np.log(prob)
            else:
                loglik += -np.inf

        # weight loglik by its proportion of length
        sum_neg_loglik += -loglik * weights[gidx]   # * dists.size
        # sum_neg_loglik += -loglik
    return sum_neg_loglik * ntrees
    #     logliks[gidx] = loglik
    # return -logliks.sum()


def test_kingman(neff: float = 1e5, nsamples: int = 10, ntrees: int = 500):
    """Return a plot of the likelihood of Ne in a single population.
    """
    import toyplot
    import toytree

    # get (sptree, gtrees, imap)
    model = ipcoal.Model(None, Ne=neff, nsamples=nsamples)
    model.sim_trees(ntrees)
    imap = model.get_imap_dict()

    # get embedding table
    emb, _ = get_genealogy_embedding_arrays(model.tree, model.df.genealogy, imap)

    # get loglik across a range of test values
    test_values = np.logspace(np.log10(neff) - 1, np.log10(neff) + 1, 20)
    logliks = []
    for val in test_values:
        emb[:, :, 3] = val
        loglik = get_msc_loglik_from_embedding(emb)
        logliks.append(loglik)

    canvas, axes, mark = toyplot.plot(
        test_values, logliks,
        xscale="log", height=300, width=400, opacity=0.7, style={'stroke-width': 4}
    )
    toytree.utils.set_axes_ticks_external(axes)
    axes.vlines([neff], style={"stroke": toytree.color.COLORS1[1], "stroke-width": 3})
    toytree.utils.show(canvas)


def test_msc(neff: float = 1e5, nsamples: int = 4, nloci: int = 500, nsites: int = 1):
    """Return a plot of the likelihood of constant Ne in multipop tree.

    This shows that the true Ne has the best likelihood score compared
    to incorrect Ne values.

    The gene tree distribution kept constant and the MSC model
    parameters are varied at several parameters.
    """
    import toyplot
    import toytree

    # get (sptree, gtrees, imap)
    sptree = toytree.rtree.imbtree(ntips=5, treeheight=1e6)
    sptree.set_node_data("Ne", {0: 1e5, 1: 2e5, 3: 3e5}, default=2.5e5)
    model = ipcoal.Model(sptree, Ne=neff, nsamples=nsamples)
    model.sim_trees(nloci, nsites, nproc=4)
    imap = model.get_imap_dict()
    logger.info("simulated trees")

    # get embedding table
    emb, enc = get_genealogy_embedding_arrays(model.tree, model.df.genealogy, imap)
    logger.info("embedded trees")

    # get loglik across a range of test values
    test_values = np.logspace(np.log10(neff) - 1, np.log10(neff) + 1, 21)
    logliks = []
    for val in test_values:
        emb[:, :, 3] = val
        loglik = get_msc_loglik_from_embedding(emb, model.df.nbps.values)
        logliks.append(loglik)
        # logger.info(f"fit value={val:.0f}: {loglik:.5e}")
        logger.info(f"fit value={val:.0f}: {loglik:.5e}")

    canvas, axes, mark = toyplot.plot(
        test_values, logliks,
        xscale="log", height=300, width=400, opacity=0.7, style={'stroke-width': 4}
    )
    toytree.utils.set_axes_ticks_external(axes)
    axes.vlines([neff], style={"stroke": toytree.color.COLORS1[1], "stroke-width": 3})
    toytree.utils.show(canvas)


if __name__ == "__main__":

    import ipcoal
    ipcoal.set_log_level("INFO")
    test_kingman(neff=1e6, nsamples=10, ntrees=500)
    test_msc(neff=1e6, nsamples=5, nloci=5000, nsites=1)
    test_msc(neff=1e6, nsamples=5, nloci=5, nsites=1e3)
    # test_msc(neff=1e6, nsamples=5, nloci=10, nsites=1e5)

    # SPTREE = toytree.rtree.baltree(2, treeheight=1e6)
    # MODEL = ipcoal.Model(SPTREE, Ne=200_000, nsamples=4, seed_trees=123)
    # MODEL = ipcoal.Model(None, Ne=200_000, nsamples=4, seed_trees=123)
    # MODEL.sim_trees(10)
    # GENEALOGIES = toytree.mtree(MODEL.df.genealogy)
    # IMAP = MODEL.get_imap_dict()
    # data = ipcoal.msc.get_genealogy_embedding_table(MODEL.tree, GENEALOGIES, IMAP, )
    # arr, _ = ipcoal.msc.get_genealogy_embedding_arrays(MODEL.tree, GENEALOGIES, IMAP, )
    # print(data.iloc[:6, :6])

    # print(_get_msc_loglik_from_embedding_table(arr))

    # # simulate genealogies
    # RECOMB = 1e-9
    # MUT = 1e-9
    # NEFF = 5e5
    # THETA = 4 * NEFF * MUT

    # # setup species tree model
    # SPTREE = toytree.rtree.unittree(ntips=3, treeheight=1e6, seed=123)
    # SPTREE = SPTREE.set_node_data("Ne", default=NEFF, data={0: 1e5})

    # # setup simulation
    # MODEL = ipcoal.Model(SPTREE, seed_trees=123, nsamples=5)
    # MODEL.sim_trees(10)
    # IMAP = MODEL.get_imap_dict()
    # GTREES = toytree.mtree(MODEL.df.genealogy)
    # # GTREE.draw(ts='c', height=400)

    # table = get_msc_embedded_gene_tree_table(SPTREE, GTREES[0], IMAP)
    # print(table)
    # print(get_loglik_gene_tree_msc_from_table(table))
    # print(get_loglik_gene_tree_msc(SPTREE, GTREES, IMAP))

    # TEST_VALUES = np.logspace(np.log10(NEFF) - 1, np.log10(NEFF) + 1, 19)
    # test_logliks = []
    # for idx in MODEL.df.index:
    #     gtree = toytree.tree(MODEL.df.genealogy[idx])
    #     table = get_msc_embedded_gene_tree_table(SPTREE, gtree, IMAP)

    #     logliks = []
    #     for ne in TEST_VALUES:
    #         table.neff = ne
    #         loglik = get_gene_tree_log_prob_msc(table)
    #         logliks.append(loglik)
    #     test_logliks.append(logliks)

    # logliks = np.array(test_logliks).sum(axis=0)

    # import toyplot
    # canvas, axes, mark = toyplot.plot(
    #     TEST_VALUES, logliks,
    #     xscale="log",
    #     height=300, width=400,
    # )
    # axes.vlines([NEFF])
    # toytree.utils.show(canvas)
