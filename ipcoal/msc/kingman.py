#!/usr/bin/env python

"""Kingman n-coalescent probability functions.

NOT YET IMPLEMENTED.

References
----------
- ...

TODO
----
- Use scipy.stats instead of np.exp
- Develop for teaching...
"""


from typing import Dict
import itertools
import numpy as np
import pandas as pd
from loguru import logger
import toytree
import ipcoal


logger = logger.bind(name="ipcoal")



def get_gene_tree_log_prob_single_pop(neff: float, coal_times: np.ndarray):
    r"""Return log prob density of a gene tree in a single population.

    All labeled histories have equal probability in a single population
    model, and so the probability of a gene tree is calculated only 
    from the coalescent times.

    Time to the first coalescent event is geometric with parameter
    k_choose_2 * 1 / 2N. Kingman's coalescent makes a continous 
    approximation of this function, where waiting time is exponentially
    distributed.

    Modified from equation 5 of Rannala et al. (2020) to use edge 
    lens in units of gens, and population neffs, instead of thetas.

    $ {2 \choose \theta}^{n-1} e^{-\frac{2}{\theta}} $

    Parameters
    ----------
    neff: float
        Effective population size
    coal_times: np.ndarray, shape=2, dtype=float
        An array of ordered coalescent times for all internal nodes
        in a tree except the root, where n = ntips in the tree the 
        order of times represents when n= [n-1, n-2, ..., 2].

    Example
    -------
    >>> import ipcoal, toytree, toyplot
    >>> neff = 1e6
    >>> model = ipcoal.Model(None, Ne=neff, nsamples=25)
    >>> model.sim_trees(1)
    >>> gtree = toytree.tree(model.df.genealogy[0])
    >>> coals = np.array(sorted(gtree.get_node_data("height")[gtree.ntips:]))
    >>> xs = np.logspace(np.log10(neff) - 1, np.log10(neff) + 1)
    >>> logliks = [get_gene_tree_log_prob_single_pop(i, coals) for i in xs]
    >>> canvas, axes, mark = toyplot.plot(
    >>>     xs, logliks,
    >>>     xscale="log", height=300, width=400,
    >>> )
    >>> axes.vlines([neff])
    """
    # TODO: this can be written much faster using broadcasting.
    nlineages = len(coal_times) + 1
    rate = (1 / (2 * neff))
    prob = 1
    for idx, nlineages in enumerate(range(nlineages, 1, -1)):
        npairs = (nlineages * (nlineages - 1)) / 2
        time = (coal_times[idx] - coal_times[idx - 1] if idx else coal_times[idx])
        opportunity = npairs * time
        prob *= rate * np.exp(-rate * opportunity)
    if prob > 0:
        return np.log(prob)
    return np.inf


def optim_func(neff: float, coal_times: np.ndarray):
    """Return the log prob density of a set of gene trees in a single pop.

    Example
    -------
    >>> import ipcoal, toytree, toyplot
    >>> neff = 1e5
    >>> model = ipcoal.Model(None, Ne=neff, nsamples=20)
    >>> model.sim_trees(100)
    >>> coal_times = np.array([
    >>>     sorted(gtree.get_node_data("height")[gtree.ntips:])
    >>>     for gtree in toytree.mtree(model.df.genealogy)
    >>> ])
    >>> xs = np.logspace(np.log10(neff) - 1, np.log10(neff) + 1)
    >>> logliks = [optim_func(i, coal_times) for i in xs]
    >>> canvas, axes, mark = toyplot.plot(
    >>>     xs, logliks,
    >>>     xscale="log", height=300, width=400,
    >>> )
    >>> axes.vlines([neff]);
    """
    assert coal_times.ndim == 2, "coal_times must be shape: (ntrees, ncoals)"
    logprobs = [get_gene_tree_log_prob_single_pop(neff, i) for i in coal_times]
    loglik = np.sum(logprobs)
    if loglik == np.inf:
        return loglik
    return -loglik


if __name__ == "__main__":
    
    ipcoal.set_log_level("INFO")

    # simulate genealogies
    RECOMB = 1e-9
    MUT = 1e-9
    NEFF = 5e5
    THETA = 4 * NEFF * MUT
    MODEL = ipcoal.Model(
        Ne=NEFF,
        seed_trees=123,
        nsamples=4,
        recomb=RECOMB,
        mut=MUT,
    )
    # MODEL.sim_trees(100, 1)
    # IMAP = MODEL.get_imap_dict()
    # GTREES = toytree.mtree(MODEL.df.genealogy.tolist())

    # get embedding table

    import ipcoal, toytree, toyplot
    import numpy as np

    neff = 1e6

    # simulate a genealogy
    model = ipcoal.Model(None, Ne=neff, nsamples=25)
    model.sim_trees(1)
    gtree = toytree.tree(model.df.genealogy[0])
    coals = np.array(sorted(gtree.get_node_data("height")[gtree.ntips:]))
    xs = np.logspace(np.log10(neff) - 1, np.log10(neff) + 1, 10)
    logliks = [get_gene_tree_log_prob_single_pop(i, coals) for i in xs]
    canvas, axes, mark = toyplot.plot(
        xs, logliks,
        xscale="log", height=300, width=400,
    )
    axes.vlines([neff])
    toytree.utils.show(canvas)
