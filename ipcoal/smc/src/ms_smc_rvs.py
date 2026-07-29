#!/usr/bin/env python

"""Return a parameterized scipy frozen probability distribution
from which waiting distances can be sampled as random variables,
and statistics can be calculated.
"""

from typing import Mapping, Sequence
from scipy import stats
from toytree import ToyTree
from ipcoal.smc.src.ms_smc_simple import (
    get_prob_tree_unchanged,
    get_prob_topo_unchanged
)


__all__ = [
    #
    "get_distribution_waiting_distance_to_recombination",
    "get_distribution_waiting_distance_to_tree_change",
    "get_distribution_waiting_distance_to_topo_change",
    #
    "get_expected_waiting_distance_to_recombination",
    "get_expected_waiting_distance_to_tree_change",
    "get_expected_waiting_distance_to_topo_change",
]


def get_distribution_waiting_distance_to_recombination(
    genealogy: ToyTree,
    recombination_rate: float,
    *args,
    **kwargs,
) -> stats._distn_infrastructure.rv_frozen:
    r"""Return the exponential probability density for waiting distance
    to next recombination event.

    Waiting distances between events are modeled as an exponentially
    distributed random variable (rv). This probability distribution
    is represented in scipy by an `rv_continous` class object. This
    function returns a "frozen" rv_continous object that has its
    rate parameter fixed, where the rate of recombination on the
    input genealogy is a product of its sum of edge lengths (L(G))
    and the per-site per-generation recombination rate (r).

    $$ \lambda_r = L(G) * r $$

    The returned frozen `rv_continous` variable can be used to
    calculate likelihoods using its `.pdf` method; to sample
    random waiting distances using its `.rvs` method; to get the
    mean expected waiting distance from `.mean`; among other things.
    See scipy docs.

    Parameters
    -----------
    genealogy: ToyTree
        A genealogy with branch lengths in generations.
    recombination rate: float
        A per-site per-generation recombination rate.

    Examples
    --------
    >>> SPTREE, GTREE, IMAP = ipcoal.msc.get_test_data()
    >>> distn = get_distribution_waiting_distance_to_recombination(
    >>>     genealogy=GTREE, recombination_rate=1e-9)
    >>> print(distn.mean())
    >>> # 310.5590062111801
    >>> print(distn.rvs(size=4, random_state=123))
    >>> # [370.27085201 104.67934245  79.90188824 248.89244777]
    """
    sumlen = sum(i._dist for i in genealogy if not i.is_root())
    lambda_ = sumlen * recombination_rate
    return stats.expon.freeze(scale=1 / lambda_)


def get_expected_waiting_distance_to_recombination(
    genealogy: ToyTree,
    recombination_rate: float,
    *args,
    **kwargs,
) -> float:
    r"""Return the expected (mean) waiting distance to next
    recombination event.

    Waiting distances between events are modeled as an exponentially
    distributed random variable with rate parameter \lambda, where the
    rate is the product of the sum of genealogy edge lengths (L(G))
    and the per-site per-generation recombination rate (r). The mean
    of an exponential distribution is 1 / lambda.

    $$ \lambda_r = L(G) * r $$

    Parameters
    -----------
    genealogy: ToyTree
        A genealogy with branch lengths in generations.
    recombination rate: float
        A per-site per-generation recombination rate.
    """
    return get_distribution_waiting_distance_to_recombination(genealogy, recombination_rate).mean()


def get_distribution_waiting_distance_to_tree_change(
    species_tree: ToyTree,
    genealogy: ToyTree,
    imap: Mapping[str, Sequence[str]],
    recombination_rate: float,
) -> stats._distn_infrastructure.rv_frozen:
    r"""Return the exponential probability density for waiting distance
    to next tree-change event.

    Waiting distances between events are modeled as an exponentially
    distributed random variable (rv). This probability distribution
    is represented in scipy by an `rv_continous` class object. This
    function returns a "frozen" rv_continous object that has its
    rate parameter fixed, where the rate of a no-change recombination
    event on the input genealogy is a product of its sum of edge
    lengths (L(G)), the per-site per-generation recombination rate
    (r) and the Prob(tree-change | S,G).

    $$ \lambda_r = L(G) * r * P(tree-change | S, G)$$

    The returned frozen `rv_continous` variable can be used to
    calculate likelihoods using its `.pdf` method; to sample
    random waiting distances using its `.rvs` method; to get the
    mean expected waiting distance from `.mean`; among other things.
    See scipy docs.

    Parameters
    -----------
    species_tree: ToyTree
        A species tree.
    genealogy: ToyTree
        A genealogy that can be embedded in the species tree.
    imap: Dict[str, Sequence[str]]
        Mapping of species tree tip names to list of gene tree tip names
    recombination rate: float
        A per-site per-generation recombination rate.

    Examples
    --------
    >>> SPTREE, GTREE, IMAP = ipcoal.msc.get_test_data()
    >>> distn = get_distribution_waiting_distance_to_tree_change(
    >>>     SPTREE, GTREE, IMAP, 1e-9)
    >>> print(distn.mean())
    >>> # 529.9365649168218
    >>> print(distn.rvs(size=4, random_state=123))
    >>> # [631.82860416 178.62438391 136.34424163 424.70901235]
    """
    sumlen = sum(i.dist for i in genealogy if not i.is_root())
    prob_u = get_prob_tree_unchanged(species_tree, genealogy, imap)
    lambda_ = sumlen * (1 - prob_u) * recombination_rate
    return stats.expon.freeze(scale=1 / lambda_)


def get_expected_waiting_distance_to_tree_change(
    species_tree: ToyTree,
    genealogy: ToyTree,
    imap: Mapping[str, Sequence[str]],
    recombination_rate: float,
) -> float:
    r"""Return the expected (mean) waiting distance to next tree-change
    recombination event.

    Waiting distances between events are modeled as an exponentially
    distributed random variable with rate parameter \lambda, where the
    rate is the product of the sum of genealogy edge lengths (L(G)),
    the P(tree-change | S, G), and the per-site per-generation
    recombination rate (r). The mean of an exponential distribution
    is 1 / lambda.

    $$ \lambda_r = L(G) * P(tree-change | S, G) * r $$

    Parameters
    -----------
    species_tree: ToyTree
        A species tree.
    genealogy: ToyTree
        A genealogy that can be embedded in the species tree.
    imap: Dict[str, Sequence[str]]
        Mapping of species tree tip names to list of gene tree tip names
    recombination rate: float
        A per-site per-generation recombination rate.
    """
    return get_distribution_waiting_distance_to_tree_change(
        species_tree, genealogy, imap, recombination_rate).mean()


def get_distribution_waiting_distance_to_topo_change(
    species_tree: ToyTree,
    genealogy: ToyTree,
    imap: Mapping[str, Sequence[str]],
    recombination_rate: float,
) -> stats._distn_infrastructure.rv_frozen:
    r"""Return the exponential probability density for waiting distance
    to next topo-change event.

    Waiting distances between events are modeled as an exponentially
    distributed random variable (rv). This probability distribution
    is represented in scipy by an `rv_continous` class object. This
    function returns a "frozen" rv_continous object that has its
    rate parameter fixed, where the rate of a no-change recombination
    event on the input genealogy is a product of its sum of edge
    lengths (L(G)), the per-site per-generation recombination rate
    (r) and the Prob(topo-change | S,G).

    $$ \lambda_r = L(G) * r * P(topo-change | S, G)$$

    The returned frozen `rv_continous` variable can be used to
    calculate likelihoods using its `.pdf` method; to sample
    random waiting distances using its `.rvs` method; to get the
    mean expected waiting distance from `.mean`; among other things.
    See scipy docs.

    Parameters
    -----------
    species_tree: ToyTree
        A species tree.
    genealogy: ToyTree
        A genealogy that can be embedded in the species tree.
    imap: Dict[str, Sequence[str]]
        Mapping of species tree tip names to list of gene tree tip names
    recombination rate: float
        A per-site per-generation recombination rate.

    Examples
    --------
    >>> SPTREE, GTREE, IMAP = ipcoal.msc.get_test_data()
    >>> distn = get_distribution_waiting_distance_to_tree_change(
    >>>     SPTREE, GTREE, IMAP, 1e-9)
    >>> print(distn.mean())
    >>> # 1008.7951259720993
    >>> print(distn.rvs(size=4, random_state=123))
    >>> # [1202.75832718  340.03203365  259.54692601  808.482392  ]
    """
    sumlen = sum(i.dist for i in genealogy if not i.is_root())
    prob_u = get_prob_topo_unchanged(species_tree, genealogy, imap)
    lambda_ = sumlen * (1 - prob_u) * recombination_rate
    return stats.expon.freeze(scale=1 / lambda_)


def get_expected_waiting_distance_to_topo_change(
    species_tree: ToyTree,
    genealogy: ToyTree,
    imap: Mapping[str, Sequence[str]],
    recombination_rate: float,
) -> float:
    r"""Return the expected (mean) waiting distance to next topo-change
    recombination event.

    Waiting distances between events are modeled as an exponentially
    distributed random variable with rate parameter \lambda, where the
    rate is the product of the sum of genealogy edge lengths (L(G)),
    the P(topo-change | S, G), and the per-site per-generation
    recombination rate (r). The mean of an exponential distribution
    is 1 / lambda.

    $$ \lambda_r = L(G) * P(topo-change | S, G) * r $$

    Parameters
    -----------
    species_tree: ToyTree
        A species tree.
    genealogy: ToyTree
        A genealogy that can be embedded in the species tree.
    imap: Dict[str, Sequence[str]]
        Mapping of species tree tip names to list of gene tree tip names
    recombination rate: float
        A per-site per-generation recombination rate.
    """
    return get_distribution_waiting_distance_to_topo_change(
        species_tree, genealogy, imap, recombination_rate).mean()


if __name__ == "__main__":

    import ipcoal
    SPTREE, GTREE, IMAP = ipcoal.msc.get_test_data()

    distn = get_distribution_waiting_distance_to_recombination(GTREE, recombination_rate=1e-9)
    e = get_expected_waiting_distance_to_recombination(GTREE, 1e-9)
    print(e)
    print(distn.mean())
    print(distn.rvs(4, random_state=123))

    distn = get_distribution_waiting_distance_to_tree_change(SPTREE, GTREE, IMAP, recombination_rate=1e-9)
    e = get_expected_waiting_distance_to_tree_change(SPTREE, GTREE, IMAP, 1e-9)
    print(e)
    print(distn.mean())
    print(distn.rvs(4, random_state=123))

    distn = get_distribution_waiting_distance_to_topo_change(SPTREE, GTREE, IMAP, recombination_rate=1e-9)
    e = get_expected_waiting_distance_to_topo_change(SPTREE, GTREE, IMAP, 1e-9)
    print(e)
    print(distn.mean())
    print(distn.rvs(4, random_state=123))
