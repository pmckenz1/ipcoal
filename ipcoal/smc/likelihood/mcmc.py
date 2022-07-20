#!/usr/bin/env python

"""Posterior sampling of demographic model parameters under the MS-SMC' by Metropolis-Hastings MCMC

"""

from typing import Tuple, Dict, Sequence, Any
from abc import ABC, abstractmethod
import argparse
import time
import sys
from pathlib import Path
from datetime import timedelta
import numpy as np
# import toyplot, toyplot.svg, toyplot.png
import toytree
from numba import set_num_threads
from loguru import logger
from scipy import stats
import ipcoal

# optional. If installed the ESS will be printed.
try:
    import arviz as az
except ImportError:
    pass

# pylint: disable=too-many-nested-blocks, too-many-statements
logger = logger.bind(name="ipcoal")


#########################################################################
#########################################################################
#########################################################################

class Prior(ABC):
    def __init__(self, *params: float):
        self.params = np.array(params, dtype=float)
        self.dist = self.get_dist_rv()

    def log_likelihood(self, params: np.ndarray) -> np.ndarray:
        """Return loglik of the params given the prior."""
        return self.dist.logpdf(params).sum()

    @abstractmethod
    def get_dist_rv(self) -> stats._distn_infrastructure.rv_frozen:
        """Return loglik of the params given the prior."""

class PriorUniform(Prior):
    """A uniform prior..."""

    def get_dist_rv(self) -> stats._distn_infrastructure.rv_frozen:
        """Return loglik of the params given the prior."""
        return stats.uniform.freeze(*self.params)

class PriorGamma(Prior):
    """A gamma prior."""
    def get_dist_rv(self) -> stats._distn_infrastructure.rv_frozen:
        """Return loglik of the params given the prior."""
        return stats.gamma.freeze(a=self.params[0], scale=self.params[1])

def get_prior(distribution: str, *params: float) -> Prior:
    """Return a Prior distribution object for the specified distribution."""
    params = [float(i) for i in params]
    if distribution.startswith("u"):
        return PriorUniform(*params)
    return PriorGamma(*params)

#########################################################################
#########################################################################
#########################################################################

class Mcmc(ABC):
    """A custom Metropolis-Hastings sampler.

    Example
    -------
    >>> mcmc = Mcmc(recomb=2e-9, lengths=lengths, embedding=edata, 
    >>>     priors=[['g', (3, 0.01)] * 3],
    >>>     init_params=[500_000] * 3,
    >>>     seed=123, jumpsize=20_000, outpath="/tmp/test"
    >>> )
    >>> mcmc.run()
    """
    def __init__(
        self,
        recomb: float,
        lengths: Sequence[np.ndarray],
        embedding: Sequence[ipcoal.smc.likelihood.Embedding],
        prior: Tuple[str, Sequence[float]],
        init_params: np.ndarray,
        seed: int,
        jumpsize: int,
        outpath: Path,
        ):
        # store the inputs
        self.recomb = recomb
        self.lengths = lengths
        self.embedding = embedding
        self.prior = get_prior(prior[0], *prior[1:])
        self.params = np.array(init_params)
        self.rng = np.random.default_rng(seed)
        self.jumpsize = jumpsize  # 20_000 #self.params * 0.125
        self.outpath = outpath

    def transition(self) -> np.ndarray:
        """Jitters the current params to new proposal values."""
        return self.rng.normal(self.params, scale=self.jumpsize)

    @classmethod
    def acceptance_ratio(cls, old_d: float, new_d: float, old_p: float, new_p) -> float:
        """Return MH ratio for accept new proposal params."""
        aratio = np.exp((old_d + old_p) - (new_d + new_p))
        # prior_ratio = np.exp(old_p - new_p)
        # prior_ll = np.exp(new_p - old_p)
        return min(1, aratio)
        # return min(1, ((new_d / old_d) * (new_p / old_p)))
        # logger.debug(f"old={old:.2f}, new={new:.2f}, accepted={accepted:.2f}")

    @abstractmethod
    def log_likelihood(self, params) -> float:
        """Return log-likelihood of the data given the params."""

    def prior_log_likelihood(self, params: np.ndarray) -> np.ndarray:
        """Return log-likelihood of the data given the params."""
        return self.prior.log_likelihood(params)

    def run(self, nsamples: int=1000, burnin=2000, sample_interval=5, print_interval=25):
        """Run to sample from the posterior distribution.

        Parameters
        ----------
        nsamples: int
            Number of accepted samples to save in the posterior distribution.
        sample_interval: int
            Number of accepted proposals between saving a sample to the posterior.
        print_interval: int
            Number of accepted proposals between printing progress to stdout.
        burnin: int
            Number of accepted samples to skip before starting sampling.

        Returns
        -------
        np.ndarray
        """
        logger.info(f"starting MCMC sampler for {nsamples} samples.")
        space = " " * (5 * len(self.params))
        logger.info(f"iter\tsample\tloglik   \tparams{space}\taccept\truntime\tcat")
        start = time.time()

        posterior = np.zeros(shape=(nsamples, len(self.params) + 1))
        old_loglik_data = self.log_likelihood(self.params)
        old_loglik_prior = self.prior_log_likelihood(self.params)

        idx = 0
        sidx = 0
        its = 0
        acc = 0
        pidx = 0

        while 1:

            # propose new params
            new_params = self.transition()

            # get likelihood
            new_loglik_prior = self.prior_log_likelihood(new_params)
            if np.isinf(new_loglik_prior):
                aratio = 0
                new_loglik_data = np.inf
            else:
                new_loglik_data = self.log_likelihood(new_params)
                args = (old_loglik_data, new_loglik_data, old_loglik_prior, new_loglik_prior)
                aratio = self.acceptance_ratio(*args)

            # accept or reject
            rand = self.rng.random()
            accept = aratio > rand
            logger.debug(
                f"loglik={new_loglik_data:.1f}/{old_loglik_data:.1f}, "
                f"aratio={aratio:.2f}, params={new_params.astype(int)}, {accept}")
            acc += aratio
            its += 1
            if accept: 

                # proposal accepted
                self.params = new_params
                old_loglik_data = new_loglik_data
                old_loglik_prior = new_loglik_prior
                # new_loglik = new_loglik_data * new_loglik_prior

                # only store every Nth accepted result
                if idx > burnin:
                    if not idx % sample_interval:
                        posterior[sidx] = list(self.params) + [new_loglik_data]
                        sidx += 1
                        if sidx == 1:
                            logger.info("---------------------------------")

                # print on interval
                if not idx % print_interval:
                    elapsed = timedelta(seconds=int(time.time() - start))
                    stype = "sample" if idx > burnin else "burnin"
                    logger.info(
                        f"{idx:>4}\t{sidx:>4}\t"
                        f"{new_loglik_data:.3f}\t"
                        f"{self.params.astype(int)}\t"
                        f"{acc/its:.2f}\t"
                        f"{elapsed}\t{stype}"
                    )

                # save to disk and print summary every 1K sidx
                if sidx and not sidx % 100:
                    if sidx != pidx:
                        np.save(self.outpath, posterior[:sidx])
                        logger.info("checkpoint saved.")
                        logger.info(f"MCMC current posterior mean={posterior[:sidx].mean(axis=0).astype(int)}")
                        logger.info(f"MCMC current posterior std ={posterior[:sidx].std(axis=0).astype(int)}")

                        # print mcmc if optional pkg arviz is installed.
                        if sys.modules.get("arviz"):
                            ess_vals = []
                            for col in range(posterior.shape[1]):
                                azdata = az.convert_to_dataset(posterior[:sidx, col])
                                ess = az.ess(azdata).x.values
                                ess_vals.append(int(ess))
                            logger.info(f"MCMC current posterior ESS ={ess_vals}\n")
                        pidx = sidx

                # advance counter and break when nsamples reached
                idx += 1
                if sidx == nsamples:
                    break
        logger.info(f"MCMC sampling complete. Means={posterior.mean(axis=0)}")
        return posterior


class McmcTree(Mcmc):
    def log_likelihood(self, params) -> float:
        """Return log-likelihood of the data (lengths, edicts) given the params."""
        loglik = 0
        for (edata, lengths) in zip(self.embedding, self.lengths):
            loglik += ipcoal.smc.likelihood.get_tree_distance_loglik(
                embedding=edata,
                params=params,
                recomb=self.recomb,
                lengths=lengths
            )
        return loglik

class McmcTopology(Mcmc):
    def log_likelihood(self, params) -> float:
        """Return log-likelihood of the data (lengths, edicts) given the params."""
        loglik = 0
        for (edata, lengths) in zip(self.embedding, self.lengths):
            loglik += ipcoal.smc.likelihood.get_topology_distance_loglik(
                embedding=edata,
                params=params,
                recomb=self.recomb,
                lengths=lengths
            )
        return loglik

class McmcCombined(Mcmc):
    def log_likelihood(self, params) -> float:
        """Return log-likelihood of the data (lengths, edicts) given the params."""
        loglik = 0
        for (edata, lengths) in zip(self.embedding, self.lengths):
            loglik += ipcoal.smc.likelihood.get_tree_distance_loglik(
                embedding=edata[0],
                params=params,
                recomb=self.recomb,
                lengths=lengths[0],
            )
            loglik += ipcoal.smc.likelihood.get_topology_distance_loglik(
                embedding=edata[1],
                params=params,
                recomb=self.recomb,
                lengths=lengths[1],
            )
        return loglik


def simulate_and_get_embeddings(
    sptree: toytree.ToyTree,
    params: Dict[str, int],
    nsamples: int,
    nsites: int,
    recomb: float,
    seed: int,
    data_type: str,
    threads: int,
    nloci: int,
    ) -> Tuple[np.ndarray, ipcoal.smc.likelihood.Embedding]:
    """Simulate a tree sequence, get embedding info, and return.
    """
    # set Ne values on species tree
    sptree.set_node_data("Ne", mapping=params, inplace=True)

    # setup a coalescent simulation model
    model = ipcoal.Model(sptree, nsamples=nsamples, recomb=recomb, seed_trees=seed)

    # get mapping of sample names to lineages
    imap = model.get_imap_dict()

    # print some details
    logger.info(f"simulating {nloci} tree sequences for {nsites:.2g} sites w/ recomb={recomb:.2g}.")

    # generate a tree sequence and store to a table
    model.sim_trees(nloci=nloci, nsites=nsites)

    # load genealogies
    genealogies = []
    for lidx in range(nloci):
        mtree = toytree.mtree(model.df[model.df.locus == lidx].genealogy)
        genealogies.append(mtree)

    # print some details
    logger.info(f"loading genealogy embedding table for {[len(i) for i in genealogies]} genealogies.")

    # get cached embedding tables
    multi_lengths = []
    multi_embeddings = []

    for lidx in range(nloci):
        args = (model.tree, genealogies[lidx], imap, threads)
        glengths = model.df[model.df.locus == lidx].nbps.values

        if data_type == "tree":
            edata = ipcoal.smc.likelihood.TreeEmbedding(*args)

        elif data_type == "topology":
            glengths = ipcoal.smc.likelihood.get_topology_interval_lengths(model, lidx)
            edata = ipcoal.smc.likelihood.TopologyEmbedding(*args)

        elif data_type == "combined":
            edata0 = ipcoal.smc.likelihood.TreeEmbedding(*args)
            edata1 = ipcoal.smc.likelihood.TopologyEmbedding(*args)
            edata = [edata0, edata1]
            tlengths = ipcoal.smc.likelihood.get_topology_interval_lengths(model, lidx)
            glengths = [glengths, tlengths]
        else:
            raise TypeError(f"data_type {data_type} arg not recognized: should be tree, topology, or combined.")

        # store results across loci
        multi_lengths.append(glengths)
        multi_embeddings.append(edata)

    # logger.info(f"embedding includes {[len(i) for i in lengths1)} sequential topology changes.")                
    return multi_lengths, multi_embeddings


def get_species_tree(
    ntips: int,
    root_height: float,
    ) -> toytree.ToyTree:
    """Return a species tree with same height given ntips."""
    if ntips == 1:
        sptree = toytree.tree("(r);")
    else:
        sptree = toytree.rtree.baltree(ntips)
    sptree = sptree.mod.edges_scale_to_root_height(treeheight=root_height, include_stem=True)
    return sptree


def main(
    ntips: int,
    root_height: float,
    params: Tuple[int],
    nsamples: int,
    nsites: int,
    recomb: float,
    seed: int,
    name: str,
    mcmc_nsamples: int,
    mcmc_sample_interval: int,
    mcmc_print_interval: int,
    mcmc_burnin: int,
    mcmc_jumpsize: int,
    force: bool,
    data_type: str,
    threads: int,
    prior: Sequence[Any],
    nloci: int,
    *args,
    **kwargs,
    ) -> None:
    """Run the main function of the script.

    This simulates a tree sequence under a given demographic model
    and generates a genealogy embedding table representing info on all
    genealogies across the chromosome, and their lengths (the ARG
    embedded in the MSC model, as a table).

    An MCMC algorithm is then run to search over the prior parameter
    space of the demographic model parameters to find the best fitting
    Ne values to explain the observed tree sequence waiting distances
    under tree changes.

    The posterior is saved to file as a numpy array.
    """
    outbase = Path(name).expanduser().absolute()
    outpath = outbase.with_suffix(".npy")
    outlog = outbase.with_suffix(".log")
    outpath.parent.mkdir(exist_ok=True)
    if force and outpath.exists():
        outpath.unlink()

    # copy command to the logger
    logfile = None if not kwargs['log_file'] else outlog
    ipcoal.set_log_level(log_level=kwargs["log_level"], log_file=logfile)
    logger.info(f"CMD: {sys.argv[0].rsplit('/')[-1]} {' '.join(sys.argv[1:])}")

    # limit parallelism
    set_num_threads(threads)

    # get species tree topology
    sptree = get_species_tree(ntips, root_height)

    # convert params to dict
    params = np.array(params)
    params_dict = {i: params[i] for i in range(sptree.nnodes)}
    logger.info(f"true simulated parameters ({params}).")

    # simulate genealogies under MSC topology and parameters
    # and get the ARG and embedding data.
    args = (sptree, params_dict, nsamples, nsites, recomb, seed, data_type, threads, nloci)
    lengths, edata = simulate_and_get_embeddings(*args)

    # initial random params
    init_params = np.repeat(5e5, len(params))

    # does a checkpoint file already exist for this run?
    # if outpath.exists():
    #     # get its length, subtract from nsamples, and set burnin to 0
    #     sampled = np.load(outpath)
    #     nsampled = sampled.shape[0]
    #     mcmc_burnin = 0
    #     mcmc_nsamples = mcmc_nsamples - nsampled
    #     init_params = sampled[-1][:-1]
    #     seed = sampled[-1][-1]
    #     logger.info(f"restarting from checkpoint (samples={nsampled})")

    # init MCMC object
    if data_type == "tree":
        mcmc_tool = McmcTree
    elif data_type == "topology":
        mcmc_tool = McmcTopology
    else:
        mcmc_tool = McmcCombined

    # init priors object
    # prior = [get_prior('g', 3, 0.001) for param in init_params]

    # init
    mcmc = mcmc_tool(
        recomb=recomb,
        lengths=lengths,
        embedding=edata,
        prior=prior,
        init_params=init_params,
        jumpsize=mcmc_jumpsize,
        seed=seed,
        outpath=outpath,
    )
    logger.info(f"({params}) loglik {mcmc.log_likelihood(params):.3f}.")

    # run MCMC chain
    posterior = mcmc.run(
        nsamples=mcmc_nsamples,
        print_interval=mcmc_print_interval,
        sample_interval=mcmc_sample_interval,
        burnin=mcmc_burnin,
    )

    # if adding to existing data then concatenate first.
    # if outpath.exists():
        # posterior = np.concatenate([sampled, posterior])
    np.save(outpath, posterior)
    logger.info(f"saved posterior w/ {posterior.shape[0]} samples to {outpath}.")


def cli():
    """Parse command line arguments and run main()."""
    parser = argparse.ArgumentParser(
        description="MCMC model fit for MS-SMC'")
    parser.add_argument(
        '--ntips', type=int, default=2, help='Number of species tree tips')
    parser.add_argument(
        '--root-height', type=float, default=1e6, help='Root height of species tree.')
    parser.add_argument(
        '--params', type=float, default=[200_000, 300_000, 400_000], nargs="*", help='True Ne values used for simulated data.')
    parser.add_argument(
        '--recomb', type=float, default=2e-9, help='Recombination rate.')
    parser.add_argument(
        '--nsites', type=float, default=1e5, help='length of simulated tree sequence')
    parser.add_argument(
        '--nsamples', type=int, default=4, help='Number of samples per species lineage')
    parser.add_argument(
        '--seed', type=int, default=666, help='Random number generator seed')
    parser.add_argument(
        '--name', type=str, default='combined-50loci-1e5sites', help='Prefix path for output files')
    parser.add_argument(
        '--mcmc-nsamples', type=int, default=10000, help='Number of samples in posterior')
    parser.add_argument(
        '--mcmc-sample-interval', type=int, default=5, help='N accepted iterations between samples')
    parser.add_argument(
        '--mcmc-print-interval', type=int, default=50, help='N accepted iterations between printing progress')
    parser.add_argument(
        '--mcmc-burnin', type=int, default=200, help='N accepted iterations before starting sampling')
    parser.add_argument(
        '--threads', type=int, default=7, help='Max number of threads (0=all detected)')
    parser.add_argument(
        '--mcmc-jumpsize', type=float, default=[20_000], nargs="*", help='MCMC jump size.')
    parser.add_argument(
        '--log-level', type=str, default="INFO", help='logger level (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument(
        '--log-file', action='store_true', help='save a log file to [name].log')
    parser.add_argument(
        '--force', action='store_true', help='Overwrite existing file w/ same name.')
    parser.add_argument(
        '--data-type', type=str, default="combined", help='tree, topology, or combined')
    parser.add_argument(
        '--prior', type=str, nargs="*", default=['u', 10, 5e6], help='prior on Ne')
    parser.add_argument(
        '--nloci', type=int, default=50, help='Number of independent loci (chromosomes)')

    cli_args = parser.parse_args()
    main(**vars(cli_args))


if __name__ == "__main__":
    cli()
