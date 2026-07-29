#!/usr/bin/env python

"""Recombination effects on MSC inference.

This runs a single job on 4 cores and is intended to be distributed
on HPC to iterate over massive combinations of parameters.

Author: Deren Eaton
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import toytree
import ipcoal

# full paths to conda-installed phylo inference binaries
HOME = Path("/moto/home/de2356/")
ASTRAL = HOME / "miniconda3/bin/astral.5.7.1.jar"
RAXML_NG = HOME / "miniconda3/bin/raxml-ng"
WORKDIR = Path("/moto/eaton/users/de2356/recomb-response/")
TMPDIR = WORKDIR / "tables"

# coalescent simulation parameters
NCORES = 2
NREPS = 100
SEEDS = np.random.default_rng(123).integers(1e12, size=NREPS)     # seed set used for NREPS @ each param setting
POPSIZE = 1e4                                                          # constant Ne across sptree
NSITES = 2000                                                          # locus length 

NLOCI = np.array([100])#, 500, 1000, 2500, 5000, 10000, 20000])   # range of N independent loci
# NLOCI = np.array([100, 250, 500, 1000, 2500, 5000, 10000, 20000])    # range of N independent loci
GTIMES = np.array([50, 100, 200, 300, 400, 500, 750, 1000]) * 1000     # range of treeheights (gens) to test
CTIMES = GTIMES / (4 * POPSIZE)                                        # range of times in coal units
MUT = 5e-8                                                             # mut rate (/site/gen)
RECOMB = [0, 5e-9]                                                     # recomb rate (/site/gen)
IMBTREE = toytree.rtree.imbtree(ntips=5)                               # an imbalanced sptree topology
RELTREE = IMBTREE.set_node_data("height", {5: 0.050, 6: 0.055, 7: 0.060, 8: 1}) # relative edge len sptree


def get_scaled_sptree(tree_height: int) -> toytree.ToyTree:
    """Return the species tree with all edges scaled to a new root height."""
    sptree = RELTREE.mod.edges_scale_to_root_height(tree_height)
    return sptree


def get_sim_loci(time: int, recomb: float, idx: int):
    """Writes simulated genealogies and seqs array to WORKDIR.
    
    This is only run on the largest NLOCI global since all smaller
    datasets can subsample from this largest one.
    """
    # params string
    params = f"th{int(time)}-nloci{NLOCI[-1]}-recomb{int(bool(recomb))}-rep{idx}"
    locpath = TMPDIR / (params + "-sim_loci.csv")
    if not locpath.exists():

        # scale species tree to root height and init coal Model
        tmp_sptree = get_scaled_sptree(time)
        model = ipcoal.Model(
            tmp_sptree, Ne=POPSIZE, 
            seed_trees=SEEDS[idx], seed_mutations=SEEDS[idx], 
            mut=MUT, recomb=recomb
        )
        model.sim_loci(nloci=NLOCI[-1], nsites=NSITES)
        model.df.to_csv(locpath)
        
        # infer gene trees and write to CSV
        raxpath = TMPDIR / (params + "-raxml_trees.csv")
        raxdf = ipcoal.phylo.infer_raxml_ng_trees(
            model, nproc=NCORES, nworkers=1, nthreads=1, seed=SEEDS[idx])
        raxdf.to_csv(raxpath)
        return raxdf


if __name__ == "__main__":

    # WORKDIR.mkdir(exist_ok=True)
    # TMPDIR.mkdir(exist_ok=True)

    jidx = sum([int(i) for i in sys.argv[1:]])
    grid = np.arange(8 * 2 * 100).reshape((8, 2, 100)) #(GTIMES, [0, RECOMB], range(NREPS))
    gidx, ridx, idx = map(int, np.where(grid == jidx))
    gtime = GTIMES[gidx]
    recomb = RECOMB[ridx]
    rep = range(NREPS)[idx]

    print(f"JOB {jidx}: gtime={gtime}, recomb={recomb}, rep={rep}")
    # raxdf = get_sim_loci(gtime, recomb, rep)
    # print(raxdf)
