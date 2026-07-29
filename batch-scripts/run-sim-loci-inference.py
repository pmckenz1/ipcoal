#!/usr/bin/env python

"""Recombination effects on MSC inference.

This script can be used to run a simulation and inference pipeline
for a single set of parameters from the command line.

Authors: Deren Eaton and Patrick McKenzie

Example
-------
>>> python run-sim-loci-inference.py --neff 100000 \
>>> --ctime 1.5 --recomb 5e-08 --nsites 2000 --nloci 20000 --rep 81
"""

import sys
from typing import List
import argparse
from pathlib import Path
import pandas as pd
import toytree
import ipcoal

CHUNKSIZE = 1_000

def run_sim_loci_and_infer_gene_trees(
    tree: toytree.ToyTree,
    ctime: int,
    recomb: float,
    mut: float,
    neff: int,
    rep: int,
    seed: int,
    nsites: int,
    outdir: Path,
    ncores: int,
    nloci: List[int],
    raxml_bin: Path,
    astral_bin: Path,
    chunksize: int=CHUNKSIZE,
    ):
    """Simulate N loci and infer gene trees.
    
    If we requested 20K gene trees for rep 0 then this will represent
    a chunk of loci from a unique seed...
    """
    # create name for this job based on params
    params = (
        f"neff{int(neff)}-ctime{ctime}-"
        f"recomb{int(bool(recomb))}-"
        f"nloci{max(nloci)}-nsites{nsites}"
    )

    # create a subdir in the outdir for this param set, all reps.
    jobdir = outdir / f"res-{params}"
    jobdir.mkdir(exist_ok=True)

    # create a tmpdir in the outdir for this param set, this rep.
    tmpdir = outdir / f"tmp-{rep}-{params}"
    tmpdir.mkdir(exist_ok=True)

    # transform species tree from units of coal time (2 * diploid Ne)
    # into units of generations.
    root_in_gens = ctime * 2 * neff
    sptree = tree.mod.edges_scale_to_root_height(root_in_gens)

    # init coal Model
    model = ipcoal.Model(
        sptree,
        Ne=neff,
        seed_trees=seed,
        seed_mutations=seed,
        mut=mut,
        recomb=recomb
    )

    # simulate the largest size dataset of NLOCI (in memory for now.)
    model.sim_loci(nloci=max(nloci), nsites=nsites)

    # break up gene tree inference into 1000 at a time, in case the 
    # it takes a long time to finish. This checks whether a saved 
    # chunk result exists and skips it if it does exist, until all
    # gene trees are inferred. Then it proceeds and uses these gene
    # trees in the astral inference.
    for lidx in range(0, max(nloci), chunksize):

        # skip the chunk if its csv already exists
        outname = jobdir / f"chunk-{lidx}-gene_trees.csv"
        if outname.exists():
            continue

        # infer gene trees for every locus and write to CSV.
        raxdf = ipcoal.phylo.infer_raxml_ng_trees(
            model,
            idxs=range(lidx, lidx + chunksize),
            nproc=ncores,
            nworkers=1,
            nthreads=1,
            seed=seed,
            binary_path=raxml_bin,
            tmpdir=tmpdir,
            cleanup=True,
        )
        raxdf.to_csv(outname)


def run_post_gene_tree_inference(
    tree: toytree.ToyTree,
    ctime: int,
    recomb: float,
    mut: float,
    neff: int,
    rep: int,
    seed: int,
    nsites: int,
    outdir: Path,
    ncores: int,
    nloci: List[int],
    raxml_bin: Path,
    astral_bin: Path,
    chunksize: int=CHUNKSIZE,
    ):
    """Writes simulated genealogies and inference results to WORKDIR.

    This only nees to be run for the largest NLOCI value.
    """
    # create name for this job based on params
    params = (
        f"neff{int(neff)}-ctime{ctime}-"
        f"recomb{int(bool(recomb))}-"
        f"nloci{max(nloci)}-nsites{nsites}"
    )

    # create a subdir in the outdir for this param set, all reps.
    jobdir = outdir / "res-" + params
    jobdir.mkdir(exist_ok=True)

    # create a tmpdir in the outdir for this param set, this rep.
    tmpdir = outdir / f"tmp-{rep}-" + params
    tmpdir.mkdir(exist_ok=True)

    # load and concatenate gene tree dataframes
    chunks = tmpdir.glob("chunk-*-gene_trees.csv")
    raxdfs = sorted(chunks, key=lambda x: int(x.name.split("-")[1]))
    raxdf = pd.concat((pd.read_csv(i) for i in raxdfs), ignore_index=True)

    # iterate over subsample sizes of NLOCI
    for numloci in sorted(nloci):
        numloci = int(numloci)

        # infer a concatenation tree for loci 0-Nloci
        ctree = ipcoal.phylo.infer_raxml_ng_tree(
            model,
            idxs=list(range(0, numloci)),
            nworkers=1,
            nthreads=ncores,
            seed=seed,
            binary_path=raxml_bin,
            tmpdir=tmpdir,
        )
        ctree.write(jobdir / f"rep{rep}-concat-subloci{numloci}.nwk")

        # infer astral species tree from true genealogies (the first)
        # genealogy at each locus, since subsequent trees are linked.
        genealogies = model.df.loc[model.df.tidx == 0].genealogy
        atree1 = ipcoal.phylo.infer_astral_tree(
            toytree.mtree(genealogies),
            binary_path=astral_bin,
            seed=seed,
            tmpdir=tmpdir,
        )
        atree1.write(jobdir / f"rep{rep}-astral-genealogy-subloci{numloci}.nwk")

        # infer astral species tree from inferred gene trees.
        genetrees = toytree.mtree(raxdf.gene_tree)[:numloci]
        atree2 = ipcoal.phylo.infer_astral_tree(
            genetrees,
            binary_path=astral_bin,
            seed=seed,
            tmpdir=tmpdir,
        )
        atree2.write(jobdir / f"rep{rep}-astral-genetree-subloci{numloci}.nwk")

    # cleanup
    jobparams = (
        f"neff{int(neff)}-ctime{ctime}-"
        f"recomb{int(bool(recomb))}-rep{rep}-"
        f"nloci{max(nloci)}-nsites{nsites}"
    )
    sh_file = outdir / (jobparams + ".sh")
    err_file = outdir / (jobparams + ".err")
    out_file = outdir / (jobparams + ".out")

    if not err_file.stat().st_size:
        err_file.unlink()
    if not out_file.stat().st_size:
        out_file.unlink()
    sh_file.unlink()

    # remove directory for ...
    tmpdir.rmdir()


def _single_command_line_parser():
    """Parse command line arguments and return.

    Example
    -------
    >>> python run-sim-loci-inference.py --neff 100000 \
    >>> --ctime 1.5 --recomb 5e-08 --nsites 2000 --nloci 20000 --rep 81
    """
    parser = argparse.ArgumentParser(
        description='Coalescent simulation and tree inference w/ recombination')
    parser.add_argument(
        '--neff', type=float, required=True, help='Effective population size')
    parser.add_argument(
        '--ctime', type=float, required=True, help='Root species tree height in coal units.')
    parser.add_argument(
        '--recomb', type=float, required=True, help='Recombination rate.')
    parser.add_argument(
        '--mut', type=float, required=True, help='Mutation rate.')
    parser.add_argument(
        '--nsites', type=int, required=True, help='length of simulated loci')
    parser.add_argument(
        '--nloci', type=int, nargs="*", required=True, help='Number of independent loci to simulate')
    parser.add_argument(
        '--rep', type=int, required=True, help='replicate id.')
    parser.add_argument(
        '--seed', type=int, required=True, help='random seed.')
    parser.add_argument(
        '--outdir', type=Path, required=True, help='directory to write output files (e.g., scratch)')
    parser.add_argument(
        '--ncores', type=int, required=True, help='number of cores.')
    parser.add_argument(
        '--node-heights', type=float, nargs=4, required=True, help='imbalanced species tree RELATIVE node heights.')
    parser.add_argument(
        '--raxml-bin', type=Path, help='path to raxml-ng binary')
    parser.add_argument(
        '--astral-bin', type=Path, help='path to astral-III binary')
    # parser.add_argument(
    #     '--astral-path', type=Path, help='directory with raxml-ng and astral3 binaries')
    return parser.parse_args()


def main():
    """Parse command line args and start an inference job."""
    args = _single_command_line_parser()

    # set node heights on a fixed 5-taxon imbalanced species tree.
    # first scale node heights to be RELATIVE, scaled by largest=1.
    # `ctime` is used to scale all branches so that root is larger 
    # than 1 coal unit. That happens in `run_sim_loci_inference()`
    imbtree = toytree.rtree.imbtree(ntips=5)
    max_height = max(args.node_heights)
    heights = [float(i) / max_height for i in args.node_heights]
    sptree = imbtree.set_node_data("height", dict(zip(range(5, 9), heights)))

    # find path to raxml-ng binary
    user_bin = Path(args.raxml_bin) if args.raxml_bin else None
    conda_bin = Path(sys.prefix) / "bin" /  "raxml-ng"
    args.raxml_bin = (user_bin if args.raxml_bin else conda_bin)
    assert args.raxml_bin.exists(), (
        f"Cannot find {args.raxml_bin}. Use conda instructions.")

    # find path to the astral binary
    user_bin = Path(args.astral_bin) if args.astral_bin else None
    conda_bin = Path(sys.prefix) / "bin" /  "astral.5.7.1.jar"
    args.astral_bin = (user_bin if args.astral_bin else conda_bin)
    assert args.astral_bin.exists(), (
        f"Cannot find {args.astral_bin}. Use conda instructions.")

    # ensure outdir exists
    Path(args.outdir).mkdir(exist_ok=True)

    # run main function
    run_sim_loci_and_infer_gene_trees(
        tree=sptree,
        ctime=args.ctime,
        recomb=args.recomb,
        mut=args.mut,
        neff=args.neff,
        rep=args.rep,
        seed=args.seed,
        nloci=args.nloci,
        nsites=args.nsites,
        outdir=args.outdir,
        ncores=args.ncores,
        raxml_bin=args.raxml_bin,
        astral_bin=args.astral_bin,
    )


if __name__ == "__main__":
    main()
