#!/usr/bin/env python

"""Recombination effects on MSC inference.


Authors: Deren Eaton and Patrick McKenzie
"""

import sys
from typing import List
import argparse
from pathlib import Path
import toytree
import ipcoal


def run_sim_loci_inference(
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
    jobdir = outdir / params
    jobdir.mkdir(exist_ok=True)

    # scale species tree to a new root height in generations
    root_in_gens = ctime * 4 * neff
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

    # simulate the largest size dataset of NLOCI
    model.sim_loci(nloci=max(nloci), nsites=nsites)
    # model.df.to_csv(locpath)  # uncomment to save genealogies

    # infer gene trees for every locus and write to CSV
    raxdf = ipcoal.phylo.infer_raxml_ng_trees(
        model,
        nproc=ncores,
        nworkers=1,
        nthreads=1,
        seed=seed,
        binary_path=raxml_bin,
        tmpdir=jobdir,
    )
    # raxdf.to_csv(gtpath)  # uncomment to save gene trees

    # iterate over subsample sizes of NLOCI
    for numloci in sorted(nloci):
        numloci = int(numloci)

        # infer a concatenation tree
        ctree = ipcoal.phylo.infer_raxml_ng_tree(
            model,
            idxs=list(range(0, numloci)),
            nworkers=1,
            nthreads=ncores,
            seed=seed,
            binary_path=raxml_bin,
            tmpdir=jobdir,
        )
        ctree.write(jobdir / f"rep{rep}-concat-subloci{numloci}.nwk")

        # infer astral species tree from true genealogies (the first)
        # genealogy at each locus, since subsequent trees are linked.
        genealogies = model.df.loc[model.df.tidx == 0].genealogy
        atree1 = ipcoal.phylo.infer_astral_tree(
            toytree.mtree(genealogies),
            binary_path=astral_bin,
            seed=seed,
            tmpdir=jobdir,
        )
        atree1.write(jobdir / f"rep{rep}-astral-genealogy-subloci{numloci}.nwk")

        # infer astral species tree from inferred gene trees.
        atree2 = ipcoal.phylo.infer_astral_tree(
            toytree.mtree(raxdf.gene_tree),
            binary_path=astral_bin,
            seed=seed,
            tmpdir=jobdir,
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




def single_command_line_parser():
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
        '--node-heights', type=float, nargs=4, required=True, help='imbalanced species tree relative node heights.')
    parser.add_argument(
        '--raxml-bin', type=Path, help='path to raxml-ng binary')
    parser.add_argument(
        '--astral-bin', type=Path, help='path to astral-III binary')
    # parser.add_argument(
    #     '--astral-path', type=Path, help='directory with raxml-ng and astral3 binaries')
    return parser.parse_args()


def main():
    """Parse command line args and start an inference job."""
    args = single_command_line_parser()

    imbtree = toytree.rtree.imbtree(ntips=5)
    sptree = imbtree.set_node_data(
        "height", dict(zip(range(5, 9), args.node_heights)))

    args.raxml_bin = (
        Path(args.raxml_bin) if args.raxml_bin
        else Path(sys.prefix) / "bin" /  "raxml-ng")
    assert args.raxml_bin.exists(), (
        f"Cannot find {args.raxml_bin}. Use conda instructions.")

    args.astral_bin = (
        Path(args.astral_bin) if args.astral_bin
        else Path(sys.prefix) / "bin" /  "astral.5.7.1.jar")
    assert args.astral_bin.exists(), (
        f"Cannot find {args.astral_bin}. Use conda instructions.")

    Path(args.outdir).mkdir(exist_ok=True)

    run_sim_loci_inference(
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
