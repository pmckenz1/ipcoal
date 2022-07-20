#!/usr/bin/env python

"""Recombination effects on MSC inference.

Authors: Deren Eaton, Patrick McKenzie, and Scarlet Au
"""

import sys
from typing import List
import argparse
from pathlib import Path
import toytree
import pandas as pd
import ipcoal


# FIXED INPUT TREE
MAMMALS_27_TIPS_RELATIVE_EDGES = toytree.tree("""(Ornithorhynchus_anatinus_ORNITHORHYNCHIDAE_MONOTREMATA:2.27219e+07,((Monodelphis_domestica_DIDELPHIDAE_DIDELPHIMORPHIA:3.7832e+07,Macropus_eugenii_MACROPODIDAE_DIPROTODONTIA:1.53532e+07)0:2.58354e+07,(((Choloepus_hoffmanni_MEGALONYCHIDAE_PILOSA:4.70065e+06,Dasypus_novemcinctus_DASYPODIDAE_CINGULATA:1.31247e+07)0:2.10194e+06,(Echinops_telfairi_TENRECIDAE_AFROSORICIDA:1.47965e+07,(Loxodonta_africana_ELEPHANTIDAE_PROBOSCIDEA:1.88991e+06,Procavia_capensis_PROCAVIIDAE_HYRACOIDEA:5.60073e+06)0:1.94372e+06)0:482422)0:138844,(((Erinaceus_europaeus_ERINACEIDAE_EULIPOTYPHLA:1.64741e+07,Sorex_araneus_SORICIDAE_EULIPOTYPHLA:5.04953e+07)0:2.12066e+06,((Pteropus_vampyrus_PTEROPODIDAE_CHIROPTERA:7.92231e+06,Myotis_lucifugus_VESPERTILIONIDAE_CHIROPTERA:5.66853e+06)0:1.34433e+06,(Sus_scrofa_SUIDAE_CETARTIODACTYLA:7.31967e+06,Tursiops_truncatus_DELPHINIDAE_CETARTIODACTYLA:3.02457e+06)0:1.90325e+06)0:168392)0:442025,(((Ochotona_princeps_OCHOTONIDAE_LAGOMORPHA:1.73297e+07,Oryctolagus_cuniculus_LEPORIDAE_LAGOMORPHA:1.19592e+07)0:6.8529e+06,(Dipodomys_ordii_HETEROMYIDAE_RODENTIA:2.17261e+07,Mus_musculus_MURIDAE_RODENTIA:3.94926e+07)0:2.3554e+06)0:276914,(Tupaia_belangeri_TUPAIIDAE_SCANDENTIA:1.67878e+07,((Otolemur_garnettii_GALAGIDAE_PRIMATES:7.60855e+06,Microcebus_murinus_CHEIROGALEIDAE_PRIMATES:8.43923e+06)0:2.72356e+06,(Tarsius_syrichta_TARSIIDAE_PRIMATES:1.00465e+07,(Callithrix_jacchus_CALLITRICHIDAE_PRIMATES:5.15897e+06,(Macaca_mulatta_CERCOPITHECIDAE_PRIMATES:1.40441e+06,(Pongo_pygmaeus_HOMINIDAE_PRIMATES:458063,(Gorilla_gorilla_HOMINIDAE_PRIMATES:260654,Pan_troglodytes_HOMINIDAE_PRIMATES:286719)0:163184)0:500365)0:619821)0:2.2589e+06)0:327620)0:404022)0:997969)0:1.50881e+06)0:726102)0:9.1514e+06)0:2.70448e+06);""")
CHUNKSIZE = 1000


def run_sim_loci_inference(
    tree: toytree.ToyTree,
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

    This only needs to be run for the largest NLOCI value.
    """
    # create name for this job based on params
    params = (
        f"neff{int(neff)}-root{int(tree.treenode.height)}-"
        f"recomb{int(bool(recomb))}-"
        f"nloci{max(nloci)}-nsites{nsites}"
    )
    jobdir = outdir / params
    jobdir.mkdir(exist_ok=True)

    # init coal Model
    model = ipcoal.Model(
        tree,
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
    # raxdf = ipcoal.phylo.infer_raxml_ng_trees(
    #     model,
    #     nproc=ncores,
    #     nworkers=1,
    #     nthreads=1,
    #     seed=seed,
    #     binary_path=raxml_bin,
    #     tmpdir=outdir,
    # )
    # raxdf.to_csv(gtpath)  # uncomment to save gene trees

    # break up gene tree inference into 1000 at a time, in case the 
    # it takes a long time to finish. This checks whether a saved 
    # chunk result exists and skips it if it does exist, until all
    # gene trees are inferred. Then it proceeds and uses these gene
    # trees in the astral inference.
    for lidx in range(0, nloci, CHUNKSIZE):

        # skip the chunk if its csv already exists
        outname = jobdir / f"rep-{lidx}-gene_trees.csv"
        if outname.exists():
            continue

        # infer gene trees for every locus and write to CSV
        raxdf = ipcoal.phylo.infer_raxml_ng_trees(
            model,
            idxs=range(lidx, lidx + CHUNKSIZE),
            nproc=ncores,
            nworkers=1,
            nthreads=1,
            seed=seed,
            binary_path=raxml_bin,
            tmpdir=jobdir,
        )
        raxdf.to_csv(outname)

    # load and concatenate gene tree dataframes
    raxdfs = sorted(jobdir.glob("rep-*-gene_trees.csv"))
    raxdf = pd.concat(raxdfs, ignore_index=True)

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
        f"neff{int(neff)}-root{int(tree.treenode.height)}-"
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
    >>> python run-mammals.py --neff 100000 \
    >>> --ctime 1.5 --recomb 5e-08 --nsites 2000 --nloci 20000 --rep 81
    """
    parser = argparse.ArgumentParser(
        description='Coalescent simulation and tree inference w/ recombination')
    parser.add_argument(
        '--neff', type=float, required=True, help='Effective population size')
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
        '--root-height', type=float, required=True, help='species tree root height (gens).')
    parser.add_argument(
        '--raxml-bin', type=Path, help='path to raxml-ng binary')
    parser.add_argument(
        '--astral-bin', type=Path, help='path to astral-III binary')
    return parser.parse_args()



if __name__ == "__main__":

    # sys.argv = """
    # run-mammals.py \
    #     --neff 100_000 \
    #     ...
    # """
    args = single_command_line_parser()

    args.raxml_bin = (
        Path(args.raxml_bin) if args.raxml_bin
        else Path(sys.prefix) / "bin" /  "raxml-ng")
    assert args.raxml_bin.exists(), f"Cannot find {args.raxml_bin}. Use conda instructions."

    args.astral_bin = (
        Path(args.astral_bin) if args.astral_bin
        else Path(sys.prefix) / "bin" /  "astral.5.7.1.jar")
    assert args.astral_bin.exists(), f"Cannot find {args.astral_bin}. Use conda instructions."

    Path(args.outdir).mkdir(exist_ok=True)

    run_sim_loci_inference(
        tree=MAMMALS_27_TIPS_RELATIVE_EDGES,
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
