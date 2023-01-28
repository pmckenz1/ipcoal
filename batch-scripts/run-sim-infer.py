#!/usr/bin/env python

"""Simulate genealogies and infer gene trees on a 5-tip species tree.

This simulation is setup so that the user can modify species tree
parameters that affect genealogical variation while keeping the 
genetic diversity constant. In other words, the user can change
the height of the species tree in coalescent units, and can set
the effective population size, and we will automatically scale the
species tree edge lengths in units of generations so that that 
theta is 2 * Ne * mu. 

To increase the genetic diversity we incrase neff, to increase the
...

Example CLI
-----------
python run-sim-infer.py \
    --neff 1e5 \
    --ctime 1.0 \
    --recomb 5e-9 \
    --mut 5e-8 \
    --nsites 1000 \
    --nloci 100 \
    --rep 0 \
    --seed 123 \
    --outdir /tmp/test \
    --ncores 4 \
    --node-heights 0.1 0.5 0.6 1.0 \
    --astral-bin ... \
    --raxml-bin ... 
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import argparse
import sys
from pathlib import Path
import pandas as pd
from loguru import logger
import toytree
import ipcoal

CHUNKSIZE = 1_000
logger = logger.bind(name="ipcoal")


@dataclass
class FiveTipImbTreeAnalyzer:
    node_heights: List[float]
    """: relative heights of internal nodes."""
    ctime: int
    """: scale all nodes of sptree so root is at ctime coal units."""
    neff: int
    """: uniform Ne applied to all edges of sptree."""
    recomb: float
    mut: float
    rep: int
    seed: int
    nsites: int
    nloci: List[int]
    outdir: Path
    """: dir for all replicate jobs of this param setting."""
    ncores: int
    raxml_bin: Path
    astral_bin: Path
    chunksize: int=CHUNKSIZE

    # attrs to be filled
    params: str = None
    """: basename for folder, does not include rep number."""    
    outfile: Path = None
    """: CSV result file written to outdir."""
    tmpdir: Path = None
    """: dir for tmp raxml files and slurm job files."""
    sptree: toytree.ToyTree = None
    """: species tree with edges in units of generations."""
    model: ipcoal.Model = None
    data: pd.DataFrame = None

    def __post_init__(self):
        self.params = (
            f"neff{int(self.neff)}-ctime{self.ctime}-"
            f"recomb{str(self.recomb).replace('-', '')}-"
            f"nloci{max(self.nloci)}-nsites{self.nsites}-"
            f"rep{self.rep}"
        )

        # create a subdir in the outdir for this param set, all reps.
        self.outdir.mkdir(exist_ok=True)
        self.outfile = self.outdir / f"res-{self.params}.csv"
        # self.jobdir = self.outdir / f"res-{self.params}"
        # self.jobdir.mkdir(exist_ok=True)

        # create a tmpdir in the outdir for this param set, this rep.
        self.tmpdir = self.outdir / f"tmp-{self.params}"
        self.tmpdir.mkdir(exist_ok=True)

        # logging file
        # self.logfile = self.outdir / f"res-{self.params}.log"
        # logger.info(f"initiating: {self.params}")

        # create a 5-tip imbalanced tree
        imbtree = toytree.rtree.imbtree(ntips=5)

        # scale relative heights to the root is at 1 unit.
        max_height = max(self.node_heights)
        heights = [float(i) / max_height for i in self.node_heights]

        # create a species tree with special attr 'height' set
        sptree = imbtree.set_node_data("height", dict(zip(range(5, 9), heights)))

        # transform all edges so that root height in generations 
        # scales to ctime coal units.
        root_in_gens = self.ctime * 2 * self.neff
        self.sptree = sptree.mod.edges_scale_to_root_height(root_in_gens)

        # init coal Model
        self.model = ipcoal.Model(
            self.sptree,
            Ne=self.neff,
            seed_trees=self.seed,
            seed_mutations=self.seed,
            mut=self.mut,
            recomb=self.recomb
        )

        theta = 2 * self.neff * self.mut
        rho = 2 * self.neff * self.recomb
        logger.info("sptree "
            f"root_tc={self.ctime:.12g}, "
            f"root_tg={root_in_gens:.12g}, "
            f"neff={self.neff:.12g}, "
            f"theta={theta:.12g}, "
            f"rho={rho:.12g}, "
            f"rho/theta={(rho / theta):.12g}"
        )

    def get_raxdf(self) -> pd.DataFrame:
        """Return concatenated dataframe of raxml gene trees."""
        # load and concatenate gene tree dataframes
        chunks = self.tmpdir.glob("chunk-*-gene_trees.csv")
        raxdfs = sorted(chunks, key=lambda x: int(x.name.split("-")[1]))
        raxdf = pd.concat((pd.read_csv(i) for i in raxdfs), ignore_index=True)
        return raxdf

    ###################################################################
    ###
    ###################################################################
    def _simulate_sequences(self) -> None:
        """Simulate the largest size dataset of NLOCI (in memory for now.)"""
        self.model.sim_loci(nloci=max(self.nloci), nsites=self.nsites)
        ngen = self.model.df.groupby('locus').size().mean()
        logger.info(f"simulated {max(self.nloci)} loci; len={self.nsites}; mean-ngenealogies-per-locus={ngen:.2f}.")

    def _infer_raxml_gene_trees(self) -> None:
        """Infer gene tree for every locus and write to a CSV in jobdir."""

        # break up gene tree inference into 1000 at a time, in case the 
        # it takes a long time to finish. TODO: This checks whether a saved 
        # chunk result exists and skips it if it does exist, until all
        # gene trees are inferred. Then it proceeds and uses these gene
        # trees in the astral inference.
        for lidx in range(0, max(self.nloci), CHUNKSIZE):

            # skip the chunk if its csv already exists
            outname = self.tmpdir / f"chunk-{lidx}-gene_trees.csv"
            # if outname.exists():
                # print("skipping gene trees")
                # continue

            # infer gene trees for every locus and write to CSV.
            raxdf = ipcoal.phylo.infer_raxml_ng_trees(
                self.model,
                idxs=range(lidx, lidx + self.chunksize),
                nproc=self.ncores,
                nworkers=1,
                nthreads=1,
                seed=self.seed,
                binary_path=self.raxml_bin,
                tmpdir=self.tmpdir,
                cleanup=True,
            )
            raxdf.to_csv(outname)
        logger.info(f"inferred gene trees for {max(self.nloci)} unlinked loci.")

    def _infer_concatenated_trees(self) -> None:
        """Infers concatenated gene tree(s) and writes to newick in jobdir."""

        # iterate over subsample sizes of NLOCI
        for numloci in sorted(self.nloci):
            numloci = int(numloci)

            # infer a concatenation tree for loci 0-Nloci
            ctree = ipcoal.phylo.infer_raxml_ng_tree(
                self.model,
                idxs=list(range(0, numloci)),
                nworkers=1,
                nthreads=self.ncores,
                seed=self.seed,
                binary_path=self.raxml_bin,
                tmpdir=self.tmpdir,
            )
            # ctree.write(self.jobdir / f"rep{self.rep}-concat-subloci{numloci}.nwk")
            self.data.loc[self.data.nloci == numloci, "concat"] = ctree.write()
        logger.info(f"inferred gene trees for concatenated sets of {self.nloci} unlinked loci.")

    def _infer_astral_trees(self) -> None:
        """Infers concatenated gene tree(s) and writes to newick in jobdir."""

        raxdf = self.get_raxdf()
        # iterate over subsample sizes of NLOCI
        for numloci in sorted(self.nloci):
            numloci = int(numloci)

            # infer astral species tree from true genealogies (the first)
            # genealogy at each locus, since subsequent trees are linked.
            genealogies = self.model.df.loc[self.model.df.tidx == 0].genealogy
            atree1 = ipcoal.phylo.infer_astral_tree(
                toytree.mtree(genealogies),
                binary_path=self.astral_bin,
                seed=self.seed,
                tmpdir=self.tmpdir,
            )
            # atree1.write(self.jobdir / f"rep{self.rep}-astral-genealogy-subloci{numloci}.nwk")
            self.data.loc[self.data.nloci == numloci, "astral_ge"] = atree1.write()

            # infer astral species tree from inferred gene trees.
            genetrees = toytree.mtree(raxdf.gene_tree)[:numloci]
            atree2 = ipcoal.phylo.infer_astral_tree(
                genetrees,
                binary_path=self.astral_bin,
                seed=self.seed,
                tmpdir=self.tmpdir,
            )
            # atree2.write(self.jobdir / f"rep{self.rep}-astral-genetree-subloci{numloci}.nwk")
            self.data.loc[self.data.nloci == numloci, "astral_gt"] = atree2.write()

        logger.info(f"inferred astral trees from sets of {self.nloci} simulated genealogies.")
        logger.info(f"inferred astral trees from sets of {self.nloci} inferred gene trees.")

    def _infer_snaq_networks(self) -> None:
        """TODO... use phylo.infer_snaq but it is still quite slow."""

    def _setup_dataframe(self):
        """..."""
        cols = ["neff", "ctime", "nloci", "nsites", "rep", "concat", "astral_ge", "astral_gt"]
        data = pd.DataFrame(columns=cols, index=range(len(self.nloci)))
        data.neff = self.neff
        data.ctime = self.ctime
        data.nloci = self.nloci
        data.nsites = self.nsites
        data.rep = self.rep
        self.data = data

    def run(self) -> None:
        """..."""
        self._setup_dataframe()
        self._simulate_sequences()
        self._infer_raxml_gene_trees()
        self._infer_concatenated_trees()
        self._infer_astral_trees()

        # remove tmpdir and write CSV to outdir
        for tmpf in self.tmpdir.glob("*"):
            tmpf.unlink()
        self.tmpdir.rmdir()
        self.data.to_csv(self.outfile)
        logger.info(f"writing CSV to {self.outfile}.")


def single_command_line_parser() -> Dict[str, Any]:
    """..."""
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
    return vars(parser.parse_args())


def command_line_tool() -> None:
    """Runs the command line tool with logging to STDOUT."""
    ipcoal.set_log_level("INFO", sys.stdout)
    kwargs = single_command_line_parser()
    tool = FiveTipImbTreeAnalyzer(**kwargs)
    tool.run()

def interactive_test() -> None:
    """Run in main while testing."""
    ipcoal.set_log_level("INFO")
    tool = FiveTipImbTreeAnalyzer(
        nloci=[20, 50, 100],
        neff=20_000, 
        node_heights=[0.01, 0.05, 0.06, 1.0],
        ctime=1.0, 
        recomb=5e-9, 
        mut=5e-8,
        rep=1,
        seed=123,
        nsites=2000, 
        outdir=Path("/tmp/itest2"), 
        ncores=4, 
        raxml_bin=None, 
        astral_bin="/home/deren/miniconda3/envs/ipyrad/bin/astral.5.7.1.jar",
    )
    tool.run()


if __name__ == "__main__":

    command_line_tool()
    # interactive_test()
