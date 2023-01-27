#!/usr/bin/env python

"""Infer a phylogenetic network using SNAQ/phylonetworks.

This is a simple wrapper to write a julia script, execute it in julia,
and parse the results file. It will check that you have julia installed
either in your $PATH or at a specified path, and that you have the 
phylonetworks julia package installed. If not an error is raised.

"""

from typing import Optional, Tuple, Union, List
from dataclasses import dataclass
import subprocess
from pathlib import Path
import numpy as np
import toytree
import toyplot
import toyplot.svg
from loguru import logger

logger = logger.bind(name="ipcoal")

REQUIRED_PACKAGES = """
#!/usr/bin/env julia

using Pkg
Pkg.add("PhyloNetworks")
Pkg.add("CSV")
"""

GENE_TREE_COUNT_SCRIPT = """
#!/usr/bin/env julia

# load required packages
using PhyloNetworks
using CSV

# load gene trees and starting tree
gtrees = readMultiTopology("{gtrees}");

# count quartet CFs
q, t = countquartetsintrees(gtrees);

# reshape into dataframe
cfdf = writeTableCF(q, t);

# save table
CSV.write("{io_table}", cfdf);
"""

NETWORK_INFER_SCRIPT = """
#!/usr/bin/env julia

# check for required packages
using PhyloNetworks
using CSV
using Distributed
using DataFrames

# parallelize
addprocs({nproc})
@everywhere using PhyloNetworks

# load quartet-CF object from table
d_sp = readTableCF("{io_table}")

# load starting network
netin = readTopology("{net_in}")

# infer the network
snaq!(netin, d_sp, hmax={nedges}, filename="{out_net}", seed={seed}, runs={nruns})
"""



@dataclass
class Snaq:
    """Run simple snaq analyses on a list of gene trees.

    The input can be either a file with newick trees on separate 
    lines, or a list of newick strings, or a list of toytree 
    objects, or a DataFrame containing a column labeled .tree.
    """
    # required inputs
    gtrees: Path
    name: str = "test"
    workdir: Path = "analysis-snaq"

    # optional with defaults
    force: bool = False
    nruns: int = 10
    nproc: int = 4
    path_to_julia: Optional[str] = None

    # i/o paths for result files in workdir
    log: Path = None
    """: phylonetworks log file of inference run."""
    network: Path = None
    """: Path for output network result."""
    _io_table: Path = None
    """: The CF table file inferred from gtrees."""    
    _io_script: Path = None
    """: The tmp julia file that is executed. Available for debugging."""
    _basename: Path = None
    """: Basename provided to phylonetworks for output file names."""

    _seed: int = 0
    """: Seed for random number generator, provided in .run()."""
    _net_in: Path = None
    """: Path for a network input file, provided in .run()."""
    _nedges: int = 0
    """: Number of admixture edges to infer, provided in .run()."""

    def __post_init__(self):
        self.workdir = Path(self.workdir)
        self.workdir.mkdir(exist_ok=True)
        self.gtrees = Path(self.gtrees).absolute()
        
        self._io_table = self.workdir / f"{self.name}.snaq.CFs.csv"
        self._io_script = self.workdir / f"{self.name}.snaq.jl"
        self.path_to_julia = self.path_to_julia if self.path_to_julia else "julia"

        self._check_julia_binary()
        # self.install_phylonetworks()
        self._run_quartet_cf_table()

    def _check_julia_binary(self) -> None:
        """Raise an exception if julia is not found."""
        cmd = ["which", str(self.path_to_julia)]
        kwargs = dict(stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        with subprocess.Popen(cmd, **kwargs) as proc:
            comm = proc.communicate()
            if not comm[0]:
                raise IOError(f"julia binary not found: {comm[0].decode()}")

    def install_phylonetworks(self) -> None:
        """Run command in julia to install or update phylonetworks"""
        with open(self._io_script, 'w', encoding="utf-8") as out:
            out.write(REQUIRED_PACKAGES)
        self._execute_script()

    def _write_script(self, script: str) -> None:
        """Writes to [workdir]/[name].jl a temp julia script."""
        with open(self._io_script, 'w', encoding="utf-8") as out:
            out.write(script)

    def _run_quartet_cf_table(self) -> None:
        """Writes a quartet CF table to workdir.

        If a table already exists at [workdir]/[name].cf_table.csv then
        this will not re-run the CF function unless force=True.
        """
        table_script = GENE_TREE_COUNT_SCRIPT.format(
            gtrees=self.gtrees, 
            io_table=self._io_table,
        )

        # remove existing cf table
        if self.force:
            if self._io_table.exists():
                self._io_table.unlink()

        # if table already exists then use it
        if self._io_table.exists():
            logger.info(f"using existing CF table: {self._io_table} (use force=True to re-run).")
        else:
            logger.info("Building quartet tree CF table.")
            self._write_script(table_script)
            self._execute_script()

    def _run_network_inference(self) -> None:
        """Run network inference. 
        
        This will load in the CF table from workdir and a starting
        network from `net_in` and infer a network with `nedges` edges
        and write the result to [workdir][name].nedges.network
        """
        logger.info(f"Inferring a network (h={self._nedges}) in SNaQ using {self.nproc} CPUs.")
        infer_script = NETWORK_INFER_SCRIPT.format(
            nproc=self.nproc, 
            nruns=self.nruns, 
            seed=self._seed, 
            io_table=self._io_table, 
            nedges=self._nedges,
            net_in=self._net_in, 
            out_net=self._basename,
        )
        self._write_script(infer_script)
        self._execute_script()
        logger.info(f"inferred network written to {self.network}")

    def _execute_script(self) -> None:
        """...        """
        cmd = [str(self.path_to_julia), str(self._io_script)]
        kwargs = dict(stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        with subprocess.Popen(cmd, **kwargs) as proc: 
            comm = proc.communicate()
            if proc.returncode:
                logger.error(f"SNAQ Error:\n{comm[0].decode()}")
            logger.debug(f"{comm[0].decode()}")

    def _get_starting_net(self, net_in: Optional[Union[Path, toytree.ToyTree]]) -> str:
        """..."""
        # user entered a newick string
        if ";" in str(net_in):
            return net_in

        # user entered None. Look for existing -1 file.
        if net_in is None:
            net_minus = self.workdir / f"{self.name}.snaq.net-{self._nedges - 1}.out"
            if not net_minus.exists():
                raise IOError("No net -1 file found in workdir. You must enter a net_in starting network.")
            logger.info(f"Using nedges-1 network as net_in ({net_minus}).")
            return net_minus

        # user passed a ToyTree as input
        if isinstance(net_in, toytree.ToyTree):
            return net_in.write(dist_formatter=None)

        # user entered a Path or filename str as input
        if not Path(net_in).exists():
            raise IOError(f"net_in file not found: {net_in}")
        return str(net_in)

    def run(self, nedges: int=0, net_in: Optional[Path]=None, seed: Optional[int]=None) -> None:
        """..."""
        # if no net_in then look for [workdir][name][nedges-1].network
        self._nedges = nedges
        self._net_in = self._get_starting_net(net_in)
        self._seed = np.random.randint(int(1e7)) if seed is None else seed
        self._basename = self.workdir / f"{self.name}.snaq.net-{nedges}"
        self.log = self.workdir / f"{self.name}.snaq.net-{nedges}.log"
        self.network = self.workdir / f"{self.name}.snaq.net-{nedges}.out"        
        self._run_network_inference()

    def plot_network_loglik(self, nedges: List[int]=None) -> Tuple["Canvas", "Axes", "Mark"]:
        """..."""
        # parse data from outfiles
        outfiles = self.workdir.glob(f"{self.name}.snaq.net-*.out")
        nedges = []
        logliks = []
        for file in sorted(outfiles):
            nedges.append(int(file.suffixes[-2].split("-")[1]))
            with open(file, 'r', encoding="utf-8") as inf:
                logliks.append(float(inf.readline().strip().split()[-1]))

        canvas = toyplot.Canvas(width=350, height=300)
        axes = canvas.cartesian(xlabel="N admixture edges", ylabel="Log-pseudolikelihood")
        axes.scatterplot(nedges, logliks, size=10)
        axes.plot(nedges, logliks)
        axes.x.ticks.show = axes.y.ticks.show = True
        toytree.utils.show(canvas)
        plot_path = self.workdir / f"{self.name}.svg"
        toyplot.svg.render(canvas, str(plot_path))
        logger.info(f"network log-pseudolikelihood plot written {plot_path}")


def test_interactive():
    """Simulate genealogies without admixture adn run SNAQ 0-2."""
    # simulate N unlinked genealogies and write to a file

    for neff in [5e5, 1e5, 5e4, 1e4]:
        sptree = toytree.rtree.imbtree(8, treeheight=2e6)
        model = ipcoal.Model(tree=sptree, Ne=neff, nsamples=1, seed_trees=123, seed_mutations=123)
        model.sim_trees(1000)
        gtrees = toytree.mtree(model.df.genealogy)
        gtrees.write("/tmp/genetrees.nwk", dist_formatter=None)

        # load trees into SNAQ and estimate network psuedolikelihoods
        tool = Snaq(
            gtrees="/tmp/genetrees.nwk",
            name=f"test-{neff:.0f}",
            workdir="/tmp/analysis-snaq",
            path_to_julia="/home/deren/local/src/julia-1.6.2/bin/julia",
            force=True,
        )
        tool.run(nedges=0, net_in=sptree, seed=123)
        tool.run(nedges=1, net_in=None, seed=123)
        tool.run(nedges=2, net_in=None, seed=123)
        tool.run(nedges=3, net_in=None, seed=123)
        tool.plot_network_loglik()


if __name__ == "__main__":

    import ipcoal
    ipcoal.set_log_level("INFO")
    test_interactive()    
