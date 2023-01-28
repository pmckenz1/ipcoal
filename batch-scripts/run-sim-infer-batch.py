#!/usr/bin/env python

"""Distributes `run-sim-infer.py` on a SLURM cluster.

This Python script includes code to run a simulation routine, to
accept arguments from the command line to parameterize this function,
and to distribute job submissions of this script to SLURM over
hundreds of combination of parameter settings.

This script sets up a total of 6400 jobs across different parameter
combinations, each of which takes a few hours to run, so it is a
good idea to use HPC to run this.

Example
-------
>>> python run-sim-loci-inference-distributed.py \
>>>   --neff 10000 100000 \
>>>   --ctime 0.1 \
>>>   --nloci 100 250 500 1000 2500 5000 10000 \
>>>   --nsites 2000 10000 \
>>>   --nreps 10 \
>>>   --mut 5e-8 \
>>>   --recomb 0 5e-9 \
>>>   --node-heights 0.01 0.05 0.06 1 \
>>>   --ncores 12 \
>>>   --outdir /moto/eaton/users/de2356/recomb-response/data5 \
>>>   --account eaton \
>>>   --delay 1

Description
-----------
The job above implements a fixed imbalanced 5-taxon species tree
with divergence times described by the `node-heights` parameter.
It will simulate 10000 loci (the largest `nloci` param) of length
2000 and 10000. This will be done in one case with recombination,
and in another case without. These sims are also each repeated
for 10 replicates from random seeds. For each dataset true genealogies
will be saved, and also empirical gene trees will be inferred for
each locus using raxml-ng. A species tree will then be inferred
from each distribution of genealogies or gene trees.

Outputs
--------
>>> outdir/rep[1-10]-concat-subloci[100-10000].nwk
>>> outdir/rep[1-10]-astral-genealogy-subloci[100-10000].nwk
>>> outdir/rep[1-10]-astral-genetree-subloci[100-10000].nwk
"""

from typing import List, Dict, Iterator, Any, Tuple
import sys
import time
import argparse
import shutil
from itertools import product
from dataclasses import dataclass
from pathlib import Path
from subprocess import Popen, STDOUT, PIPE
from loguru import logger
import numpy as np
# import pandas as pd
import ipcoal

logger = logger.bind(name="ipcoal")
ROOT = str(Path(__file__).parent)
SBATCH = """\
#!/bin/sh

#SBATCH --account={account}
#SBATCH --job-name={jobname}
#SBATCH --output=log-{outpath}.out
#SBATCH --error=log-{outpath}.err
#SBATCH --time=11:59:00
#SBATCH --ntasks={ncores}
#SBATCH --mem=12G

# run the command to write and submit a shell script
{python} {root}/run-sim-infer.py \
  --neff {neff} \
  --ctime {ctime} \
  --mut {mut} \
  --recomb {recomb} \
  --nsites {nsites} \
  --nloci {nloci} \
  --rep {rep} \
  --seed {seed} \
  --outdir {outdir} \
  --ncores {ncores} \
  --node-heights {node_heights} \
  --raxml-bin {raxml_bin} \
  --astral-bin {astral_bin}
"""


@dataclass
class SlurmDistribute:
    # user params with one or more values
    neff: List[int]
    ctime: List[int]
    recomb: List[float]
    nsites: List[int]
    nloci: List[int]

    # individual params
    mut: float
    nreps: int
    seed: int
    ncores: int
    outdir: Path
    account: str
    node_heights: List[float]

    # params with defaults, or to be filled
    raxml_bin: Path = None
    astral_bin: Path = None
    delay: float = 0.1

    def __post_init__(self):
        self.outdir = Path(self.outdir)
        bindir = Path(sys.prefix) / "bin"
        if self.astral_bin is None:
            self.astral_bin = bindir / "astral.5.7.1.jar"
        else:
            self.astral_bin = Path(self.astral_bin)
        assert self.astral_bin.exists(), f"cannot find {self.astral_bin}. Use conda instructions."

        if self.raxml_bin is None:
            self.raxml_bin = bindir / "raxml-ng"
        else:
            self.raxml_bin = Path(self.raxml_bin)
        assert self.raxml_bin.exists(), f"cannot find {self.raxml_bin}. Use conda instructions."

    def iter_jobs(self) -> Iterator[Tuple[str, List[Any]]]:
        """Yield Tuples iterating over parameters combinations."""
        combs = product(self.nsites, self.ctime, self.recomb, self.neff)
        for nsi, cti, rec, nef in combs:
            # basename of the params used across a set of replicates.
            params_basename = (
                f"neff{int(nef)}-ctime{cti}-"
                f"recomb{str(rec).replace('-', '')}-"
                f"nloci{max(self.nloci)}-nsites{nsi}"
            )
            yield params_basename, [nsi, cti, rec, nef]

    def iter_params(self) -> Iterator[Dict[str, Any]]:
        """Yield Dicts with all params for each job replicate."""
        # all runs use the same set of seeds across replicates
        seeds = np.random.default_rng(self.seed).integers(1e12, size=self.nreps)

        # iterate over jobs to be submitted
        for params_basename, [nsi, cti, rec, nef] in self.iter_jobs():

            # run this set for nreplicate times.
            for rep in range(self.nreps):

                # get name of this job
                jobname = f"{params_basename}-rep{rep}"
                outpath = self.outdir / jobname # for .err and .out files

                # submit job to run...
                kwargs = dict(
                    account=self.account,
                    jobname=jobname,
                    outpath=outpath,
                    ncores=self.ncores,
                    root=ROOT,
                    neff=nef,
                    ctime=cti,
                    mut=self.mut,
                    recomb=rec,
                    nsites=nsi,
                    nloci=" ".join([str(i) for i in self.nloci]),
                    rep=rep,
                    seed=seeds[rep],
                    outdir=self.outdir,
                    node_heights=" ".join([str(i) for i in self.node_heights]),
                    raxml_bin=self.raxml_bin,
                    astral_bin=self.astral_bin,
                    python=Path(sys.prefix) / 'bin' / 'python',
                )
                yield kwargs

    def iter_slurm_scripts(self) -> Iterator[Tuple[str, str]]:
        """Yield SLURM scripts (bash w/ #HEADER) for all job params."""
        for params in self.iter_params():
            yield params['jobname'], SBATCH.format(**params)

    def submit_subprocess(self, name: str, script: str, cmd: str="sbatch") -> None:
        """..."""
        # b/c the params string name has a '.' in it for decimal ctime.
        tmpfile = self.outdir / f"job-{name}.sh"
        with open(tmpfile, 'w', encoding='utf-8') as out:
            out.write(script)

        # submit job to bash or SLURM job manager
        cmd = [cmd, str(tmpfile)]
        with Popen(cmd, stdout=PIPE, stderr=STDOUT) as proc:
            out, _ = proc.communicate()
        if proc.returncode:
            logger.error(f"{out.decode()}")
        tmpfile.unlink()

    def _count_njobs(self) -> int:
        """Return number of jobs."""
        nlen = len(self.neff)
        rlen = len(self.recomb)
        clen = len(self.ctime)
        slen = len(self.nsites)
        ilen = self.nreps
        njobs = nlen * rlen * clen * slen * ilen
        return njobs

    def run(self, cmd: str="sbatch", resume: bool=False, force: bool=False) -> None:
        """Calls submit_subprocess() on jobs in iter_slurm_script().

        This command also enforces the resume/force options to continue
        or restart a set of jobs, and ensures the outdir exists.
        """
        # if outdir exists an error is raised unless the user specified
        # either resume or force. The former will resume the run, the
        # latter will remove any files and restart the run.
        if self.outdir.exists():
            if not (resume or force):
                raise IOError(f"Output directory {self.outdir} exists.\n"
                    "Use --resume to continue running remaining jobs without existing results.\n"
                    "Or use --force to clear the output directory and restart."
                )
        if force and resume:
            raise ValueError("You must select --force or --resume, not both.")

        # outdir must exist.
        self.outdir.mkdir(exist_ok=True)

        # force removes everything inside the outdir.
        if force:
            for path in self.outdir.glob("*-neff*-ctime*-recomb*-nloci*"):
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()

        # resume removes .sh files and tmpdirs/ but leaves .csv
        if resume:
            for path in self.outdir.glob("*-neff*-ctime*-recomb*-nloci*"):
                if not path.suffix == ".csv":
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()

        # iterate over all jobs to submit
        nfin = len(list(self.outdir.glob("*-neff*-ctime*-recomb*-nloci*.csv")))
        njobs = self._count_njobs()
        resuming = "Resuming." if resume else ""
        logger.info(f"Submitting {njobs - nfin} jobs. {resuming}")
        for name, script in self.iter_slurm_scripts():

            # skip if results exist for this rep
            resfile = self.outdir / f"res-{name}.csv"
            if resfile.exists():
                logger.info(f"skipping {name}.")
                continue

            # submit job to run and remove .sh file when done.
            logger.info(f"starting job {name}")
            self.submit_subprocess(name, script, cmd)

            # .out file contains log, .err file is errors; remove if empty.
            logfile = self.outdir / f"log-{name}.err"
            if logfile.exists():
                if not logfile.stat().st_size():
                    logfile.unlink()

            # use short delay between job submissions to be nice.
            time.sleep(self.delay)

    # def combine(self) -> None:
    #     # # if it is the final rep then perform concatenation on CSVs
    #     base, rep = name.rsplit("-rep", 1)
    #     logger.info(f"REP-{rep}")
    #     if int(rep) == self.nreps - 1:
    #         print(f"*{base}-rep*.csv")
    #         chunks = self.outdir.glob(f"*{base}-rep*.csv")
    #         print(list(chunks))
    #         dfs = sorted(chunks, key=lambda x: int(x.name.rsplit("-rep", 1)[-1]))
    #         cdf = pd.concat((pd.read_csv(i) for i in dfs), ignore_index=True)
    #         logger.info(f"concatenated FINAL\n{cdf}")
    #     #     # unlink...

    # def null(self):
    #     # print summary of jobs that will be started.
    #     nlen = len(self.neff)
    #     rlen = len(self.recomb)
    #     clen = len(self.ctime)
    #     ilen = self.nreps
    #     slen = len(self.nsites)
    #     njobs = nlen * rlen * clen * slen * ilen
    #     print(f"Submitting {njobs} sbatch jobs at {self.delay} second intervals.")

    #     # check whether the summary file for this paramjob exists
    #     result = self.outdir / paramdir / 'result.csv'


def distributed_command_line_parser():
    """Parse command line arguments and return.

    Example
    -------
    >>> python run-sim-loci-inference-distributed.py  \
    >>>     --ncores 2 \
    >>>     --nreps 100 \
    >>>     --nsites 2000 10000 \
    >>>     --neff 1e4 1e5 \
    >>>     --ctimes 0.1 0.2 0.3 0.4 0.5 0.75 1.0 1.25 \
    >>>     --mut 5e-8 \
    >>>     --recomb 0 5e-9 \
    >>>     --node-heights 0.01 0.05 0.06 1 \
    >>>     --outdir /scratch/recomb/ \
    >>>     --account eaton \
    >>>     --step 1
    """
    parser = argparse.ArgumentParser(
        description='Coalescent simulation and tree inference w/ recombination')
    parser.add_argument(
        '--neff', type=int, default=[10000, 100000], nargs="*", help='Effective population sizes')
    parser.add_argument(
        '--ctime', type=float, default=[0.1, 1.5], nargs="*", help='Root species tree height in coalescent units')
    parser.add_argument(
        '--recomb', type=float, default=[0, 5e-9], nargs=2, help='Recombination rate.')
    parser.add_argument(
        '--mut', type=float, default=5e-8, help='Mutation rate.')
    parser.add_argument(
        '--node-heights', type=float, default=[0.05, 0.055, 0.06, 1], nargs=4, help='Internal relative node heights')
    parser.add_argument(
        '--nsites', type=int, default=[2000], nargs="*", help='length of simulated loci')
    parser.add_argument(
        '--nloci', type=int, default=[20000], nargs="*", help='number of independent simulated loci.')
    parser.add_argument(
        '--nreps', type=int, default=10, help='number replicate per param setting.')
    parser.add_argument(
        '--outdir', type=Path, default=Path("/tmp/test"), help='directory to write output files (e.g., scratch)')
    parser.add_argument(
        '--account', type=str, default="free", help='Account name for SLURM job submission')
    parser.add_argument(
        '--ncores', type=int, default=2, help='Number of cores per job (recommended=2)')
    parser.add_argument(
        '--resume', action='store_true', help='Resume an interrupted run with some existing results.')
    parser.add_argument(
        '--force', action='store_true', help='Restart an run overwriting any existing results.')
    parser.add_argument(
        '--delay', type=float, default=0.5, help='Number of seconds delay between SLURM job submissions.')
    parser.add_argument(
        '--seed', type=int, default=123, help='Random number generator seed.')
    parser.add_argument(
        '--cmd', type=str, default="bash", help="Execute 'bash' for local or 'sbatch' for SLURM.")
    parser.add_argument(
        '--log-level', type=str, default="INFO", help="logging level: DEBUG, INFO, WARNING, ERROR.")
    parser.add_argument(
        '--astral-bin', type=Path, default=None, help="Full path to astral binary.")
    parser.add_argument(
        '--raxml-bin', type=Path, default=None, help="Full path to raxml-ng binary.")
    return parser.parse_args()


def main():
    """Command line utility to accept arguments.

    """
    args = distributed_command_line_parser()
    kwargs = vars(args)
    resume = kwargs.pop("resume")
    force = kwargs.pop("force")
    cmd = kwargs.pop("cmd")
    log_level = kwargs.pop("log_level")
    ipcoal.set_log_level(log_level)
    tool = SlurmDistribute(**kwargs)
    tool.run(cmd=cmd, resume=resume, force=force)

def interactive():
    """Substitute this command for main() in __main__ when testing."""
    ipcoal.set_log_level("INFO")
    tool = SlurmDistribute(
        neff=[10_000, 100_000],
        ctime=[0.1, 1.5],
        recomb=[0, 5e-8, 5e-9],
        node_heights=[0.05, 0.055, 0.06, 1],
        nsites=[1_000, 2_000],
        nloci=[200, 300],
        mut=5e-8,
        nreps=3,
        outdir=Path("/tmp/tester"),
        account="free",
        ncores=6,
        seed=123,
        delay=0.1,
    )
    tool.run(cmd="bash", resume=True)

    # jobs = tool.iter_slurm_scripts()
    # print(next(jobs))
    # print(next(jobs))

    # # for job in tool.iter_slurm_scripts():
    # name, script = next(jobs)
    # tool.submit_slurm_subprocess(name, script, 'bash')

    # name, script = next(jobs)
    # tool.submit_slurm_subprocess(name, script, 'bash')

    # name, script = next(jobs)
    # tool.submit_slurm_subprocess(name, script, 'bash')


if __name__ == "__main__":

    # interactive()
    main()
