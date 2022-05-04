#!/usr/bin/env python

"""...

This Python script includes code to run a simulation routine, to 
accept arguments from the command line to parameterize this function,
and to distribute job submissions of this script to SLURM over 
hundreds of combination of parameter settings.

This script sets up a total of 6400 jobs across different parameter
combinations, each of which takes a few hours to run, so it is a 
good idea to use HPC to run this.

"""

from typing import List
import sys
import time
import argparse
from pathlib import Path
from subprocess import Popen, STDOUT, PIPE
import numpy as np


SBATCH = """\
#!/bin/sh

#SBATCH --account={account}
#SBATCH --job-name={jobname}
#SBATCH --output={outpath}.out
#SBATCH --error={outpath}.err
#SBATCH --time=11:59:00
#SBATCH --ntasks={ncores}
#SBATCH --mem=12G

python {root}/run-sim-loci-inference.py \
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
--astral-bin {astral_bin} \
--outdir {outdir}

"""


def write_and_submit_sbatch_script(
    neff: int, 
    ctime: int, 
    mut: float,
    recomb: float, 
    rep: int,
    seed: int,
    nsites: int,
    nloci: int,
    ncores: int,
    outdir: Path,
    account: str,
    node_heights: List[float],
    raxml_bin: Path,
    astral_bin: Path,    
    ):
    """Submit an sbatch job to the cluster with these params."""
    # build parameter name string
    params = (
        f"neff{neff}-ctime{int(ctime)}-"
        f"recomb{int(bool(recomb))}-rep{rep}-"
        f"nloci{max(nloci)}-nsites{nsites}"
    )

    # check for existing output files and skip this job if present
    # paths = [outdir / (params + f"-astral-genetree-subloci{i}.nwk") for i in nloci]
    # if all(i.exists() for i in paths):
    #     print(f"skipping job {params}, result files exist.")
    #     return 

    # expand sbatch shell script with parameters
    sbatch = SBATCH.format(**dict(
        account=account,
        jobname=params,
        ncores=ncores,
        neff=neff,
        ctime=ctime,
        mut=mut,
        recomb=recomb,
        nsites=nsites,
        nloci=" ".join([str(i) for i in nloci]),
        rep=rep,
        seed=seed,
        node_heights=" ".join([str(i) for i in node_heights]),
        raxml_bin=raxml_bin,
        astral_bin=astral_bin,
        outdir=outdir,
        outpath=outdir / params,
        root=str(Path(__file__).parent),
    ))
    # print(sbatch)

    # write the sbatch shell script
    tmpfile = (outdir / params).with_suffix(".sh")
    with open(tmpfile, 'w', encoding='utf-8') as out:
        out.write(sbatch)

    # submit job to HPC SLURM job manager
    cmd = ['sbatch', str(tmpfile)]
    with Popen(cmd, stdout=PIPE, stderr=STDOUT) as proc:
        out, _ = proc.communicate()


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
    """
    parser = argparse.ArgumentParser(
        description='Coalescent simulation and tree inference w/ recombination')
    parser.add_argument(
        '--neff', type=int, default=[10000, 100000], nargs="*", help='Effective population size')
    parser.add_argument(
        '--ctime', type=float, default=[0.1, 1.5], nargs="*", help='Root species tree height in coalescent units')
    parser.add_argument(
        '--recomb', type=float, default=[0, 5e-8], nargs=2, help='Recombination rate.')
    parser.add_argument(
        '--mut', type=float, default=5e-9, help='Mutation rate.')
    parser.add_argument(
        '--node-heights', type=float, default=[0.05, 0.055, 0.06, 1], nargs=4, help='Internal relative node heights')
    parser.add_argument(
        '--nsites', type=int, default=[2000], nargs="*", help='length of simulated loci')
    parser.add_argument(
        '--nloci', type=int, default=[20000], nargs="*", help='number of independent simulated loci.')
    parser.add_argument(
        '--nreps', type=int, default=100, help='number replicate per param setting.')
    parser.add_argument(
        '--outdir', type=Path, default=Path("."), help='directory to write output files (e.g., scratch)')
    parser.add_argument(
        '--account', type=str, default="free", help='Account name for SLURM job submission')
    parser.add_argument(
        '--ncores', type=int, default=2, help='Number of cores per job (recommended=2)')
    return parser.parse_args()


if __name__ == "__main__":

    # parse command line args
    args = distributed_command_line_parser()

    # build grid of all jobs
    nlen = len(args.neff)
    rlen = len(args.recomb)
    clen = len(args.ctime)
    ilen = args.nreps
    slen = len(args.nsites)
    njobs = nlen * rlen * clen * slen * ilen
    print(f"Submitting {njobs} sbatch jobs at 1 second intervals.")

    # find conda installed packages
    Path(args.outdir).mkdir(exist_ok=True)
    BINDIR = Path(sys.prefix) / "bin"
    ASTRAL_BIN = BINDIR / "astral.5.7.1.jar"
    RAXML_BIN = BINDIR / "raxml-ng"
    assert ASTRAL_BIN.exists(), f"cannot find {ASTRAL_BIN}. Use conda instructions."
    assert RAXML_BIN.exists(), f"cannot find {RAXML_BIN}. Use conda instructions."

    # distribute jobs over all params except NLOCI (pass whole list).
    SEEDS = np.random.default_rng(123).integers(1e12, size=args.nreps)
    for nsites in args.nsites:
        for neff in args.neff:
            for ctime in args.ctime:
                for recomb in args.recomb:                
                    for rep in range(args.nreps):

                        # skip submitting job if all outfiles exist.
                        params = (
                            f"neff{neff}-ctime{int(ctime)}-"
                            f"recomb{int(bool(recomb))}-rep{rep}-"
                            f"nloci{max(args.nloci)}-nsites{nsites}"
                        )

                        # check for existing output files and skip this job if present
                        paths = [args.outdir / (params + f"-astral-genetree-subloci{i}.nwk") for i in args.nloci]
                        if all(i.exists() for i in paths):
                            print(f"skipping job {params}, result files exist.")
                            continue

                        # for nloci in args.nloci:
                        # gtime = int(ctime * 4 * neff)
                        write_and_submit_sbatch_script(
                            neff=neff,
                            ctime=ctime, 
                            mut=args.mut,
                            recomb=recomb, 
                            nloci=args.nloci,
                            nsites=nsites,
                            rep=rep,
                            seed=SEEDS[rep],
                            ncores=args.ncores,
                            outdir=args.outdir,
                            account=args.account,
                            node_heights=args.node_heights,
                            raxml_bin=RAXML_BIN,
                            astral_bin=ASTRAL_BIN,                            
                        )
                        time.sleep(0.1)
    print(f"{njobs} jobs submitted.")
