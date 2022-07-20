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

python {root}/run-mammals.py \
--neff {neff} \
--mut {mut} \
--recomb {recomb} \
--nsites {nsites} \
--nloci {nloci} \
--rep {rep} \
--seed {seed} \
--outdir {outdir} \
--ncores {ncores} \
--root-height {root_height} \
--raxml-bin {raxml_bin} \
--astral-bin {astral_bin} \
--outdir {outdir}

"""


def write_and_submit_sbatch_script(
    neff: int, 
    mut: float,
    recomb: float, 
    rep: int,
    seed: int,
    nsites: int,
    nloci: int,
    ncores: int,
    outdir: Path,
    account: str,
    root_height: float,
    raxml_bin: Path,
    astral_bin: Path,    
    ):
    """Submit an sbatch job to the cluster with these params."""
    # build parameter name string
    params = (
        f"neff{neff}-root{root_height:.2e}-"
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
        mut=mut,
        recomb=recomb,
        nsites=nsites,
        nloci=" ".join([str(i) for i in nloci]),
        rep=rep,
        seed=seed,
        root_height=root_height,
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

    The fixed mammal tree will be scaled to the root_height arg in
    units of generations.

    Example
    -------
    >>> python run-sim-loci-inference-distributed.py  \
    >>>     --ncores 2 \
    >>>     --nreps 100 \
    >>>     --nsites 2000 10000 \
    >>>     --neff 1e4 1e5 \
    >>>     --mut 5e-8 \
    >>>     --recomb 0 5e-9 \
    >>>     --node-heights 0.01 0.05 0.06 1 \
    >>>     --outdir /scratch/recomb/ \
    >>>     --account eaton \
    """
    parser = argparse.ArgumentParser(
        description='Coalescent simulation and tree inference w/ recombination')
    parser.add_argument(
        '--root-height', default=[66_371_836], nargs="*", type=float, help='Scale relative sptree edges so root height is at this.')
    parser.add_argument(
        '--neff', type=int, default=[10000, 100000], nargs="*", help='Effective population size')
    parser.add_argument(
        '--recomb', type=float, default=[0, 5e-9], nargs=2, help='Recombination rate.')
    parser.add_argument(
        '--mut', type=float, default=5e-8, help='Mutation rate.')
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
    # sys.argv = """
    # python run-mammals-distributed.py  \
    #      --ncores 2 \
    #      --nreps 100 \
    #      --root-height 66e6 \
    #      --neff 1e4 1e5 \
    #      --nsites 2_000 10_000 \
    #      --nloci 20_000 \
    #      --mut 5e-8 \
    #      --recomb 0 5e-9 \
    #      --outdir /scratch/recomb/ \
    #      --account eaton \
    # """
    args = distributed_command_line_parser()

    # build grid of all jobs
    nlen = len(args.neff)
    rlen = len(args.recomb)
    ilen = args.nreps
    slen = len(args.nsites)
    njobs = nlen * rlen * slen * ilen
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
    for nsites_ in args.nsites:
        for neff_ in args.neff:
            for root_ in args.root_height:
                for recomb_ in args.recomb:                
                    for rep_ in range(args.nreps):

                        # skip submitting job if all outfiles exist.
                        params_ = (
                            f"neff{neff_}-root{root_:.2e}-"
                            f"recomb{int(bool(recomb_))}-rep{rep_}-"
                            f"nloci{max(args.nloci)}-nsites{nsites_}"
                        )

                        # check for existing output files and skip this job if present
                        paths = [args.outdir / (params_ + f"-astral-genetree-subloci{i}.nwk") for i in args.nloci]
                        if all(i.exists() for i in paths):
                            njobs -= 1
                            print(f"skipping job {params_}, result files exist.")
                            continue

                        # gtime = int(ctime * 4 * neff)
                        write_and_submit_sbatch_script(
                            neff=neff_,
                            mut=args.mut,
                            recomb=recomb_, 
                            nloci=args.nloci,
                            nsites=nsites_,
                            rep=rep_,
                            seed=SEEDS[rep_],
                            ncores=args.ncores,
                            outdir=args.outdir,
                            account=args.account,
                            root_height=root_,
                            raxml_bin=RAXML_BIN,
                            astral_bin=ASTRAL_BIN,                            
                        )
                        time.sleep(0.5)
    print(f"{njobs} jobs submitted.")
