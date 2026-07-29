#!/usr/bin/env python

"""Infer tree by maximum-likelihood in iqtree

IQtree may be preferable to raxml-ng in a number of situations. Most
notably for me has been for inferring many small trees, where the
enforced and non-suppressable nature of raxml-ng to write many many
many many small temporary files causes extreme bottlenecks even on
HPC systems.


TODO: Not YET IMPLEMENTED. Just copied from raxml code.
"""

from typing import Union, Sequence, Optional
import sys
import tempfile
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT
from concurrent.futures import ProcessPoolExecutor

from loguru import logger
import numpy as np
import pandas as pd
import toytree
import ipcoal
from ipcoal.utils.utils import IpcoalError

logger = logger.bind(name="ipcoal")
IQTREE = Path(sys.prefix) / "bin" / "iqtree"


def infer_iqtree_trees(
    model: ipcoal.Model,
    idxs: Union[Sequence[int], None] = None,
    nboots: int = 0,
    nproc: int = 1,
    seed: int = None,
    diploid: bool = False,
    subst_model: str = "GTR+G",
    binary_path: Union[str, Path] = None,
    tmpdir: Optional[Path] = None,
    nthreads: int = 4,
    nworkers: int = 4,
) -> pd.DataFrame:
    r"""Return a DataFrame w/ inferred gene trees at every locus.

    Sequence data is extracted from the model.seqs array and written
    as concatenated data to a phylip file, either for individual
    haplotypes or diploid genotypes if diploid=True. If `idxs=None`
    all data is concatenated, else a subset of one or more loci can
    be selected to be concatenated.

    CMD: iqtree -s {phy} --quiet -T 1 --threads-max 1 --subst_model {GTR+G} --redo

    Parameters
    ----------
    model: str or Path
        An ipcoal.Model object with simulated locus data.
    idxs: Sequence[int], int or None
        The index of one or more loci from an `ipcoal.Model.sim_loci`
        dataset which will be concatenated and passed to raxml. If
        None then all loci are concatenated.
    nboots: int
        Number of bootstrap replicates to run.
    nthreads: int
        Number of threads used for parallelization.
    seed: int or None
        ...
    subst_model: str
        ...
    binary_path: None, str, or Path
        Path to the raxml binary.
    nworkers: int
        ...
    tmpdir: Path or None
        Path to store temporary files. Default is tempdir (/tmp).

    Note
    ----
    The parallelization does not provide a significant speedup if the
    inference jobs take less than one second or so to run, since the
    setup of writing/organizing files takes time as well.
    """
    assert model.seqs is not None, "must first call Model.sim_loci."
    assert model.seqs.ndim == 3, "must first call Model.sim_loci."

    # store arguments to infer method
    kwargs = dict(
        nboots=nboots, nthreads=nthreads, nworkers=nworkers,
        seed=seed, subst_model=subst_model, 
        binary_path=binary_path, tmpdir=tmpdir)

    # distribute jobs in parallel
    rng = np.random.default_rng(seed)
    empty = 0
    rasyncs = {}

    # loci for which trees can be inferred
    pidxs = set(model.df.locus.unique())

    # which loci to do
    if idxs is None:
        idxs = sorted(pidxs)
    if isinstance(idxs, int):
        idxs = [idxs]
    if not isinstance(idxs, list):
        idxs = list(idxs)

    # TODO: asynchrony so that writing and processing are not limited.
    with ProcessPoolExecutor(max_workers=nproc) as pool:
        for lidx in idxs:
            if lidx not in pidxs:
                continue
            locus = model.df[model.df.locus == lidx]

            # if no data then return a star tree.
            if not locus.nsnps.sum():
                tree = toytree.tree(locus.genealogy.iloc[0])
                tree = tree.mod.collapse_nodes(*range(tree.ntips, tree.nnodes))
                rasyncs[lidx] = tree.write(None)
                empty += 1
            else:
                # disk-crushing mode.
                # fname = _write_tmp_phylip_file(model, int(lidx), diploid, tmpdir)
                # kwargs['alignment'] = fname
                # kwargs['seed'] = rng.integers(1e12)
                # rasync = pool.submit(infer_raxml_ng_tree_from_phylip, **kwargs)
                # rasyncs[lidx] = rasync

                # disk-friendly mode, but higher memory-usage.
                ali = model.write_concat_to_phylip(idxs=int(lidx), diploid=diploid, quiet=True)
                kwargs['alignment'] = ali
                kwargs['seed'] = rng.integers(1e12)
                rasync = pool.submit(infer_raxml_ng_tree_from_alignment, **kwargs)
                rasyncs[lidx] = rasync

    # log report of empty windows.
    if empty:
        logger.warning(
            f"{empty} loci ({empty / model.df.locus.iloc[-1]:.2f}%) "
            "contain 0 SNPs and were returned as star trees.")

    # create results as a dataframe
    data = model.df[model.df.locus.isin(idxs)]
    data = (data
        .groupby("locus")
        .agg({"start": "min", "end": "max", "nbps": "sum", "nsnps": "sum"})
        .reset_index()
    )
    data['gene_tree'] = [
        rasyncs[i] if isinstance(rasyncs[i], str) else
        rasyncs[i].result().write() for i in sorted(rasyncs)
    ]
    return data