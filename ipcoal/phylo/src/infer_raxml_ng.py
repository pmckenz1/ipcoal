#!/usr/bin/env python

"""Infer a concatenation tree from a sequence alignment using raxml-ng.

"""

from typing import Union, Sequence, Optional
import os
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
RAXML = Path(sys.prefix) / "bin" / "raxml-ng"
BINARY_MISSING = """
    Cannot find raxml-ng binary {}. Make sure you have
    raxml-ng installed, which you can do with the following command
    which also installs required dependencies:

    >>> conda install raxml-ng -c conda-forge -c bioconda

    Then find the full path where the binary is installed and
    enter it to this function as the `binary_path` argument.
"""

def _write_tmp_phylip_file(
    model: ipcoal.Model,
    idxs: Union[int, Sequence[int], None]=None,
    diploid: bool=False,
    tmpdir: Optional[Path]=None,
    ) -> Path:
    """Write concat matrix to a random/tmp named file and return Path.

    This is used to separate the phylip writing step from the
    tree inference step, so that the Model object does not need to
    be passed to the other function. This makes it easier to pickle
    the arguments and parallelize it.
    """
    tmpdir = tmpdir if tmpdir is not None else tempfile.gettempdir()
    with tempfile.NamedTemporaryFile(dir=tmpdir, suffix=f"_{os.getpid()}") as tmp:
        model.write_concat_to_phylip(
            name=tmp.name,
            idxs=idxs,
            diploid=diploid,
            quiet=True,
        )
    fname = Path(tmp.name).with_suffix(".phy")
    return fname

def infer_raxml_ng_tree_from_alignment(
    alignment: str,
    nboots: int=0,
    nthreads: int=4,
    nworkers: int=4,
    seed: Optional[int]=None,
    subst_model: str="GTR+G",
    binary_path: Union[str, Path]=None,
    tmpdir: Optional[Path]=None,
    ) -> toytree.ToyTree:
    """Return a single ML tree inferred by raxml-ng from a phylip string."""
    tmpdir = tmpdir if tmpdir is not None else tempfile.gettempdir()
    with tempfile.NamedTemporaryFile(dir=tmpdir) as tmp:

        # write phylip data to the tmpfile
        fname = Path(tmp.name).with_suffix(".phy")
        with open(fname, 'w', encoding="utf-8") as out:
            out.write(alignment)

        tree = infer_raxml_ng_tree_from_phylip(
            alignment=fname, nboots=nboots, nthreads=nthreads,
            nworkers=nworkers, seed=seed, subst_model=subst_model, 
            binary_path=binary_path,
        )
    fname.unlink()
    return tree

def infer_raxml_ng_tree_from_phylip(
    alignment: Union[str, Path],
    nboots: int=0,
    nthreads: int=4,
    nworkers: int=4,
    seed: Optional[int]=None,
    subst_model: str="GTR+G",
    binary_path: Union[str, Path]=None,
    ) -> toytree.ToyTree:
    """Return a single ML tree inferred by raxml-ng.
    """
    binary_path = binary_path if binary_path else RAXML
    assert Path(binary_path).exists(), BINARY_MISSING.format(binary_path)
    fpath = Path(alignment)
    assert fpath.exists(), f"{fpath} alignment input file does not exist"

    # run `raxml-ng [search|all] ...`
    cmd = [
        binary_path,
        "--msa", str(fpath),
        "--model", str(subst_model),
        "--redo",
        "--threads", str(nthreads),
        "--workers", str(nworkers) if nworkers else 'auto',
        "--nofiles", "interim",
    ]
    if seed:
        cmd.extend(["--seed", str(seed)])
    if nboots:
        cmd.extend(["--all", "--bs-trees", str(nboots)])
        treefile = fpath.with_suffix(".phy.raxml.support")
    else:
        treefile = fpath.with_suffix(".phy.raxml.bestTree")

    with Popen(cmd, stderr=STDOUT, stdout=PIPE) as proc:
        out, _ = proc.communicate()
        if proc.returncode:
            raise IpcoalError(out.decode())

    # parse result from treefile and cleanup
    tree = toytree.tree(treefile)
    tmpfiles = fpath.parent.glob(fpath.name + ".*")
    for tmp in tmpfiles:
        tmp.unlink()
    return tree


def infer_raxml_ng_tree(
    model: ipcoal.Model,
    idxs: Union[int, Sequence[int], None]=None,
    nboots: int=0,
    nthreads: int=4,
    nworkers: Optional[int]=None,
    seed: int=None,
    diploid: bool=False,
    subst_model: str="GTR+G",
    binary_path: Union[str, Path]=None,
    tmpdir: Optional[Path]=None,
    ) -> toytree.ToyTree:
    """Return a single ML tree inferred by raxml-ng.

    Sequence data is extracted from the model.seqs array and written
    as concatenated data to a phylip file, either for individual
    haplotypes or diploid genotypes if diploid=True. If `idxs=None`
    all data is concatenated, else a subset of one or more loci can
    be selected to be concatenated.

    CMD: raxml-ng --all --msa {phy} --subst_model {GTR+G} --redo

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
    binary_path: None, str, or Path
        Path to the ASTRAL binary that is called by `java -jar binary`.

    Examples
    --------
    >>> sptree = toytree.rtree.unittree(10, treeheight=1e5, seed=123)
    >>> model = ipcoal.Model(tree=sptree, Ne=1e4, nsamples=2)
    >>> model.sim_loci(nloci=100, nsites=10_000)
    >>> tree = ipcoal.phylo.infer_raxml_ng_tree(model.seqs, 0)
    >>> tree.draw();
    """
    fname = _write_tmp_phylip_file(model, idxs, diploid, tmpdir)
    kwargs = dict(
        alignment=fname, nboots=nboots, nthreads=nthreads,
        nworkers=nworkers, seed=seed, subst_model=subst_model,
        binary_path=binary_path)
    tree = infer_raxml_ng_tree_from_phylip(**kwargs)
    for tmp in fname.parent.glob(fname.name + "*"):
        tmp.unlink()
    return tree

# def infer_raxml_ng_tree_from_window():
#     pass

# def infer_raxml_ng_trees_from_sliding_windows():
#     pass

def infer_raxml_ng_trees(
    model: ipcoal.Model,
    nboots: int=0,
    nproc: int=1,
    seed: int=None,
    diploid: bool=False,
    subst_model: str="GTR+G",
    binary_path: Union[str, Path]=None,
    tmpdir: Optional[Path]=None,
    nthreads: int=4,
    nworkers: int=4,
    ) -> pd.DataFrame:
    """Return a DataFrame w/ inferred gene trees at every locus.

    Sequence data is extracted from the model.seqs array and written
    as concatenated data to a phylip file, either for individual
    haplotypes or diploid genotypes if diploid=True. If `idxs=None`
    all data is concatenated, else a subset of one or more loci can
    be selected to be concatenated.

    CMD: raxml-ng --all --msa {phy} --subst_model {GTR+G} --redo

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
        Path to the ASTRAL binary that is called by `java -jar binary`.
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

    # TODO: asynchrony so that writing and processing are not limited.
    with ProcessPoolExecutor(max_workers=nproc) as pool:
        for lidx in model.df.locus.unique():
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

                # disk-friendly mode
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
    data = (model.df
        .groupby("locus")
        .agg({"start": "min", "end": "max", "nbps": "sum", "nsnps": "sum"})
        .reset_index()
    )
    data['gene_tree'] = [
        rasyncs[i] if isinstance(rasyncs[i], str) else
        rasyncs[i].result().write() for i in sorted(rasyncs)
    ]
    return data


if __name__ == "__main__":

    BIN = "/home/deren/miniconda3/envs/ipyrad/bin/raxml-ng"
    TREE = toytree.rtree.unittree(ntips=5, seed=123, treeheight=1e6)
    MODEL = ipcoal.Model(TREE, Ne=5e4, subst_model="jc69")
    MODEL.sim_loci(nloci=10, nsites=20)
    TREES = infer_raxml_ng_trees(MODEL, binary_path=BIN, nthreads=2, nboots=10, seed=1)
    print(TREES)
    # print({i: j.result().write(None) for (i, j) in TREES.items()})
    # TREE._draw_browser(ts='s', node_labels="support")
