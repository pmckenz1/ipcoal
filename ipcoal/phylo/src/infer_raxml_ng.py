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
    cleanup: bool=True,
    ) -> toytree.ToyTree:
    """Return a single ML tree inferred by raxml-ng from a phylip string.

    This takes an alignment string and creates a temporary phylip file
    that is automatically cleaned up afterwards. The other tmp and result
    files created by raxml-ng can also be cleaned up automatically.
    """
    tmpdir = tmpdir if tmpdir is not None else tempfile.gettempdir()
    with tempfile.NamedTemporaryFile(dir=tmpdir, suffix=f"_{os.getpid()}") as tmp:

        # write phylip data to the tmpfile
        fname = Path(tmp.name).with_suffix(".phy")
        with open(fname, 'w', encoding="utf-8") as out:
            out.write(alignment)

        tree = infer_raxml_ng_tree_from_phylip(
            alignment=fname, nboots=nboots, nthreads=nthreads,
            nworkers=nworkers, seed=seed, subst_model=subst_model,
            binary_path=binary_path,
        )
    if cleanup:
        for tmpf in fname.parent.glob(fname.name + "*"):
            tmpf.unlink()
    else:
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

    This is a convenience function for inferring gene trees for loci
    simulated ipcoal using raxml-ng. It is not intended as a generic
    wrapper around raxml-ng to support its many capabilities.
    """
    binary_path = binary_path if binary_path else RAXML
    assert Path(binary_path).exists(), BINARY_MISSING.format(binary_path)
    fpath = Path(alignment)
    assert fpath.exists(), f"{fpath} alignment input file does not exist"

    # remove any fpath.raxml.* existing result files.
    for tmp in fpath.parent.glob(f"{fpath.name}.*"):
        tmp.unlink()

    # run `raxml-ng [search|all] ...`
    # I believe '--threads auto{X}' is max X threads.
    cmd = [
        str(binary_path),
        "--msa", str(fpath),
        "--model", str(subst_model),
        "--redo",
        "--threads", f"auto{{nthreads}}",
        "--workers", str(nworkers) if nworkers else 'auto',
        "--nofiles", "interim",
        "--log", "ERROR"
    ]
    if seed:
        cmd.extend(["--seed", str(seed)])
    if nboots:
        cmd.extend(["--all", "--bs-trees", str(nboots)])

    #
    with Popen(cmd, stderr=STDOUT, stdout=PIPE) as proc:
        out, _ = proc.communicate()

        # raise an error. raxml-ng uses returncode 1 even for warnings
        # which is really annoying. We asked it only to log errors, so
        # we will only raise an exception if the 'out' variable contains
        # any logged errors.
        if proc.returncode:
            error = out.decode()
            logger.error(f"ERROR:\n{error}\n\n{' '.join(cmd)}")
            if error:
                raise IpcoalError(error)

    # raxml bestTree has randomly resolved nodes when no info exists
    # but we instead want unresolved info for unresolved nodes which,
    # if any exist, it writes to the Collapsed output file.
    if nboots:
        treefile = fpath.with_suffix(".phy.raxml.support")
    else:
        if fpath.with_suffix(".phy.raxml.bestTreeCollapsed").exists():
            treefile = fpath.with_suffix(".phy.raxml.bestTreeCollapsed")
        else:
            treefile = fpath.with_suffix(".phy.raxml.bestTree")

    # parse result from treefile and return
    tree = toytree.tree(treefile)
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
    remove_tmp_files: bool=True,
    ) -> toytree.ToyTree:
    """Return a gene tree inferred by raxml-ng for one or more loci.

    Sequence data is extracted from the model.seqs array and written
    as concatenated data to a phylip file, either for individual
    haplotypes or diploid genotypes if diploid=True. If `idxs=None`
    all data is concatenated, else a subset of one or more loci can
    be selected to be concatenated.

    CMD: raxml-ng --all --msa {phy} --subst_model {GTR+G} --redo

    Note
    ----
    This function will return a ToyTree object parsed from the results
    and then remove all temporary files. The tree is parsed from either
    the .support, .bestTree.Collapsed, or .bestTree files, in that
    order.

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
    diploid: bool
        If True then pairs of samples belonging to the same lineage
        are grouped to create diploid genotype calls in the phylip
        file. This will raise an error if nsamples is not even.
    binary_path: None, str, or Path
        Path to the raxa binary.

    Examples
    --------
    >>> sptree = toytree.rtree.unittree(10, treeheight=1e5, seed=123)
    >>> model = ipcoal.Model(tree=sptree, Ne=1e4, nsamples=2)
    >>> model.sim_loci(nloci=100, nsites=10_000)
    >>> tree = ipcoal.phylo.infer_raxml_ng_tree(model.seqs, 0)
    >>> tree.draw();
    """
    # write selected loci to a concatenated phylip TMP file.
    fpath = _write_tmp_phylip_file(model, idxs, diploid, tmpdir)

    # dict w/ TMP file path and params args.
    kwargs = dict(
        alignment=fpath, nboots=nboots, nthreads=nthreads,
        nworkers=nworkers, seed=seed, subst_model=subst_model,
        binary_path=binary_path)

    # get result as ToyTree and remove all tmp files.
    tree = infer_raxml_ng_tree_from_phylip(**kwargs)

    # cleanup
    if remove_tmp_files:
        for tmp in fpath.parent.glob(f"{fpath.name}*"):
            tmp.unlink()
    return tree

# def infer_raxml_ng_tree_from_window():
#     pass

# def infer_raxml_ng_trees_from_sliding_windows():
#     pass

def infer_raxml_ng_trees(
    model: ipcoal.Model,
    idxs: Union[Sequence[int], None]=None,
    nboots: int=0,
    nproc: int=1,
    seed: int=None,
    diploid: bool=False,
    subst_model: str="GTR+G",
    binary_path: Union[str, Path]=None,
    tmpdir: Optional[Path]=None,
    nthreads: int=4,
    nworkers: int=4,
    cleanup: bool=True,
    ) -> pd.DataFrame:
    r"""Return a DataFrame w/ inferred gene trees at every locus.

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
        Path to the raxml binary.
    nworkers: int
        ...
    tmpdir: Path or None
        Path to store temporary files. Default is tempdir (/tmp).
    cleanup: bool
        If True then each tmpfile is removed immediately after use and
        the tmpdir is removed at the end.

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
        binary_path=binary_path, tmpdir=tmpdir, cleanup=cleanup)

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
            # if no data then return a star tree.
            # locus = model.df[model.df.locus == lidx]
            # if not locus.nsnps.sum():
            #     names = list(chain(*model.get_imap_dict(diploid=diploid).values()))
            #     rasyncs[lidx] = pool.submit(get_star_tree, names)
            #     empty += 1
            # else:
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

    # check for failures
    for job, rasync in rasyncs.items():
        if rasync.exception():
            logger.error(f"raxml-ng error on loc {job}: {kwargs}")
            _ = rasync.result()

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

def get_star_tree(names: Sequence[str]) -> toytree.ToyTree:
    """Return a ToyTree w/ start topology for a list of names."""
    treenode = toytree.Node(dist=0)
    for tip in names:
        treenode._add_child(toytree.Node(name=tip, dist=0))
    tree = toytree.ToyTree(treenode)
    return tree


if __name__ == "__main__":

    BIN = "/home/deren/miniconda3/envs/ipyrad/bin/raxml-ng"
    TREE = toytree.rtree.unittree(ntips=5, seed=123, treeheight=1e6)
    MODEL = ipcoal.Model(TREE, Ne=5e4, subst_model="jc69")
    MODEL.sim_loci(nloci=10, nsites=200)
    TREES = infer_raxml_ng_trees(MODEL, binary_path=BIN, nthreads=2, nworkers=2, nboots=10, seed=1)
    print(TREES)

    # print({i: j.result().write(None) for (i, j) in TREES.items()})
    # TREE._draw_browser(ts='s', node_labels="support")
