#!/usr/bin/env python

"""Infer a species tree from a collection of gene trees using astral.

"""

from typing import Union, TypeVar, Optional, Dict, Sequence
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT
import tempfile
import toytree
# from loguru import logger
from ipcoal.utils.utils import IpcoalError

MultiTree = TypeVar("MultiTree")


BINARY_MISSING = """
    Cannot find astral binary {}. Make sure you have
    astral installed, which you can do with the following command
    which also installs required dependencies (e.g., java):

    >>> conda install astral3 -c conda-forge -c eaton-lab

    Then find the full path where the binary is installed and 
    enter it to this function as the `binary_path` argument.
"""


def infer_astral_tree(
    trees: Union[str, Path, MultiTree],
    binary_path: Union[str, Path],
    nboots: int=1000,
    annotation: int=3,
    seed: Optional[int]=None,
    imap: Dict[str, Sequence[str]]=None,
    tmpdir: Optional[Path]=None,
    ) -> toytree.ToyTree:
    """Return tree inferred by astral from a collection of gene trees.

    java -jar {binary_path} -i {trees} -t {annotation} -r {nboots} ...

    Parameters
    ----------
    model: str, toytree.MultiTree
        A collection of trees input as either a filepath to a text
        file with newick strings on separate lines, or input can be
        a toytree.MultiTree object containing multiple trees.
    nboots: int
        Number of bootstrap replicates to run.
    annotation: int
        Instructions to ASTRAL for recording statistics in the
        newick tree structure.
    binary_path: None, str, or Path
        Path to the ASTRAL binary that is called by `java -jar binary`.

    Examples
    --------
    >>> sptree = toytree.rtree.unittree(ntips=10, treeheight=1e5)
    >>> model = ipcoal.Model(tree=sptree, Ne=1e5, nsamples=2)

    (1) infer sptree from true genealogies
    >>> model.sim_trees(nloci=1000)
    >>> imap = model.get_imap_dict()
    >>> mtree = toytree.mtree(model.df.genealogy)
    >>> astree = ipcoal.phylo.infer_astral_tree(mtree)
    >>> astree.draw();

    (2) infer sptree from inferred gene trees
    >>> model.sim_loci(nloci=1000, nsites=300)
    >>> imap = model.get_imap_dict(diploid=True)
    >>> raxdata = ipcoal.phylo.infer_raxml_ng_...(model, diploid=True)
    >>> mtree = toytree.mtree(raxdata.gene_tree)
    >>> astree = ipcoal.phylo.infer_astral_tree(mtree)
    >>> astree.draw();
    """
    tmpdir = tmpdir if tmpdir is not None else tempfile.gettempdir()
    binary_path = binary_path if binary_path else "NO PATH PROVIDED"
    assert Path(binary_path).exists(), BINARY_MISSING.format(binary_path)

    # write trees input as a newline separated file
    with tempfile.NamedTemporaryFile(dir=tmpdir) as tmpfile:
        fname = Path(tmpfile.name)

        # write trees to a file separated by newlines w/o edge lens/labels
        if isinstance(trees, toytree.MultiTree):
            trees.write(fname.with_suffix(".trees"), None, None)
        else:
            toytree.mtree(trees).write(fname.with_suffix(".trees"), None, None)

    # run ASTRAL on treefile
    tree_file = fname.with_suffix(".astral")
    cmd = [
        "java", "-jar", str(binary_path),
        "--input", str(fname.with_suffix(".trees")),
        "--output", str(tree_file),
        "--branch-annotate", str(annotation),
        "--reps", str(nboots),
    ]
    if seed:
        cmd.extend(["--seed", str(seed)])
    if imap:
        # write imap to astral's pop format: "species: a,b,c,d"
        fimap = fname.with_suffix('.imap')
        with open(fimap, 'w', encoding='utf-8') as out:
            for key, vals in imap.items():
                out.write(f"{key}: {','.join(vals)}\n")
        cmd.extend(["--namemapfile", str(fimap)])
        
    # run ASTRAL and parse and return the result
    with Popen(cmd, stderr=STDOUT, stdout=PIPE) as proc:
        out, _ = proc.communicate()
        if proc.returncode:
            raise IpcoalError(out.decode())
    tree = toytree.io.read_newick(tree_file, internal_labels="support")

    # cleanu tmpfiles
    for tmp in fname.parent.glob(fname.name + "*"):
        tmp.unlink()
    return tree

# def sim_test():
#     import ipcoal

#     sptree = toytree.rtree.imbtree(ntips=5)
#     sptree.mod.edges_scale_to_root_height(1e6, inplace=True)
#     model = ipcoal.Model(sptree, Ne=1e5, nsamples=2)
#     model.sim_trees(100)
#     imap = model.get_imap_dict()
#     atre = infer_astral_tree(trees, binary_path=ASTRAL, imap=imap)



if __name__ == "__main__":

    ASTRAL = "/home/deren/miniconda3/envs/ipyrad/bin/astral.5.7.1.jar"
    TREE = infer_astral_tree("/tmp/test.trees", binary_path=ASTRAL)
    TREE._draw_browser(ts='s', node_labels="support")
