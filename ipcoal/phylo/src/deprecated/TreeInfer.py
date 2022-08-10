#!/usr/bin/env python

"""Tree inference wrappers for running raxml, mb or others.

Inference can be run on discrete loci or sliding windows of
sequence data for comparing genealogies with inferred gene trees.
"""

from typing import Sequence, Dict
import os
import sys
import glob
import tempfile
import subprocess as sps
import numpy as np
from loguru import logger

import toytree
from ipcoal.io.writer import Writer
# from .mrbayes import MrBayes as mrbayes
from ipcoal.utils.utils import IpcoalError

logger = logger.bind(name="ipcoal")

SUPPORTED = {
    "raxml": "raxmlHPC-PTHREADS",
    "iqtree": "iqtree",
    "mb": "mb",
    "mrbayes": "mb",
}

DEFAULT_RAX_KWARGS = {
    "f": "d",
    "N": "10",
    "T": "4",
    "m": "GTRGAMMA",
    "w": tempfile.gettempdir(),
}

DEFAULT_MB_KWARGS = {
    "clockratepr": "lognorm(-7,0.6)",
    "clockvarpr": "tk02",
    "tk02varpr": "exp(1.0)",
    "brlenspr": "clock:birthdeath",
    "samplestrat": "diversity",
    "sampleprob": "0.1",
    "speciationpr": "exp(10)",
    "extinctionpr": "beta(2, 200)",
    "treeagepr": "offsetexp(1, 5)",
    "ngen": 100000,
    "nruns": "1",
    "nchains": 4,
    "samplefreq": 1000,
}


class TreeInfer:
    """Select and implement phylogenetic inference methods.

    Takes a seqs array and names List from the Model object, and
    a method name and args dict for the inference method to employ.
    """
    def __init__(
        self,
        model: 'ipcoal.Model',
        inference_method: str = "raxml",
        inference_args: Dict[str,str] = None,
        diploid: bool = False,
        **kwargs,
        ):

        self.model = model
        self.method = inference_method.lower()
        self.inference_args = {} if inference_args is None else inference_args
        self.diploid = diploid

        self.rng = np.random.default_rng()
        self.seqs = model.seqs
        self.names = model.alpha_ordered_names
        self.binary = ""

        # for windows inference method a kwargs['newseqs'] will exist.
        self.alt_seqs = kwargs.get("alt_seqs", None)

        self._check_method_and_binary()
        self._update_inference_kwargs_with_defaults()

    def _update_inference_kwargs_with_defaults(self):
        """Takes dict w/ default options and updates with user options"""
        if self.method == "raxml":
            kwargs = DEFAULT_RAX_KWARGS.copy()
            kwargs.update(self.inference_args)
            self.inference_args = kwargs
        elif self.method == "mb":
            kwargs = DEFAULT_MB_KWARGS.copy()
            kwargs.update(self.inference_args)
            self.inference_args = kwargs

    def _check_method_and_binary(self):
        """Check that the 'method' is supported and find binary.

        Currently searches only in CONDAPATH/BIN/...
        """
        if self.method not in SUPPORTED:
            raise IpcoalError(
                f"method {self.method} not currently supported")

        # check for binary of this method (assumes conda install)
        self.binary = os.path.join(sys.prefix, "bin", SUPPORTED[self.method])
        if not os.path.exists(self.binary):
            raise IpcoalError(
                f"binary {self.binary} not found, please install using conda.")

    def write_tempfile(self, idx):
        """Writes a phylip or nexus file for xxx or mrbayes."""

        # create a writer object with seqs and names
        writer = Writer(self.model, self.alt_seqs)

        # write the locus idx to .phy format
        if self.method == "raxml":
            writer.write_concat_to_phylip(
                outdir=tempfile.gettempdir(),
                name=str(os.getpid()) + ".phy",
                idxs=[idx],
                quiet=True,
                diploid=self.diploid,
            )
            # return filepath to the phy file
            path = os.path.join(
                tempfile.gettempdir(),
                str(os.getpid()) + ".phy"
            )
            return path

        # mrbayes is only alternative righ tnow...
        if self.method == "mb":
            writer.write_concat_to_nexus(
                outdir=tempfile.gettempdir(),
                name=str(os.getpid()) + ".nex",
                idxs=[idx],
                diploid=self.diploid,
            )
            path = os.path.join(
                tempfile.gettempdir(),
                str(os.getpid()) + ".nex",
            )
            return path

        # other
        raise IpcoalError(f"inference_method not supported: {self.method}.")

    def run(self, idx):
        """Calls write_tempfile and runs binary on the file."""

        # write the tempfile for locus idx
        tmp = self.write_tempfile(idx)

        # send tempfile to be processed
        if self.method == "raxml":
            tree = self.infer_raxml(tmp)
        if self.method == "iqtree":
            tree = self.infer_iqtree(tmp)
        if self.method == "mb":
            tree = self.infer_mb(tmp)

        # cleanup (TODO) remove phy file extensions
        #os.remove(tmp)

        # return result
        return tree

    def infer_raxml(self, tmp: str) -> str:
        """Runs raxml on tmp file and returns newick str.

        Writes a tmp file in phylip format, infers raxml tree, cleans up,
        and returns a newick string of the full tree as a result.
        """
        # remove the results files if they exist in outdir already
        raxfiles = glob.glob(os.path.join(self.inference_args["w"], "RAxML_*"))
        raxfiles = [i for i in raxfiles if i.endswith(os.path.basename(tmp))]
        for rfile in raxfiles:
            os.remove(rfile)

        # create the command line
        cmd = [
            self.binary,
            "-f", self.inference_args["f"],
            "-T", self.inference_args["T"],
            "-m", "GTRGAMMA",
            "-n", os.path.basename(tmp),
            "-w", self.inference_args["w"],
            "-s", tmp,
            "-p", str(self.rng.integers(0, 1e9)),
        ]

        # additional allowed arguments
        if "N" in self.inference_args:
            cmd += ["-N", str(self.inference_args["N"])]
        if "x" in self.inference_args:
            cmd += ["-x", str(self.inference_args["x"])]
        if "o" in self.inference_args:
            cmd += ["-o", str(self.inference_args["o"])]
        if "a" in self.inference_args:
            cmd += ["-a", str(self.inference_args["a"])]

        # call the command
        with sps.Popen(cmd, stdout=sps.PIPE, stderr=sps.STDOUT) as proc:
            out, _ = proc.communicate()
            if proc.returncode:
                logger.error(f"raxml error: {out}, returncode={proc.returncode}\n cmd={cmd}")
            #raise ipcoalError("error in raxml: {}".format(out.decode()))

        # read in the full tree or bipartitions
        best = os.path.join(
            self.inference_args["w"],
            "RAxML_bestTree.{}".format(os.path.basename(tmp))
        )
        with open(best, 'r', encoding="utf-8") as res:
            tree = res.read().strip()
        return tree

    def infer_iqtree(self):
        raise NotImplementedError("iqtree not yet supported")

    def infer_mb(self, tmp):
        """
        Call mb on phy and returned parse tree result
        """
        raise NotImplementedError("mb not currently supported")

        # call mb on the input phylip file with inference args
        mb = mrbayes(
            data=tmp,
            name="temp_" + str(os.getpid()),
            workdir=tempfile.gettempdir(),
            **self.inference_args
        )
        mb.run(force=True, quiet=True, block=True)

        # get newick string from result
        tree = toytree.tree(mb.trees.constre, tree_format=10).newick

        # cleanup remote tree files
        for tup in mb.trees:
            tpath = tup[1]
            if os.path.exists(tpath):
                os.remove(tpath)

        # remove the TEMP phyfile in workdir/tmpdir
        #os.remove(tmp)

        # return results
        return tree
