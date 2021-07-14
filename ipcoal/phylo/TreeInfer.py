#!/usr/bin/env python


"""
Tree inference wrappers for running raxml, mb or others 
on loci or sliding windows of sequence data for comparing
genealogies with inferred gene trees.
"""


import os
import sys
import glob
import tempfile
import subprocess as sps
import numpy as np
import toytree

from ipcoal.io.writer import Writer
# from .mrbayes import MrBayes as mrbayes
from ipcoal.utils.utils import IpcoalError



SUPPORTED = {
    "raxml": "raxmlHPC-PTHREADS",
    "iqtree": "iqtree",
    "mb": "mb",
    "mrbayes": "mb",
}


class TreeInfer:
    """
    Class for selecting and implementing phylogenetic inference methods.
    """
    def __init__(self, seqs, names, inference_method="raxml", inference_args={}):
        """
        DocString...
        """
        self.rng = np.random.default_rng()
        self.seqs = seqs
        self.names = names  # model.alpha_ordered_names
        self.binary = ""
        self.method = inference_method.lower()        
        self.inference_args = inference_args
        self.check_method_and_binary()

        # default options that user can override
        self.raxml_kwargs = {
            "f": "d", 
            "N": "10",
            "T": "4", 
            "m": "GTRGAMMA",
            "w": tempfile.gettempdir()
        }
        self.raxml_kwargs.update(inference_args)

        self.mb_kwargs = {
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
        self.mb_kwargs.update(inference_args)


    def check_method_and_binary(self):
        """
        Checks that the 'method' is supported and finds existing binary
        """
        # check if method is supported
        if self.method not in SUPPORTED:
            raise ipcoalError(
                "method {} not currently supported".format(self.method)
            )

        # check for binary of this method (assumes conda install)
        self.binary = os.path.join(sys.prefix, "bin", SUPPORTED[self.method])
        if not os.path.exists(self.binary): 
            raise ipcoalError(
                "binary {} not found, please install using conda."
                .format(self.binary)
            )


    def write_tempfile(self, idx):
        """
        Writes a phylip or nexus file for xxx or mrbayes.
        """
        # create a writer object with seqs and names
        writer = Writer(self.seqs, self.names)

        if self.method == "raxml":

            # call write with selected idx 
            writer.write_concat_to_phylip(
                outdir=tempfile.gettempdir(),
                name=str(os.getpid()) + ".phy",
                idxs=[idx],
                quiet=True,
            )
            # return filepath to the phy file
            path = os.path.join(
                tempfile.gettempdir(), 
                str(os.getpid()) + ".phy"
            )
            return path

        # mrbayes is only alternative righ tnow...
        else:
            writer.write_concat_to_nexus(
                outdir=tempfile.gettempdir(),
                name=str(os.getpid()) + ".nex",
                idxs=[idx],
            )
            path = os.path.join(
                tempfile.gettempdir(), 
                str(os.getpid()) + ".nex",
            )
            return path



    def run(self, idx):
        """
        Runs the appropriate binary call
        """

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



    def infer_raxml(self, tmp):
        """
        Writes a tmp file in phylip format, infers raxml tree, cleans up, 
        and returns a newick string of the full tree as a result.
        """

        # remove the results files if they exist in outdir already
        raxfiles = glob.glob(os.path.join(self.raxml_kwargs["w"], "RAxML_*"))
        raxfiles = [i for i in raxfiles if i.endswith(os.path.basename(tmp))]
        for rfile in raxfiles:
            os.remove(rfile)

        # create the command line
        cmd = [
            self.binary, 
            "-f", self.raxml_kwargs["f"],
            "-T", self.raxml_kwargs["T"],
            "-m", "GTRGAMMA", 
            "-n", os.path.basename(tmp),
            "-w", self.raxml_kwargs["w"],
            "-s", tmp,
            "-p", str(self.rng.integers(0, 1e9)),
        ]

        # additional allowed arguments
        if "N" in self.raxml_kwargs:
            cmd += ["-N", str(self.raxml_kwargs["N"])]
        if "x" in self.raxml_kwargs:
            cmd += ["-x", str(self.raxml_kwargs["x"])]
        if "o" in self.raxml_kwargs:
            cmd += ["-o", str(self.raxml_kwargs["o"])]
        if "a" in self.raxml_kwargs:
            cmd += ["-a", str(self.raxml_kwargs["a"])]

        # call the command
        proc = sps.Popen(cmd, stdout=sps.PIPE, stderr=sps.STDOUT)
        out, err = proc.communicate()
        if proc.returncode:
            print(out, proc.returncode, cmd)
            #raise ipcoalError("error in raxml: {}".format(out.decode()))

        # read in the full tree or bipartitions
        best = os.path.join(
            self.raxml_kwargs["w"], 
            "RAxML_bestTree.{}".format(os.path.basename(tmp))
        )
        with open(best, 'r') as res:
            tree = res.read().strip()
        return tree



    def infer_iqtree(self):
        raise NotImplementedError("iqtree not yet supported")


    def infer_mb(self, tmp):
        """
        Call mb on phy and returned parse tree result
        """

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
