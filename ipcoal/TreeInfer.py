#!/usr/bin/env python


import os
import sys
import glob
import tempfile
import subprocess as sps

from .Writer import Writer
from .utils import ipcoalError


SUPPORTED = {
    "raxml": "raxmlHPC-PTHREADS-AVX2",
    "iqtree": "iqtree",
}


class TreeInfer:

    def __init__(self, model, inference_method="raxml", inference_args={}):
        """
        DocString...
        """
        self.model = model
        self.seqs = model.seqs
        self.names = model.names
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
        writer = Writer(self.seqs, self.names)
        writer.write_concat_to_phylip(
            outdir=tempfile.gettempdir(), 
            name=str(os.getpid()) + ".phy",
            idxs=[idx],
        )
        return os.path.join(tempfile.gettempdir(), str(os.getpid()) + ".phy")



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

        # cleanup (TODO) remove phy file extensions
        os.remove(tmp)

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
            "-p", str(self.model.random.randint(0, 1e9)),
        ]
        if "N" in self.raxml_kwargs:
            cmd += ["-N", str(self.raxml_kwargs["N"])]
        if "x" in self.raxml_kwargs:
            cmd += ["-x", str(self.raxml_kwargs["x"])]
        if "o" in self.raxml_kwargs:
            cmd += ["-o", str(self.raxml_kwargs["o"])]

        # call the command
        proc = sps.Popen(cmd, stdout=sps.PIPE)  # stderr=sps.STDOUT, 
        out, err = proc.communicate()
        if proc.returncode:
            raise ipcoalError("error in raxml: {}".format(out.decode()))

        # read in the full tree or bipartitions
        best = os.path.join(
            self.raxml_kwargs["w"], 
            "RAxML_bestTree.{}".format(os.path.basename(tmp))
        )
        with open(best, 'r') as res:
            tree = res.read().strip()
        return tree



    def infer_iqtree(self):
        pass