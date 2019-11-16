#!/usr/bin/env python


import os
import sys
import subprocess as sps
from .utils import ipcoalError



SUPPORTED = {
    "raxml": "raxmlHPC-PTHREADS-AVX2",
    "iqtree": "iqtree",
}



class TreeInfer:

    def __init__(self, seqs, method="raxml"):
        """
        DocString...
        """
        self.seqs = seqs
        self.binary = ""
        self.method = method.lower()
        self.inference_args = {}
        self.check_method_and_binary()
        self.run()



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



    def run(self, idx):
        """
        Runs the appropriate binary call
        """

        # write the tempfile for locus idx
        tempfile = self.write_tempfile(idx)

        # send tempfile to be processed
        if self.method == "raxml":
            tree = self.infer_raxml(tempfile)
        if self.method == "iqtree":
            tree = self.infer_iqtree(tempfile)

        # cleanup 


        # return result
        return tree
                


    def infer_raxml(self, tempfile):
        """
        Writes a tmp file in phylip format, infers raxml tree, cleans up, 
        and returns a newick string of the full tree as a result.
        """

        # create the command line
        cmd = [
            self.binary, 
            "-f", self.kwargs["a"],
            "-T", "0", 
            "-m", "GTRGAMMA", 
            "-n", self.kwargs["n"],
            "-w", self.kwargs["w"],
            "-s", tempfile,
        ]
        if "N" in self.kwargs:
            cmd += ["-N", str(self.kwargs["N"])]
        if "x" in self.kwargs:
            cmd += ["-x", str(self.kwargs["x"])]
        if "o" in self.kwargs:
            cmd += ["-o", str(self.kwargs["o"])]

        # call the command
        proc = sps.Popen(cmd, stderr=sps.STDOUT, stdout=sps.PIPE)
        out, err = proc.communicate()
        if proc.returncode:
            raise ipcoalError("error in raxml: {}".format(out.decode()))

        # read in the full tree or bipartitions
        tree = ""

        return tree



    def infer_iqtree(self):
        pass