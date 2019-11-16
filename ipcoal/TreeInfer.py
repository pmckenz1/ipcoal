#!/usr/bin/env python


import os
import sys
from .utils import ipcoalError



SUPPORTED = {
    "raxml": "raxmlHPC-PTHREADS-AVX2",
    "iqtree": "iqtree",
}


class TreeInfer:

    def __init__(self, method="raxml"):
        """
        DocString...
        """

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



    def run(self):
        """
        Runs the appropriate binary call
        """
        if self.method == "raxml":
            self.infer_raxml()
        if self.method == "iqtree":
            self.infer_iqtree()
                


    def infer_raxml(self):
        
        cmd = [
            self.binary, 

        ]



    def infer_iqtree(self):
        pass