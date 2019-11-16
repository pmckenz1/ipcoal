#!/usr/bin/env python

import os
import re
import sys
import numpy as np
import subprocess as sps
from .utils import ipcoalError


class SeqGen:
    """
    Opens a view to seq-gen in a subprocess so that many gene trees can be 
    cycled through without the overhead of opening/closing subprocesses.
    """

    def __init__(self):

        # set binary path for conda env and check for binary
        self.binary = os.path.join(sys.prefix, "bin", "seq-gen")
        assert os.path.exists(self.binary), (
            "binary {} not found".format(self.binary))

        # call open_subprocess to set the shell 
        self.shell = None


    def open_subprocess(self):
        """
        Open a persistent Popen bash shell
        """
        # open 
        self.shell = sps.Popen(
            ["bash"], stdin=sps.PIPE, stdout=sps.PIPE, bufsize=0)


    def close_subprocess(self):
        """
        Cleanup and shutdown the subprocess shell.
        """
        self.shell.stdin.close()
        self.shell.terminate()
        self.shell.wait(timeout=1.0)


    def feed_tree(self, newick, nsites, mut, seed, **kwargs):
        """
        Feed a command string a read results until empty line.
        TODO: allow kwargs to add additional seq-gen args.
        """
        # command string
        cmd = (
            "{} -mGTR -l {} -s {} -z {} <<< \"{}\"; echo done\n"
            .format(self.binary, nsites, mut, seed, newick)
        )

        # feed to the shell
        self.shell.stdin.write(cmd.encode())
        self.shell.stdin.flush()

        # catch returned results until done\n
        hold = []
        for line in iter(self.shell.stdout.readline, b"done\n"):
            hold.append(line.decode())

        # remove the "Time taken: 0.0000 seconds" bug in seq-gen
        hold = "".join(hold)
        hold = re.sub(
            pattern=r"Time\s\w+:\s\d.\d+\s\w+\n",
            repl="",
            string=hold,
        )

        # if no sequence was produce then raise an error
        if not hold:
            raise ipcoalError(
                "seq-gen error; no sites generated\ncmd: {}\n"
                .format(cmd)
                )

        # make seqs into array, sort it, and count differences
        names = []
        seqs = []
        for line in hold.split("\n")[1:-1]:
            name, seq = line.split()
            names.append(name)
            seqs.append(list(seq))

        # convert seqs to int array 
        arr = np.array(seqs)
        arr[arr == "A"] = 0
        arr[arr == "C"] = 1
        arr[arr == "G"] = 2
        arr[arr == "T"] = 3
        arr = arr.astype(np.uint8)

        # reorder rows to be in alphanumeric order of the names
        return arr[np.argsort(names), :]




    # def seqgen_on_tree(self, newick, ntaxa, nsites, mut, **kwargs):
    #     """
    #     Call seqgen on a newick tree to generate nsites of data using a given
    #     mutation rate and return an integer sequence array for ntaxa.

    #     Parameters
    #     ----------
    #     newick: str
    #         A newick string with branch lengths in coalescent units. Here we 
    #         expect the trees will be ultrametric.
    #     ntaxa: int
    #         The number of tips in the newick tree. 
    #     nsites: int
    #         The number of sites to simulate. Depending on mutation rate and 
    #         tree size, shape, and height these may evolve SNPs or not. 
    #     mut: float
    #         The per site per generation mutation rate. 
    #     kwargs: dict
    #         Additional arguments to the GTR model (TODO).
    #     """

    #     # write the newick string to a temporary file. This is required as far
    #     # as I can tell, cannot get piped input to work in seqgen, ugh.
    #     fname = os.path.join(tempfile.gettempdir(), str(os.getpid()) + ".tmp")
    #     with open(fname, 'w') as temp:
    #         temp.write(newick)

    #     # set up the seqgen CLI call string
    #     proc1 = subprocess.Popen([
    #         "seq-gen",
    #         "-m", "GTR",
    #         "-l", str(nsites),        # seq length
    #         "-s", str(self.mut),      # mutation rate
    #         fname,
    #         # ... other model params...,
    #         ],
    #         stderr=subprocess.STDOUT,
    #         stdout=subprocess.PIPE,
    #     )

    #     # run seqgen and check for errors
    #     out, _ = proc1.communicate()
    #     if proc1.returncode:
    #         raise Exception("seq-gen error: {}".format(out.decode()))

    #     # remove the "Time taken: 0.0000 seconds" bug in seq-gen
    #     physeq = re.sub(
    #         pattern=r"Time\s\w+:\s\d.\d+\s\w+\n",
    #         repl="",
    #         string=out.decode())

    #     # make seqs into array, sort it, and count differences
    #     physeq = physeq.strip().split("\n")[-(self.ntips + 1):]

    #     arr = np.array([list(i.split()[-2:]) for i in physeq[1:]], dtype=bytes)
    #     names = [arr_ele[0].astype(str) for arr_ele in arr]
    #     seqs = [arr_ele[1].astype(str) for arr_ele in arr]

    #     final_seqs = []
    #     for indv_seq in seqs:
    #         orig_arrseq = np.array([i for i in indv_seq])
    #         arrseq = np.zeros(orig_arrseq.shape, dtype=np.int8)
    #         arrseq[orig_arrseq == "A"] = 0
    #         arrseq[orig_arrseq == "C"] = 1
    #         arrseq[orig_arrseq == "G"] = 2
    #         arrseq[orig_arrseq == "T"] = 3
    #         final_seqs.append(arrseq)

    #     return(dict(zip(names, final_seqs)))




