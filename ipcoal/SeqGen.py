#!/usr/bin/env python

import os
import re
import sys
import numpy as np
import subprocess as sps
from .utils import ipcoalError
import toytree


class SeqGen:
    """
    Opens a view to seq-gen in a subprocess so that many gene trees can be 
    cycled through without the overhead of opening/closing subprocesses.
    """

    def __init__(self, **kwargs):

        # set binary path for conda env and check for binary
        self.kwargs = kwargs
        self.binary = os.path.join(sys.prefix, "bin", "seq-gen")
        assert os.path.exists(self.binary), (
            "binary {} not found".format(self.binary))

        # call open_subprocess to set the shell 
        self.shell = None

        # store substitution model kwargs
        self.kwargs = {
            "kappa": 1.0,
            "state_frequencies": (0.25, 0.25, 0.25, 0.25),
        }
        self.kwargs.update(kwargs)

        # set tstv based on kappa
        self.state_frequencies = self.kwargs["state_frequencies"]
        freqR = self.state_frequencies[0] + self.state_frequencies[2]
        freqY = self.state_frequencies[1] + self.state_frequencies[3]
        self.kwargs["tstv"] = (
            self.kwargs["kappa"] * sum([
                self.state_frequencies[0] * self.state_frequencies[2],
                self.state_frequencies[1] * self.state_frequencies[3]
            ])) / (freqR * freqY)



    def open_subprocess(self):
        """
        Open a persistent Popen bash shell on a new thread.
        """
        # open shell arg with line buffering
        self.shell = sps.Popen(
            ["bash"], stdin=sps.PIPE, stdout=sps.PIPE, bufsize=1)


    def close(self):
        """
        Cleanup and shutdown the subprocess shell.
        """
        self.shell.stdin.close()
        self.shell.terminate()
        self.shell.wait(timeout=1.0)


    def feed_tree(self, newick, nsites, mut, seed):
        """
        Feed a command string a read results until empty line.
        TODO: allow kwargs to add additional seq-gen args.
        """
        # command string
        cmd = (
            "{} -mHKY -l {} -s {} -z {} -t {} -f {} {} {} {} -q <<< \"{}\"; echo done\n"
            .format(
                self.binary,
                nsites,
                mut, 
                seed,
                self.kwargs["tstv"], 
                self.kwargs["state_frequencies"][0],
                self.kwargs["state_frequencies"][1],
                self.kwargs["state_frequencies"][2],
                self.kwargs["state_frequencies"][3],
                newick)
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

        # store names and seqs to a dict (names are 1-indexed msprime tips)
        seqd = {}
        for line in hold.split("\n")[1:-1]:
            name, seq = line.split()
            seqd[int(name)] = list(seq)
        
        # convert seqs to int array 
        rt_genealogy = toytree._rawtree(newick)
        arr = np.array([seqd[int(i)] for i in rt_genealogy.treenode.get_leaf_names()[::-1]])
        arr[arr == "A"] = 0
        arr[arr == "C"] = 1
        arr[arr == "G"] = 2
        arr[arr == "T"] = 3
        arr = arr.astype(np.uint8)

        # reorder rows to return 1-indexed numeric tip name order
        return arr
