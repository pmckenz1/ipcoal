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
        Open a persistent Popen bash shell on a new thread.
        """
        # open shell arg with line buffering
        self.shell = sps.Popen(
            ["bash"], stdin=sps.PIPE, stdout=sps.PIPE, bufsize=1)


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
            "{} -mGTR -l {} -s {} -z {} -q <<< \"{}\"; echo done\n"
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

        # store names and seqs to a dict (names are 1-indexed msprime tips)
        seqd = {}
        for line in hold.split("\n")[1:-1]:
            name, seq = line.split()
            seqd[int(name)] = list(seq)

        # convert seqs to int array 
        arr = np.array([seqd[i] for i in range(1, len(seqd) + 1)])
        arr[arr == "A"] = 0
        arr[arr == "C"] = 1
        arr[arr == "G"] = 2
        arr[arr == "T"] = 3
        arr = arr.astype(np.uint8)

        # reorder rows to return 1-indexed numeric tip name order
        return arr
