#!/usr/bin/env python

"""
Class for converting int seq arrays to string arrays, and optionally
for combining two haploid sequences into a diploid sequence with
IUPAC ambiguity codes for heterozygous sites.
"""

from itertools import groupby
import numpy as np
from ipcoal.utils.utils import convert_intarr_to_bytearr, IpcoalError
from ipcoal.utils.utils import convert_intarr_to_bytearr_diploid


class Transformer:
    """Converts seqs from ints to strings including diploid base calls
    for substitution models that support ambiguities.

    seqs: ndarray
        The .seqs array of simulated alleles from a Model object.
    names: ndarray
        The .alpha_ordered_names array from a Model object.
    diploid: bool
        Randomly join haploid samples to write diploid genotypes.
    """
    def __init__(
        self,
        seqs,
        names,
        alleles,
        diploid=True,
        ):

        # store input params
        self.seqs = seqs
        self.orig_names = names.copy()
        self.names = names.copy()
        self.alleles = alleles
        self.diploid = diploid
        self.diploid_map = {}
        self.dindex_map = {}

        # setup functions
        self.get_diploid_map()
        # self.transform_seqs()

    def get_diploid_map(self):
        """Combine haploid names and indices to map pairs as new diploids.

        diploid_map = {'A': ['A-0-1', 'A-2-3], 'B': ['B-0-1', 'B-2-3'], ...]}
        dindex_map = {1: [(0,1), (2,3)], 2: [(4,5), (6,7)], ...}
        """
        # haploid indices simply repeat itself twice.
        if not self.diploid:
            pidx = 0
            for idx, name in enumerate(self.orig_names):
                key = name.rsplit("_", 1)[0]
                self.diploid_map[key] = (name, name)
                self.dindex_map[idx] = (pidx, pidx)
                pidx += 1

        # diploid indices increase in pairs: (0,1), (2,3), (4,6)...
        else:
            # group names by prefix
            groups = groupby(self.orig_names, key=lambda x: x.rsplit("_", 1)[0])

            # arrange into an IMAP dictionary: {r0: [r0-0, r0-1, r0-2, ...]}
            imap = {i[0]: list(i[1]) for i in groups}

            # if all nsamples are 2 then we will not add name suffix
            suffix = True
            if all(len(imap[i]) == 2 for i in imap):
                suffix = False

            # iterate over tips and increment diploids and haploid pair idxs
            didx = 0
            for sppname in imap:

                # nsamples matched to this tip
                samples = imap[sppname]
                samples = sorted(samples, key=lambda x: int(x.rsplit("_", 1)[-1]))

                # iterate 0, 2, 4
                for pidx in range(0, len(samples), 2):

                    # the local idx of these sample names
                    ps0 = samples[pidx]
                    ps1 = samples[pidx + 1]

                    # the global idx of these sample names
                    pidx0 = self.orig_names.index(ps0)
                    pidx1 = self.orig_names.index(ps1)

                    # fill dicts
                    if suffix:
                        newname = "{}_{}".format(sppname, int(pidx / 2))
                    else:
                        newname = sppname
                    self.diploid_map[newname] = (ps0, ps1)
                    self.dindex_map[didx] = (pidx0, pidx1)
                    didx += 1

            # store new diploid-collapsed names
            self.names = sorted(self.diploid_map)

    def transform_seqs(self):
        """
        Transforms seqs from ints to strings. If using diploid map this also
        combines the two alleles to represent ambiguity codes for hetero sites,
        which changes the dimension of both .seqs and .names.
        """
        # simply convert to bytes
        if not self.diploid:
            self.seqs = convert_intarr_to_bytearr(self.seqs)

        # combine alleles to get heterozygotes
        else:
            # raise an error if allele-type cannot be converted to
            # ambiguities, as with, for example 'pam' or 'binary' models.
            if set(self.alleles.values()) != set("ACGT"):
                raise IpcoalError(
                    "Alleles from this substition model cannot be "
                    "converted to ambiguities to represent diploid "
                    "genotypes: {}".format(tuple(self.alleles.values()))
                )

            # temporary store diploid copies
            dnames = sorted(self.diploid_map)
            nloci = self.seqs.shape[0]
            nsamples = int(self.seqs.shape[1] / 2)
            nsites = self.seqs.shape[2]
            dseqs = np.zeros((nloci, nsamples, nsites), dtype=bytes)

            # fill diploid seqs
            for key, val in self.diploid_map.items():
                didx = dnames.index(key)
                arr0 = self.seqs[:, self.orig_names.index(val[0])]
                arr1 = self.seqs[:, self.orig_names.index(val[1])]
                carr = np.char.array(arr0) + np.char.array(arr1)
                carr = convert_intarr_to_bytearr_diploid(carr)
                dseqs[:, didx] = carr

            # store diploid copy over the original
            self.seqs = dseqs
