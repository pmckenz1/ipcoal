#!/usr/bin/env python

"""
A class for encoding genotype calls for writing 
data to VCF or HDF5 output formats.
"""

import numpy as np
import pandas as pd
from ipcoal.utils import convert_intarr_to_bytearr



class Genos:
    """
    Get genotype calls by comparing the simulated sequence to the 
    ancestral sequence. These geno calls are used in VCF and HDF5 
    output files, and often filter out non-biallelic variants.

    Parameters
    ==========
    concatseqs (array shape=(nsamples, nsites), dtype=uint8)
    ancestral (array shape=(1, nsites), dtype=uint8)
    snpidxs (array shape=(nsites,), dtype=np.int)
    dindex_map (dict): map of diploid index to haploid indices.
    fill_missing_alleles: write diploids with missing alleles (0|.) as (0|0).    
    """
    def __init__(self, seqs, ancestral, snpidxs, dindex_map, fill_missing_alleles=False):
        self.seqs = seqs
        self.aseq = ancestral
        self.snpidxs = snpidxs
        self.dindex_map = dindex_map
        self.fill_missing_alleles = fill_missing_alleles



    def get_alts_and_genos_matrix(self):
        """
        Returns genos matrix a bit slower b/c accomodates missing values (9),
        snpidxs has already been computed on a masked array.
        """
        # subsample to varible sites if SNPs only
        if self.snpidxs is not None:
            self.aseq = self.aseq[self.snpidxs]
            self.seqs = self.seqs[:, self.snpidxs]

        # the geno matrix to fill starts out all 9s
        gmat = np.zeros(self.seqs.shape, dtype=np.uint8)
        gmat.fill(9)

        # fill 0 -------------------------------------------
        # mask all bases that are missing
        marr = np.ma.array(self.seqs, mask=(self.seqs == 9))

        # set to 0 all non-masked sites that match the ref
        gmat[marr == self.aseq] = 0

        # --------
        # if we assume non-biallele sites will be filtered then it does
        # not matter which non-ref allele is set to 1,2 or 3.
        # --------

        # fill 1 -------------------------------------------
        marr = np.ma.array(
            data=self.seqs,
            mask=(gmat == 0) | (self.seqs == 9)
        )
        colmask = np.all(marr.mask, axis=0)
        maxa = marr.max(axis=0)
        maxa[colmask] = 9
        gmat[marr == maxa] = 1
        alts1 = np.zeros(self.seqs.shape[1], dtype="S1")
        alts1[~colmask] = convert_intarr_to_bytearr(maxa[~colmask])


        # fill 2 --------------------------------------------
        marr = np.ma.array(
            data=self.seqs,
            mask=(gmat == 0) | (gmat == 1) | (self.seqs == 9)
        )
        colmask = np.all(marr.mask, axis=0)
        maxa = marr.max(axis=0)
        maxa[colmask] = 9
        gmat[marr == maxa] = 2
        alts2 = np.zeros(self.seqs.shape[1], dtype="S1")
        alts2[~colmask] = convert_intarr_to_bytearr(maxa[~colmask])


        # fill 3 --------------------------------------------
        marr = np.ma.array(
            data=self.seqs,
            mask=(gmat == 0) | (gmat == 1) | (gmat == 2) | (self.seqs == 9)
        )
        colmask = np.all(marr.mask, axis=0)
        maxa = marr.max(axis=0)
        maxa[colmask] = 9
        gmat[marr == maxa] = 3
        alts3 = np.zeros(self.seqs.shape[1], dtype="S1")
        alts3[~colmask] = convert_intarr_to_bytearr(maxa[~colmask])

        # combine alts
        alts = (
            np.char.array(alts1) + b"," + \
            np.char.array(alts2) + b"," + \
            np.char.array(alts3)
        ).strip(b",")

        # re-shape gmat into (n, m, 2) array
        garr = np.zeros((self.seqs.shape[1], len(self.dindex_map), 2), dtype=np.uint8)
        for idx in self.dindex_map:

            # get the haplotypes indices
            left, right = self.dindex_map[idx]

            # get data arranged to right shape
            garr[:, idx, :] = gmat[(left, right), :].T

            # copy other allele over 9 if it is not 9
            if self.fill_missing_alleles:

                garr[garr[:, :, 0] == 9, 0] = garr[garr[:, :, 0] == 9, 1]
                garr[garr[:, :, 1] == 9, 1] = garr[garr[:, :, 1] == 9, 0]

        return alts, garr



    def get_alts_and_genos_string_matrix(self):
        """
        Get string representation of genotype calls with missing data.
        """
        alts, garr = self.get_alts_and_genos_matrix()
        garrdf = (
            pd.DataFrame(garr[:, :, 0]).astype(str).replace("9", ".") + "|" + \
            pd.DataFrame(garr[:, :, 1]).astype(str).replace("9", ".")
        )
        return alts, garrdf
