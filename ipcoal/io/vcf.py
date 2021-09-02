#!/usr/bin/env python

"""
Write SNPs to VCF format.
"""

import datetime
import numpy as np
import pandas as pd
from ipcoal.io.transformer import Transformer
from ipcoal.io.genos import Genos
from ipcoal.utils.utils import convert_intarr_to_bytearr
import ipcoal


# TODO add an attribution of the ipcoal version and list sim parameters.
VCFHEADER = """\
##fileformat=VCFv4.2
##fileDate={date}
##source=ipcoal-v.{version}
##reference={reference}
{contig_lines}
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t\
"""


class VCF:
    """
    Write SNPs in VCF format. Note: we use the true ancestral sequence to
    represent the reference such that matching the reference of not 
    is the same as ancestral vs. derived. However, we only include sites
    that are variable among the samples, i.e., a SNP is not defined if 
    all samples are derived relative to the reference (ancestral seq).

    Parameters
    ==========
    seqs: ndarray
        int array (nloci, nsamples, nsites).
    names: ndarray
        haploid sample names.
    diploid: bool
        Make diploid genos.
    ancestral: ndarray
        The ancestral seq (nloci, nsites)
    fill_missing_alleles: bool
        Write diploids with missing alleles (0|.) as (0|0).
    """
    def __init__(self, seqs, names, diploid, ancestral, fill_missing_alleles):
        self.names = names
        self.seqs = seqs
        self.aseqs_ints = ancestral
        self.aseqs_bytes = convert_intarr_to_bytearr(self.aseqs_ints)
        self.fill_missing_alleles = fill_missing_alleles

        # do not combine for ambiguity codes, but get diploid_map and names.
        txf = Transformer(self.seqs, self.names, diploid)
        self.dindex_map = txf.dindex_map
        self.dnames = txf.names


    def get_header(self):
        """
        Called AFTER the .df vcf is built.
        """
        # build the header
        contig_lines = []
        for loc in range(self.seqs.shape[0]):
            arr = self.seqs[loc]
            if np.any(arr != arr[0], axis=0).sum():
                contig_lines.append(
                    "##contig=<ID={},length={}>".format(loc, arr.shape[1])) 

        header = VCFHEADER.format(**{
            "date": datetime.datetime.now(),
            "version": ipcoal.__version__, 
            "reference": "true_simulated_ancestral_sequence",
            "contig_lines": "\n".join(contig_lines)
        })
        header = "{}{}\n".format(header, "\t".join(self.dnames))
        return header



    def vcf_chunk_generator(self):
        """
        Build a DF of genotypes and metadata.
        """
        # iterate over loci building vcf dataframes
        for lidx in range(self.seqs.shape[0]):

            # get array of sequence data
            arr = self.seqs[lidx]

            # get indices of variable sites while allowing for missing data       
            marr = np.ma.array(data=arr, mask=(arr == 9))
            common = marr.mean(axis=0).round().astype(int)
            varsites = np.where(np.any(marr != common, axis=0).data)[0]
            nsites = varsites.size

            # vcf dataframe
            vdf = pd.DataFrame({
                "CHROM": np.repeat(0, nsites),
                "POS": np.repeat(1, nsites),
                "ID": np.repeat(".", nsites),
                "REF": "N", 
                "ALT": "A,C,G",
                "QUAL": 99,
                "FILTER": "PASS",
                "INFO": ".",
                "FORMAT": "GT",
            })

            # fill in the reference allele using the known ancestral seq
            vdf.loc[:, "REF"] = self.aseqs_bytes[lidx][varsites].astype(str)

            # get genos and alts            
            genos = Genos(
                arr, 
                self.aseqs_ints[lidx],
                varsites, 
                self.dindex_map,
                self.fill_missing_alleles,
            )
            alts, garrdf = genos.get_alts_and_genos_string_matrix()
            garrdf.columns = self.dnames

            # fill vcf chunk dataframe
            vdf.loc[:, "CHROM"] = lidx + 1
            vdf.loc[:, "POS"] = varsites + 1
            vdf.loc[:, "ALT"] = alts.astype(str)
            vdf = pd.concat([vdf, garrdf], axis=1)

            # yield result to act as a generator of chunks
            yield vdf
