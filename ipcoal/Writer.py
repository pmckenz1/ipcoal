#!/usr/bin/env python


import os
import numpy as np
import pandas as pd
from .utils import ipcoalError


class Writer:

    def __init__(self, seqs, names):
        """
        Writer class object to write ipcoal seqs in a variety of formats.

        Parameters
        ----------
        seqs (ndarray)
            A .seqs array from ipcoal of dimensions (nloci, ntaxa, nsites). 
            The data for the ntaxa is ordered by their names alphanumerically.
        names (list)
            A list of the taxon names ordered alphanumerically.
        """
        # both are already ordered alphanumerically
        self.seqs = seqs
        self.names = names
        self.outdir = None
        self.outfile = None


    def write_loci_to_phylip(self, outdir, idxs=None, name_prefix=None, name_suffix=None):
        """
        Write all seq data for each locus to a separate phylip file in a shared
        directory with each locus named by locus index. If you want to write
        only a subset of loci to file you can list their index

        Parameters:
        -----------
        outdir (str):
            A directory in which to write all the phylip files. It will be 
            created if it does not yet exist. Default is "./ipcoal_loci/".
        idx (list):
            Numeric indices of the rows (loci) to be written to file. 
            Default=None meaning that all loci will be written to file. 
        """
        # make outdir if it does not yet exist
        self.outdir = os.path.realpath(os.path.expanduser(outdir))
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # set names parts to empty string if None
        name_suffix = ("" if name_suffix is None else name_suffix)
        name_prefix = ("" if name_prefix is None else name_prefix)

        # get loci to write
        if idxs is None:
            lrange = range(self.seqs.shape[0])
        else:
            # if int make it iterable
            if isinstance(idxs, int):
                lrange = [idxs]
            else:
                lrange = list(idxs)

            # check that idxs exist
            for loc in lrange:
                if loc not in range(self.seqs.shape[0]):
                    raise ipcoalError("idx {} is not in the data set")

        # iterate over loci (or single selected locus)
        for loc in lrange:

            # get locus and convert to bases
            arr = self.seqs[loc].astype(bytes)
            arr = convert_intarr_to_bytearr(arr)

            # open file handle numbered unless user
            fhandle = os.path.join(
                self.outdir, 
                "{}{}{}.phy".format(name_prefix, loc, name_suffix),
            )

            # build list of line strings
            phystring = self.build_phystring_from_loc(arr)

            # write to file
            with open(fhandle, 'w') as out:
                out.write(phystring)

        self.written = len(lrange)

    def write_loci_to_vcf(self, filename=None, outdir=".", idxs=None, reference=None):
        return_dataframe = False
        if not filename:
            return_dataframe = True
        if not return_dataframe:
            # make outdir if it does not yet exist
            self.outdir = os.path.realpath(os.path.expanduser(outdir))
            if not os.path.exists(self.outdir):
                os.makedirs(self.outdir)


        # get loci to write
        if idxs is None:
            lrange = range(self.seqs.shape[0])
        else:
            # if int make it iterable
            if isinstance(idxs, int):
                lrange = [idxs]
            else:
                lrange = list(idxs)

            # check that idxs exist
            for loc in lrange:
                if loc not in range(self.seqs.shape[0]):
                    raise ipcoalError("idx {} is not in the data set")

        vcfstr = "{}{}{}".format(VCFHEADER, '\t'.join(self.names), '\n')

        # object to hold number of SNPs.
        nsnps = 0

        if return_dataframe:
            CHROM_full = []
            POS_full = []
            ID_full = []
            REF_full = []
            ALT_full = []
            QUAL_full = []
            FILTER_full = []
            INFO_full = []
            SAMPLES_full = []

        # iterate over loci (or single selected locus)
        for loc in lrange:

            # get locus and convert to bases
            arr = self.seqs[loc].astype(bytes)
            arr = convert_intarr_to_bytearr(arr)

            # get snp locations
            snp_locs = np.where(np.array([len(np.unique(i)) for i in arr.T]) > 1)[0]

            # get (nsamps x nsnps) array of snps
            snps = arr[:, snp_locs]
            nsnps += snps.shape[1]

            # chrom column (ZERO INDEXED)
            CHROM = np.repeat(str(loc), snps.shape[1])

            # pos column (ONE INDEXED)
            POS = (snp_locs+1).astype(str)

            # ID
            ID = np.repeat('.', snps.shape[1])
            # REF
            if reference:
                REF = snps[np.argmax(np.array(self.names) == reference)].astype(str)
            else:
                REF = snps[0].astype(str)

            # ALT
            ALT = []
            allele_dict = []
            for i in range(len(snps.T)):
                curr_snp = snps.T[i].astype(str)
                alls = set(curr_snp)
                alls.remove(REF[i].astype(str))
                ALT.append(','.join(list(alls)))
                # add to dict mapping letters to allele numbers for each SNP
                tmp_allele_dict = {REF[i].astype(str): "0"}
                for idx in range(len(list(alls))):
                    tmp_allele_dict[list(alls)[idx]] = str(idx+1)
                allele_dict.append(tmp_allele_dict)
            ALT = np.array(ALT)

            # QUAL
            QUAL = np.repeat('.', snps.shape[1])

            # FILTER
            FILTER = np.repeat('PASS', snps.shape[1])

            # INFO
            INFO = np.repeat('.', snps.shape[1])

            # FORMAT
            #FORMAT = np.repeat('GT', snps.shape[1])

            # SAMPLES
            SAMPLES = np.zeros((snps.shape), dtype=str)
            for sample_rows in range(SAMPLES.shape[0]):
                for base_columns in range(SAMPLES.shape[1]):
                    SAMPLES[sample_rows, base_columns] = allele_dict[base_columns][snps[sample_rows, base_columns].astype(str)]
            SAMPLES = SAMPLES.T

            loclines = ''
            for i in range(snps.shape[1]):
                SAMP = "\t".join(SAMPLES[i])
                loclines = loclines + "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(CHROM[i],
                                                            POS[i],
                                                            ID[i],
                                                            REF[i],
                                                            ALT[i],
                                                            QUAL[i],
                                                            FILTER[i],
                                                            INFO[i],
                                                            #FORMAT[i],
                                                            SAMP
                                                            )
            vcfstr = vcfstr + loclines

            if return_dataframe:
                CHROM_full.append(CHROM)
                POS_full.append(POS)
                ID_full.append(ID)
                REF_full.append(REF)
                ALT_full.append(ALT)
                QUAL_full.append(QUAL)
                FILTER_full.append(FILTER)
                INFO_full.append(INFO)
                SAMPLES_full.append(SAMPLES)

        self.written = len(lrange)
        self.nsnps = nsnps
        self.vcf = vcfstr

        CHROM_full = np.concatenate(CHROM_full)
        POS_full = np.concatenate(POS_full)
        ID_full = np.concatenate(ID_full)
        REF_full = np.concatenate(REF_full)
        ALT_full = np.concatenate(ALT_full)
        QUAL_full = np.concatenate(QUAL_full)
        FILTER_full = np.concatenate(FILTER_full)
        INFO_full = np.concatenate(INFO_full)
        SAMPLES_full = np.concatenate(SAMPLES_full)

        df = pd.DataFrame([CHROM_full,
                           POS_full,
                           ID_full,
                           REF_full,
                           ALT_full,
                           QUAL_full,
                           FILTER_full,
                           INFO_full],
                          index=["#CHROM",
                                   "POS",
                                   "ID",
                                   "REF",
                                   "ALT",
                                   "QUAL",
                                   "FILTER",
                                   "INFO"])
        samples_df = pd.DataFrame(SAMPLES_full,
                                  columns=self.names).T

        df = df.append(samples_df).T

        if not return_dataframe:
            # open file handle numbered unless user
            fhandle = os.path.join(
                self.outdir, 
                "{}.vcf".format(filename),
            )

            # write to file
            with open(fhandle, 'w') as out:
                out.write(vcfstr)
        else:
            self.df = df

    #def write_concat_to_vcf(self)


    def write_concat_to_phylip(self, outdir, name, idxs=None):
        """
        Write all seq data (loci or snps) concated to a single phylip file.

        Parameters:
        -----------
        outfile (str):
            The name/path of the outfile to write. 
        idxs (list):
            A list of locus indices to subselect which will be concatenated.
        """
        # create a directory if it doesn't exist
        outdir = os.path.realpath(os.path.expanduser(outdir))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outfile = os.path.join(outdir, name)

        # concat loci (unless SNPs already) and convert to bases
        if self.seqs.ndim == 2:
            if idxs is not None:
                arr = self.seqs[:, idxs].astype(bytes)
            else:
                arr = self.seqs.astype(bytes)
        else:
            if idxs is not None:
                arr = np.concatenate(self.seqs[idxs], axis=1).astype(bytes)
            else:
                arr = np.concatenate(self.seqs, axis=1).astype(bytes)
        arr = convert_intarr_to_bytearr(arr)

        # build list of line strings
        phystring = self.build_phystring_from_loc(arr)

        # write to file
        with open(outfile, 'w') as out:
            out.write(phystring)
        self.outfile = outfile
        self.shape = arr.shape


    def write_concat_to_nexus(self, outdir, name, idxs=None):
        """
        Write all seq data (loci or snps) concated to a single phylip file.

        Parameters:
        -----------
        outfile (str):
            The name/path of the outfile to write. 
        idxs (list):
            A list of locus indices to subselect which will be concatenated.
        """
        # create a directory if it doesn't exist

        outdir = os.path.realpath(os.path.expanduser(outdir))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outfile = os.path.join(outdir, name)

        # concat loci (unless SNPs already) and convert to bases
        if self.seqs.ndim == 2:
            if idxs is not None:
                arr = self.seqs[:, idxs].astype(bytes)
            else:
                arr = self.seqs.astype(bytes)
        else:
            if idxs is not None:
                arr = np.concatenate(self.seqs[idxs], axis=1).astype(bytes)
            else:
                arr = np.concatenate(self.seqs, axis=1).astype(bytes)
        arr = convert_intarr_to_bytearr(arr)

        nexstring = self.build_nexstring_from_loc(arr)
        with open(outfile, 'w') as out:
            out.write(nexstring)

        self.outfile = outfile
        self.shape = arr.shape


    def build_phystring_from_loc(self, arr):
        """
        Builds phylip format string with 10-spaced names.

        3 200
        aaaaaaaaaa ATCTCTACAT...
        bbbbbbb    ATCTCTACAT...
        cccccccc   ATCACTACAT...
        """
        loclist = ["{} {}".format(arr.shape[0], arr.shape[1])]
        for row in range(arr.shape[0]):
            line = "{:<10} {}".format(
                self.names[row], b"".join(arr[row]).decode())
            loclist.append(line)
        return "\n".join(loclist)


    def build_nexstring_from_loc(self, arr):
        """
        Builds nexus format string
        """
        # write the header
        lines = []
        lines.append(
            NEXHEADER.format(arr.shape[0], arr.shape[1])
        )

        # grab a big block of data
        for block in range(0, arr.shape[1], 100):           
            # store interleaved seqs 100 chars with longname+2 before
            stop = min(block + 100, arr.shape[1])
            for idx, name in enumerate(self.names):  

                # py2/3 compat --> b"TGCGGG..."
                seqdat = arr[idx, block:stop]
                lines.append(
                    "  {}\t{}\n".format(
                        name,
                        #"".join(seqdat)
                        "".join([base_.decode() for base_ in seqdat])
                        )
                    )
            lines.append("\n")
        lines.append("\t;\nend;")
        return("".join(lines))


    # def write_seqs_as_fasta(self, loc, path):
    #     fastaseq = deepcopy(self.seqs[loc]).astype(str)

    #     fastaseq[fastaseq == '0'] = "A"
    #     fastaseq[fastaseq == '1'] = "C"
    #     fastaseq[fastaseq == '2'] = "G"
    #     fastaseq[fastaseq == '3'] = "T"

    #     fasta = []
    #     for idx, name in enumerate(self.names):
    #         fasta.append(('>' + name + '\n'))
    #         fasta.append("".join(fastaseq[idx])+'\n')

    #     with open(path, 'w') as file:
    #         for line in fasta:
    #             file.write(line)

    def _deprecated(self):

        fastapath = "tempfile" + str(np.random.randint(0, 99999)) + ".fasta"
        self.write_fasta(seqnum, fastapath)

        self._call_iq(['iqtree',
                       '-s', fastapath,
                       '-m', 'MFP',
                       '-bb', '1000'])
        if os.path.isfile(fastapath+".treefile"):
            with open(fastapath+".treefile", 'r') as treefile:
                newick = treefile.read()
            self.df.loc[(self.df['locus_idx'] == seqnum),
                        'inferred_trees'] = newick
            self.newicklist.append(newick)
        for filename in glob.glob(fastapath+"*"):
            os.remove(filename)
       # storage for output
        # self.nquarts = int(comb(N=self.ntips, k=4))  # scipy.special.comb

        # temporarily format these as stacked matrices


def convert_intarr_to_bytearr(arr):
    arr[arr == b"0"] = b"A"
    arr[arr == b"1"] = b"C"
    arr[arr == b"2"] = b"G"
    arr[arr == b"3"] = b"T"
    return arr


NEXHEADER = """#nexus
begin data;
  dimensions ntax={} nchar={};
  format datatype=DNA missing=N gap=- interleave=yes;
  matrix\n
"""

VCFHEADER = """
##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT QUAL\tFILTER\tINFO\tFORMAT\t"""