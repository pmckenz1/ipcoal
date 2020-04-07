#!/usr/bin/env python


import os
import datetime
import numpy as np
import pandas as pd
import ipcoal
from .utils import ipcoalError




class Writer:
    def __init__(self, seqs, names, ancestral_seq=None):
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
        self.seqs = seqs.copy()
        self.names = names.copy()
        self.outdir = None
        self.outfile = None
        self.idxs = None
        self.ancestral_seq = ancestral_seq


    def _subset_loci(self, idxs):
        """
        If datasets is dim=3 then subset loci by idxs argument.
        """
        # subselect the linkage groups to write (default is to select all)
        if idxs is not None:
            # ensure it is iterable
            if isinstance(idxs, int):
                self.idxs = [self.idxs]
            else:
                self.idxs = idxs

            # check that idxs exist
            for loc in np.array(self.idxs):
                if loc not in range(self.seqs.shape[0]):
                    raise ipcoalError(
                        "idx {} is not in the data set".format(loc))

            # subset self.seqs to selected loci
            idxs = np.array(sorted(self.idxs))
            self.seqs = self.seqs[idxs]
        else:
            self.idxs = range(self.seqs.shape[0])


    def _transform_seqs(self, diploid, diploid_map, seed):
        """
        Transform seqs from int type to str type. Also optionally combine 
        haploid samples into diploids and use IUPAC ambiguity codes to 
        represent hetero sites.
        """
        txf = Transformer(self.seqs, self.names, diploid, diploid_map, seed)
        self.seqs = txf.seqs
        self.names = txf.names


    def write_loci_to_phylip(
        self, 
        outdir, 
        idxs=None, 
        name_prefix=None, 
        name_suffix=None, 
        diploid=False, 
        diploid_map=None, 
        seed=None,
        quiet=False):
        """
        Write all seq data for each locus to a separate phylip file in a shared
        directory with each locus named by locus index. If you want to write
        only a subset of loci to file you can list their index

        Parameters:
        -----------
        outdir (str):
            A directory in which to write all the phylip files. It will be 
            created if it does not yet exist. Default is "./ipcoal_loci/".
        idxs (list):
            Numeric indices of the rows (loci) to be written to file. 
            Default=None meaning that all loci will be written to file. 
        ...
        """
        # bail out and complain if sim_snps() or sim_trees().
        if self.seqs is None:
            raise ipcoalError("cannot write 'loci' for sim_trees() result.")

        if self.seqs.ndim == 2:
            raise ipcoalError("cannot write 'loci' for sim_snps() result.")

        # make outdir if it does not yet exist
        self.outdir = os.path.realpath(os.path.expanduser(outdir))
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # set names parts to empty string if None
        name_suffix = ("" if name_suffix is None else name_suffix)
        name_prefix = ("" if name_prefix is None else name_prefix)

        # self.seqs is reduced to the loci in idxs, and self.idxs is set.
        self._subset_loci(idxs)

        # self.seqs and self.names are reduced to diploid samples
        self._transform_seqs(diploid, diploid_map, seed)

        # iterate over loci writing each one individually
        for loc in self.idxs:

            # get locus as bytes
            arr = self.seqs[loc]

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

        # report
        if not quiet:
            print("wrote {} loci ({} x {}bp) to {}/[...].phy".format(
                len(self.idxs), self.seqs.shape[1], self.seqs.shape[2],
                self.outdir.rstrip("/")
                ),
            )


    def write_concat_to_phylip(
        self, 
        outdir="./", 
        name=None,
        idxs=None, 
        diploid=False, 
        diploid_map=None, 
        seed=None, 
        quiet=False):
        """
        Write all seq data (loci or snps) concated to a single phylip file.

        Parameters:
        -----------
        outfile (str):
            The name/path of the outfile to write. 
        idxs (list):
            A list of locus indices to subselect which will be concatenated.
        """
        # reshape SNPs array to be like loci 
        if self.seqs.ndim == 2:
            self.seqs = self.seqs.T.reshape(
                self.seqs.shape[1], self.seqs.shape[0], 1)

        # subset selected loci
        self._subset_loci(idxs)

        # transform data to string type and ploidy
        self._transform_seqs(diploid, diploid_map, seed)

        # concatenate sequences
        arr = np.concatenate(self.seqs, axis=1)

        # build list of line strings
        phystring = self.build_phystring_from_loc(arr)

        # return result as a string
        if not name:
            return phystring

        else:
            # create a directory if it doesn't exist
            outdir = os.path.realpath(os.path.expanduser(outdir))
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            outfile = os.path.join(outdir, name.rstrip(".phy") + ".phy")

            # write to file
            with open(outfile, 'w') as out:
                out.write(phystring)

            # report 
            if not quiet:
                print("wrote concat locus ({} x {}bp) to {}"
                      .format(arr.shape[0], arr.shape[1], outfile))


    def write_concat_to_nexus(
        self, 
        outdir="./", 
        name=None,
        idxs=None, 
        diploid=False, 
        diploid_map=None, 
        seed=None, 
        quiet=False):
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
        # reshape SNPs array to be like loci 
        if self.seqs.ndim == 2:
            self.seqs = self.seqs.T.reshape(
                self.seqs.shape[1], self.seqs.shape[0], 1)

        # subset selected loci
        self._subset_loci(idxs)

        # transform data to string type and ploidy
        self._transform_seqs(diploid, diploid_map, seed)

        # concatenate sequences
        arr = np.concatenate(self.seqs, axis=1)

        # build list of line strings
        nexstring = self.build_nexstring_from_loc(arr)

        # return result as a string
        if not name:
            return nexstring

        else:
            # create a directory if it doesn't exist
            outdir = os.path.realpath(os.path.expanduser(outdir))
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            outfile = os.path.join(outdir, name.rstrip(".nex") + ".nex")

            # write to file
            with open(outfile, 'w') as out:
                out.write(nexstring)

            # report 
            if not quiet:
                print("wrote concat locus ({} x {}bp) to {}"
                      .format(arr.shape[0], arr.shape[1], outfile))


    def write_vcf(
        self, 
        name=None, 
        outdir=None, 
        diploid=None, 
        diploid_map=None,
        seed=None,
        bgzip=False,
        quiet=False,
        ):
        """
        ...
        """
        # reshape SNPs array to be like loci 
        if self.seqs.ndim == 2:
            self.seqs = self.seqs.T.reshape(
                self.seqs.shape[1], self.seqs.shape[0], 1)
            self.ancestral_seq = self.ancestral_seq.reshape(
                self.ancestral_seq.size, 1)

        # make into genotype calls relative to reference
        vcf = VCF(
            self.seqs, 
            self.names, 
            diploid, 
            diploid_map, 
            self.ancestral_seq,
            seed)
        vcfdf = vcf.build_vcf()

        # return dataframe if no filename
        if name is None:
            return vcfdf

        # create a directory if it doesn't exist
        else:
            # get filepath
            outdir = (outdir if outdir else "./")
            outdir = os.path.realpath(os.path.expanduser(outdir))
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            outfile = os.path.join(outdir, name.rstrip(".vcf") + ".vcf")

            # write to filepath
            with open(outfile, 'wt') as vout:
                vcf.build_header()
                vout.write(vcf.header)
                vout.write(vcfdf.to_csv(header=False, index=False, sep="\t"))

            # call bgzip from tabix so that bcftools can be used on VCF.
            # this will not work with normal gzip compression.
            if bgzip:
                import subprocess
                subprocess.call(["bgzip", outfile])
                outfile = outfile.rstrip(".gz") + ".gz"

            # report
            if not quiet:
                print(
                    "wrote {} SNPs across {} linkage blocks to {}"
                    .format(
                        vcfdf.shape[0],
                        vcfdf.CHROM.unique().shape[0], 
                        outfile)
                )


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
                        "".join([base_.decode() for base_ in seqdat])
                        )
                    )
            lines.append("\n")
        lines.append("\t;\nend;")
        return("".join(lines))


    # def write_seqs_as_fasta(self, loc, path):
    #     fastaseq = deepcopy(self.seqs[loc]).astype(str)
    #     fasta = []
    #     for idx, name in enumerate(self.names):
    #         fasta.append(('>' + name + '\n'))
    #         fasta.append("".join(fastaseq[idx])+'\n')

    #     with open(path, 'w') as file:
    #         for line in fasta:
    #             file.write(line)



class Transformer:
    def __init__(
        self,
        seqs,
        names,
        diploid=True,
        diploid_map=False, 
        seed=None):
        """
        writer: (class)
            A writer class object from ipcoal with .seqs and .names.
        diploid: (bool)
            Form diploids by randomly joining 2 samples from a population.
            If no diploid_map is provided then it is expected that individuals
            in a population are named [prefix]-[0-n] with a matching prefix.
        diploid_map: (dict) 
            If using diploid=True but sample names do not match the naming
            convention then you can use a dictionary to select haploid
            individuals that should be combined into diploids.
            Example: diploid_map={'1': [1A-0, 1A-1], '2': [1B-0, 1B-1], ... }
        idxs: (list, ndarray)
            A list of ndarray of the indices of a subset of loci to write.
            Default is to write all loci in .seqs.
        seed: (int) 
            seed random number generator.
        """
        # set random seed
        if seed:
            np.random.seed(seed)

        # store input params
        self.seqs = seqs
        self.names = names
        self.diploid = diploid
        self.diploid_map = diploid_map
        self.dindex_map = {}

        # setup functions
        self.get_diploid_map()
        self.transform_seqs()


    def get_diploid_map(self):
        "randomly sample two haploids to make diploids without replacement."

        if self.diploid:

            # try to auto-generate a diploid map
            if not self.diploid_map:

                # check if names can work for auto mapping
                name_pre = [i.rsplit("-", 1)[0] for i in self.names]
                name_suf = [i.rsplit("-", 1)[1] for i in self.names]
                pcount = [name_pre.count(i) % 2 == 0 for i in set(name_pre)]
                assert all(pcount), (
                    "to make diploids nsamples must all be multiples of 2")
                try:
                    [int(i) for i in name_suf]
                except TypeError as inst:
                    print("sample names are not formatted for diploid=True")
                    raise inst

                # get diploids inds lists: {ind1: [], ind2: [], ...}
                self.diploid_map = {}
                for name in name_pre:
                    for dipcount in range(int(name_pre.count(name) / 2)):
                        newname = "{}-{}".format(name, dipcount)
                        self.diploid_map[newname] = []

                # fill the map: {ind1: [sample, sample], ...}
                names = self.names.copy()
                keys = sorted(self.diploid_map.keys())
                for dname in keys:

                    # get name prefix
                    dpre, dpost = dname.rsplit("-", 1)

                    # find all matching name prefixes
                    npres = [i.rsplit("-", 1)[0] for i in names]
                    match = [i == dpre for i in npres]

                    # randomly sample one matching sample 
                    midxs = np.where(match)[0]
                    sidxs = np.random.choice(midxs, size=2, replace=False)
                    allele_0 = names[sidxs[0]]
                    allele_1 = names[sidxs[1]]

                    # remove selected from names
                    names.remove(allele_0)
                    names.remove(allele_1)

                    # store results using informative name about samples
                    self.diploid_map.pop(dname)
                    subs = sorted(
                        [i.rsplit("-", 1)[-1] for i in (allele_0, allele_1)])
                    newname = "{}-{}.{}".format(dpre, subs[0], subs[1])

                    # map names to dname
                    self.diploid_map[newname] = [allele_0, allele_1]
                    self.dindex_map[newname] = [
                        self.names.index(allele_0), self.names.index(allele_1)
                    ]

                # reassign dindex names by sorted now that all dnames exist
                keys = sorted(self.dindex_map.keys())
                for kidx, key in enumerate(keys):
                    self.dindex_map[kidx] = self.dindex_map[key]
                    self.dindex_map.pop(key)

                # check that population sizes are multiple of 2
            else:
                assert all([len(i) == 2 for i in self.diploid_map.values()]), (
                    "all populations must have an even number of samples.")
                assert set(len(self.diploid_map)) != len(self.diploid_map), (
                    "all keys in diploid map must be unique.")
                assert set(len(self.diploid_map.values())) != len(self.diploid_map.values()), (
                    "all values in diploid map must be unique.")


    def transform_seqs(self):
        """
        Transforms seqs from ints to strings. If using diploid map this also
        combines the two alleles to represent ambiguity codes for hetero sites,
        which changes the dimension of both .seqs and .names.
        """
        # simply convert to bytes
        if not self.diploid:
            self.seqs = convert_intarr_to_bytearr(self.seqs.astype(bytes))

        # combine alleles to get heterozygotes
        else:
            # temporary store diploid copies
            self.dnames = sorted(self.diploid_map)
            self.dseqs = np.zeros(
                (self.seqs.shape[0], int(self.seqs.shape[1] / 2), self.seqs.shape[2]),
                dtype=bytes,
            )

            # fill diploid seqs
            for key, val in self.diploid_map.items():
                didx = self.dnames.index(key)
                arr0 = self.seqs[:, self.names.index(val[0])]
                arr1 = self.seqs[:, self.names.index(val[1])]
                carr = np.char.array(arr0) + np.char.array(arr1)
                carr = convert_intarr_to_bytearr_diploid(carr)
                self.dseqs[:, didx] = carr

            # store diploid copy over the original
            self.seqs = self.dseqs
            self.names = self.dnames
            del self.dseqs
            del self.dnames



class VCF:
    def __init__(self, seqs, names, diploid, diploid_map, ancestral, seed):
        """
        Write SNPs in VCF format. Note: we use the true ancestral sequence to
        represent the reference such that matching the reference of not 
        is the same as ancestral vs. derived. However, we only include sites
        that are variable among the samples, i.e., a SNP is not defined if 
        all samples are derived relative to the reference (ancestral seq).
        """
        self.names = names
        self.seqs = seqs
        self.diploid = diploid
        self.diploid_map = diploid_map
        self.aseqs = convert_intarr_to_bytearr(ancestral.astype(bytes))

        # do not combine for ambiguity codes, but get diploid_map and names.
        txf = Transformer(self.seqs, self.names, diploid, diploid_map, seed)
        self.dindex_map = txf.dindex_map
        self.dnames = txf.names


    def build_header(self):
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

        self.header = VCFHEADER.format(**{
            "date": datetime.datetime.now(),
            "version": ipcoal.__version__, 
            "reference": "true_simulated_ancestral_sequence",
            "contig_lines": "\n".join(contig_lines)
        })
        self.header = "{}{}\n".format(self.header, "\t".join(self.dnames))


    def build_vcf(self):

        # get nrows (snps) in VCF
        arr = np.concatenate(self.seqs, axis=1)
        varsites = np.where(np.any(arr != arr[0], axis=0))[0]
        nsites = varsites.size

        # vcf dataframe
        df = pd.DataFrame({
            "CHROM": np.repeat(0, nsites),
            "POS": np.repeat(1, nsites),
            "ID": np.repeat(".", nsites),
            "REF": np.concatenate(self.aseqs)[varsites].astype(str),
            "ALT": "A,C,G",
            "QUAL": 99,
            "FILTER": "PASS",
            "INFO": ".",
            "FORMAT": "GT",
        })

        # haploid samples genotype dataframe
        samples = pd.DataFrame(
            np.zeros((nsites, len(self.names)), dtype=int),
            columns=self.names,
        )

        # count up from first row
        snpidx = 0

        # iterate over loci to fill dataframe
        for loc in range(self.seqs.shape[0]):

            # get locus and SNP indices
            arr = convert_intarr_to_bytearr(self.seqs[loc].astype(bytes))
            slocs = np.where(np.any(arr != arr[0], axis=0))[0]
            nsnps = len(slocs)

            # if any snps
            if nsnps:

                # fill the ALT field
                altstr = []
                for cidx, col in enumerate(slocs):

                    # get the alt alleles
                    site = arr[:, col]
                    alts = sorted(set(site) - set([self.aseqs[loc, col]]))
                    altstr.append(b",".join(alts).decode())                    

                    # fill sample genos with alt index
                    for adx, alt in enumerate(alts):
                        match = site == alt
                        samples.iloc[snpidx + cidx, match] = adx + 1

                # fill in dataframe (remember .loc includes last row index)
                df.loc[snpidx:snpidx + nsnps - 1, "CHROM"] = loc
                df.loc[snpidx:snpidx + nsnps - 1, "POS"] = slocs + 1
                df.loc[snpidx:snpidx + nsnps - 1, "ALT"] = altstr

                # advance counter
                snpidx += nsnps

        # convert to diploid genotypes
        if self.diploid:
            dsamples = pd.DataFrame(
                np.zeros((nsites, len(self.dnames)), dtype=str), 
                columns=self.dnames, 
            )
            for didx, sidxs in self.dindex_map.items():
                sidxs = sorted(sidxs)
                col0 = samples.iloc[:, sidxs[0]].astype(str)
                col1 = samples.iloc[:, sidxs[1]].astype(str)
                dsamples.iloc[:, didx] = col0 + "|" + col1
            return pd.concat([df, dsamples], axis=1)
        return pd.concat([df, samples], axis=1)



def convert_intarr_to_bytearr(arr):
    "An array of ints was turned into bytes and this converts to bytestrings"
    arr[arr == b"0"] = b"A"
    arr[arr == b"1"] = b"C"
    arr[arr == b"2"] = b"G"
    arr[arr == b"3"] = b"T"
    return arr


def convert_intarr_to_bytearr_diploid(arr):
    """
    Two arrays of ints were turned into bytes and joined (e.g., b'00') and
    this converts it to a single bytestring IUPAC code for diploids.
    """
    arr[arr == b"00"] = b"A"
    arr[arr == b"11"] = b"C"
    arr[arr == b"22"] = b"G"
    arr[arr == b"33"] = b"T"
    arr[arr == b"01"] = b"K"
    arr[arr == b"10"] = b"K"    
    arr[arr == b"02"] = b"Y"
    arr[arr == b"20"] = b"Y"    
    arr[arr == b"03"] = b"W"
    arr[arr == b"30"] = b"W"    
    arr[arr == b"12"] = b"S"
    arr[arr == b"21"] = b"S"    
    arr[arr == b"13"] = b"R"
    arr[arr == b"31"] = b"R"    
    arr[arr == b"23"] = b"M"
    arr[arr == b"32"] = b"M"    
    return arr



NEXHEADER = """#nexus
begin data;
  dimensions ntax={} nchar={};
  format datatype=DNA missing=N gap=- interleave=yes;
  matrix\n
"""


# TODO add an attribution of the ipcoal version and list sim parameters.
VCFHEADER = """\
##fileformat=VCFv4.2
##fileDate={date}
##source=ipcoal-v.{version}
##reference={reference}
{contig_lines}
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT QUAL\tFILTER\tINFO\tFORMAT\t\
"""
