#!/usr/bin/env python


import os
import datetime
import numpy as np
import pandas as pd
import ipcoal

from itertools import groupby
from .utils import ipcoalError


"""
Use Genos in VCF to make it much faster.
"""


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
        self.ancestral_seq = None
        if ancestral_seq is not None:
            self.ancestral_seq = ancestral_seq.copy()


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


    def _transform_seqs(self, diploid):
        """
        Transform seqs from int type to str type. Also optionally combine 
        haploid samples into diploids and use IUPAC ambiguity codes to 
        represent hetero sites.
        """
        txf = Transformer(self.seqs, self.names, diploid)
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
        self._transform_seqs(diploid)

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
        self._transform_seqs(diploid)

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
        self._transform_seqs(diploid)

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


    def write_loci_to_hdf5(self, name, outdir, diploid, quiet):
        """
        Optional writing option to write output to HDF5 database format 
        used by ipyrad analysis toolkit. This will check that you have 
        the h5py package installed and raise an exception if it is missing.

        The .snps.hdf5 output file can be used in ipa.tetrad, 
        ipa.window_extracter, ipa.treeslider, etc.
        """
        # if non-dependency h5py is not installed then raise exception.
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "Writing to HDF5 format requires the additional dependency "
                "'h5py' which you can install with the following command:\n "
                "  conda install h5py -c conda-forge \n"
                "After installing you will need to restart your notebook."
            )

        # get seqs as bytes 
        txf = Transformer(self.seqs, self.names, diploid)

        # open h5py database handle
        if name is None:
            name = "test"
        if outdir is None:
            outdir = "."
        outdir = os.path.realpath(os.path.expanduser(outdir))
        h5file = os.path.join(outdir, name + ".seqs.hdf5")
        with h5py.File(h5file, 'w') as io5:

            # write the concatenated seqs bytes array to 'seqs'
            io5.create_dataset("phy", data=np.concatenate(txf.seqs, 1).view(np.uint8))

            # write the phymap array
            nloci = txf.seqs.shape[0]
            loclen = txf.seqs.shape[2]
            phymap = io5.create_dataset(
                "phymap", shape=(nloci, 5), dtype=np.int64)
            phymap[:, 0] = range(1, nloci + 1)  # 1-indexed 
            phymap[:, 1] = range(0, nloci * loclen, loclen)
            phymap[:, 2] = phymap[:, 1] + loclen
            phymap[:, 3] = 0
            phymap[:, 4] = phymap[:, 1] + loclen

            # placeholders for now
            io5.create_dataset("scaffold_lengths", data=np.repeat(loclen, nloci))
            io5.create_dataset("scaffold_names", data=(
                ['loc-{}'.format(i).encode() for i in range(1, nloci + 1)]))

            # meta info stored to phymap
            phymap.attrs["columns"] = (b'chroms', b'phy0', b'phy1', b'pos0', b'pos1')
            phymap.attrs["phynames"] = [i.encode() for i in txf.names]
            phymap.attrs["reference"] = 'ipcoal-simulation'

        # report
        if not quiet:
            print("wrote {} loci to {}".format(nloci, h5file))


    def write_snps_to_hdf5(self, name, outdir, diploid, quiet):
        """

        """
        # if non-dependency h5py is not installed then raise exception.
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "Writing to HDF5 format requires the additional dependency "
                "'h5py' which you can install with the following command:\n "
                "  conda install h5py -c conda-forge \n"
                "After installing you will need to restart your notebook."
            )

        # reshape SNPs to be like loci.
        if self.seqs.ndim == 2:
            self.seqs = self.seqs.T.reshape(
                self.seqs.shape[1], self.seqs.shape[0], 1)
            self.ancestral_seq = self.ancestral_seq.reshape(
                self.ancestral_seq.size, 1)

        # get seqs as bytes (with optional diploid collapsing)
        txf = Transformer(self.seqs, self.names, diploid)
        tarr = np.concatenate(txf.seqs, axis=1)

        # get indices of variable sites (while allowing missing data)
        arr = np.concatenate(self.seqs, axis=1)
        marr = np.ma.array(data=arr, mask=(arr == 9))
        common = marr.mean(axis=0).round().astype(int)
        varsites = np.where(np.any(marr != common, axis=0).data)[0]
        nsites = varsites.size

        # get genos as string array [0|0, 0|1, 1|1, ...]
        genos = Genos(arr, self.ancestral_seq, varsites, txf.dindex_map)
        if 9 in arr:
            gmat = genos.get_genos_matrix_missing()
        else:
            gmat = genos.get_genos_matrix()

        # get snpsmap (chrom, locidx, etc.)
        smap = np.zeros((nsites, 5), dtype=np.uint32)
        gidx = 0
        for loc in range(self.seqs.shape[0]):

            # mask array to select only variable sites (while allow missing)
            larr = np.ma.array(self.seqs[loc], mask=(self.seqs[loc] == 9))
            lcom = larr.mean(axis=0).round().astype(int)
            lvar = np.where(np.any(larr != lcom, axis=0).data)[0]
            lidx = 0

            # enter variants to snpmap
            for snpidx in lvar:
                smap[gidx] = loc + 1, lidx, snpidx, 0, gidx + 1
                lidx += 1
                gidx += 1

        # open h5py database handle
        if name is None:
            name = "test"
        if outdir is None:
            outdir = "."
        outdir = os.path.realpath(os.path.expanduser(outdir))
        h5file = os.path.join(outdir, name + ".snps.hdf5")

        # write datasets to database
        with h5py.File(h5file, 'w') as io5:

            # write the concatenated seqs as bytes->uint8 to snps
            snps = io5.create_dataset(
                name="snps",
                data=tarr[:, varsites].view(np.uint8),
            )

            # write snpsmap [chrom1, locsnpidx0, locsnppos0, 0, snpidx1]
            snpsmap = io5.create_dataset("snpsmap", data=smap)

            # genotype calls (derived/ancestral) compared to known ancestral
            io5.create_dataset(name="genos", data=gmat)

            # placeholders for now
            io5.create_dataset(name="psuedoref", shape=(nsites, 2))

            # meta info stored to phymap
            snpsmap.attrs["columns"] = (
                b'locus', b'locidx', b'locpos', b'scaf', b'scafpos')
            snps.attrs['names'] = [i.encode() for i in txf.names]

        # report
        if not quiet:
            print("wrote {} SNPs to {}".format(nsites, h5file))


    def write_vcf(self, name=None, outdir=None, diploid=None, bgzip=False, quiet=False):
        """
        Passes data to VCF object for conversion and writes resulting table 
        to CSV. 

        TODO: bgzip option may be overkill, build_vcf could be much faster
        using methods like Genos.

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
            self.ancestral_seq,
        )
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


class Genos:
    """

    """
    def __init__(self, concatseqs, anc, snpidxs, dindex_map):

        self.seqs = concatseqs
        self.anc = anc
        self.snpidxs = snpidxs
        self.dindex_map = dindex_map


    def get_genos_matrix(self):
        """
        Returns genos matrix as ints array (nsnps, nsamples, 2)
        """
        # concatenate seqs to -1 dim
        aseq = np.concatenate(self.anc)
        tseq = self.seqs

        # subsample to varible sites if SNPs only
        if self.snpidxs is not None:
            aseq = aseq[self.snpidxs]
            tseq = tseq[:, self.snpidxs]

        # get genotype calls (derived or ancestral)
        genos = np.invert(tseq == aseq).astype(int)

        # derived (1) or another derived (2); compare to the low allele in data
        # this does not worry about which is the more common allele, since we
        # expect any downstream method that will use 'geno' calls will simply
        # use the presence of multiple genos as a filtering mechanism. 
        # Thus, the x/0 in genos still just means derived/ancestral.

        # get derived alleles at every site
        marr = np.ma.array(data=tseq, mask=genos == 0)

        # if any derived alleles are not the max allele at that site
        maxa = marr.max(axis=0)
        alts = marr != maxa
        alts[marr.mask] = False
        genos[alts] = 2

        # shape into char array 
        gmat = np.zeros((tseq.shape[1], len(self.dindex_map), 2), dtype=np.uint8)
        for idx in self.dindex_map:
            left, right = self.dindex_map[idx]
            gmat[:, idx, :] = genos[(left, right), :].T
        return gmat


    def get_genos_matrix_missing(self):
        """
        Returns genos matrix a bit slower b/c accomodates missing values (9),
        snpidxs has already been computed on a masked array.
        """
        # concatenate seqs to -1 dim
        aseq = np.concatenate(self.anc)
        tseq = self.seqs

        # subsample to varible sites if SNPs only
        if self.snpidxs is not None:
            aseq = aseq[self.snpidxs]
            tseq = tseq[:, self.snpidxs]

        # get genotype calls (inverts data but not mask)
        tseqm = np.ma.array(tseq, mask=(tseq == 9))
        genos = np.invert(tseqm == aseq).astype(int)

        # if any derived alleles are not the max allele at that site
        marr = np.ma.array(data=tseq, mask=(genos == 0) | (tseq == 9))
        maxa = marr.max(axis=0)
        alts = marr != maxa
        alts[marr.mask] = False
        genos[alts] = 2

        # shape into char array 
        gmat = np.zeros((tseq.shape[1], len(self.dindex_map), 2), dtype=np.uint8)
        for idx in self.dindex_map:

            # get the haplotypes indices
            left, right = self.dindex_map[idx]

            # get data arranged to right shape
            mdata = genos[(left, right), :].T

            # fill masked values to 9
            idat = mdata.data.astype(int)
            idat[mdata.mask] = 9

            # copy other allele over 9 if it is not 9
            idat[idat[:, 0] == 9, 0] = idat[idat[:, 0] == 9, 1]
            idat[idat[:, 1] == 9, 1] = idat[idat[:, 1] == 9, 0]

            # store the results
            gmat[:, idx, :] = idat
        return gmat





    def get_genos_string(self):
        """
        Return string representation of genotypes, e.g., 0|0, 1|0, ...
        """
        # concatenate seqs to -1 dim
        aseq = np.concatenate(self.anc)
        seqs = np.concatenate(self.seqs, axis=1)

        # subsample to varible sites if SNPs only
        if self.snpidxs is not None:
            aseq = aseq[self.snpidxs]
            seqs = seqs[:, self.snpidxs]

        # get genotype calls
        genos = np.invert(seqs == aseq).astype(int)

        # shape into char array 
        if self.dindex_map is None:
            gmat = np.char.array(genos) + b"|" + np.char.array(genos)
        else:
            gmat = np.chararray(seqs.shape, 3)
            for idx in self.dindex_map:
                left, right = self.dindex_map[idx]
                gmat[idx] = ["{}|{}".format(i, j) for (i, j) in zip(genos)]
        return gmat



class Transformer:
    """
    seqs: (ndarray)
    names: (ndarray)
    diploid: (bool)
    """    
    def __init__(
        self,
        seqs,
        names,
        diploid=True,
        ):

        # store input params
        self.seqs = seqs
        self.names = names
        self.diploid = diploid
        self.diploid_map = {}
        self.dindex_map = {}

        # setup functions
        self.get_diploid_map()
        self.transform_seqs()


    def get_diploid_map(self):
        """
        Combine haploid names and indices to map pairs as new diploids.

        diploid_map = {'A': ['A-0-1', 'A-2-3], 'B': ['B-0-1', 'B-2-3'], ...]}
        dindex_map = {1: [(0,1), (2,3)], 2: [(4,5), (6,7)], ...}
        """

        # haploid indices simply repeat itself twice. 
        # TODO: NOT TESTED, OR USED YET, CHECK ORDER OF DINDEX
        if not self.diploid:
            pidx = 0
            for idx, name in enumerate(self.names):
                self.diploid_map[name] = (name, name)
                self.dindex_map[idx] = (pidx, pidx)
                pidx += 1

        # diploid indices increase in pairs: (0,1), (2,3), (4,6)...
        else:

            # group names by prefix
            groups = groupby(self.names, key=lambda x: x.split("-")[0])

            # arrange into an IMAP dictionary: {r0: [r0-0, r0-1, r0-2, ...]}
            imap = {i[0]: list(i[1]) for i in groups}

            # if all nsamples are 2 then we will not add name suffix
            suffix = True
            if all([len(imap[i]) == 2 for i in imap]):
                suffix = False

            # iterate over tips and increment diploids and haploid pair idxs
            didx = 0
            for sppname in imap:

                # nsamples matched to this tip
                samples = imap[sppname]
                samples = sorted(samples, key=lambda x: int(x.rsplit("-", 1)[-1]))

                # must be x2
                assert len(samples) % 2 == 0, (
                    "nsamples args must be multiples of 2 to form diploids" 
                    "sample {} has {} samples".format(sppname, len(samples)))

                # iterate 0, 2, 4
                for pidx in range(0, len(samples), 2):

                    # the local idx of these sample names
                    p0 = samples[pidx]
                    p1 = samples[pidx + 1]

                    # the global idx of these sample names
                    pidx0 = self.names.index(p0)
                    pidx1 = self.names.index(p1)

                    # fill dicts
                    if suffix:
                        newname = "{}-{}".format(sppname, int(pidx / 2))
                    else:
                        newname = sppname
                    self.diploid_map[newname] = (p0, p1)
                    self.dindex_map[didx] = (pidx0, pidx1)
                    didx += 1



    def transform_seqs(self):
        """
        Transforms seqs from ints to strings. If using diploid map this also
        combines the two alleles to represent ambiguity codes for hetero sites,
        which changes the dimension of both .seqs and .names.
        """
        # simply convert to bytes
        if not self.diploid:
            self.seqs = convert_intarr_to_bytearr(self.seqs)  # .astype(bytes))

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
    def __init__(self, seqs, names, diploid, ancestral):
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
        self.diploid_map = {}
        self.aseqs = convert_intarr_to_bytearr(ancestral)  # .astype(bytes))

        # do not combine for ambiguity codes, but get diploid_map and names.
        txf = Transformer(self.seqs, self.names, diploid)
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
        """
        Build a DF of genotypes and metadata.
        """

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
            arr = convert_intarr_to_bytearr(self.seqs[loc])
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



def convert_intarr_to_bytearr(iarr):
    "An array of ints converted to bytes"
    barr = np.zeros(iarr.shape, dtype="S1")
    barr[iarr == 0] = b"A"
    barr[iarr == 1] = b"C"
    barr[iarr == 2] = b"G"
    barr[iarr == 3] = b"T"
    barr[iarr == 9] = b"N"
    return barr


# def convert_intarr_to_bytearr(arr):
#     "An array of ints was turned into bytes and this converts to bytestrings"
#     arr[arr == b"0"] = b"A"
#     arr[arr == b"1"] = b"C"
#     arr[arr == b"2"] = b"G"
#     arr[arr == b"3"] = b"T"
#     return arr


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

    arr[arr == b"90"] = b"A"
    arr[arr == b"09"] = b"A"
    arr[arr == b"91"] = b"C"
    arr[arr == b"19"] = b"C"
    arr[arr == b"92"] = b"G"
    arr[arr == b"29"] = b"G"
    arr[arr == b"93"] = b"T"
    arr[arr == b"39"] = b"T"
    arr[arr == b"99"] = b"N"

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
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t\
"""
