#!/usr/bin/env python

"""
Classes for writing seqs or snps to popular data formats
like VCF, PHY, NEXUS, or HDF5, while also optionally
combining haplotypes into diploid base calls.
"""

import os
from typing import Optional, Iterable

import numpy as np
import pandas as pd

# import ipcoal
from ipcoal.io.genos import Genos
from ipcoal.io.transformer import Transformer
from ipcoal.io.vcf import VCF
from ipcoal.utils.utils import IpcoalError



NEXHEADER = """#nexus
begin data;
  dimensions ntax={} nchar={};
  format datatype=DNA missing=N gap=- interleave=yes;
  matrix\n
"""


class Writer:
    """Writer class to write ipcoal seqs in a variety of formats.

    This class is used by the Model write functions to convert data
    to output formats and is not intended to be accessed by users
    directly.

    Parameters
    ----------
    seqs: ndarray
        A .seqs array from ipcoal of dimensions (nloci, ntaxa, nsites).
        The data for the ntaxa is ordered by their names alphanumerically.
    names: List[str]
        A list of the taxon names ordered alphanumerically.
    """
    def __init__(self, model):
        # both are already ordered alphanumerically
        self.seqs = model.seqs.copy()
        self.names = model.alpha_ordered_names.copy()
        self.alleles = model.alleles.copy()
        self.outdir = None
        self.outfile = None
        self.idxs = None
        self.ancestral_seq = None
        if model.ancestral_seq is not None:
            self.ancestral_seq = model.ancestral_seq.copy()


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
                    raise IpcoalError(
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
        txf = Transformer(
            self.seqs,
            self.names,
            alleles=self.alleles,
            diploid=diploid,
        )
        txf.transform_seqs()
        self.seqs = txf.seqs
        self.names = txf.names


    def write_loci_to_phylip(
        self,
        outdir,
        idxs=None,
        name_prefix=None,
        name_suffix=None,
        diploid=False,
        quiet=False,
        ):
        """Write seq data for each locus to a separate phylip file.

        Files are written to a shared directory with each locus named
        by locus index. If you want to write only a subset of loci to
        file you can list their indices.

        Parameters:
        -----------
        outdir: str
            A directory in which to write all the phylip files. It will be
            created if it does not yet exist. Default is "./ipcoal_loci/".
        idxs: List[int]
            Numeric indices of the rows (loci) to be written to file.
            Default=None meaning that all loci will be written to file.
        ...
        """
        # bail out and complain if sim_snps() or sim_trees().
        if self.seqs is None:
            raise IpcoalError("cannot write 'loci' for sim_trees() result.")

        if self.seqs.ndim == 2:
            raise IpcoalError("cannot write 'loci' for sim_snps() result.")

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
        outdir: Optional[str]="./",
        name: Optional[str]=None,
        idxs: Optional[Iterable[int]]=None,
        diploid: bool=False,
        quiet: bool=False,
        ):
        """
        Write seq data concatenated to a single phylip file.

        If you want to write only a subset of loci to file you can
        select by their indices.

        Parameters:
        -----------
        outfile: str
            The name/path of the outfile to write.
        idxs: List[int]
            A list of locus indices to subselect which will be concatenated.
        ...
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
        return None


    def write_concat_to_nexus(
        self,
        outdir="./",
        name=None,
        idxs=None,
        diploid=False,
        quiet=False,
        ):
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
        return None


    def write_loci_to_hdf5(
        self,
        name: str,
        outdir: str,
        diploid: bool,
        quiet: bool,
        ):
        """Write data to ipyrad HDF5 database format.

        This format is used by ipyrad-analysis toolkit to efficiently
        extract, filter, and format data from many loci. It is used
        in tools like ipa.tetrad, ipa.window_extracter, ipa.tree_slider.

        This writer function requires the dependency h5py. If this
        package is not installed it will raise an ImportError and
        give recommended instructions for installing it.

        Parameters
        ----------
        name: str
        outdir: str
        diploid: bool
        quiet: bool
        """
        # if non-dependency h5py is not installed then raise exception.
        try:
            import h5py
        except ImportError as err:
            raise ImportError(
                "Writing to HDF5 format requires the additional dependency "
                "h5py which you can install with the following command:\n "
                "  conda install h5py -c conda-forge \n"
                "After installing you will need to restart your notebook."
            ) from err

        # get seqs as bytes
        self._transform_seqs(diploid)
        # txf = Transformer(self.seqs, self.names, self.alleles, diploid)
        # txf.transform_seqs()

        # open h5py database handle
        if name is None:
            name = "test"
        if outdir is None:
            outdir = "."
        outdir = os.path.realpath(os.path.expanduser(outdir))
        h5file = os.path.join(outdir, name + ".seqs.hdf5")
        with h5py.File(h5file, 'w') as io5:

            # write the concatenated seqs bytes array to 'seqs'
            io5.create_dataset("phy",
                data=np.concatenate(self.seqs, 1).view(np.uint8)
            )

            # write the phymap array
            nloci = self.seqs.shape[0]
            loclen = self.seqs.shape[2]
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
            phymap.attrs["columns"] = ['chroms', 'phy0', 'phy1', 'pos0', 'pos1']
            phymap.attrs["phynames"] = self.names
            phymap.attrs["reference"] = 'ipcoal-simulation'

        # report
        if not quiet:
            print("wrote {} loci to {}".format(nloci, h5file))


    def write_snps_to_hdf5(self, name, outdir, diploid, quiet):
        """Writes SNP data to the ipyrad snps HDF5 database format.

        This function can be called whether the data was simulated
        using sim_snps or sim_loci. In the latter case it also stores
        information about where on each locus the SNP was located,
        which can be used by other tools to filter for linkage
        disequilibrium. This file format is used by the ipyrad-analysis
        toolkit for PCA, popgen, and other analyses.

        Parameters
        ----------
        name: str
        outdir: str
        diploid: bool
        quiet: bool
        """
        # if non-dependency h5py is not installed then raise exception.
        try:
            import h5py
        except ImportError as err:
            raise ImportError(
                "Writing to HDF5 format requires the additional dependency "
                "'h5py' which you can install with the following command:\n "
                ">>> conda install h5py -c conda-forge \n"
                "After installing you will need to restart your notebook."
            ) from err

        # reshape SNPs to be like loci.
        if self.seqs.ndim == 2:
            self.seqs = self.seqs.T.reshape(
                self.seqs.shape[1], self.seqs.shape[0], 1)
            self.ancestral_seq = self.ancestral_seq.reshape(
                self.ancestral_seq.size, 1)

        # get seqs as bytes (with optional diploid collapsing). In this
        # case we want this not to transform the seqs data in place.
        txf = Transformer(self.seqs, self.names, self.alleles, diploid)
        txf.transform_seqs()

        # get indices of variable sites (while allowing missing data
        # which might have made some SNP sim sites no longer variable.
        arr = np.concatenate(self.seqs, axis=1)
        marr = np.ma.array(data=arr, mask=(arr == 9))
        common = marr.mean(axis=0).round().astype(int)
        varsites = np.where(np.any(marr != common, axis=0).data)[0]
        nsites = varsites.size

        # get genos as string array _, ([[[9, 9], [0, 1], [1, 1], ...]], [[9, 0], ...])
        genos = Genos(
            arr,
            np.concatenate(self.ancestral_seq),
            varsites,
            txf.dindex_map,
        )
        _, gmat = genos.get_alts_and_genos_matrix()

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
            for snppos in lvar:
                smap[gidx] = loc + 1, lidx, snppos + 1, 0, gidx + 1
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
        tarr = np.concatenate(txf.seqs, axis=1)
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


    def write_vcf(self, name=None, outdir=None, diploid=None, bgzip=False, fill_missing_alleles=True, quiet=False):
        """
        Passes data to VCF object for conversion and writes resulting table
        to CSV.

        Parameters
        ----------
        name (str):
            Prefix name for output file. Will write {name}.vcf.
        outdir (str):
            The directory to write the output file to.
        diploid (bool):
            Combine haploid samples into diploid genotypes.
        bgzip (bool):
            Call bgzip to block compress the file (writes as .vcf.gz).
        fill_missing_alleles (bool):
            If there is missing data this will fill diploid missing alleles.
            e.g., the call (0|.) will be written as (0|0). This is meant to
            emulate real data where we often do not know the other allele
            is missing (also, some software tools do not accept basecalls
            with one missing allele, such as vcftools).
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
            fill_missing_alleles,
        )

        # return dataframe if no filename
        if name is None:
            # concatenate all chunks and return as full dataframe
            fullv = pd.concat(vcf.vcf_chunk_generator(), axis=0)
            fullv.reset_index(drop=True, inplace=True)
            return fullv

        # create a directory if it doesn't exist
        outdir = (outdir if outdir else "./")
        outdir = os.path.realpath(os.path.expanduser(outdir))
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # get output filepath
        outfile = os.path.join(outdir, name.rstrip(".vcf") + ".vcf")

        # write to filepath and record stats while doing it.
        nchroms = 0
        nsnps = 0
        with open(outfile, 'wt') as vout:
            vout.write(vcf.get_header())
            for vchunk in vcf.vcf_chunk_generator():
                vout.write(vchunk.to_csv(header=False, index=False, sep="\t"))
                nsnps += vchunk.shape[0]
                nchroms += vchunk.CHROM.unique().shape[0]

        # call bgzip from tabix so that bcftools can be used on VCF.
        # this will not work with normal gzip compression.
        if bgzip:
            import subprocess
            subprocess.run(["bgzip", "-f", outfile], check=True)
            outfile = outfile.rstrip(".gz") + ".gz"

        # report
        if not quiet:
            print("wrote {} SNPs across {} linkage blocks to {}"
                .format(nsnps, nchroms, outfile))
        return None


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
        return "".join(lines)


    # def write_seqs_as_fasta(self, loc, path):
    #     fastaseq = deepcopy(self.seqs[loc]).astype(str)
    #     fasta = []
    #     for idx, name in enumerate(self.names):
    #         fasta.append(('>' + name + '\n'))
    #         fasta.append("".join(fastaseq[idx])+'\n')

    #     with open(path, 'w') as file:
    #         for line in fasta:
    #             file.write(line)







if __name__ == "__main__":

    import h5py
    import ipcoal
    import toytree

    TREE = toytree.rtree.unittree(6, 1e6)
    MOD = ipcoal.Model(TREE, Ne=1e6, nsamples=2)
    MOD.sim_loci(10, 100)
    MOD.apply_missing_mask(coverage=0.5)

    # test vcf writing to file, and to diploid and haploid dataframes
    WRITER = Writer(MOD.seqs, MOD.alpha_ordered_names, MOD.ancestral_seq)
    df = WRITER.write_vcf(
        name='test',
        outdir='/tmp/',
        diploid=True,
        bgzip=True,
        quiet=False,
    )
    df = WRITER.write_vcf(diploid=True)
    print(df.head())
    df = WRITER.write_vcf(diploid=False)
    print(df.head())

    # test writing loci to hdf5 and show snpsmap
    WRITER.write_loci_to_hdf5(
        name="test",
        outdir="/tmp",
        diploid=True,
        quiet=False,
    )
    with h5py.File("/tmp/test.seqs.hdf5", 'r') as io5:
        print(io5["phymap"][:5])


    WRITER.write_snps_to_hdf5(
        name="test",
        outdir="/tmp",
        diploid=True,
        quiet=False,
    )
    with h5py.File("/tmp/test.snps.hdf5", 'r') as io5:
        print(io5["snps"].shape)
        print(io5["snpsmap"][:20])



