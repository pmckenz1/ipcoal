#!/usr/bin/env python


import os
import numpy as np



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



    def write_loci_to_phylip(self, outdir, idx=None, outfile=None):
        """
        Write all seq data for each locus to a separate phylip file in a shared
        directory with each locus named by ids locus index. 

        Parameters:
        -----------
        outdir (str):
            A directory in which to write all the phylip files. It will be 
            created if it does not yet exist. Default is "./ipcoal_loci/".
        idx (int):
            To write a single locus file provide the idx. If None then all loci
            are written to separate files.
        outfile (str):
            Only used if idx is not None. Set the name of the locus file being
            written. This is used internally to write tmpfiles for TreeInfer.
        """

        # make outdir if it does not yet exist
        self.outdir = os.path.realpath(os.path.expanduser(outdir))
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # get loci to write
        if not idx:
            lrange = range(self.seqs.shape[0])
        else:
            lrange = range(idx, idx + 1)

        # iterate over loci (or single selected locus)
        for loc in lrange:

            # get locus and convert to bases
            arr = self.seqs[loc].astype(bytes)
            arr = convert_intarr_to_bytearr(arr)

            # open file handle numbered unless user
            fhandle = os.path.join(self.outdir, "{}-ipcoal.phy".format(loc))
            if (idx and outfile):
                fhandle = fhandle

            # build list of line strings
            phystring = self.build_phystring_from_loc(arr)

            # write to file
            with open(fhandle, 'w') as out:
                out.write(phystring)



    def write_seqs_to_phylip(self, outfile):
        """
        Write all seq data (loci or snps) concated to a single phylip file.

        Parameters:
        -----------
        outfile (str):
            The name/path of the outfile to write. 
        """
        # create a directory if it doesn't exist
        outfile = os.path.realpath(os.path.expanduser(outfile))
        directory = os.path.dirname(outfile)
        directory = os.path.realpath(os.path.expanduser(directory))
        if not os.path.exists(directory):
            os.makedirs(directory)

        # concat loci (unless SNPs already) and convert to bases
        if self.seqs.ndim == 2:
            arr = self.seqs.astype(bytes)
        else:
            arr = np.concatenate(self.seqs, axis=1).astype(bytes)
        arr = convert_intarr_to_bytearr(arr)

        # build list of line strings
        phystring = self.build_phystring_from_loc(arr)

        # write to file
        with open(outfile, 'w') as out:
            out.write(phystring)
        self.outfile = outfile



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



    def write_seqs_to_countmatrix(self, outfile):
        pass            
        # # iterator for quartets, e.g., (0, 1, 2, 3), (0, 1, 2, 4)...
        # quartidx = 0
        # qiter = itt.combinations(range(self.ntips), 4)
        # for currquart in qiter:
        #     # cols indices match tip labels b/c we named tips node.idx
        #     quartsnps = snparr[:, currquart]
        #     # save as stacked matrices
        #     tmpcounts[quartidx] = count_matrix_int(quartsnps)
        #     # save flattened to counts
        #     quartidx += 1
        # return(np.ravel(tmpcounts))



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
