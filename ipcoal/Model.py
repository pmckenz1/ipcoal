#!/usr/bin/env python

"""
Generate large database of site counts from coalescent simulations
based on msprime + toytree for using in machine learning algorithms.
"""

# imports for py3 compatibility
from __future__ import print_function
from builtins import range

# imports
import sys
from copy import deepcopy

import toytree
import numpy as np
import pandas as pd
import msprime as ms

from .utils import get_all_admix_edges, ipcoalError
from .TreeInfer import TreeInfer
from .Writer import Writer
from .SeqModel import SeqModel
from .SeqGen import SeqGen

# set global display preference to make tree columns look nice
pd.set_option("max_colwidth", 28)



class Model:
    """
    An ipcoal.Model object for defining demographic models for coalesent 
    simulation in msprime. 
    """
    def __init__(
        self,
        tree,
        Ne=10000,   
        admixture_edges=None,
        admixture_type=0,
        samples=1,
        recomb=1e-9,
        mut=1e-8,
        seed=None,
        seed_mutations=None,
        debug=False,
        ):

        """
        Takes an input topology with edge lengths in units of generations
        entered as either a newick string or as a Toytree object, and defines
        a demographic model based on Ne (or Ne mapped to tree nodes) and 
        admixture edge arguments. Genealogies and sequence data is then 
        generated with msprime and seq-gen, respectively.

        Parameters:
        -----------
        tree: (str)
            A newick string or Toytree object of a species tree with edges in
            coalescent units.

        admixture_edges (list, tuple):
            A list of admixture events in the 'admixture interval' format:
            (source, dest, (edge_min, edge_max), (rate_min, rate_max)).
            e.g., (3, 5, 0.5, 0.01)
            e.g., (3, 5, (0.5, 0.5), (0.05, 0.5))
            e.g., (1, 3, (0.1, 0.9), 0.05)
            The source sends migrants to destination **backwards in time.**
            The edge min, max are *proportions* of the length of the edge that
            overlaps between source and dest edges over which admixture can
            occur. If None then default values of 0.25 and 0.75 are used,
            meaning introgression can occur over the middle 50% of the edge.
            The rate min, max are migration rates or proportions that will be
            either a single value or sampled from a range. For 'rate' details
            see the 'admixture type' parameter.

        admixture_type (str, int):
            Either "pulsed" (0; default) or "interval" (1).
            If "pulsed" then admixture occurs at a single time point
            selected uniformly from the admixture interval (e.g., (0.1, 0.9)
            can select over 90% of the overlapping edge; (0.5, 0.5) would only
            allow admixture at the midpoint). The 'rate' parameter is the
            proportion of one population that will be introgressed into the
            other.
            If "interval" then admixture occurs uniformly over the entire
            admixture interval and 'rate' is a constant migration rate over
            this time period.

        Ne (float, int): default=10000
            The effective population size. This value will be set to all edges
            of the tree. If you want to set different Ne values to different
            edges then you should add Ne node attributes to your input tree
            which will override the global value at those nodes. For example, 
            if you set Ne=5000, and on your tree set Ne values for a few nodes
            with (tre.set_node_values("Ne", {1:1000, 2:10000})) then all edges
            will use Ne=5000 except those nodes.

        mut (float): default=1e-8
            ...

        recomb (float): default=1e-9
            ...

        seed (int):
            Random number generator used for msprime (and seqgen unless a 
            separate seed is set for seed_mutations.

        seed_mutations (int):
            Random number generator used for seq-gen. If not set then the
            generic seed is used for both msprime and seq-gen.
        """

        # initialize random seed for msprime and seq-gen
        self.random = np.random.RandomState(seed)
        self.random_mut = (
            np.random.RandomState(seed_mutations) if seed_mutations 
            else self.random
        )

        # hidden argument to turn on debugging
        self._debug = debug

        # parse the input tree (and store original)
        if isinstance(tree, toytree.Toytree.ToyTree):
            self.treeorig = tree
            self.tree = deepcopy(self.treeorig)
        elif isinstance(tree, str):
            self.treeorig = toytree.tree(tree)
            self.tree = deepcopy(self.treeorig)
        else:
            raise TypeError("input tree must be newick str or Toytree object")

        # the order of samples given tree (5 tips) and samples [3, 2, 1, 2, 3]
        # ladderized tree tip order from top to bottom.
        # 
        # |--------0  0-0, 0-1, 0-2
        #  |-------1  1-0, 1-1
        #   |------2  2-0, 
        #     |----3  3-0, 3-1
        #       |--4  4-0, 4-1, 4-2
        #
        # namedict: {0: '0-0', 1: '0-1', 2: '0-2', 3: '1-0', 4: '1-1'...}
        self.ntips = len(self.tree)
        if isinstance(samples, int):
            self.samples = [int(samples) for i in range(self.ntips)]
            self.nstips = int(samples) * self.ntips
        else:
            assert isinstance(samples, (list, tuple)), (
                "samples should be a list")
            assert len(samples) == self.ntips, (
                "samples list should be same length as ntips in tree.")
            self.samples = samples
            self.nstips = sum(self.samples)

        # store sim params: fixed mut, Ne, recomb
        self.mut = mut
        self.recomb = recomb

        # global Ne will be overwritten by Ne attrs in .tree. This sets node.Ne
        self.Ne = Ne
        self._get_Ne()

        # store tip names for renaming on the ms tree (ntips * nsamples)
        # these are 1-indexed because they msprime trees tips are.
        if samples == 1:
            self.tipdict = {
                i + 1: j for (i, j) in enumerate(self.tree.get_tip_labels())
            }
        else:
            self.tipdict = {}
            idx = 1
            for tip, ns in zip(self.tree.get_tip_labels(), self.samples):
                for nidx in range(ns):
                    self.tipdict[idx] = "{}-{}".format(tip, nidx)
                    idx += 1

        # alphanumeric ordered tip names -- order of printing to seq files
        _tmp = {j: i for (i, j) in self.tipdict.items()}       
        self.names = sorted(self.tipdict.values())
        self.order = [_tmp[i] - 1 for i in self.names]

        # check formats of admixture args
        self.admixture_edges = (admixture_edges if admixture_edges else [])
        self.admixture_type = (1 if admixture_type in (1, "interval") else 0)
        if self.admixture_edges:
            if not isinstance(self.admixture_edges[0], (list, tuple)):
                self.admixture_edges = [self.admixture_edges]
            for edge in self.admixture_edges:
                if len(edge) != 4:
                    raise ValueError(
                        "admixture edges should each be a tuple with 4 values")
        self.aedges = (
            0 if not self.admixture_edges else len(self.admixture_edges))

        # demography info to fill
        self.ms_migrate = []
        self.ms_migtime = []
        self.ms_demography = set()
        self.ms_popconfig = ms.PopulationConfiguration()

        # get migration time, rate {mrates: [], mtimes: []}
        self._get_migration()

        # get demography dict for msprime input
        self._get_demography()

        # get popconfig as msprime input
        self._get_popconfig()

        # hold the model outputs
        self.df = None
        self.seqs = None



    def _get_Ne(self):
        """
        If Ne node attrs are present in the input tree these override the 
        global Ne argument which is set to all other nodes. Every node should
        have an Ne value at the end of this. Sets node.Ne attrs and sets max
        value to self.Ne.
        """
        # get map of {nidx: node}
        ndict = self.tree.get_node_dict(True, True)

        # set user entered arg (self.Ne) to any node without a Ne attr
        for nidx, node in ndict.items():
            if not hasattr(node, "Ne"):
                setattr(node, "Ne", int(self.Ne))
            else:
                setattr(node, "Ne", int(node.Ne))

        # check that all nodes have .Ne
        nes = self.tree.get_node_values("Ne", True, True)
        if not all(nes):
            raise ipcoalError(
                "Ne must be provided as an argument or tree node attribute.")

        # set to the max value        
        self.Ne = max(nes)



    def _get_migration(self):
        """
        Generates mrates, mtimes, and thetas arrays for simulations.

        Migration times are uniformly sampled between start and end points that
        are constrained by the overlap in edge lengths, which is automatically
        inferred from 'get_all_admix_edges()'. Migration rates are in [0, 1)

        # rates are proportions, times are in generations
        # single edge: 
        self.ms_migrate = [0.05]
        self.ms_migtime = [12000]

        # two edges:
        self.ms_migrate = [0.05, 0.05]
        self.ms_migtime = [12000, 20000]
        """
        # sample times and proportions/rates for admixture intervals
        for iedge in self.admixture_edges:

            # mtimes: if None then sample from uniform.
            if iedge[2] is None:
                mi = (0.0, 1.0)
            # if int or float then sample from one position
            elif isinstance(iedge[2], (float, int)):
                mi = (iedge[2], iedge[2])
            # if an iterable then use min max edge overlaps
            else:
                mi = iedge[2]
            intervals = get_all_admix_edges(self.tree, mi[0], mi[1])

            # mrates: if None then sample from uniform
            if iedge[3] is None:
                if self.admixture_type:
                    mr = (0.0, 0.1)
                else:
                    mr = (0.05, 0.5)
            # if a float then use same for all
            elif isinstance(iedge[3], (float, int)):
                mr = (iedge[3], iedge[3])
            # if an iterable then sample from range
            else:
                mr = iedge[3]
            # migrate uniformly drawn from range
            mrate = self.random.uniform(mr[0], mr[1])

            # intervals are overlapping edges where admixture can occur.
            # lower and upper restrict the range along intervals for each
            snode = self.tree.treenode.search_nodes(idx=iedge[0])[0]
            dnode = self.tree.treenode.search_nodes(idx=iedge[1])[0]
            ival = intervals.get((snode.idx, dnode.idx))
            dist_ival = ival[1] - ival[0]

            # intervals mode
            if self.admixture_type:
                ui = self.random.uniform(
                    ival[0] + mi[0] * dist_ival,
                    ival[0] + mi[1] * dist_ival, 2,
                )
                ui = ui.reshape((1, 2))
                mtime = np.sort(ui, axis=1).astype(int)

            # pulsed mode
            else:
                ui = self.random.uniform(
                    ival[0] + mi[0] * dist_ival,
                    ival[0] + mi[1] * dist_ival)
                mtime = int(ui)

            # store values only if migration is high enough to be detectable
            self.ms_migrate.append(mrate)
            self.ms_migtime.append(mtime)



    def _get_demography(self):
        """
        Returns demography scenario based on an input tree and admixture
        edge list with events in the format (source, dest, start, end, rate).
        Time on the tree is defined in units of generations.
        """
        # Define demographic events for msprime
        demog = set()

        # tag min index child for each node, since at the time the node is
        # called it may already be renamed by its child index b/c of
        # divergence events.
        for node in self.tree.treenode.traverse():
            if node.children:
                node._schild = min([i.idx for i in node.get_descendants()])
            else:
                node._schild = node.idx

        # Add divergence events (converts time to N generations)
        for node in self.tree.treenode.traverse():
            if node.children:
                dest = min([i._schild for i in node.children])
                source = max([i._schild for i in node.children])
                time = int(node.height)
                demog.add(ms.MassMigration(time, source, dest))
                demog.add(ms.PopulationParametersChange(
                    time,
                    initial_size=node.Ne,
                    population=dest),
                )
                if self._debug:
                    print(
                        'div time:  {:>9}, {:>2} {:>2}, {:>2} {:>2}, Ne={}'
                        .format(
                            int(time), source, dest,
                            node.children[0].idx, node.children[1].idx,
                            node.Ne,
                            ),
                        file=sys.stderr,
                    )

        # Add migration pulses
        if not self.admixture_type:
            for evt in range(self.aedges):

                # rate is prop. of population, time is prop. of edge
                rate = self.ms_migrate[evt]
                time = self.ms_migtime[evt]
                source, dest = self.admixture_edges[evt][:2]

                # rename nodes at time of admix in case diverge renamed them
                snode = self.tree.treenode.search_nodes(idx=source)[0]
                dnode = self.tree.treenode.search_nodes(idx=dest)[0]
                children = (snode._schild, dnode._schild)
                demog.add(
                    ms.MassMigration(time, children[0], children[1], rate))
                if self._debug:
                    print(
                        'mig pulse: {:>9}, {:>2} {:>2}, {:>2} {:>2}, rate={:.2f}'
                        .format(
                            time, 
                            "", "",  # node.children[0].idx, node.children[1].idx, 
                            snode.name, dnode.name, 
                            rate),
                        file=sys.stderr,
                    )

        # Add migration intervals
        else:
            for evt in range(self.aedges):
                rate = self.ms_migration[evt]['mrates']
                time = (self.ms_migration[evt]['mtimes']).astype(int)
                source, dest = self.admixture_edges[evt][:2]

                # rename nodes at time of admix in case diverg renamed them
                snode = self.tree.treenode.search_nodes(idx=source)[0]
                dnode = self.tree.treenode.search_nodes(idx=dest)[0]
                children = (snode._schild, dnode._schild)
                demog.add(ms.MigrationRateChange(time[0], rate, children))
                demog.add(ms.MigrationRateChange(time[1], 0, children))
                if self._debug:
                    print("mig interv: {}, {}, {}, {}, {:.3f}".format(
                        time[0], time[1], children[0], children[1], rate),
                        file=sys.stderr)

        # sort events by type (so that mass migrations come before pop size
        # changes) and time
        demog = sorted(list(demog), key=lambda x: x.type)
        demog = sorted(demog, key=lambda x: x.time)
        self.ms_demography = demog



    def _get_popconfig(self):
        """
        returns population_configurations for N tips of a tree
        """       
        # pull Ne values from the toytree nodes attrs.
        if not self.Ne:
            # get Ne values from tips of the tree
            nes = self.tree.get_node_values("Ne", show_tips=True)
            nes = nes[-self.tree.ntips:][::-1]

            # list of popconfig objects for each tip
            population_configurations = [
                ms.PopulationConfiguration(
                    sample_size=self.samples[i], initial_size=nes[i])
                for i in range(self.ntips)]

            # set the max Ne value as the global Ne
            self.Ne = max(nes)

        # set user-provided Ne value to all edges of the tree
        else:
            population_configurations = [
                ms.PopulationConfiguration(
                    sample_size=self.samples[i], initial_size=self.Ne)
                for i in range(self.ntips)]

        # debug printer
        if self._debug:
            print(
                "pop: Ne:{:.0f}, mut:{:.2E}".format(self.Ne, self.mut),
                file=sys.stderr)
        self.ms_popconfig = population_configurations



    def _get_locus_sim(self, nsites=1, snp=False):
        """
        Returns a msprime.simulate() generator object that can generate 
        treesequences under the demographic model parameters. 

        Parameters:
        -----------
        nsites (int):
            The number of sites to simulate a tree_sequence over with 
            recombination potentially creating multiple genealogies.
        snp (bool):
            Sets length=None, recombination_rate=None, and num_replicates=big,
            as a way to ensure we will get a single gene tree. This is mostly
            used internally for simcat SNP sims.
        """
        # snp flag to ensure a single genealogy is returned
        if snp and nsites != 1:
            raise ipcoalError(
                "The flag snp=True should only be used with nsites=1")

        # migration scenarios from admixture_edges, used in demography
        migmat = np.zeros((self.ntips, self.ntips), dtype=int).tolist()

        # msprime simulation to make tree_sequence generator
        sim = ms.simulate(
            length=nsites,
            random_seed=self.random.randint(1e9),
            recombination_rate=(None if snp else self.recomb),
            migration_matrix=migmat,
            num_replicates=(int(1e20) if snp else 1),        # ensures SNPs
            population_configurations=self.ms_popconfig,     # applies Ne
            demographic_events=self.ms_demography,           # applies popst
        )
        return sim



    def _sim_locus(self, nsites, locus_idx=0, **kwargs):
        """
        Simulate tree sequence for each locus and sequence data for each 
        genealogy and return all in a dataframe. 
        """

        # initialize a sequence simulator unless provided as a hidden arg.
        # this is just a convenience for testing, users should call .run().
        if not kwargs.get("seqgen"):
            mkseq = SeqGen()
            mkseq.open_subprocess()
        else:
            mkseq = kwargs.get("seqgen")

        # get the msprime ts generator 
        msgen = self._get_locus_sim(nsites)

        # get the treesequence and its breakpoints
        msts = next(msgen)
        breaks = list(msts.breakpoints())

        # get start and stop indices
        starts = breaks[0:len(breaks) - 1]
        ends = breaks[1:len(breaks)]

        # what is the appropriate rounding? (some trees will not exist...)
        bps = (np.round(ends) - np.round(starts)).astype(int)

        # init dataframe
        df = pd.DataFrame({
            "start": np.round(starts).astype(int),
            "end": np.round(ends).astype(int),
            "genealogy": "",
            "nbps": bps,
            "nsnps": 0,
            "locus": locus_idx,
            },
            columns=['locus', 'start', 'end', 'nbps', 'nsnps', 'genealogy'],
        )

        # the full sequence array to fill
        bidx = 0
        seqarr = np.zeros((self.nstips, nsites), dtype=np.uint8)

        # iterate over the index of the dataframe to sim for each genealogy
        for idx, mstree in zip(df.index, msts.trees()):

            # get the number of base pairs taken up by this gene tree
            gtlen = df.loc[idx, 'nbps']

            # only simulate data if there is bp 
            if gtlen:
                # parse the mstree
                nwk = mstree.newick()

                # sim locus
                seed = self.random_mut.randint(1e9)
                seq = mkseq.feed_tree(nwk, gtlen, self.mut, seed)

                # store locus
                seqarr[:, bidx:bidx + gtlen] = seq[self.order, :]

                # record snps
                subseq = seqarr[:, bidx:bidx + gtlen]
                df.loc[idx, 'nsnps'] = (
                    np.any(subseq != subseq[0], axis=0).sum())

                # advance site counter
                bidx += gtlen

                # reset .names on msprime tree with node_labels 1-indexed
                gtree = toytree.tree(nwk)
                for node in gtree.treenode.get_leaves():
                    node.name = self.tipdict[int(node.name)]
                newick = gtree.write(tree_format=5)
                df.loc[idx, "genealogy"] = newick

        # drop intervals that are 0 bps in length (sum bps will still = nsites)
        df = df.drop(index=df[df.nbps == 0].index).reset_index(drop=True)        

        # clean and close subprocess if not still using
        if not kwargs.get("seqgen"):
            mkseq.close_subprocess()

        # return the dataframe and seqarr
        return df, seqarr



    def sim_loci(self, nloci=1, nsites=1):
        """
        Simulate tree sequence for each locus and sequence data for each 
        genealogy and return all genealogies and their summary stats in a 
        dataframe and the concatenated sequences in an array with rows ordered
        by sample names alphanumerically.
        """
        # multidimensional array of sequence arrays to fill 
        seqarr = np.zeros((nloci, self.nstips, nsites), dtype=np.uint8)

        # a list to be concatenated into the final dataframe of genealogies
        dflist = []

        # open the subprocess to seqgen
        mkseq = SeqGen()
        mkseq.open_subprocess()

        # iterate over nloci to simulate, get df and arr to store.
        for lidx in range(nloci):

            # returns genetree_df and seqarray
            df, arr = self._sim_locus(nsites, lidx, **{'seqgen': mkseq})

            # store seqs in a list for now
            seqarr[lidx] = arr

            # store the genetree df in a list for now
            dflist.append(df)

        # concatenate all of the genetree dfs
        df = pd.concat(dflist)
        df = df.reset_index(drop=True)

        # clean and close subprocess 
        mkseq.close_subprocess()

        # store values to object
        self.df = df
        self.seqs = seqarr



    def sim_snps(self, nsnps=1, repeat_on_trees=False, seqgen=False):
        """
        Run simulations until nsnps _unlinked_ SNPs are generated. If the tree
        is shallow and the mutation rate is low this can take a long time b/c
        most genealogies will produce an invariant SNP. There are two options
        for how to simulate snps:

        nsnps (int):
            The number of SNPs to produce.
        repeat_on_trees (bool):
            If True then sequence simulations repeat on a genealogy until it 
            produces a SNP. If False then if a genealogy does not produce
            a SNP we move on to the next simulated genealogy. This may be
            more correct since shallow trees are less likely to contain SNPs.
        """

        # initialize a sequence simulator
        if seqgen:
            mkseq = SeqGen()
            mkseq.open_subprocess()
        else:
            mkseq = SeqModel()

        # get the msprime ts generator 
        msgen = self._get_locus_sim(1, snp=True)

        # store results (nsnps, ntips); def. 1000 SNPs
        newicks = []
        snpidx = 0
        snparr = np.zeros((self.nstips, nsnps), dtype=np.uint8)

        # continue until we get nsnps
        while 1: 

            # bail out if nsnps finished
            if snpidx == nsnps:
                break

            # get first tree from next tree_sequence
            newick = next(msgen).first().newick()

            # simulate first base
            seed = self.random_mut.randint(1e9)    
            seq = mkseq.feed_tree(newick, 1, self.mut, seed)

            # if repeat_on_trees then keep sim'n til we get a SNP
            if repeat_on_trees:
                # if not variable
                while np.all(seq == seq[0]):
                    seed = self.random_mut.randint(1e9)    
                    seq = mkseq.feed_tree(newick, 1, self.mut, seed)

            # otherwise just move on to the next generated tree
            else:
                if np.all(seq == seq[0]):
                    continue

            # reset .names on msprime tree with node_labels 1-indexed
            gtree = toytree.tree(newick)
            for node in gtree.treenode.get_leaves():
                node.name = self.tipdict[int(node.name)]
            newick = gtree.write(tree_format=5)

            # reorder SNPs to be alphanumeric nameordered by tipnames 
            seq = seq[self.order, :]

            # Store result and advance counter
            snparr[:, snpidx] = seq.flatten()
            snpidx += 1
            newicks.append(newick)

        # close subprocess
        mkseq.close()

        # init dataframe
        self.df = pd.DataFrame({
            "start": 0,
            "end": 1,
            "genealogy": newicks,
            "nbps": 1, 
            "nsnps": 1,
            "locus": range(nsnps),
            },
            columns=['locus', 'start', 'end', 'nbps', 'nsnps', 'genealogy'],
        )
        self.seqs = snparr



    def write_loci_to_phylip(
        self, 
        outdir="./ipcoal-sims/", 
        idxs=None, 
        name_prefix=None, 
        name_suffix=None,
        ):
        """
        Write all seq data for each locus to a separate phylip file in a shared
        directory with each locus named by ids locus index. 

        Parameters:
        -----------
        outdir (str):
            A directory in which to write all the phylip files. It will be 
            created if it does not yet exist. Default is "./ipcoal_loci/".
        outfile (str):
            Only used if idx is not None. Set the name of the locus file being
            written. This is used internally to write tmpfiles for TreeInfer.
        idx (int):
            To write a single locus file provide the idx. If None then all loci
            are written to separate files.
        """
        writer = Writer(self.seqs, self.names)
        writer.write_loci_to_phylip(outdir, idxs, name_prefix, name_suffix)

        # report
        print("wrote {} loci ({} x {}bp) to {}/[...].phy".format(
            writer.written, self.seqs.shape[1], self.seqs.shape[2],
            writer.outdir.strip("/")
            ),
        )



    def write_concat_to_phylip(
        self, 
        outdir="./",
        name="test.phy",
        idxs=None,
        ):
        """
        Write all seq data (loci or snps) concated to a single phylip file.

        Parameters:
        -----------
        outfile (str):
            The name/path of the outfile to write. Default is "./test.phy"
        """       
        writer = Writer(self.seqs, self.names)
        writer.write_concat_to_phylip(outdir, name, idxs)

        # report 
        print(
            "wrote concatenated loci ({} x {}bp) to {}"
            .format(writer.shape[0], writer.shape[1], writer.outfile),
            )



    def infer_gene_trees(self, inference_method='raxml', inference_args={}):
        """
        Infer gene trees at every locus using the sequence in the locus 
        interval. 

        Parameters
        ----------
        method (str):
            options include "iqtree", "raxml", "mrbayes".
        kwargs (dict):
            a limited set of supported inference options. See docs.
        """

        # bail out if the data is only unlinked SNPs
        if self.df.nbps.max() == 1:
            raise ipcoalError(
                "gene tree inference cannot be performed on individual SNPs\n"
                "perhaps you meant to run .sim_loci() instead of .sim_snps()."
                )

        # expand self.df to include an inferred_trees column
        self.df["inferred_tree"] = np.nan  # or should we use ""?

        # init the TreeInference object (similar to ipyrad inference code)
        ti = TreeInfer(
            self, 
            inference_method=inference_method, 
            inference_args=inference_args,
        )

        # iterate over nloci. This part could be easily parallelized...
        for lidx in range(self.seqs.shape[0]):

            # skip invariable loci
            if self.df.nsnps[lidx]:

                # let low data fails return NaN
                try:
                    # write data to a temp phylip or nexus file
                    tree = ti.run(lidx)

                    # enter result
                    self.df.loc[self.df.locus == lidx, "inferred_tree"] = tree

                # caught raxml exception (prob. low data)
                # except ipcoalError:
                   # pass
                except ipcoalError as err:
                    # self.df.loc[self.df.locus == lidx, "inferred_tree"] = np.nan
                    # print(err)
                    raise err
