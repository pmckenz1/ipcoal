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
import toyplot
import toytree
import numpy as np
import pandas as pd
import msprime as ms
import os
import glob
import re
import subprocess
import tempfile
import itertools as itt
from scipy.special import comb
from copy import deepcopy

from .jitted import count_matrix_int, base_to_int
from .utils import get_all_admix_edges, SimcatError
from .SeqModel import SeqModel


class Model:
    """
    A coalescent model for returning ms simulations.
    """
    def __init__(
        self,
        tree,
        admixture_edges=None,
        admixture_type=0,
        Ne=10000,
        recomb=1e-9,
        mut=1e-8,
        nreps=1,
        seed=None,
        debug=False,
        run=False,
        ):

        """
        An object used for demonstration and testing only. The real simulations
        use the similar object Simulator.

        Takes an input topology with edge lengths in coalescent units (2N)
        entered as either a newick string or as a Toytree object,
        and generates 'ntests' parameter sets for running msprime simulations
        which are stored in the the '.test_values' dictionary. The .run()
        command can be used to execute simulations to fill count matrices
        stored in .counts. Admixture events (intervals or pulses) from source
        to dest are described as viewed backwards in time.

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

        theta (float, tuple):
            Mutation parameter. Enter a float, or a tuple of floats to supply
            a range to sample from over ntests. If None then values will be
            extracted from the Toytree if it has a 'theta' feature on each
            internal node. Else an errror will be raise if no thetas found.

        nsnps (int):
            Number of unlinked SNPs simulated (e.g., counts is (nsnps, 16, 16))

        ntests (int):
            Number of parameter sets to sample for each event, i.e., given
            a theta range and admixture events range multiple sets of parameter
            values could be sampled. The counts array is expanded to be
            (ntests, nsnps, 16, 16)

        nreps (int):
            Number of technical replicates to run using the same param sets.
            The counts array is expanded to be (nreps * ntests, nsnps, 16, 16)

        seed (int):
            Random number generator for numpy.
        """
        # init random seed: all np and ms random draws proceed from this.
        # TODO: still not working...

        self.newicklist = []

        self.random = np.random.RandomState()

        # hidden argument to turn on debugging
        self._debug = debug

        # parse the input tree
        if isinstance(tree, toytree.Toytree.ToyTree):
            self.treeorig = tree
            self.tree = deepcopy(self.treeorig)
        elif isinstance(tree, str):
            self.treeorig = toytree.tree(tree)
            self.tree = deepcopy(self.treeorig)
        else:
            raise TypeError("input tree must be newick str or Toytree object")
        self.ntips = len(self.tree)

        if Ne:
            for node in tree.treenode.traverse():
                node.add_feature('Ne', Ne)

        # store sim params: fixed mut, Ne, recomb
        self.mut = mut
        self.recomb = recomb
        if Ne:
            self.Ne = Ne
        else:
            self.Ne = None

        # storage for output
        self.nquarts = int(comb(N=self.ntips, k=4))  # scipy.special.comb

        # store node.name as node.idx, save old names in a dict.
        self.namedict = {}
        for node in self.tree.treenode.traverse():
            if node.is_leaf():
                # store old name and set new one
                self.namedict[node.idx] = node.name
                node.name = node.idx
        self.names = np.sort(list(self.namedict.values()))

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
        self.aedges = (0 if not self.admixture_edges else len(self.admixture_edges))

        # generate sim parameters from the tree and admixture scenarios
        # stores in self.sims 'mrates' and 'mtimes'
        self._get_test_values()

        # get demography as msprime input
        self.ms_demography = self._get_demography()

        # get popconfig as msprime input
        self.ms_popconfig = self._get_popconfig()

        # hold the model outputs
        self.df = None
        self.seqs = None

        # fill the counts matrix or call run later
        if run:
            self.run()

    def _get_test_values(self):
        """
        Generates mrates, mtimes, and thetas arrays for simulations.

        Migration times are uniformly sampled between start and end points that
        are constrained by the overlap in edge lengths, which is automatically
        inferred from 'get_all_admix_edges()'. migration rates are drawn
        uniformly between 0.0 and 0.5. thetas are drawn uniformly between
        theta0 and theta1, and Ne is just theta divided by a constant.

        self.test_values = {
            thetas: [1, 2, 0.2, .1, .5],
            1: {mrates: [.5, .2, .3], mtimes: [(2, 3), (4, 5), (1, 2)]},
            2: {mrates: [.01, .05,], mtimes: [(0.5, None), 0.1, None)]
            3: {...}
            ...
        }
        """

        self.test_values = {}

        # sample times and proportions/rates for admixture intervals
        idx = 0
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
            mrates = self.random.uniform(mr[0], mr[1], size=1)[0]

            # intervals are overlapping edges where admixture can occur.
            # lower and upper restrict the range along intervals for each
            snode = self.tree.treenode.search_nodes(idx=iedge[0])[0]
            dnode = self.tree.treenode.search_nodes(idx=iedge[1])[0]
            ival = intervals.get((snode.idx, dnode.idx))
            dist_ival = ival[1]-ival[0]
            # intervals mode
            if self.admixture_type:
                ui = self.random.uniform(ival[0]+mi[0]*dist_ival,
                                         ival[0]+mi[1]*dist_ival, 2)
                ui = ui.reshape((1, 2))
                mtimes = np.sort(ui, axis=1)
            # pulsed mode
            else:
                ui = self.random.uniform(ival[0]+mi[0]*dist_ival,
                                         ival[0]+mi[1]*dist_ival, 1)
                mtimes = int(ui[0])

            # store values only if migration is high enough to be detectable
            self.test_values[idx] = {
                "mrates": mrates,
                "mtimes": mtimes,
            }
            idx += 1

            # print info
            if self._debug:
                print("migration: edge({}->{}) time({:.3f}, {:.3f}), rate({:.3f}, {:.3f})"
                      .format(snode.idx, dnode.idx, ival[0], ival[1], mr[0], mr[1]),
                      file=sys.stderr)

    def _get_demography(self):
        """
        returns demography scenario based on an input tree and admixture
        edge list with events in the format (source, dest, start, end, rate).
        Time on the tree is defined in coalescent units, which here is
        converted to time in 2Ne generations as an int.
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
                demog.add(ms.PopulationParametersChange(time,
                          initial_size=node.Ne,
                          population=dest))
                if self._debug:
                    print('div time: {} {} {}'
                          .format(int(time), source, dest), file=sys.stderr)

        # Add migration pulses
        if not self.admixture_type:
            for evt in range(self.aedges):
                rate = self.test_values[evt]['mrates']
                time = int(self.test_values[evt]['mtimes'])
                source, dest = self.admixture_edges[evt][:2]

                # rename nodes at time of admix in case divergences renamed them
                snode = self.tree.treenode.search_nodes(idx=source)[0]
                dnode = self.tree.treenode.search_nodes(idx=dest)[0]
                children = (snode._schild, dnode._schild)
                demog.add(ms.MassMigration(time, children[0], children[1], rate))
                if self._debug:
                    print('mig pulse: {} ({:.3f}), {}, {}, {:.3f}'.format(
                        time, self._mtimes[evt][0], source, dest, rate),
                        file=sys.stderr)

        # Add migration intervals
        else:
            for evt in range(self.aedges):
                rate = self.test_values[evt]['mrates']
                time = (self.test_values[evt]['mtimes']).astype(int)
                source, dest = self.admixture_edges[evt][:2]

                # rename nodes at time of admix in case divergences renamed them
                snode = self.tree.treenode.search_nodes(idx=source)[0]
                dnode = self.tree.treenode.search_nodes(idx=dest)[0]
                children = (snode._schild, dnode._schild)
                demog.add(ms.MigrationRateChange(time[0], rate, children))
                demog.add(ms.MigrationRateChange(time[1], 0, children))
                if self._debug:
                    print("mig interv: {}, {}, {}, {}, {:.3f}".format(
                        time[0], time[1], children[0], children[1], rate),
                        file=sys.stderr)

        # sort events by type (so that mass migrations come before pop size changes) and time
        demog = sorted(sorted(list(demog), key=lambda x: x.type), key=lambda x: x.time)
        return demog

    def _get_popconfig(self):
        """
        returns population_configurations for N tips of a tree
        """
        Ne_vals = []
        for node in self.tree.treenode.traverse():
            if node.is_leaf():
                Ne_vals.append(node.Ne)
        inv_Ne_vals = Ne_vals[::-1]  # this is so they're added to the tree in the right order...
        population_configurations = [
            ms.PopulationConfiguration(sample_size=1,
                                       initial_size=inv_Ne_vals[ntip])
            for ntip in range(self.ntips)]

        return population_configurations

    def _get_SNP_sims(self, nsnps):
        """
        Performs simulations with params varied across input values.
        """
        # migration scenarios from admixture_edges, used in demography
        migmat = np.zeros((self.ntips, self.ntips), dtype=int).tolist()
        self._mtimes = [
            self.test_values[evt]['mtimes'] for evt in
            range(len(self.admixture_edges))
        ]
        self._mrates = [
            self.test_values[evt]['mrates'] for evt in
            range(len(self.admixture_edges))
        ]

        # debug printer
        if self._debug:
            print("pop: Ne:{}, mut:{:.2E}"
                  .format(self.Ne, self.mut),
                  file=sys.stderr)

        # msprime simulation to make tree_sequence generator
        sim = ms.simulate(
            random_seed=self.random.randint(1e9),
            migration_matrix=migmat,
            num_replicates=nsnps * 10000,                 # ensures SNPs
            population_configurations=self._get_popconfig(),  # applies Ne
            demographic_events=self._get_demography(),        # applies popst.
        )
        return sim

    def _get_locus_sim(self, locus_len):
        """
        Performs simulations with params varied across input values.
        """
        # migration scenarios from admixture_edges, used in demography
        migmat = np.zeros((self.ntips, self.ntips), dtype=int).tolist()
        self._mtimes = [
            self.test_values[evt]['mtimes'] for evt in
            range(len(self.admixture_edges))
        ]
        self._mrates = [
            self.test_values[evt]['mrates'] for evt in
            range(len(self.admixture_edges))
        ]

        # debug printer
        if self._debug:
            print("pop: Ne:{}, mut:{:.2E}"
                  .format(self.Ne, self.mut),
                  file=sys.stderr)

        # msprime simulation to make tree_sequence generator
        sim = ms.simulate(
            length=locus_len,
            random_seed=self.random.randint(1e9),
            recombination_rate=self.recomb,
            migration_matrix=migmat,
            num_replicates=1,                 # ensures SNPs
            population_configurations=self._get_popconfig(),  # applies Ne
            demographic_events=self._get_demography(),        # applies popst.
        )
        return sim

    def run_locus(self,
                  size,
                  seqgen=True,
                  locus_idx=0):

        msinst = self._get_locus_sim(size)
        msmod = next(msinst)
        breaks = []
        for breakpts in msmod.breakpoints():
            breaks.append(breakpts)

        starts = breaks[0:(len(breaks)-1)]
        stops = breaks[1:len(breaks)]
        bps = (np.round(stops)-np.round(starts)).astype(int)

        trees = msmod.trees()
        newicks = []
        for atree in trees:
            newicks.append(atree.newick(node_labels=dict(zip([i for i in atree.leaves()], [self.namedict[i] for i in atree.leaves()]))))

        if not seqgen:
            sm = SeqModel()

        seqlist = []
        for num in range(len(newicks)):
            # get each gene tree
            gtree = toytree.tree(newicks[num])

            # get the number of base pairs taken up by this gene tree
            gtlen = bps[num]
            if gtlen:
                if seqgen:
                    seqdict = self.seqgen_on_tree(newick=gtree.write(tree_format=5),
                                                  seqlength=gtlen)
                else:
                    # simulate the sequence for the gene tree
                    seqdict = sm.run(ttree=gtree,
                                     seq_length=gtlen,
                                     return_leaves=True)

                seqlist.append(seqdict)

        # concatenate the gene tree sequences together
        dnadict = {}
        for key in seqlist[0].keys():
            dnadict.update({key: np.concatenate([i[key] for i in seqlist])})

        df = pd.DataFrame({"locus_idx": np.repeat(locus_idx, len(starts)),
                           "starts": starts,
                           "stops": stops,
                           "genealogies": newicks,
                           "bps": bps},
                          columns=['locus_idx',
                                   'starts',
                                   'stops',
                                   'bps',
                                   'genealogies'])
        return([df, dnadict])

    def run(self,
            num_loci,
            size,
            outfile=None,
            seqgen=True,
            force=False):

        loci_list = []
        seq_list = []
        res_arr = np.zeros((num_loci, self.ntips, size), dtype=np.int8)
        for locusrun in range(num_loci):
            gt_df, seqs = self.run_locus(size=size,
                                         seqgen=seqgen,
                                         locus_idx=locusrun)

            seq_list.append(seqs)
            for keyidx, key in enumerate(self.names):
                res_arr[locusrun, keyidx, :] = seqs[key]

            # count the number of snps in this locus
            _nsnps = 0
            for column in res_arr[locusrun].T:
                if len(np.unique(column)) > 1:
                    _nsnps += 1

            gt_df['nsnps'] = np.repeat(_nsnps, len(gt_df))
            loci_list.append(gt_df)

        loci_result = pd.concat(loci_list)

        cumulative_bps = 0
        cumulative_list = []
        for i in loci_result['bps']:
            cumulative_bps += i
            cumulative_list.append(cumulative_bps)
        loci_result['cumulative_bps'] = cumulative_list
        loci_result['inferred_trees'] = np.repeat(None, len(cumulative_list))

        # reindex
        loci_result.set_index(pd.Series(range(len(loci_result))))

        self.df = loci_result
        self.seqs = res_arr

    def plot_test_values(self):

        """
        Returns a toyplot canvas
        """

        # canvas, axes = plot_test_values(self.tree)
        if not self.counts.sum():
            raise SimcatError("No mutations generated. First call '.run()'")

        # setup canvas
        canvas = toyplot.Canvas(height=250, width=800)

        ax0 = canvas.cartesian(
            grid=(1, 3, 0))
        ax1 = canvas.cartesian(
            grid=(1, 3, 1),
            xlabel="simulation index",
            ylabel="migration intervals",
            ymin=0,
            ymax=self.tree.treenode.height)  # * 2 * self._Ne)
        ax2 = canvas.cartesian(
            grid=(1, 3, 2),
            xlabel="proportion migrants",
            # xlabel="N migrants (M)",
            ylabel="frequency")

        # advance colors for different edges starting from 1
        colors = iter(toyplot.color.Palette())

        # draw tree
        self.tree.draw(
            tree_style='c',
            node_labels=self.tree.get_node_values("idx", 1, 1),
            tip_labels=False,
            axes=ax0,
            node_sizes=16,
            padding=50)
        ax0.show = False

        # iterate over edges
        for tidx in range(self.aedges):
            color = next(colors)

            # get values for the first admixture edge
            mtimes = self.test_values[tidx]["mtimes"]
            mrates = self.test_values[tidx]["mrates"]
            mt = mtimes[mtimes[:, 0].argsort()]
            boundaries = np.column_stack((mt[:, 0], mt[:, 1]))

            # plot
            for idx in range(boundaries.shape[0]):
                ax1.fill(
                    # boundaries[idx],
                    (boundaries[idx][0], boundaries[idx][0] + 0.1),
                    (idx, idx),
                    (idx + 0.5, idx + 0.5),
                    along='y',
                    color=color,
                    opacity=0.5)

            # migration rates/props
            ax2.bars(
                np.histogram(mrates, bins=20),
                color=color,
                opacity=0.5,
            )

        return canvas, (ax0, ax1, ax2)

    def seqgen_on_tree(self, newick, seqlength):
        fname = os.path.join(tempfile.gettempdir(), str(os.getpid()) + ".tmp")
        with open(fname, 'w') as temp:
            temp.write(newick)

        # write sequence data to a tempfile
        proc1 = subprocess.Popen([
            "seq-gen",
            "-m", "GTR",
            "-l", str(seqlength),  # seq length
            "-s", str(self.mut),  # mutation rate
            fname,
            # ... other model params...,
            ],
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
        )

        # check for errors
        out, _ = proc1.communicate()
        if proc1.returncode:
            raise Exception("seq-gen error: {}".format(out.decode()))

        # remove the "Time taken: 0.0000 seconds" bug in seq-gen
        physeq = re.sub(
            pattern=r"Time\s\w+:\s\d.\d+\s\w+\n",
            repl="",
            string=out.decode())

        # make seqs into array, sort it, and count differences
        physeq = physeq.strip().split("\n")[-(self.ntips + 1):]
        arr = np.array([list(i.split()[-2:]) for i in physeq[1:]], dtype=bytes)
        names = [arr_ele[0].astype(str) for arr_ele in arr]
        seqs = [arr_ele[1].astype(str) for arr_ele in arr]

        final_seqs = []
        for indv_seq in seqs:
            orig_arrseq = np.array([i for i in indv_seq])
            arrseq = np.zeros(orig_arrseq.shape, dtype=np.int8)
            arrseq[orig_arrseq == "A"] = 0
            arrseq[orig_arrseq == "C"] = 1
            arrseq[orig_arrseq == "G"] = 2
            arrseq[orig_arrseq == "T"] = 3
            final_seqs.append(arrseq)

        return(dict(zip(names, final_seqs)))

    def write_fasta(self, loc, path):
        fastaseq = deepcopy(self.seqs[loc]).astype(str)

        fastaseq[fastaseq == '0'] = "A"
        fastaseq[fastaseq == '1'] = "C"
        fastaseq[fastaseq == '2'] = "G"
        fastaseq[fastaseq == '3'] = "T"

        fasta = []
        for idx, name in enumerate(self.names):
            fasta.append(('>' + name + '\n'))
            fasta.append("".join(fastaseq[idx])+'\n')

        with open(path, 'w') as file:
            for line in fasta:
                file.write(line)

    def _call_iq(self, command_list):
        """ call the command as sps """
        proc = subprocess.Popen(
            command_list,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE
            )
        proc.communicate()
        # return comm[0].decode()

    def infer_trees(self, method='iqtree'):
        for seqnum in range(len(self.seqs)):
            fastapath = "tempfile.fasta"
            self.write_fasta(seqnum, fastapath)

            self._call_iq(['iqtree',
                           '-s', fastapath,
                           '-m', 'MFP',
                           '-bb', '1000'])
            with open(fastapath+".treefile", 'r') as treefile:
                newick = treefile.read()
            self.df.loc[(self.df['locus_idx'] == seqnum),
                        'inferred_trees'] = newick
            self.newicklist.append(newick)
            for filename in glob.glob(fastapath+"*"):
                os.remove(filename)

    def _run_snps(self, nsnps):

        # temporarily format these as stacked matrices
        tmpcounts = np.zeros((self.nquarts, 16, 16), dtype=np.int64)

        # get tree_sequence for this set
        sims = self._get_SNP_sims(nsnps)

        # store results (nsnps, ntips); def. 1000 SNPs
        snparr = np.zeros((nsnps, self.ntips), dtype=np.int64)

        # continue until all SNPs are sampled from generator
        n_counted_snps = 0

        countem = 0
        while n_counted_snps < nsnps:
            try:
                newtree = next(next(sims).trees()).newick()

                filename = str(np.random.randint(1e10)) + '.newick'
                with open(filename, 'w') as f:
                    f.write(str(newtree))

                process = subprocess.Popen(['seq-gen',
                                            '-m', 'GTR',
                                            '-l', '1',
                                            '-s', str(self.mut),
                                            filename,
                                            '-or',
                                            '-q'],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()

                result = stdout.decode("utf-8").split('\n')[:-1]
                geno = dict([i.split(' ') for i in result[1:]])

                ordered = [geno[np.str(i)] for i in range(1, len(geno)+1)]

                if len(np.unique(ordered)) > 1:
                    snparr[n_counted_snps] = base_to_int(ordered)
                    n_counted_snps += 1

                    countem += 1
                if os.path.isfile(filename):
                    os.remove(filename)
                else:  # Show an error
                    print("Error: %s file not found" % filename)
            except:
                countem += 1
                pass

        # iterator for quartets, e.g., (0, 1, 2, 3), (0, 1, 2, 4)...
        quartidx = 0
        qiter = itt.combinations(range(self.ntips), 4)
        for currquart in qiter:
            # cols indices match tip labels b/c we named tips node.idx
            quartsnps = snparr[:, currquart]
            # save as stacked matrices
            tmpcounts[quartidx] = count_matrix_int(quartsnps)
            # save flattened to counts
            quartidx += 1
        return(np.ravel(tmpcounts))
