#!/usr/bin/env python

"""
Generate large database of site counts from coalescent simulations
based on msprime + toytree for using in machine learning algorithms.
"""

from typing import List, Tuple, Optional, Dict
import os
import numpy as np
import pandas as pd
import msprime as ms
import toytree
from loguru import logger

from ipcoal.markov.SeqModel import SeqModel
from ipcoal.phylo.TreeInfer import TreeInfer
from ipcoal.io.writer import Writer
from ipcoal.utils.utils import get_all_admix_edges, IpcoalError
from ipcoal.draw.seqview import draw_seqview
from ipcoal.utils.utils import calculate_pairwise_dist
# from .SeqGen import SeqGen


# set global display preference to make tree columns look nice
pd.set_option("max_colwidth", 28)


class Model:
    """
    An ipcoal.Model object for defining demographic models for coalesent 
    simulation in msprime. 

    Takes an input topology with edge lengths in units of generations
    entered as either a newick string or as a Toytree object, and defines
    a demographic model based on Ne (or Ne mapped to tree nodes) and 
    admixture edge arguments. Genealogies and sequence data is then 
    generated with msprime and seq-gen, respectively.

    Parameters:
    -----------
    tree: (str)
        A newick string or Toytree object of a species tree with edges in
        coalescent units. Default is an empty string ("") which means no 
        species tree and thus a single panmictic population coalescent.

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
        The per-site per-generation mutation rate

    recomb (float): default=1e-9
        The per-site per-generation recombination rate.

    recomb_map (DataFrame): default=None
        A recombination map in hapmap format as a pandas dataframe.
        Columns should be as desired by msprime.

        Example:
        Chromosome\tPosition\tRate\tMap\n
        chr1\t0\t0\t0\n
        chr1\t55550\t2.981822\t0.000000

    seed (int):
        Random number generator used for msprime (and seqgen unless a 
        separate seed is set for seed_mutations.

    nsamples (int or list):
        An integer for the number of samples from each lineage, or a list
        of the number of samples from each lineage ordered by the tip
        order of the tree when plotted.

    seed_mutations (int):
        Random number generator used for seq-gen. If not set then the
        generic seed is used for both msprime and seq-gen.

    substitution_model (dict):
        A dictionary of arguments to the markov process mutation model.
        example:
        substitution_model = {
            state_frequencies=[0.25, 0.25, 0.25, 0.25],
            kappa=3,
            gamma=4,
        }
    """
    def __init__(
        self,
        tree: Optional[toytree.ToyTree]=None,
        Ne: Optional[int]=None,   
        admixture_edges: Optional[List[Tuple[int,int,float,float]]]=None,
        admixture_type:int=0,  # FIXME: this could be dropped since we can infer interval if admixture_edge uses a tuple instead of float for the edge min,max
        nsamples: int=1,
        mut: float=1e-8,
        recomb: float=1e-9,
        recomb_map=None,
        seed: Optional[int]=None,
        seed_mutations: Optional[int]=None,
        substitution_model: str=None,
        **kwargs,
        ):

        # legacy support warning messages
        self._warn_bad_kwargs(kwargs)

        # initialize random seed for msprime and SeqModel
        self._init_seed = seed
        self._init_mseed = seed_mutations
        self.rng_trees: np.random.Generator = None
        self.rng_muts: np.random.Generator = None
        self._reset_random_generators()

        # parse input tree, store orig, resolv poly if needed to get copy
        self.treeorig = toytree.tree(tree)
        self.tree = self.treeorig.resolve_polytomy(dist=0.00001)
        self.ntips: int = self.tree.ntips
        self.nsamples: List[int] = None
        self.nstips: int = None
        # expand nsamples to ordered list, e.g., [2, 1, 1, 2, 10, 10]
        self._set_samples(nsamples)

        # store sim params: fixed mut, Ne, recomb
        self.mut = mut
        self.recomb = recomb
        self.recomb_map = (
            None if recomb_map is None 
            else ms.RecombinationMap(
                list(recomb_map['position']), 
                list(recomb_map['recomb_rate'])
            )
        )

        # global Ne will be overwritten by Ne attrs in .tree. This sets node.Ne
        self.neff: int = None
        self._set_neff(Ne)

        # store tip names for renaming on the ms tree (ntips * nsamples)
        self.tipdict: dict = None
        self.sampledict: dict = None
        self.tip_to_heights: dict = None
        self._tips_are_ultrametric: bool = None
        self.alpha_ordered_names: List[str] = None
        self._order: Dict[str,int] = None
        self._get_tip_names_for_renaming()

        # check formats of admixture args
        self.admixture_edges: List[Tuple(int,int,float,float)] = None
        self.admixture_type: int = 1 if admixture_type in (1, "interval") else 0
        self.aedges: int = None
        self._get_admixture_edges(admixture_edges)

        # demography info to fill (UPDATED FOR MSPRIME VERSION X)
        self.ms_migrate = []
        self.ms_migtime = []
        self.ms_demography = set()
        self.ms_popconfig = ms.PopulationConfiguration()

        # get migration time, rate {mrates: [], mtimes: []}
        self._get_migration()

        # get .ms_demography dict for msprime input
        self._get_demography()

        # get .ms_popconfig as msprime input
        self._get_popconfig()

        # this is used when tips are not ultrametric
        self._get_nsamples()

        # to hold the model outputs
        self.df = None
        self.seqs = None
        self.ancestral_seq = None

        # check substitution model kwargs and assert it is a dict
        self.substitution_model = substitution_model
        self._check_substitution_kwargs()

    @staticmethod
    def _warn_bad_kwargs(kwargs):
        """
        Warn user that args are being skipped
        """
        if kwargs:
            logger.warning(
                f"Parameters {list(kwargs)} are not supported. See "
                "documentation, argument names may have changed."
            )

    def _reset_random_generators(self):
        """
        Called after sim_trees(), sim_snps() or sim_loci to return all seeds
        to their state during init so that an init'd mod object that was init'd
        with some random seed will always return the same results even if 
        run multiple times.
        """
        self.rng_trees = np.random.default_rng(self._init_seed)
        self.rng_muts = (
            self.rng_trees if not self._init_mseed
            else np.random.default_rng(self._init_mseed)
        )

    def _set_samples(self, nsamples):
        """
        Get a list with number of sampled gene copies per tip species.
        """
        if isinstance(nsamples, int):
            self.nsamples = [int(nsamples) for i in range(self.ntips)]
            self.nstips = int(nsamples) * self.ntips
        else:
            assert isinstance(nsamples, (list, tuple)), (
                "nsamples should be a list")
            assert len(nsamples) == self.ntips, (
                "nsamples list should be same length as ntips in tree.")
            self.nsamples = nsamples
            self.nstips = sum(self.nsamples)

    def _get_tip_names_for_renaming(self):
        """
        Fills tipdict and sampledict for renaming the genealogy tips
        to match the species tree names if a single sample per tip, 
        or to match with additional integers (A-0, A-1) if multiple 
        samples per tip.
        """
        _tlabels = self.tree.get_tip_labels()
        if all([i == 1 for i in self.nsamples]):
            self.tipdict = dict(enumerate(_tlabels))
            self.sampledict = dict(zip(_tlabels, self.nsamples))
        else:
            self.tipdict = {}
            self.sampledict = {}
            idx = 0
            for tip, nsamp in zip(_tlabels, self.nsamples):
                for nidx in range(nsamp):
                    self.tipdict[idx] = "{}-{}".format(tip, nidx)
                    idx += 1
                self.sampledict[tip] = nsamp

        # get map of tips to heights for trees with generation time 
        # differences such that the tree is not ultrametric. Also store
        # a boolean for whether or not tips are ultrametric.x
        self.tip_to_heights = self.tree.get_feature_dict("name", "height")
        self.tip_to_heights = {
            i: j for (i, j) in self.tip_to_heights.items() if i in _tlabels}
        self._tips_are_ultrametric: bool = (
            len(set(self.tip_to_heights.values())) == 1)
        logger.debug(f"tree is ultrametric: {self._tips_are_ultrametric}")

        # alphanumeric ordered tip names -- order of printing to seq files
        self.alpha_ordered_names = sorted(self.tipdict.values())

        # for reordering seq array (in 1-indexed tip order) to alpha tipnames
        self._order = {
            i: self.alpha_ordered_names.index(j) for (i, j) 
            in self.tipdict.items()
        }

    def _get_admixture_edges(self, admixture_edges):
        """
        Parse admixture_edges list of tuples to check proper types.
        """
        if admixture_edges is None:
            self.admixture_edges = []
        else:
            if not isinstance(admixture_edges[0], (list, tuple)):
                raise TypeError("admixture_edges should be a list of tuples.")
            if isinstance(admixture_edges, tuple):
                admixture_edges = [admixture_edges]
            for edge in self.admixture_edges:
                if len(edge) != 4:
                    raise ValueError(
                        "admixture edges should each be a tuple with 4 values")
            self.admixture_edges = admixture_edges
        # store number of edges
        self.aedges = (
            0 if not self.admixture_edges else len(self.admixture_edges))

    def _check_substitution_kwargs(self):
        """
        check that user supplied substitution kwargs make sense.
        """
        # must be a dictionary
        if self.substitution_model is None:
            self.substitution_model = {}

        # check each item
        for key in self.substitution_model:

            # do not allow typos or unsupported params
            supp = ["state_frequencies", "kappa", "gamma", "gamma_categories"]
            if key not in supp:
                raise TypeError(
                    "substitution_model param {} not currently supported."
                    .format(key))

            # check that state_frequencies sum to 1
            if self.substitution_model.get("state_frequencies"):
                fsum = sum(self.substitution_model['state_frequencies'])
                if not fsum == 1:
                    raise IpcoalError("state_frequencies must sum to 1.0")

            # check that kappa in the range...
            pass

    def _set_neff(self, Ne):
        """
        If an Ne argument was entered then it overrides any setting
        on the TreeNodes, else check that all nodes in the tree have 
        an Ne setting, or raise an error.
        """
        if Ne is None:
            node_neffs = self.tree.get_node_values("Ne")
            if not all(node_neffs):
                msg = (
                    "if Ne=None then Ne values must be set to all "
                    "nodes of the input tree (e.g., using .set_node_values()."
                )
                logger.error(msg)
                raise IpcoalError(msg)
            self.neff = max(node_neffs)

        else:
            self.neff = Ne
            self.tree = self.tree.set_node_values("Ne", default=Ne)

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
            mrate = self.rng_trees.uniform(mr[0], mr[1])

            # intervals are overlapping edges where admixture can occur.
            # lower and upper restrict the range along intervals for each
            snode = self.tree.treenode.search_nodes(idx=iedge[0])[0]
            dnode = self.tree.treenode.search_nodes(idx=iedge[1])[0]
            ival = intervals.get((snode.idx, dnode.idx))
            dist_ival = ival[1] - ival[0]

            # intervals mode
            if self.admixture_type:
                ui = self.rng_trees.uniform(
                    ival[0] + mi[0] * dist_ival,
                    ival[0] + mi[1] * dist_ival, 2,
                )
                ui = ui.reshape((1, 2))
                mtime = np.sort(ui, axis=1).astype(int)

            # pulsed mode
            else:
                ui = self.rng_trees.uniform(
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
        demog = list()

        # tag min index child for each node, since at the time the node is
        # called it may already be renamed by its child index b/c of
        # divergence events.
        for node in self.tree.treenode.traverse():
            if node.children:
                node._schild = min([i.idx for i in node.get_descendants()])
            else:
                node._schild = node.idx

        # traverse tree from root to tips
        for node in self.tree.treenode.traverse():

            # if children add div events
            if node.children:
                dest = min([i._schild for i in node.children])
                source = max([i._schild for i in node.children])
                time = node.height  # int(node.height)
                demog.append(ms.MassMigration(time, source, dest))

                # for all nodes set Ne changes
                demog.append(ms.PopulationParametersChange(
                    time,
                    initial_size=node.Ne,
                    population=dest),
                )

            # tips set populations sizes (popconfig seemings does this too,
            # but it didn't actually work for tips until I added this...
            else:
                time = node.height  # int(node.height)
                demog.append(ms.PopulationParametersChange(
                    time,
                    initial_size=node.Ne,
                    population=node.idx,
                ))

            # debugging helper
            if node.children:
                logger.debug((
                    f"div time: {time:>8.0f}, ({source:>2} {dest:>2}), "
                    f"({node.children[0].idx:>2} {node.children[1].idx:>2}), "
                    f"Ne={node.Ne:.0f}, "
                    f"θ={4*node.Ne*self.mut:.5f}, τ={node.dist / (2 * node.Ne):.2f}"
                ))

        # Add migration pulses
        if not self.admixture_type:
            for evt in range(self.aedges):

                # rate is prop. of population, time is prop. of edge
                rate = self.ms_migrate[evt]
                time = self.ms_migtime[evt]
                source, dest = self.admixture_edges[evt][:2]

                # rename nodes at time of admix in case diverge renamed them
                snode = self.tree.idx_dict[source]
                dnode = self.tree.idx_dict[dest]
                children = (snode._schild, dnode._schild)

                demog.append(
                    ms.MassMigration(time, children[0], children[1], rate))

                logger.debug(
                    f"mig pulse: {time:>9}, {'':>2} {'':>2}, "
                    f"{snode.name:>2} {dnode.name:>2}, rate={rate:.2f}"
                )

        # Add migration intervals
        else:
            raise NotImplementedError("interval migration currently deprecated...")
            for evt in range(self.aedges):
                rate = self.ms_migration[evt]['mrates']
                time = (self.ms_migration[evt]['mtimes']).astype(int)
                source, dest = self.admixture_edges[evt][:2]

                # rename nodes at time of admix in case diverg renamed them
                snode = self.tree.treenode.search_nodes(idx=source)[0]
                dnode = self.tree.treenode.search_nodes(idx=dest)[0]
                children = (snode._schild, dnode._schild)
                demog.append(ms.MigrationRateChange(time[0], rate, children))
                demog.append(ms.MigrationRateChange(time[1], 0, children))

                logger.debug(
                    f"mig interv: {time[0]:>9}, {time[1]:>9} "
                    f"{children[0]:>2} {children[1]:>2} {rate:.2f}"
                    f"{snode.name:>2} {dnode.name:>2}, rate={rate:.2f}"
                )

        # sort events by type (so that mass migrations come before pop size
        # changes) and time
        if ms.__version__ < "1":
            demog = sorted(demog, key=lambda x: x.type)
        else:
            demog = sorted(demog, key=lambda x: x._type_str)
        demog = sorted(demog, key=lambda x: x.time)
        self.ms_demography = demog

    def _get_nsamples(self):
        """
        If tips are not ultrametric then individuals must be entered to 
        sim using the samples=[ms.Sample(popname, time), ...] format. If 
        tips are ultrametric then this should be empty (None).
        """
        # set to None and return
        if self._tips_are_ultrametric:
            self._samples = None
            return

        # create a list of sample tuples: [(popname, time), ...]
        self._samples = []

        # iterate over all sampled tips
        for otip, tip in enumerate(self.tree.get_tip_labels()):

            # get height of this tip
            height = int(self.tip_to_heights[tip])
            nsamples = self.sampledict[tip]

            # add for each nsamples
            for _ in range(nsamples):
                self._samples.append(ms.Sample(otip, height))

    def _get_popconfig(self):
        """
        Sets self.ms_popconfig as a list of msprime objects.

        In the strange case that tips are not ultrametric
        then we need to return a list of empty Popconfig objects.
        """
        # set to list of empty pops and return
        if not self._tips_are_ultrametric:
            self.ms_popconfig = [
                ms.PopulationConfiguration() for i in range(self.ntips)
            ]
            return 

        # pull Ne values from the toytree nodes attrs.
        if self.neff is None:
            # get Ne values from tips of the tree
            nes = self.tree.get_node_values("Ne")
            nes = nes[-self.tree.ntips:][::-1]

            # list of popconfig objects for each tip
            population_configurations = [
                ms.PopulationConfiguration(
                    sample_size=self.nsamples[i], initial_size=nes[i])
                for i in range(self.ntips)
            ]

        # set user-provided Ne value to all edges of the tree
        else:
            population_configurations = [
                ms.PopulationConfiguration(
                    sample_size=self.nsamples[i], initial_size=self.neff)
                for i in range(self.ntips)]

        # debug printer
        logger.debug(f"pop: Ne:{self.neff:.0f}, mut:{self.mut:.2E}")         
        self.ms_popconfig = population_configurations


    # ----------------------------------------------------------------
    # end init methods
    # ----------------------------------------------------------------

    def draw_seqview(
        self, 
        idx: Optional[int]=None, 
        start: Optional[int]=None, 
        end: Optional[int]=None, 
        width: Optional[int]=None, 
        height: Optional[int]=None, 
        show_text: bool=False, 
        **kwargs,
        ) -> ('toyplot.Canvas', 'toyplot.Table'):
        """
        Returns a (Canvas, Table) tuple as a drawing of the sequence 
        array.

        Parameters
        ----------
        idx: (int)
            The locus index of a locus to draw. If None and multiple 
            loci are present then it draws the first locus. If SNPs 
            were simulated then all SNPs are concatenated into a 
            single 'locus'.
        start: (int)
            Slice start position of sequence array to draw. Default=0.
        end: (int)
            Slice end position of sequence array to draw. Default=end.
        width: (int)
            Width of the canvas drawing in pixels. Default: auto
        height: (int)
            Height of the canvas drawing in pixels. Deafult: auto
        show_text: (bool)
            Whether to show base labels on table cells.
        kwargs: (dict)
            Additional drawing arguments to toyplot table.
        """
        canvas, table = draw_seqview(
            self, idx, start, end, width, height, show_text, **kwargs)
        return canvas, table


    def draw_genealogy(self, idx: Optional[int]=None, **kwargs):
        """
        Returns a (Canvas, Axes) tuple as a drawing of the genealogy.

        Parameters
        ----------
        idx: (int)
            The index of the genealogy to draw from the (Model.df) dataframe.
        """
        tree = toytree.tree(self.df.genealogy[idx])
        canvas, axes, mark = tree.draw(ts='c', tip_labels=True, **kwargs)
        return canvas, axes, mark


    def draw_genealogies(self, idxs: Optional[List[int]]=None, **kwargs):
        """
        Returns a (Canvas, Axes) tuple as a drawing of the genealogy.

        Parameters
        ----------
        idx: (None or list of ints)
            The index of the genealogies to draw from the (Model.df)
            dataframe.
        """
        if idxs is None:
            idxs = range(4)
        mtre = toytree.mtree(self.df.genealogy[idxs])
        canvas, axes, mark = mtre.draw(ts='c', tip_labels=True, **kwargs)
        return canvas, axes, mark


    def draw_sptree(self, **kwargs):
        """
        Returns a 
        """
        # tree = toytree.tree(self.df.genealogy[idx])
        admix = [i[:2] for i in self.admixture_edges]
        canvas, axes, mark = self.tree.draw(
            ts='p', 
            tip_labels=True, 
            admixture_edges=admix,
            **kwargs)
        return canvas, axes, mark


    def draw_demography(self, idx=None, spacer=0.25, ymax=None, **kwargs):
        """
        ...
        """
        # bail out if self.admixture_edges
        if self.admixture_edges:
            raise NotImplementedError(
                "admixture_edges are not yet supported in demography drawing.")

        # return only the container
        if idx is None:
            ctre = toytree.container(self.tree, idx=0, spacer=spacer, **kwargs)

        # return genealogy within container
        else:
            ctre = toytree.container(self, idx=idx, spacer=spacer, **kwargs)

        # scale ymax to not require all coalescences
        if ymax is None:
            ctre.axes.y.domain.max = self.tree.treenode.height
        elif isinstance(ymax, (int, float)):
            ctre.axes.y.domain.max = ymax
        return ctre.canvas, ctre.axes


    def get_substitution_model_summary(self):
        """
        Returns a summary of the demographic model and sequence substitution
        model that will used in simulations.
        TODO: make this a method of SeqModel.
        """
        # init a seqmodel to get calculated matrices
        seqmodel = SeqModel(**self.substitution_model)

        # print state frequencies
        print(
            "state_frequencies:\n{}"
            .format(
                pd.DataFrame(
                    [seqmodel.state_frequencies],
                    columns=list("ACGT"),
                ).to_string(index=False))
            )

        # the tstv parameters
        print("\nkappa: {}".format(seqmodel.kappa))
        print("ts/tv: {}".format(seqmodel.tstv))

        # the Q matrix 
        print(
            "\ninstantaneous transition rate matrix:\n{}".
            format(
                pd.DataFrame(
                    seqmodel.Q, 
                    index=list("ACGT"), 
                    columns=list("ACGT"),
                ).round(4)
            ))

        # the alpha and gamma ...
        if seqmodel.gamma:
            if seqmodel.gamma_categories:
                gtype = "({} discrete categories)".format(seqmodel.gamma_categories)
            else:
                gtype = "(continuous)"
            print(
                "\ngamma rate var. {} alpha: {}"
                .format(gtype, seqmodel.gamma)
            )


    # ----------------------------------------------------------------
    # MSPRIME simulation methods
    # ----------------------------------------------------------------


    def _get_tree_sequence_generator(self, nsites=1, snp=False):
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
            raise IpcoalError(
                "The flag snp=True should only be used with nsites=1")

        # migration scenarios from admixture_edges, used in demography
        migmat = np.zeros((self.ntips, self.ntips), dtype=int).tolist()

        # msprime simulation to make tree_sequence generator
        sim = ms.simulate(
            length=(None if self.recomb_map else nsites),
            random_seed=self.rng_trees.integers(1e9),
            recombination_rate=(None if snp or self.recomb_map else self.recomb),
            migration_matrix=migmat,
            num_replicates=(int(1e20) if snp else 1),        # ensures SNPs
            demographic_events=self.ms_demography,
            population_configurations=self.ms_popconfig,
            samples=self._samples,  # None if tips are ultrametric
            recombination_map=self.recomb_map,  # None unless specified
        )
        return sim



    def _sim_locus(self, nsites, locus_idx, mkseq):
        """
        Simulate tree sequence for each locus and sequence data for each
        genealogy and return all in a dataframe.
        """
        # get the msprime ts generator (np.random val is pulled here)
        msgen = self._get_tree_sequence_generator(nsites)

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
            "tidx": 0,
            },
            columns=[
                'locus', 'start', 'end', 'nbps', 
                'nsnps', 'tidx', 'genealogy',
            ],
        )

        # the full sequence array to fill
        bidx = 0
        seqarr = np.zeros((self.nstips, nsites), dtype=np.uint8)
        aseqarr = np.zeros((1, nsites), dtype=np.uint8)

        # iterate over the index of the dataframe to sim for each genealogy
        pseudoindex = 0
        for idx, mstree in zip(df.index, msts.trees()):

            # get the number of base pairs taken up by this gene tree
            gtlen = df.loc[idx, 'nbps']

            # only simulate data if there is bp 
            if gtlen:
                # write mstree to newick with original labels mapped on tips
                nwk = mstree.newick(node_labels=self.tipdict, precision=14)

                # parse the newick to toytree
                gtree = toytree.rawtree(nwk, tree_format=5)

                # mutate sequences on this tree; return array alphanum-ordered
                seed = self.rng_muts.integers(1e9)
                seq = mkseq.feed_tree(gtree, gtlen, self.mut, seed)

                # store the seqs to locus array
                seqarr[:, bidx:bidx + gtlen] = seq
                aseqarr[:, bidx:bidx + gtlen] = mkseq.ancestral_seq

                # record the number of snps in this locus
                df.loc[idx, 'nsnps'] = (np.any(seq != seq[0], axis=0).sum())

                # advance site counter
                bidx += gtlen

                # store newick string 
                df.loc[idx, "genealogy"] = gtree.write(tree_format=5)

                # this will skip zero length segments to we use pseudoindex
                df.loc[idx, "tidx"] = pseudoindex
                pseudoindex += 1

        # drop intervals that are 0 bps in length (sum bps will still = nsites)
        df = df.drop(index=df[df.nbps == 0].index).reset_index(drop=True)        

        # return the dataframe and seqarr
        return df, seqarr, aseqarr



    def sim_loci(self, nloci=1, nsites=1, seqgen=False):
        """
        Simulate tree sequence for each locus and sequence data for each 
        genealogy and return all genealogies and their summary stats in a 
        dataframe and the concatenated sequences in an array with rows ordered
        by sample names alphanumerically.

        Parameters
        ----------
        nloci (int):
            The number of loci to simulate.

        nsites (int):
            The length of each locus.

        seqgen (bool):
            Use seqgen as simulator backend. TO BE REMOVED.
        """
        # check conflicting args
        if self.recomb_map is not None:
            if nsites:
                raise IpcoalError(
                    "Both nsites and recomb_map cannot be used together since"
                    "the recomb_map also specifies nsites. To use a recomb_map"
                    "specify nsites=None.")
            nsites = self.recomb_map.get_length()

        # allow scientific notation, e.g., 1e6
        nsites = int(nsites)
        nloci = int(nloci)        

        # multidimensional array of sequence arrays to fill 
        seqarr = np.zeros((nloci, self.nstips, nsites), dtype=np.uint8)
        aseqarr = np.zeros((nloci, nsites), dtype=np.uint8)

        # a list to be concatenated into the final dataframe of genealogies
        dflist = []

        # open the subprocess to seqgen
        if seqgen:
            mkseq = SeqGen(**self.substitution_model)
            mkseq.open_subprocess()
        else:
            mkseq = SeqModel(**self.substitution_model)

        # iterate over nloci to simulate, get df and arr to store.
        for lidx in range(nloci):

            # returns genetree_df and seqarray
            df, arr, anc = self._sim_locus(nsites, lidx, mkseq)

            # store seqs in a list for now
            seqarr[lidx] = arr
            aseqarr[lidx] = anc

            # store the genetree df in a list for now
            dflist.append(df)

        # concatenate all of the genetree dfs
        df = pd.concat(dflist)
        df = df.reset_index(drop=True)

        # clean and close subprocess 
        mkseq.close()

        # store values to object
        self.df = df
        self.seqs = seqarr
        self.ancestral_seq = aseqarr

        # reset random seeds
        self._reset_random_generators()



    def sim_trees(self, nloci=1, nsites=1):
        """
        Record tree sequence without simulating any sequence data.
        This is faster than simulating snps or loci when you are only 
        interested in the tree sequence. To examine genealogical variation 
        within the same locus use nsites.

        Parameters:
        -----------
        See sim_loci()
        """
        # check conflicting args
        if self.recomb_map is not None:
            if nsites:
                raise IpcoalError(
                    "Both nsites and recomb_map cannot be used together since"
                    "the recomb_map also specifies nsites. To use a recomb_map"
                    "specify nsites=None.")
            nsites = self.recomb_map.get_length()

        # allow scientific notation, e.g., 1e6
        nsites = int(nsites)
        nloci = int(nloci)        

        # store dfs
        dflist = []

        # iterate over nloci to simulate, get df and arr to store.
        for lidx in range(nloci):

            # get the msprime ts generator 
            msgen = self._get_tree_sequence_generator(nsites)

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
                "tidx": 0,
                "locus": lidx,
                },
                columns=[
                    'locus', 'start', 'end', 'nbps', 
                    'nsnps', 'tidx', 'genealogy'
                ],
            )

            # iterate over the index of the dataframe to sim for each genealogy
            for idx, mstree in zip(df.index, msts.trees()):

                # get the number of base pairs taken up by this gene tree
                gtlen = df.loc[idx, 'nbps']

                # only simulate data if there is bp 
                if gtlen:
                    # convert nwk to original names
                    nwk = mstree.newick(node_labels=self.tipdict, precision=14)
                    df.loc[idx, "genealogy"] = nwk
                    df.loc[idx, "tidx"] = mstree.index

            # drop intervals 0 bps in length (sum bps will still = nsites)
            df = df.drop(index=df[df.nbps == 0].index).reset_index(drop=True)        

            # store the genetree df in a list for now
            dflist.append(df)

        # concatenate all of the genetree dfs
        df = pd.concat(dflist)
        df = df.reset_index(drop=True)

        # store values to object
        self.df = df

        # allows chaining funcs
        # return self



    def sim_snps(self, nsnps=1, repeat_on_trees=False, seqgen=False):
        """
        Run simulations until nsnps _unlinked_ SNPs are generated. If the tree
        is shallow and the mutation rate is low this can take a long time b/c
        most genealogies will produce an invariant site (i.e., not a SNP). 
        Take note that sim_tree() and sim_loci() do not condition on whether
        any mutations fall on the tree, whereas sim_snps() does. This means 
        that the distribution of trees from sim_snps will vary from 
        the others unless you use 'repeat_on_tree=True' which will force a 
        mutation to occur on every visited tree.

        nsnps (int):
            The number of SNPs to produce.

        repeat_on_trees (bool):
            If True then sequence simulations repeat on a genealogy until it 
            produces a SNP. If False then if a genealogy does not produce
            a SNP we move on to the next simulated genealogy. This may be
            more correct since shallow trees are less likely to contain SNPs.

        seqgen (bool):
            A (hidden) argument to use seqgen to test our mutation
            models against its results.
        """
        # allow scientific notation, e.g., 1e6
        nsnps = int(nsnps)

        # initialize a sequence simulator
        if seqgen:
            mkseq = SeqGen(**self.substitution_model)
            mkseq.open_subprocess()
        else:
            mkseq = SeqModel(**self.substitution_model)

        # get the msprime ts generator 
        msgen = self._get_tree_sequence_generator(1, snp=True)

        # store results (nsnps, ntips); def. 1000 SNPs
        newicks = []
        snpidx = 0
        snparr = np.zeros((self.nstips, nsnps), dtype=np.uint8)
        ancarr = np.zeros(nsnps, np.uint8)

        # continue until we get nsnps
        while 1:

            # bail out if nsnps finished
            if snpidx == nsnps:
                break

            # get first tree from next tree_sequence and parse it
            mstree = next(msgen).first()

            # write mstree to newick with original labels mapped on tips
            nwk = mstree.newick(node_labels=self.tipdict, precision=14)

            # parse the newick to toytree
            gtree = toytree.rawtree(nwk, tree_format=5)

            # simulate evolution of 1 base
            seed = self.rng_muts.integers(1e9)    
            seq = mkseq.feed_tree(gtree, 1, self.mut, seed)

            # if repeat_on_trees then keep sim'n til we get a SNP
            if repeat_on_trees:
                # if not variable
                while np.all(seq == seq[0]):
                    seed = self.rng_muts.integers(1e9)    
                    seq = mkseq.feed_tree(gtree, 1, self.mut, seed)

            # otherwise just move on to the next generated tree
            else:
                if np.all(seq == seq[0]):
                    continue

            # get newick string to store the tree the SNP landed on            
            newick = gtree.write(tree_format=5, dist_formatter="%0.3f")

            # Store result and advance counter
            snparr[:, snpidx] = seq.flatten()
            ancarr[snpidx] = mkseq.ancestral_seq
            snpidx += 1
            newicks.append(newick)

        # close subprocess is seqgen, or nothing if seqmodel
        mkseq.close()

        # init dataframe
        self.df = pd.DataFrame({
            "start": 0,
            "end": 1,
            "genealogy": newicks,
            "nbps": 1, 
            "nsnps": 1,
            "tidx": 0,
            "locus": range(nsnps),
            },
            columns=[
                'locus', 'start', 'end', 'nbps', 
                'nsnps', 'tidx', 'genealogy',
            ],
        )
        self.seqs = snparr
        self.ancestral_seq = ancarr

        # reset random seeds
        # self._reset_random_generators()


    # ---------------------------------------------------------------
    # i/o methods.
    # ---------------------------------------------------------------

    def write_loci_to_hdf5(self, name=None, outdir=None, diploid=False):
        """
        Writes a database file in .seqs.hdf5 format which is compatible with
        the ipyrad-analysis toolkit. This requires the additional dependency
        h5py and will raise an exception if the library is missing.
        """
        writer = Writer(self.seqs, self.alpha_ordered_names, self.ancestral_seq)
        writer.write_loci_to_hdf5(name, outdir, diploid, quiet=False)

    def write_snps_to_hdf5(self, name=None, outdir=None, diploid=False):
        """
        Writes a database file in .snps.hdf5 format which is compatible with
        the ipyrad-analysis toolkit. This requires the additional dependency
        h5py and will raise an exception if the library is missing.
        """
        writer = Writer(self.seqs, self.alpha_ordered_names, self.ancestral_seq)
        writer.write_snps_to_hdf5(name, outdir, diploid, quiet=False)

    def write_vcf(
        self, 
        name:str=None,
        outdir:str="./",
        diploid:bool=False,
        bgzip:bool=False,
        quiet:bool=False,
        fill_missing_alleles:bool=True,
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
        diploid (bool): 
            Combine haploid pairs into diploid genotypes.
        bgzip (bool):
            Call bgzip to block compress the output file (create .vcf.gz).
        quiet (bool):
            Suppress printed info.
        fill_missing_alleles (bool):
            If there is missing data this will fill diploid missing alleles. 
            e.g., the call (0|.) will be written as (0|0). This is meant to
            emulate real data where we often do not know the other allele
            is missing (also, some software tools do not accept basecalls 
            with one missing allele, such as vcftools).        
        """
        writer = Writer(self.seqs, self.alpha_ordered_names, self.ancestral_seq)
        vdf = writer.write_vcf(
            name, 
            outdir, 
            diploid, 
            bgzip,
            fill_missing_alleles,
            quiet,
        )
        return vdf

    def write_loci_to_phylip(
        self, 
        outdir="./",
        idxs=None, 
        name_prefix=None, 
        name_suffix=None,
        diploid=False,
        quiet=False,
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
        idxs (int):
            To write a single locus file provide the idx. If None then all loci
            are written to separate files.
        """
        writer = Writer(self.seqs, self.alpha_ordered_names, self.ancestral_seq)
        writer.write_loci_to_phylip(
            outdir, 
            idxs, 
            name_prefix, 
            name_suffix,
            diploid,
            quiet,
        )

    def write_concat_to_phylip(
        self, 
        name=None, 
        outdir="./",
        idxs=None,
        diploid=None, 
        ):
        """
        Write all seq data (loci or snps) concated to a single phylip file.

        Parameters:
        -----------
        outfile (str):
            The name/path of the outfile to write. Default is "./test.phy"
        """       
        writer = Writer(self.seqs, self.alpha_ordered_names, self.ancestral_seq)
        phystring = writer.write_concat_to_phylip(outdir, name, idxs, diploid)           
        if name is None:
            return phystring
        return None

    def write_concat_to_nexus(
        self, 
        name=None,
        outdir="./",
        idxs=None,
        diploid=None, 
        ):
        """
        Write all seq data (loci or snps) concated to a single phylip file.

        Parameters:
        -----------
        outfile (str):
            The name/path of the outfile to write. Default is "./test.phy"
        """       
        writer = Writer(self.seqs, self.alpha_ordered_names, self.ancestral_seq)
        nexstring = writer.write_concat_to_nexus(outdir, name, idxs, diploid)            
        if name is None:
            return nexstring
        return None

    def write_popfile(self, name:str, outdir:str="./", diploid:bool=False):
        """
        Writes popfile mapping sample names to species tree tip names
        to a file.

        Parameters:
        -----------
        name: str
            The name prefix of the file: {name}.popfile.tsv
        outdir: str
            The directory name (filepath prefix) to write the file to.
        diploid: bool
            If diploid=True was used when writing the data files then
            it should also be used when writing the popfile.
        """
        name = name.rstrip(".tsv").rstrip(".popfile")
        popdata = []
        writer = Writer(self.seqs, self.alpha_ordered_names, self.ancestral_seq)
        writer._transform_seqs(diploid=diploid)
        for tip in writer.names:
            popdata.append(f"{tip.rsplit('-')[0]}\t{tip}")
        outname = os.path.join(outdir, name + ".popfile.tsv")
        with open(outname, 'w') as out:
            out.write("\n".join(popdata))
        print(f"wrote popfile to {outname}")

    # ---------------------------------------------------------------
    # post-sim methods
    # ---------------------------------------------------------------    

    def infer_gene_tree_windows(
        self, 
        window_size:Optional[int]=None, 
        inference_method:str='raxml', 
        inference_args:Dict=None,
        ):
        """
        Infer gene trees at every locus using the sequence in the locus 
        interval. If no window_size is entered then a single full
        concatenation tree is inferred.

        Parameters
        ----------
        window_size: 
            The size of non-overlapping windows to be applied across the 
            sequence alignment to infer gene tree windows. If None then 
            a single gene tree is inferred for the entire concatenated seq.
        method (str):
            options include "iqtree", "raxml", "mrbayes".
        kwargs (dict):
            a limited set of supported inference options. See docs.

        Returns
        ----------
        pd.DataFrame is returned, example below:
        """
        # bail out if the data is only unlinked SNPs
        if self.df.nbps.max() == 1:
            raise IpcoalError(
                "gene tree inference cannot be performed on individual SNPs\n"
                "perhaps you meant to run .sim_loci() instead of .sim_snps()."
                )
        # complain if no seq data exists
        if self.seqs is None:
            raise IpcoalError(
                "Cannot infer trees because no seq data exists. "
                "You likely called sim_trees() instead of sim_loci()."
            )

        # if window_size is None then use entire chrom
        if window_size is None:
            window_size = self.df.end.max()

        # create the results dataframe
        resdf = pd.DataFrame({
            "start": np.arange(0, self.df.end.max(), window_size),
            "end": np.arange(window_size, self.df.end.max() + window_size, window_size),
            "nbps": window_size,
            "nsnps": 0,
            "inferred_tree": np.nan,
        })

        # reshape seqs: (nloc, ntips, nsites) to (nwins, ntips, win_size)
        newseqs = np.zeros((resdf.shape[0], self.ntips, window_size), dtype=int)
        for idx in resdf.index:
            # TODO: HERE IT'S ONLY INFERRING AT LOC 0
            loc = self.seqs[0, :, resdf.start[idx]:resdf.end[idx]]
            newseqs[idx] = loc
            resdf.loc[idx, "nsnps"] = (np.any(loc != loc[0], axis=0).sum())

        # init the TreeInference object (similar to ipyrad inference code)
        tool = TreeInfer(
            newseqs,
            self.alpha_ordered_names,
            inference_method=inference_method,
            inference_args=inference_args,
        )

        # iterate over nloci. This part could be easily parallelized...
        for idx in resdf.index:
            resdf.loc[idx, "inferred_tree"] = tool.run(idx)
        return resdf

    def infer_gene_trees(
        self, 
        inference_method:str='raxml', 
        inference_args:Optional[Dict[str,str]]=None,
        ):
        """
        Infer gene trees at every locus using the sequence in the locus 
        interval. 

        Parameters
        ----------
        method (str):
            options include "iqtree", "raxml", "mrbayes".
        kwargs (dict):
            a limited set of supported inference options. See docs.
            Default:
            raxml_kwargs = {
                "f": "d", 
                "N": "10",
                "T": "4", 
                "m": "GTRGAMMA",
                "w": tempfile.gettempdir()
            }
        """
        inference_args = inference_args if inference_args is not None else {}

        # bail out if the data is only unlinked SNPs
        if self.df.nbps.max() == 1:
            raise IpcoalError(
                "gene tree inference cannot be performed on individual SNPs\n"
                "perhaps you meant to run .sim_loci() instead of .sim_snps()."
                )

        # expand self.df to include an inferred_trees column
        self.df["inferred_tree"] = np.nan

        # init the TreeInference object (similar to ipyrad inference code)
        tool = TreeInfer(
            self.seqs, 
            self.alpha_ordered_names,
            inference_method=inference_method, 
            inference_args=inference_args,
        )

        # complain if no seq data exists
        if self.seqs is None:
            raise IpcoalError(
                "Cannot infer trees because no seq data exists. "
                "You likely called sim_trees() instead of sim_loci()."
            )

        # TODO; if sim_snps() infer one concatenated tree.
        # ...

        # iterate over nloci. This part could be easily parallelized...
        for lidx in range(self.seqs.shape[0]):

            # skip invariable loci
            if self.df.nsnps[self.df.locus == lidx].sum():
                # let low data fails return NaN
                try:
                    tree = tool.run(lidx)
                    # enter result
                    self.df.loc[self.df.locus == lidx, "inferred_tree"] = tree

                # caught raxml exception (prob. low data)
                except IpcoalError as err:
                    print(err)
                    raise err

    def get_pairwise_distances(self, model=None):
        """
        Returns pairwise distance matrix.

        Parameters:
        -----------
        model (str):
            Default is None meaning the Hamming distance. Supported options:
                None: Hamming distance, i.e., proportion of differences.
                "JC": Jukes-Cantor distance, i.e., -3/4 ln((1-(4/3)d))
                "HKY": Not yet implemented.  
        """
        # requires data
        if self.seqs is None:
            raise IpcoalError("You must first run .sim_snps() or .sim_loci")
        return calculate_pairwise_dist(self, model)

    def apply_missing_mask(self, coverage=1.0, cut1=0, cut2=0, distance=0.0, coverage_type='locus'):
        """
        Mask data by marking it as missing based on a number of possible 
        models for dropout. Uses the random tree seed.

        Parameters:
        -----------
        coverage (float):
            This emulates sequencing coverage. A value of 1.0 means that all
            loci have a 100% probabilty of being sampled. A value of 0.5 
            would lead to 50% of (haploid) samples to be missing at every 
            locus due to sequencing coverage. The resulting pattern of missing
            data is stochastic.

        cut1 (int):
            This emulates allele dropout by restriction digestion (e.g., a 
            process that could occur in RAD-seq datasets). This is the 
            length of the cutsite at the 5' end. When the value is 0 no
            dropout will occur. If it is 10 then the haplotype will be 
            dropped if any mutations occurred within the first 10 bases of 
            this allele relative to the known ancestral sequence. 

        cut2 (int):
            The same as cut 1 but applies to the 3' end to allow emulating 
            double-digest methods.

        distance (float):
            Not Yet Implemented. 
            This emulates sequence divergence as would apply to RNA bait 
            capture approaches where capture decreases with disimilarity from
            the bait sequence.

        coverage_type (str):
            By default coverage assumes that reads cover the entire locus,
            e.g., RAD-seq, but alternatively you may wish for coverage to 
            apply to every site randomly. This can be toggled by changing
            the coverage_type='locus' to coverage_type='site'

        """
        # do not allow user to double-apply
        if 9 in self.seqs:
            raise IpcoalError(
                "Missing data can only be applied to a dataset once.")

        # fix a seed generator
        # if seed:
            # np.random.seed(seed)

        # iterate over each locus converting missing to 9
        for loc in range(self.seqs.shape[0]):
            arr = self.seqs[loc]

            # apply coverage mask
            if coverage_type == "site":
                mask = self.rng_trees.binomial(1, 1.0 - coverage, arr.shape).astype(bool)
                arr[mask] = 9

            # implement 'locus' coverage as default
            else:
                mask = self.rng_trees.binomial(1, 1.0 - coverage, arr.shape[0]).astype(bool)
                arr[mask, :] = 9

            # apply dropout cut1
            if cut1:
                mask = np.any(arr[:, :cut1] != self.ancestral_seq[loc, :cut1], axis=1)
                arr[mask, :] = 9

            # apply dropout cut2
            if cut2:
                mask = np.any(arr[:, -cut2:] != self.ancestral_seq[loc, -cut2:], axis=1)
                arr[mask, :] = 9


if __name__ == "__main__":
    
    import ipcoal
    ipcoal.set_loglevel("DEBUG")

    TREE = toytree.rtree.imbtree(ntips=10, treeheight=1e6)
    print(TREE.get_feature_dict("idx", "height"))
    MODEL = ipcoal.Model(TREE, Ne=10000)
    # MODEL.sim_trees(5)
    MODEL.sim_loci(1, 10)
    print(MODEL.df)
    print(MODEL.seqs)
    MODEL.sim_snps(10)
    print(MODEL.df)
    print(MODEL.seqs)
    print(MODEL.write_vcf())
    # MODEL.sim_loci(1, 10)
    # MODEL.write_loci_to_phylip()
