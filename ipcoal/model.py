#!/usr/bin/env python

"""The core ipcoal class for coalescent simulations.

The Model class is used for setting up demographic models, running
coalescent simulations, summarizing results, and running
downstream analyses.
"""

from typing import List, Tuple, Optional, Dict, Union, Iterator
import os
import numpy as np
import pandas as pd
import msprime as ms
from msprime.mutations import mutation_model_factory
import toytree
from loguru import logger

from ipcoal.io.writer import Writer
from ipcoal.io.transformer import Transformer
from ipcoal.draw import draw_seqview, draw_genealogy
from ipcoal.core import sim_trees, sim_loci, sim_snps
# from ipcoal.utils.utils import calculate_pairwise_dist
# from ipcoal.phylo.TreeInfer import TreeInfer
from ipcoal.utils.utils import get_admix_interval_as_gens, IpcoalError

# pylint: disable=too-many-public-methods, invalid-name, too-many-lines, too-many-statements

# set display preference to make tree columns look nice
pd.set_option("max_colwidth", 28)

# register logger to module
logger = logger.bind(name="ipcoal")


class Model:
    """Core ipcoal class for coalescent simulations.

    Takes an input topology with edge lengths in units of generations
    entered as either a newick string or as a Toytree object, and
    defines a demographic model with a single Ne (or different Ne
    values mapped to tree nodes) and admixture edge arguments.
    Genealogies are generated with msprime and mutations are added
    under a finite-site Markov substitution model.

    Parameters
    ----------
    tree: str or toytree.ToyTree
        A newick string or ToyTree object of a species tree with edges
        in units of generations. Default is an empty string ("") which
        means no species tree and thus a single panmictic population
        coalescent with no divergences (no time limit).
    nsamples: int or Dict[str, int]:
        An integer for the number of samples *per lineage*, or a
        dict mapping the tip names (or int idx labels) to an integer
        for the number of sampled individuals (haploid genomes) to
        sample from each lineage. Examples: 2, or {0: 2, 1: 2, 2: 4}.
        If using a dict you must enter a value for every tip.
    Ne: float, int, or None
        The _diploid_ effective population size (coalescent probs are
        scaled to 1/2N). If you are simulating asexual haploids then
        you should double Ne. Ipcoal does not currently support ploidy
        >2 (see msprime which can). If a value is entered here it will
        be set for all edges of the tree. To set different Ne to
        different edges you must add Ne node attributes to an input
        ToyTree. For example, tre.set_node_data("Ne", {1:1000,
        2:10000}, default=5000)) will set all edges to 5000 except
        those with specific values specified by their node idx or name.
    admixture_edges: List[Tuple[...]]
        A list of admixture events in the 'admixture tuple format':
        (source, dest, time, rate), where the time argument can be
        entered as a float, int, or tuple.\n
            e.g., (3, 5, 0.5, 0.01);\n
            e.g., (3, 5, 200000, 0.25);\n
            e.g., (3, 5, (0.4, 0.6), 0.0001);\n
            e.g., (3, 5, (200000, 300000), 0.0001);\n
        The source sends migrants to dest **backwards in time.** Use
        the .draw_demography() function to verify your model setup.
        The time value is entered as a *proportion* of the length of
        the edge that overlaps between source and dest edges over
        which admixture can occur. If time=None it defaults to a pulse
        of migration at the midpoint of the shared edge. If a floating
        point value (e.g., 0.25) then admixture occurs this proportion
        of time back from the recent end of the shared interval
        (e.g., 25%). If an integer value is entered (10000) a pulse of
        migration occurs at this time point. If the time point is
        not in the shared interval between source and dest nodes, or
        they do not share any interval, an error is raised. If a tuple
        interval is entered for time then admixture occurs uniformly
        over the entire admixture interval and 'rate' is a constant
        _migration rate_ (as opposed to proportion) over this time
        period.
    mut: Union[float, ms.RateMap]
        The per-site per-generation mutation rate. Default=1e-8.
    recomb: Union[float, ms.RateMap]
        The per-site per-generation recombination rate. Default=1e-9.
    ancestry_model: str or AncestryModel
        The type of ancestry model from msprime (default="hudson").
    subst_model: Union[str, ms.MutationModel]
        A finite-site Markov substitution model supported by msprime.
        Default="JC69"
    seed_trees: int
        Random number generator used for msprime (and seqgen unless a
        separate seed is set for seed_mutations. Default=None.
    seed_mutations: int
        Random number generator used for seq-gen. If not set then the
        generic seed is used for trees and mutations. Default=None
    store_tree_sequences: bool
        If True the TreeSequence objects are stored in `Model.ts_dict`.
        If you do not plan to access and use these objects then
        setting this to False can greatly reduce memory usage.
    record_full_arg: bool
        If True then "invisible recombination" events which have no
        effect on the genealogy are also recorded.

    Attributes
    ----------
    df: pandas.DataFrame
        A DataFrame with a summary of the simulated data.
    seqs: ndarray
        An int array of shape (nloci, nsamples, nsites) with simulated
        data recorded as the index of alleles in the subst_model.
        See .alleles attribute for the alleles given the subst_model
        that was set on the Model object.
    ts_dict: Dict[int, tskit.trees.TreeSequence]
        A dictionary mapping the locus index to the tskit TreeSequence
        object generated by msprime, if Model is initialized with the
        option `store_tree_sequences=True`, else this is an empty dict
        (much more memory efficient by default).

    Example
    -------
    >>> tree = toytree.rtree.unittree(ntips=2, treeheight=1e6)
    >>> model = ipcoal.Model(tree=tree, Ne=1000, nsample=2)
    """
    def __init__(
        self,
        tree: Optional[toytree.ToyTree]=None,
        Ne: Optional[int] = None,
        nsamples: Union[int, Dict[Union[str,int],int]]=1,
        admixture_edges: Optional[List[Tuple[int,int,float,float]]]=None,
        mut: Union[float, ms.RateMap]=1e-8,
        recomb: Union[float, ms.RateMap]=1e-9,
        ancestry_model: Union[str, ms.AncestryModel]="hudson",
        subst_model: Union[str, ms.MutationModel]="JC69",
        seed_trees: Optional[int]=None,
        seed_mutations: Optional[int]=None,
        store_tree_sequences: bool=False,
        record_full_arg: bool=False,
        **kwargs,
        ):

        # legacy support warning messages
        self._warn_bad_kwargs(kwargs)

        # store user input args
        self.tree = tree if tree is not None else "(p);"
        self.neff = Ne       # upper-case only used in input arg.
        self.nsamples = nsamples
        self.admixture_edges = admixture_edges
        self.mut = mut
        self.recomb = recomb
        self.ancestry_model = ancestry_model
        self.subst_model = subst_model
        self.store_tree_sequences = store_tree_sequences
        self.record_full_arg = record_full_arg

        # public attrs to be filled, or storing init args.
        self.samples: List[ms.SampleSet] = None
        """: List of SampleSet objects with ploidy info."""
        self.rng_trees: np.random.Generator = None
        """: Random Generator used for sampling genealogies"""
        self.rng_muts: np.random.Generator = None
        """: Random Generator used for sampling mutations"""
        self.alleles: Dict[str, int]
        """: Dict mapping str alleles to int indices in the subst_model"""
        self.nstips: int = None
        """: Number of samples"""
        self.tipdict: Dict[str, List[str]] = {}
        """: Dict mapping species tree tip names to lists of sample tip names"""
        self.alpha_ordered_names: List[str] = None
        """: List of sample tip names in alphanumeric order"""
        self.ms_admix: List[Tuple] = []
        """: List of admix tuples as (src, dest, interval, rate)."""
        self.ms_demography: ms.Demography = None
        """: List of ms Demography events..."""

        # private attrs to be filled, or storing init args.
        self._init_tseed: Optional[int] = seed_trees
        """: Init seed for rng_trees"""
        self._init_mseed: Optional[int] = seed_mutations
        """: Init seed for rng_muts"""
        self._recomb_is_map: bool = isinstance(recomb, ms.RateMap)
        """: Boolean for whether recomb rate is a RateMap"""
        self._alleles: Tuple[str] = None
        """: Tuple of the alleles in subst_model in index order"""
        self._reorder: List[str]
        """: List or sample names ordered by ..."""

        # results
        self.ts_dict: Dict[int, 'tskit.trees.TreeSequence'] = {}
        """: Dict mapping locus int idx labels to TreeSequence objects"""
        self.df: pd.DataFrame = None
        """: Pandas DataFrame with summary of simulated data."""
        self.seqs: np.ndarray = None
        """: Numpy array of simulated sequence data: (nloci, ntaxa, nsites)"""
        self.ancestral_seq: np.ndarray = None
        """: Numpy array of simulated ancestral sequence: (nloci, 1, nsites)"""

        # functions to check input args and set values to Model attrs
        self._reset_random_generators()    # .rng_trees, .rng_muts
        self._set_species_tree()           # .tree
        self._set_mutation_model()         # .subst_model, .alleles, ._alleles
        self._set_samples()                # .samples, .nstips
        self._set_neff()                   # .tree (sets Node.Ne attrs)
        self._set_tip_names()              # .tipdict, .alpha_ordered_names, ._reorder
        self._set_admixture_edges()        # .admixture_edges,
        self._set_migration()              # .ms_admix
        self._set_new_demography()         # .ms_demography

    @staticmethod
    def _warn_bad_kwargs(kwargs):
        """Warn user that args are being skipped."""
        if kwargs:
            logger.warning(
                f"Parameters {list(kwargs)} are not supported. See "
                "documentation, argument names may have changed."
            )

    def _reset_random_generators(self):
        """Set RNGs to their starting states.

        Called after sim_trees(), sim_snps() or sim_loci to return all
        RNGs to their state during init so that a Model object that
        was init'd with some random seed will always return the same
        results even if run multiple times.
        """
        self.rng_trees = np.random.default_rng(self._init_tseed)
        self.rng_muts = (
            self.rng_trees if not self._init_mseed
            else np.random.default_rng(self._init_mseed)
        )

    def _set_species_tree(self):
        """set .tree as ToyTree by parsing newick str, or copying a ToyTree."""
        if isinstance(self.tree, toytree.ToyTree):
            self.tree = self.tree.mod.resolve_polytomies(dist=1e-5)
        else:
            self.tree = self.tree if self.tree else "n0;"
            self.tree = toytree.tree(self.tree).mod.resolve_polytomies(dist=1e-5)

    def _set_mutation_model(self):
        """Set subst_model as ms.MutationModel, and store alleles.

        Check the MutationModel is compatible with msprime and
        supported by ipcoal, and store a dict mapping alleles to
        their integer index.
        """
        try:
            self.subst_model = mutation_model_factory(self.subst_model)
        except ValueError as inst:
            msg = (
                "Model can be a string for 'JC69', 'binary', 'pam', or blosum62; "
                "or it must be a msprime.MutationModel instance to describe a "
                "more complex substitution model."
            )
            raise IpcoalError(msg) from inst
        self._alleles = tuple(self.subst_model.alleles)
        self.alleles = dict(enumerate(self.subst_model.alleles))

    def _set_samples(self):
        """Set .samples as List[ms.SampleSet] and .nsamples as int.

        The SampleSet 'ploidy' arg is always set to 1 in ipcoal,
        meaning that the user always enters a number referring to the
        number of haploid genomes to sample, using an int or dict.

        The 'population ploidy' can also be set by the user, usually
        to either 1 or 2. This will affect the time scale of coalescence

        Get a dict mapping tree tip idxs to the number of sampled
        haploid genomes. This will be passed to ms.sim_ancestry().
        We could alternatively use ms.SampleSet objects, which are
        more explicit, but I prefer the dict option.

        The demography model renames populations according to their
        toytree idx label with an 'n' prefix. This is required since
        msp has very strict pop naming conventions. Thus, here I
        rename the pops (tree tips) to match those names. The final
        outputs will be converted back to the tip names on the tree.
        """
        # user entered an integer: 2 -> {n0: 2, n1: 2, n2: 2, ...}
        if isinstance(self.nsamples, int):
            samples = {
                f"n{idx}": self.nsamples for idx in range(self.tree.ntips)
            }

        # user entered a dict with str or ints:
        # {0: 2, 1: 4, 2: 2} -> {n0: 2, n1: 4, n2: 2, ...}
        # {A: 2, B: 4, C: 2} -> {n0: 2, n1: 4, n2: 2, ...}
        elif isinstance(self.nsamples, dict):
            # keys are integers
            if all(isinstance(i, int) for i in self.nsamples):
                samples = {
                    f"n{idx}": val for idx, val in self.nsamples.items()
                }
            # keys are string names
            else:
                samples = {}
                for name in self.nsamples:
                    nodeobj = self.tree.get_nodes(name)[0]
                    samples[f"n{nodeobj.idx}"] = self.nsamples[name]

            # N keys in samples must match N tips in the species tree.
            if len(samples) != self.tree.ntips:
                raise IpcoalError(
                    "N keys in `nsamples` dict must match number of "
                    "lineages (N tips in tree).")

        else:
            raise TypeError(
                "The 'nsamples' arg must be either an int or a dict "
                "mapping tree tip names or idxs to integers. Examples:\n"
                "  nsamples=2 \n"
                "  nsamples={'A': 2, 'B': 2} \n"
                "  nsamples={0: 2, 1: 2}\n"
                )

        # samples is now a dict {spnames: int}, e.g., {n0: 2, n1: 2, ...}
        # store sum n samples
        self.nstips = sum(samples.values())

        # convert samples to an ordered list of ms.SampleSet objects
        # because these are most explicit about ploidy.
        self.samples = []
        for nidx in samples:
            sset = ms.SampleSet(
                num_samples=samples[nidx],
                population=nidx,
                ploidy=1,
            )
            self.samples.append(sset)

    def _set_neff(self):
        """Set `.Ne` attrs on all Nodes of `.tree`.

        Sets Ne values on all nodes of self.tree from neff arg, or
        checks for values on the existing tree. If an Ne argument was
        entered then it overrides any setting on the Nodes, else check
        that all nodes in the tree have an Ne setting, or raise an error.
        """
        # user entered an Ne arg to init, override any Ne data on tree.
        if self.neff is not None:
            self.tree = self.tree.set_node_data("Ne", default=self.neff)

        # Ne values exist on tree, ensure present for all Nodes.
        else:
            if "Ne" not in self.tree.features:
                raise IpcoalError(
                    "You must either enter an Ne argument or set Ne values "
                    "to all nodes of the input tree using ToyTree, e.g., "
                    "tree.set_node_data('Ne', mapping=..., default=...). "
                )
            neffs = self.tree.get_node_data("Ne", missing=np.nan)
            if neffs.isna().any():
                raise IpcoalError(
                    "You must either enter an Ne argument or set Ne values "
                    "to all nodes of the input tree using ToyTree, e.g., "
                    "tree.set_node_data('Ne', mapping=..., default=...).\n"
                    f"Your tree has NaN for Ne at some nodes:\n{neffs}")

    def _set_tip_names(self):
        """Fill .tipdict to map 0-indexed ints to ordered samples.

        NB: even though the msprime trees write labels 1-indexed, the
        newick node_labels arg in msprime expects a 0-indexed dict.
        """
        # are we sampling only 1 sample per lineage? If so, names will
        # be simpler, since we don't add a counter after the str name.
        is_singles = all(i.num_samples == 1 for i in self.samples)

        # get {1: "A", 2: "B"}
        if is_singles:
            ordered_idxs = [int(i.population[1:]) for i in self.samples]
            self.tipdict = {
                odx: self.tree[idx].name for (odx, idx) in enumerate(ordered_idxs)
            }

        # get {1: "A_0", 2: "A_1", 3: "B_0"}
        else:
            idx = 0
            for sset in self.samples:
                nidx = int(sset.population[1:])
                tipname = self.tree[nidx].name
                for sdx in range(sset.num_samples):
                    self.tipdict[idx] = f"{tipname}_{sdx}"
                    idx += 1

        # alphanumeric ordering of tipnames for seqs and outfiles.
        revdict = {j: i for (i, j) in self.tipdict.items()}
        # e.g., ["A", "B", "C", "D"]
        self.alpha_ordered_names = sorted(self.tipdict.values())
        # e.g., [0, 1, 2, 3]
        self._reorder = [revdict[i] for i in self.alpha_ordered_names]

    def _set_admixture_edges(self):
        """Checks that admixture_edges is a list of tuples."""
        if self.admixture_edges is None:
            self.admixture_edges = []
            return

        if not isinstance(self.admixture_edges[0], (list, tuple)):
            raise TypeError("admixture_edges should be a list of tuples.")
        if isinstance(self.admixture_edges, tuple):
            self.admixture_edges = [self.admixture_edges]
        for edge in self.admixture_edges:
            if len(edge) != 4:
                raise ValueError(
                    "admixture edges should each be a tuple with 4 values")

    def _set_migration(self):
        """Checks admixture tuples for proper configuration.

        Fills the admixture_edges list as int generations.
        >>> [(src, dest, interval-time-in-gens, rate), ...]
        """
        # sample times and proportions/rates for admixture intervals
        for iedge in self.admixture_edges:
            # expand args
            src, dest, time, rate = iedge

            # mtimes: if None then sample from uniform.
            if time is None:
                props = (0.25, 0.75)
                heights = None
            elif isinstance(time, int):
                props = None
                heights = (time, time)
            elif isinstance(time, float):
                props = (time, time)
                heights = None
            elif isinstance(time, tuple):
                if isinstance(time[0], int) and isinstance(time[1], int):
                    props = None
                    heights = (time[0], time[1])
                else:
                    props = (time[0], time[1])
                    heights = None
            interval = get_admix_interval_as_gens(
                self.tree, src, dest, heights, props)

            # store values in a new list
            self.ms_admix.append((src, dest, interval, rate))

    def _set_new_demography(self):
        """Demography updated for  msprime v.1.0+."""
        # init a demography model
        demography = ms.Demography()

        # traverse tree adding TIP pops with ID and name as toytree idxs.
        # also create .current to keep track of admixed edge fragments.
        for node in self.tree.traverse():
            if not node.children:
                name = f"n{node.idx}"
                demography.add_population(
                    name=name,
                    initial_size=node.Ne,
                    description=node.name,
                    default_sampling_time=node.height,
                )
                node.current = name

        # traverse tree from tips to root adding admixture and/or
        # population split events. These must be added in order since
        # admixture events require creating new internal nodes. By
        # contrast migration rate intervals can be added later at end.
        events = []

        # add split events
        for node in self.tree.traverse("levelorder"):
            if len(node.children) > 1:
                events.append({
                    'type': 'split',
                    'time': node.height,
                    'derived': [i.idx for i in node.children],
                    'ancestral': node.idx,
                })

        # add admixture events.
        for event in self.ms_admix:
            src, dest, time, rate = event
            if time[0] == time[1]:
                events.append({
                    'type': 'admixture',
                    'time': time[0],
                    'ancestral': [src, dest],
                    'derived': dest,
                    'proportions': [rate, 1 - rate],
                })
            else:
                events.append({
                    'type': 'migration',
                    'time': time[0],
                    'source': src,
                    'dest': dest,
                    'end': False,
                    'rate': rate,
                })
                events.append({
                    'type': 'migration',
                    'time': time[1],
                    'source': src,
                    'dest': dest,
                    'end': True,
                    'rate': 0,
                })

        # sort events by time then type
        events.sort(key=lambda x: (x['time'], x['type']))

        # create demography calls for each event, update 'current' attr
        # of nodes to point to their new reference as its created.
        for event in events:
            if event['type'] == "split":
                node = self.tree[event['ancestral']]  # index by idx label
                node.current = "_".join(
                    sorted([i.current for i in node.children])
                )
                demography.add_population(
                    name=node.current,
                    initial_size=node.Ne,
                )
                demography.add_population_split(
                    time=node.height,
                    derived=[i.current for i in node.children],
                    ancestral=node.current,
                )

            if event['type'] == "admixture":
                # get nodes info
                src, dest = event['ancestral']
                node_src = self.tree[src]
                node_dest = self.tree[dest]
                newname = node_src.current + "a"

                # create new node that inherits its Ne from dest
                demography.add_population(
                    name=newname,
                    initial_size=node_src.Ne,
                    initially_active=False,
                )

                # create new node ancestry
                demography.add_admixture(
                    time=event['time'],
                    derived=node_src.current,
                    ancestral=[node_dest.current, newname],
                    proportions=[rate, 1 - rate],
                )

                # set new current name for dest node
                node_src.current = newname

            # add migration interval events (start, end)
            if event['type'] == 'migration':
                # set migration rate start event
                demography.add_migration_rate_change(
                    time=event['time'],
                    source=event['source'],
                    dest=event['dest'],
                    rate=event['rate'],
                )

        # store and sort
        self.ms_demography = demography
        self.ms_demography.sort_events()

    def debug_demography(self):
        """Returns the msprime demography debugger.

        This represents a summary of the currently described
        demographic model.

        See Also
        --------
        ms_demography: Attribute list of demographic events.
        draw_demography: Function to visualize demographic model.
        draw_sptree: Function to visualize species tree model.
        """
        return self.ms_demography.debug()

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
        scrollable: bool=True,
        max_width: int=1_000,
        **kwargs,
        ) -> ('toyplot.Canvas', 'toyplot.Table'):
        """Returns a toyplot visualization of the simulated sequences.

        Parameters
        ----------
        idx: int
            The locus index of a locus to draw. If None and multiple
            loci are present then it draws the first locus. If SNPs
            were simulated then all SNPs are concatenated into a
            single 'locus'.
        start: int or None
            Slice start position of sequence array to draw. Default=0.
        end: int or None
            Slice end position of sequence array to draw. Default=end.
        width: int or None
            Width of the canvas drawing in pixels. Default: auto
        height: int or None
            Height of the canvas drawing in pixels. Deafult: auto
        show_text: bool
            Whether to show base labels on table cells.
        scrollable: bool
            If plot width exceeds the window width in pixel units a
            scroll bar will allow scrolling on horizontal axis.
        max_width: int
            The maximum number of columns that will be plotted. Much
            larger plots will likely slow down your browser.
        **kwargs: dict
            Additional drawing arguments to toyplot table.

        Returns
        -------
        canvas: toyplot.Canvas, will autorender in jupyter notebooks
        table: toyplot.Table, allows further editing on figure.
        """
        canvas, table = draw_seqview(
            self, idx, start, end, width, height, show_text,
            scrollable, max_width,
            **kwargs)
        return canvas, table

    def draw_genealogy(
        self, 
        idx: Optional[int]=None, 
        show_substitutions: bool=False, 
        **kwargs,
        ):
        """Return a toytree drawing of the genealogy.

        Parameters
        ----------
        idx: int
            index of the genealogy to draw from the (Model.df) dataframe.
        show_substitutions: bool
            If True then substitutions are shown on the branches of the
            genealogy. For this you must have initialized the Model 
            with `store_tree_sequences=True` to retain substitutions.
        """
        return draw_genealogy(self, idx, show_substitutions, **kwargs)
        # idx = idx if idx else 0
        # tree = toytree.tree(self.df.genealogy[idx])
        # canvas, axes, mark = tree.draw(ts='c', tip_labels=True, **kwargs)
        # return canvas, axes, mark

    def draw_genealogies(self, idxs: Optional[List[int]]=None, **kwargs):
        """Returns a toytree multitree drawing of several genealogies.

        Parameters
        ----------
        idx: list of ints, or None
            The index of the genealogies to draw from the (Model.df)
            dataframe.
        """
        if idxs is None:
            idxs = list(self.df.index)[:4]
        mtre = toytree.mtree(self.df.genealogy[idxs].tolist())
        canvas, axes, mark = mtre.draw(ts='c', tip_labels=True, **kwargs)
        return canvas, axes, mark

    def draw_sptree(self, **kwargs):
        """Returns a toytree drawing of the species tree.

        The value of Ne is shown by edge widths.
        """
        assert self.tree.ntips > 1, "No species tree. Use `draw_demography`."
        admix = [i[:2] for i in self.admixture_edges]
        canvas, axes, mark = self.tree.draw(
            ts='p',
            tip_labels=True,
            admixture_edges=admix,
            **kwargs)
        return canvas, axes, mark

    def draw_demography(self, idx=None, spacer=0.25, ymax=None, **kwargs):
        """Return drawing of parameterized demographic model.

        A genealogy can also be embedded in the demographic model by
        selecting a genealogy index (idx).

        Parameters
        ----------
        ...

        Examples
        --------
        >>> ...
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

    def draw_tree_sequence(self, **kwargs):
        """Return a toytree TreeSequence drawing."""
        raise NotImplementedError("TODO..")

    # ----------------------------------------------------------------
    # MSPRIME simulation methods
    # ----------------------------------------------------------------

    def get_tree_sequence(self, nsites: int=1) -> 'tskit.trees.TreeSequence':
        """Return a mutated TreeSequence from this demographic model.

        Note
        ----
        The TreeSequence object's underlying simulations can be also
        accessed from a Model object after running any of the core
        ipcoal simulation methods (`sim_trees`, `sim_loci`, `sim_snps`)
        from the `.ts_dict` attribute of the Model object. In these
        methods the `ts_dict` can be indexed to access TreeSequences
        associated with each simulated locus.

        Parameters
        ----------
        nsites: int
            The number of sites to simulate.
        """
        treeseq = ms.sim_ancestry(
            samples=self.samples,
            demography=self.ms_demography,
            sequence_length=(None if self._recomb_is_map else nsites),
            recombination_rate=(None if self._recomb_is_map else self.recomb),
            random_seed=self.rng_trees.integers(2**31),
            discrete_genome=True,
            record_full_arg=self.record_full_arg,
            model=self.ancestry_model,
        )
        mutated_ts = ms.sim_mutations(
            tree_sequence=treeseq,
            rate=self.mut,
            model=self.subst_model,
            random_seed=self.rng_muts.integers(2**31),
            discrete_genome=True,
        )
        return mutated_ts

    def _get_tree_sequence_generator(
        self,
        # seed: int,
        nsites: int=1,
        snp: bool=False,
        ) -> Iterator['tskit.trees.TreeSequence']:
        """Return a TreeSequence generator from `ms.sim_ancestry`.

        This function is used internally in `sim_trees`, `sim_loci`
        and `sim_snps` to sample genealogies for N unlinked regions.

        Parameters
        ----------
        seed: int
            This requires a seed, and the internal functions that
            use this function should pre-generate the seeds
        """
        # snp flag to ensure a single genealogy is returned
        if snp and nsites != 1:
            raise IpcoalError(
                "The flag snp=True can only be used with nsites=1")
        tsgen = ms.sim_ancestry(
            samples=self.samples,
            demography=self.ms_demography,
            sequence_length=(None if self._recomb_is_map else nsites),
            recombination_rate=(None if snp or self._recomb_is_map else self.recomb),
            num_replicates=(int(1e20) if snp else 1),
            random_seed=self.rng_trees.integers(2**31),
            discrete_genome=True,
            record_full_arg=self.record_full_arg,
            model=self.ancestry_model,
        )
        return tsgen

    def sim_trees(
        self,
        nloci: int=1,
        nsites: int=1,
        precision: int=14,
        ) -> None:
        """Simulate tree sequence without mutations

        Record tree sequence without simulating any sequence data.
        This is faster than simulating snps or loci when you are only
        interested in the tree sequence. To examine genealogical
        variation within a locus (with recombination) use nsites > 1.

        Parameters
        ----------
        nloci: int
            Number of unlinked loci to simulate.
        nsites: int
            The length of each locus in number of sites.
        precision: float
            The precision of float values recorded in newick tree
            strings in the .df output summary table.

        See Also
        --------
        sim_loci: Simulate N loci with mutated sequences of length M.
        sim_snps: Simulate N unlinked mutated sites (SNPs).

        Example
        -------
        >>> tree = toytree.rtree.unittree(ntips=2, treeheight=1e6)
        >>> mod = ipcoal.Model(tree=tree, Ne=5e5, nsamples=10)
        >>> mod.sim_trees(nloci=1, nsites=1000)
        >>> mod.df
        """
        sim_trees(self, nloci, nsites, precision)

    def sim_loci(
        self,
        nloci: int=1,
        nsites: int=1,
        precision: int=14,
        ) -> None:
        """Simulate tree sequence for N loci of length M sites.

        Parameters
        ----------
        nloci: int
            The number of loci to simulate.

        nsites: int
            The length of each locus.

        precision: int
            Floating point precision of blens in newick tree strings.

        See Also
        --------
        sim_trees: Simulate genealogies spanning N loci of length M.
        sim_snps: Simulate N unlinked mutated sites (SNPs).

        Example
        -------
        >>> tree = toytree.rtree.unittree(ntips=10, treeheight=1e6)
        >>> model = ipcoal.Model(tree=tree, Ne=5e5, nsamples=2)
        >>> model.sim_loci(nloci=10, nsites=1000)
        >>> model.write_loci_to_phylip(outdir="/tmp")
        """
        sim_loci(self, nloci, nsites, precision)

    def sim_snps(
        self,
        nsnps: int = 1,
        min_alleles: int = 2,
        max_alleles: Optional[int] = None,
        min_mutations: int = 1,
        max_mutations: Optional[int] = None,
        repeat_on_trees: bool = False,
        precision: int = 14,
        # exclude_fixed: bool = False,
        ) -> None:
        """Simulate N *unlinked* variable sites (SNPs).

        Parameters
        ----------
        nsnps: int
            The number of unlinked SNPs to produce.
        min_alleles: int=2
            A site is discarded if the number of observed alleles is
            less than min_alleles. A setting of 2 ensures SNPs will
            be **at least** bi-allelic, thus preventing a case where
            multiple mutations could revert a site to its ancestral
            state.
        max_alleles: int=None
            A site is discarded if the number of observed alleles is
            greater than max_alleles. A setting of 2 ensures that
            SNPs will be **at most** bi-allelic, and would discard
            multi-allelic sites produced by multiple mutations. See
            also max_mutations.
        min_mutations: int=1
            A site is discarded if the number of mutations is less
            than min_mutations. This ensures that only variant sites
            will be returned. Multiple mutations can revert a site to
            its ancestral state under many substitution models, thus
            it is useful to also use the min_alleles arg.
        max_mutations: int=None
            A site is discarded if the number of mutation is greater
            than max_mutations.
        repeat_on_trees: bool
            If True then the mutation process is repeated on each
            visited genealogy until a SNP is observed. If False
            (default) then genealogies are discarded, and new ones
            drawn, if a SNP does not occur on the first attempt.
            This is faster but introduces a bias.

        Note
        ----
        Depending on the simulation params (e.g., Ne, demography, mut)
        the probability that a mutation occurs on a simulated
        genealogy may be very low. All else being equal, genealogies
        with longer branch lengths should have a higher probability
        of containing SNPs. Thus, if we were to force mutations onto
        every sampled genealogy (optional argument 'repeat_on_trees'
        here) this would bias shorter branches towards having more
        mutations than they actually should have. Instead, this method
        samples unlinked genealogies and tests each once for the
        occurence of a mutation, and if one does not occur, it is
        discarded, until the requested number of unlinked SNPs is
        observed.

        Example
        -------
        >>> tree = toytree.rtree.baltree(ntips=2, treeheight=1e5)
        >>> model = ipcoal.Model(Ne=5e4, nsamples=6)
        >>> model.sim_snps(nsnps=10)
        >>> model.draw_seqview()
        """
        sim_snps(
            model=self, nsnps=nsnps,
            min_alleles=min_alleles, max_alleles=max_alleles,
            min_mutations=min_mutations, max_mutations=max_mutations,
            repeat_on_trees=repeat_on_trees, precision=precision,
        )

    # ---------------------------------------------------------------
    # i/o methods.
    # ---------------------------------------------------------------

    def write_loci_to_hdf5(self, name=None, outdir=None, diploid=False):
        """Write a database file in ipyrad seqs HDF5 format.

        This file format is compatible with the ipyrad-analysis
        toolkit and allows for fast lookups in even very large files.
        This function requires the additional dependency 'h5py' and
        will raise an exception if the library is missing.

        Parameters
        ----------
        name: str
            The prefix name of the output file.
        outdir: str
            The name of the directory in which to write the file. It
            will attempt to be created if it does not yet exist.
        diploid: bool
            Randomly join haploid samples to write diploid genotypes.
        """
        if self.seqs.ndim < 2:
            raise IpcoalError("No sequence data. Try Model.sim_loci().")
        if self.seqs.ndim == 2:
            raise IpcoalError(
                "Simulated data are not loci. "
                "See Model.write_snps_to_hdf5() for writing a SNP database.")
        writer = Writer(self)
        writer.write_loci_to_hdf5(name, outdir, diploid, quiet=False)

    def write_snps_to_hdf5(self, name=None, outdir=None, diploid=False):
        """Write a database file in .snps.hdf5 format

        This file format is compatible with the ipyrad-analysis
        toolkit and allows for very fast lookups even in very large
        files. This requires the additional dependency 'h5py' and
        will raise an exception if the library is missing.

        All SNPs will be written to file if data were generated using
        `sim_snps`, whereas if `sim_loci` was called then only the
        variable sites will be written, and non-variant sites will be
        excluded.

        Parameters
        ----------
        name: str=None
            The prefix name of the output file
        outdir: str=None
            The name of the directory in which to write the file. It
            will attempt to be created if it does not yet exist.
        diploid: bool=False
            Randomly join haploid samples to write diploid genotypes.
        """
        if self.seqs.ndim < 2:
            raise IpcoalError("No sequence data. Try Model.sim_snps().")
        writer = Writer(self)
        writer.write_snps_to_hdf5(name, outdir, diploid, quiet=False)

    def write_vcf(
        self,
        name: str=None,
        outdir: str="./",
        diploid: bool=False,
        bgzip: bool=False,
        quiet: bool=False,
        fill_missing_alleles: bool=True,
        ):
        """Writes variant sites to VCF (variant call format) file.

        Parameters
        ----------
        name: str
            Name prefix for vcf file.
        outdir: str
            A directory name where will be written, created if it
            does not yet exist.
        diploid: bool
            Combine haploid pairs into diploid genotypes.
        bgzip: bool
            Call bgzip to block compress output file (to .vcf.gz).
        quiet: bool
            Suppress printed info.
        fill_missing_alleles: bool
            If there is missing data this will fill diploid missing
            alleles. e.g., the call (0|.) will be written as (0|0).
            This is meant to emulate real data where we often do not
            know the other allele is missing (also, some software
            tools do not accept basecalls with one missing allele,
            such as vcftools).
        """
        if self.seqs.ndim < 2:
            raise IpcoalError("No sequence data. Try Model.sim_snps().")
        writer = Writer(self)
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
        outdir: str="./",
        idxs: List[int]=None,
        name_prefix: str=None,
        name_suffix: str=None,
        diploid: bool=False,
        quiet: bool=False,
        ):
        """Write loci in .phy format to separate files in shared dir.

        Parameters
        ----------
        outdir: str
            A directory in which to write all the phylip files. It
            will be created if it does not yet exist. Default is ./
        idxs: int
            To write a single locus file provide the idx. If None
            then all loci are written to separate files.
        name_prefix: str
            Prefix used in file names before locus index.
        name_suffix: str
            Suffix used in file names after locus index.
        diploid: bool
            Randomly combine haploid genotypes into diploid genotypes.
        quite: bool
            Suppress statements.
        """
        writer = Writer(self)
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
        name: str=None,
        outdir: str="./",
        idxs: List[int]=None,
        diploid: bool=False,
        quiet: bool=False,
        ):
        """Write concatenated sequence data to a single phylip file.

        Parameters
        ----------
        name: str
            Prefix name for output file.
        outdir: str
            Directory name where file will be written, created if it
            does not yet exist.
        idxs: List[int]
            A list of locus indices to include in the concatenated
            output file. Default is None, which writes all loci.
        diploid: bool
            Randomly combine haploid samples into diploid genotypes.
        """
        writer = Writer(self)
        phystring = writer.write_concat_to_phylip(outdir, name, idxs, diploid, quiet)
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
        """Write concatenated sequence data to a single nexus file.

        Parameters
        ----------
        name: str
            Prefix name for output file.
        outdir: str
            Directory name where file will be written, created if it
            does not yet exist.
        idxs: List[int]
            A list of locus indices to include in the concatenated
            output file. Default is None, which writes all loci.
        diploid: bool
            Randomly combine haploid samples into diploid genotypes.
        """
        writer = Writer(self)
        nexstring = writer.write_concat_to_nexus(outdir, name, idxs, diploid)
        if name is None:
            return nexstring
        return None

    def get_imap_dict(self, diploid: bool=False):
        """Return a dictionary mapping population names to sample names

        Parameters
        ----------
        diploid: bool
            If diploid is True then pairs of haploid samples are joined
            into single sample names.

        Returns
        -------
        Dict: example {'A': ['A_0', 'A_1'], 'B': ['B_0', 'B_1']}
        """
        txf = Transformer(
            self.seqs,
            self.alpha_ordered_names,
            alleles=self.alleles,
            diploid=diploid,
        )
        imap = {}
        for name in txf.names:
            key = name.split("_", 1)[0]
            if key not in imap:
                imap[key] = [name]
            else:
                imap[key].append(name)
        return imap

    def write_popfile(self, name:str, outdir:str="./", diploid:bool=False):
        """Writes popfile mapping species tree tips to sample names.

        The sample names here represent those that are in the written
        sequence files, which will be different from those in the
        genealogies if the files were written with the option
        diploid=True, which combines pairs of samples into a single
        representative. The diploid flag has the same effect here.

        The format of the output file looks like this:\n
            species_A\tA_0
            species_A\tA_1
            species_B\tB_0
            species_B\tB_1
            ...\n

        Parameters
        ----------
        name: str
            The name prefix of the file: {name}.popfile.tsv
        outdir: str
            The directory name (filepath prefix) to write the file to.
        diploid: bool
            If diploid=True was used when writing the data files then
            it should also be used when writing the popfile.
        """
        name = name.rsplit(".tsv")[0].rsplit(".popfile")[0]
        popdata = []
        txf = Transformer(
            self.seqs,
            self.alpha_ordered_names,
            alleles=self.alleles,
            diploid=diploid,
        )
        for tip in txf.names:
            popdata.append(f"{tip.rsplit('_')[0]}\t{tip}")
        outname = os.path.join(outdir, name + ".popfile.tsv")
        with open(outname, 'w', encoding="utf-8") as out:
            out.write("\n".join(popdata))
        print(f"wrote popfile to {outname}")

    # ---------------------------------------------------------------
    # post-sim methods
    # ---------------------------------------------------------------


    # def get_pairwise_distances(self, model=None):
    #     """
    #     Returns pairwise distance matrix.

    #     Parameters:
    #     -----------
    #     model (str):
    #         Default is None meaning the Hamming distance. Supported options:
    #             None: Hamming distance, i.e., proportion of differences.
    #             "JC": Jukes-Cantor distance, i.e., -3/4 ln((1-(4/3)d))
    #             "HKY": Not yet implemented.
    #     """
    #     # requires data
    #     if self.seqs is None:
    #         raise IpcoalError("You must first run .sim_snps() or .sim_loci")
    #     return calculate_pairwise_dist(self, model)

    def apply_missing_mask(
        self,
        coverage=1.0,
        cut_sites: Tuple[int,int]=(0, 0),
        distance: Optional[Tuple[str,float]]=None,
        coverage_type: str='locus',
        seed: Optional[int]=None,
        ):
        """Mask some data by marking it as missing.

        Several optional are available for dropping genotypes to
        appear as missing data. See coverage, cut_sites, and distance
        arguments.

        Parameters
        ----------
        coverage: float
            This emulates sequencing coverage. A value of 1.0 means that all
            loci have a 100% probabilty of being sampled. A value of 0.5
            would lead to 50% of (haploid) samples to be missing at every
            locus due to sequencing coverage. The resulting pattern of missing
            data is stochastic.
        cut_sites: Tuple[int, int]
            This emulates allele dropout by restriction digestion (e.g., a
            process that could occur in RAD-seq datasets). This is the
            length of the cutsite at the 5' end. When the value is 0 no
            dropout will occur. If it is 10 then the haplotype will be
            dropped if any mutations occurred within the first 10 bases of
            this allele relative to the known ancestral sequence.
        distance: Tuple[str, float]:
            Not Yet Implemented.
            This emulates sequence divergence as would apply to RNA bait
            capture approaches where capture decreases with disimilarity from
            the bait sequence.
        coverage_type: str
            For loci data it is assumed that reads cover the entire
            locus, e.g., RAD-seq, but alternatively you may wish for
            coverage to apply to every site randomly. This can be
            toggled by changing the coverage_type='locus' to
            coverage_type='site'. For SNP data coverage always applies
            to site.
        seed: Optional[int]
            Random generator seed.
        """
        # do not allow user to double-apply
        if 9 in self.seqs:
            raise IpcoalError(
                "Missing data can only be applied to a dataset once.")

        # missing mask cut_sites method can only apply to locus data.
        if self.seqs.ndim < 2:
            if any(cut_sites):
                raise IpcoalError(
                    "cut_sites method can only apply to data simulated with "
                    "Model.sim_loci()."
                )

        if distance is not None:
            raise NotImplementedError("distance method not yet implemented. TODO.")

        # fix a seed generator
        rng = np.random.default_rng(seed)

        # iterate over each locus converting missing to 9
        for loc in range(self.seqs.shape[0]):
            arr = self.seqs[loc]

            # apply coverage mask
            if (coverage_type == "site") or (self.seqs.ndim == 2):
                mask = rng.binomial(1, 1.0 - coverage, arr.shape).astype(bool)
                arr[mask] = 9

            # implement 'locus' coverage as default
            else:
                mask = rng.binomial(1, 1.0 - coverage, arr.shape[0]).astype(bool)
                arr[mask, :] = 9

            # apply dropout cut1
            if cut_sites[0]:
                cut1 = cut_sites[0]
                mask = np.any(arr[:, :cut1] != self.ancestral_seq[loc, :cut1], axis=1)
                arr[mask, :] = 9

            # apply dropout cut2
            if cut_sites[1]:
                cut2 = cut_sites[1]
                mask = np.any(arr[:, -cut2:] != self.ancestral_seq[loc, -cut2:], axis=1)
                arr[mask, :] = 9


if __name__ == "__main__":

    import ipcoal
    ipcoal.set_log_level("DEBUG")

    TREE = toytree.rtree.imbtree(ntips=10, treeheight=1e6)
    TREE = TREE.set_node_data("Ne", {0: 1000}, default=10000)
    print(TREE.get_feature_dict("idx", "height"))

    MODEL = ipcoal.Model(TREE, Ne=10000, nsamples=2)
    MODEL = ipcoal.Model(TREE, nsamples=2)

    MODEL.sim_trees(5)
    print(MODEL.df)
    print(MODEL.seqs)
    MODEL.sim_snps(10)
    print(MODEL.df)
    print(MODEL.seqs)
    print(MODEL.write_vcf())

    MODEL.sim_loci(2, 1000)
    MODEL.infer_gene_trees(diploid=True)
    print(MODEL.df)
