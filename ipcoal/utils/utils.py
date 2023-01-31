#!/usr/bin/env python

"""
Miscellaneous functions
"""

from typing import Tuple, Optional
import time
import datetime
import itertools

import toytree
import numpy as np
import pandas as pd
from numba import njit
from ipcoal.utils.jitted import count_matrix_int

try:
    from IPython.display import display
    from ipywidgets import IntProgress, HTML, Box
except ImportError:
    pass



ABBA_IDX = [
    (1, 4), (2, 8), (3, 12), (4, 1),
    (6, 9), (7, 13), (8, 2), (9, 6),
    (11, 14), (12, 3), (13, 7), (14, 11),
]
BABA_IDX = [
    (1, 1), (2, 2), (3, 3), (4, 4), 
    (6, 6), (7, 7), (8, 8), (9, 9),
    (11, 11), (12, 12), (13, 13), (14, 14),
]
FIXED_IDX = [
    (0, 0), (5, 5), (10, 10), (15, 15),
]



class IpcoalError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)



class Progress:
    """
    Interactive progress bar for jupyter notebooks.
    """
    def __init__(self, njobs, message, children):

        # data
        self.njobs = njobs
        self.message = message
        self.start = time.time()

        # the progress bar 
        self.prog = IntProgress(
            value=0, min=0, max=self.njobs, 
            layout={
                "width": "350px",
                "height": "30px",
                "margin": "5px 0px 0px 0px",
            })

        # the message above progress bar
        self.label = HTML(
            self.printstr, 
            layout={
                "height": "25px",
                "margin": "0px",
            })

        # the box widget container
        heights = [
            int(i.layout.height[:-2]) for i in 
            children + [self.label, self.prog]
        ]
        self.widget = Box(
            children=children + [self.label, self.prog],
            layout={
                "display": "flex",
                "flex_flow": "column",
                "height": "{}px".format(sum(heights) + 5),
                "margin": "5px 0px 5px 0px",
            })

    @property
    def printstr(self):
        """
        message as html
        """
        elapsed = datetime.timedelta(seconds=int(time.time() - self.start))
        str1 = "<span style='font-size:14px; font-family:monospace'>"
        str2 = "</span>"
        inner = "{} | {:>3}% | {}".format(
            self.message, 
            int(100 * (self.prog.value / self.njobs)),
            elapsed,
        )
        return str1 + inner + str2

    def display(self):
        """Show html"""
        display(self.widget)

    def increment_all(self, value=1):
        """adds value to prog"""
        self.prog.value += value
        if self.prog.value == self.njobs:
            self.prog.bar_style = "success"
        self.increment_time()

    def increment_time(self):
        """sets label value to printstr"""
        self.label.value = self.printstr



def get_admix_interval_as_gens(
    tree: 'ToyTree', 
    idx0:int, 
    idx1:int, 
    heights:Optional[Tuple[int, int]]=None,
    props:Optional[Tuple[float, float]]=None,
    ) -> Tuple[int, int]:
    """
    Returns the branch interval in units of generations that two 
    edges of a tree are overlapping, with the lower and upper edges
    optionally trimmed. If user enters admix times as integers then 
    they are checked only for validation, no trimming.
    """
    if tree[idx0].is_root() or tree[idx1].is_root():
        raise IpcoalError(f"no shared admix interval for idxs: {idx0} {idx1}")

    # get full possible intervals for these two nodes from the tree
    node0 = tree[idx0]
    ival0 = (node0.height, node0.up.height)
    node1 = tree[idx1]
    ival1 = (node1.height, node1.up.height)

    low_bin = max([ival0[0], ival1[0]])
    top_bin = min([ival0[1], ival1[1]])
    if top_bin < low_bin:
        raise IpcoalError(f"no shared admix interval for idxs: {idx0} {idx1}")

    # if user entered a time in gens then check if it works
    if heights is not None:
        if not ((heights[0] >= low_bin) and (heights[1] <= top_bin)):
            raise IpcoalError(
                f"admix interval ({heights}) not within a shared "
                f"edge interval for idxs: {idx0} {idx1}")
        return max(low_bin + 1e-3, heights[0]), min(top_bin - 1e-3, heights[1])

    # restrict migration within bin to a smaller interval
    length = top_bin - low_bin
    low_limit = max(low_bin + 1e-3, low_bin + (length * props[0]))
    top_limit = min(top_bin - 1e-3, low_bin + (length * props[1]))
    return low_limit, top_limit


def get_all_admix_edges(ttree, lower=0.25, upper=0.75, exclude_sisters=False):
    """
    Find all possible admixture edges on a tree.

    Edges are unidirectional, so the source and dest need to overlap in
    time interval. To retrict migration to occur away from nodes (these 
    can be harder to detect when validating methods) you can set upper 
    and lower limits. For example, to make all source migrations to occur
    at the midpoint of overlapping intervals in which migration can occur
    you can set upper=.5, lower=.5.   
    """
    # bounds on edge overlaps
    if lower is None:
        lower = 0.0
    if upper is None:
        upper = 1.0

    # for all nodes map the potential admixture interval
    for snode in ttree.treenode.traverse():
        if snode.is_root():
            snode.interval = (None, None)
        else:
            snode.interval = (snode.height, snode.up.height)

    # for all nodes find overlapping intervals
    intervals = {}
    for snode in ttree.treenode.traverse():
        for dnode in ttree.treenode.traverse():
            if not any([snode.is_root(), dnode.is_root(), dnode == snode]):

                # [option] skip sisters
                if (exclude_sisters) & (dnode.up == snode.up):
                    continue

                # check for overlap
                smin, smax = snode.interval
                dmin, dmax = dnode.interval

                # find if nodes have interval where admixture can occur
                low_bin = np.max([smin, dmin])
                top_bin = np.min([smax, dmax])              
                if top_bin > low_bin:

                    # restrict migration within bin to a smaller interval
                    length = top_bin - low_bin
                    low_limit = low_bin + (length * lower)
                    top_limit = low_bin + (length * upper)
                    intervals[(snode.idx, dnode.idx)] = (low_limit, top_limit)
    return intervals



def get_snps_count_matrix(tree, seqs):
    """
    Return a multidimensional SNP count matrix (sensu simcat and SVDquartets).    
    Compiles SNP data into a nquartets x 16 x 16 count matrix with the order
    of quartets determined by the shape of the tree.
    """
    # get all quartets for this size tree
    if isinstance(tree, toytree.ToyTree):
        quarts = list(itertools.combinations(range(tree.ntips), 4))
    else:
        # or, can be entered as tuples directly, e.g., [(0, 1, 2, 3)]
        quarts = tree

    # shape of the arr (count matrix)
    arr = np.zeros((len(quarts), 16, 16), dtype=np.int64)

    # iterator for quartets, e.g., (0, 1, 2, 3), (0, 1, 2, 4)...
    quartidx = 0
    for currquart in quarts:
        # cols indices match tip labels b/c we named tips node.idx
        quartsnps = seqs[currquart, :]
        # save as stacked matrices
        arr[quartidx] = count_matrix_int(quartsnps)
        # save flattened to counts
        quartidx += 1
    return arr



def calculate_dstat(seqs, p1, p2, p3, p4):
    """
    Calculate ABBA-BABA (D-statistic) from a count matrix. 
    """
    # order tips into ab|cd tree based on hypothesis
    mat = get_snps_count_matrix([(0, 1, 2, 3)], seqs[[p1, p2, p3, p4], :])[0]

    # calculate
    abba = sum([mat[i] for i in ABBA_IDX])
    baba = sum([mat[i] for i in BABA_IDX])
    if abba + baba == 0:
        dstat = 0.
    else:
        dstat = (abba - baba) / float(abba + baba)
    return pd.DataFrame({'dstat': [dstat], 'baba': [baba], 'abba': [abba]})



def abba_baba(model, testtuples):
    """
    Calculate ABBA/BABA statistic (D) as (ABBA - BABA) / (ABBA + BABA)

    Parameters:
    -----------
    model (ipcoal.Model Class object):
        A model class object from ipcoal that has generated sequence data by 
        calling either .sim_loci() or .sim_snps(). 

    testtuples (tuple, list):
        A tuple or list of tuples with the ordered taxon names for each test.
        The order should be (P1, P2, P3, P4). You can see the names of taxa 
        from the tree on which data were simulated from the model object using
        model.treeorig.draw();

    Returns: 
    ---------
    pandas.DataFrame

    """
    # check that data was simulated
    if not model.seqs:
        raise ipcoalError(
            "you must first simulate data with .sim_snps() or .sim_loci()")

    # ensure testtuples is a list of tuples
    if isinstance(testtuples, tuple):
        testtuples = [testtuples]

    # get tip order of tree and check that testtuple names are in tips
    tips = [i for i in model.treeorig.get_tip_labels()]
    for tup in testtuples:
        for name in tup:        
            if name not in tips:
                raise ipcoalError(
                    "name {} is not in the tree {}"
                    .format(name, tips))

    # get counts matrix
    counts = get_snps_count_matrix(model.tree, model.seqs)

    # store vals
    abbas = np.zeros(counts.shape[0], dtype=int)
    babas = np.zeros(counts.shape[0], dtype=int)
    dstats = np.zeros(counts.shape[0], dtype=float)
    p1 = np.zeros(counts.shape[0], dtype="U10")
    p2 = np.zeros(counts.shape[0], dtype="U10")
    p3 = np.zeros(counts.shape[0], dtype="U10")
    p4 = np.zeros(counts.shape[0], dtype="U10")

    # quartet iterator
    quarts = itertools.combinations(range(len(tips)), 4)

    # iterate over each (mat, quartet)
    idx = 0
    for count, qrt in zip(counts, quarts):

        # calculate
        abba = sum([count[i] for i in ABBA_IDX])
        baba = sum([count[i] for i in BABA_IDX])
        dstat = abs(abba - baba) / (abba + baba)

        # store stats
        abbas[idx] = abba
        babas[idx] = baba
        dstats[idx] = dstat

        # store taxa
        p1[idx] = tips[qrt[0]]
        p2[idx] = tips[qrt[1]]
        p3[idx] = tips[qrt[2]]
        p4[idx] = tips[qrt[3]]
        idx += 1

    # convert to dataframe   
    data = pd.DataFrame({
        "ABBA": np.array(abbas, dtype=int),
        "BABA": np.array(babas, dtype=int),
        "D": dstats,
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        }, 
        columns=["ABBA", "BABA", "D", "p1", "p2", "p3", "p4"],
    )
    return data



class Params(object):
    """ 
    A dict-like object for storing params values with a custom repr
    that shortens file paths, and which makes attributes easily viewable
    through tab completion in a notebook while hiding other funcs, attrs, that
    are in normal dicts. 
    """
    def __len__(self):
        return len(self.__dict__)

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __repr__(self):
        _repr = ""
        keys = sorted(self.__dict__.keys())
        if keys:
            _printstr = "{:<" + str(2 + max([len(i) for i in keys])) + "} {:<20}\n"
            for key in keys:
                _val = str(self[key]).replace(os.path.expanduser("~"), "~")
                _repr += _printstr.format(key, _val)
        return _repr



def calculate_pairwise_dist(mod, model=None, locus=None):
    """
    Return a pandas dataframe with pairwise distances between taxa.
    The model object should have already run sim.snps or sim.loci to 
    generate sequence data in .seqs. The returned distance is approx
    2X the distance from each tip to their common ancestor, so you can
    divide by 2 to get dist to mrca.
    """
    # a dataframe to fill with distances
    data = pd.DataFrame(
        index=mod.alpha_ordered_names,
        columns=mod.alpha_ordered_names,
        dtype=float,
    )

    # use either all loci concatenated, or a single locus
    if locus:
        arr = mod.seqs[locus]
    else:
        arr = np.concatenate(mod.seqs, axis=1)

    # calculate all pairs
    for pair in itertools.product(range(mod.nstips), range(mod.nstips)):

        # sample taxa
        idx0, idx1 = pair
        seq0 = arr[idx0]
        seq1 = arr[idx1]

        # hamming distance (proportion that are not matching)
        if model is None:
            dist = hamming_distance(seq0, seq1)
        elif model.upper().startswith("JC"):
            dist = jukes_cantor_distance(seq0, seq1)
        else:
            raise NotImplementedError("model not supported.")
        data.iloc[idx0, idx1] = dist
        data.iloc[idx1, idx0] = dist
    return data


@njit
def jukes_cantor_distance(seq0, seq1):
    "calculate the jukes cantor distance"
    dist = np.sum(seq0 != seq1) / seq0.size
    jcdist = (-3. / 4.) * np.log(1. - ((4. / 3.) * dist))
    return jcdist

@njit
def hamming_distance(seq0, seq1):
    "calculate hamming distance"
    return np.sum(seq0 != seq1) / seq0.size



def generate_recomb_map(length, num_pos, num_peaks, min_rate, max_rate, even_spacing=True):
    """
    Generates a discrete recombination map in the shape of a sine wave, with
    troughs going down to min_rate and peaks going up to max_rate. There
    are `num_peaks` peaks in the sine wave, and (`num_peaks`+1) troughs. The
    rates are unitless, and can be cM/Mb or crossovers/bp/gen, it's up to you.

    Parameters:
    -----------
    length (integer):
        The number of base pairs in the chromosome

    num_pos (integer):
        Number of positions delimiting windowed regions with estimated rates.

    num_peaks (integer):
        Number of peaks to the sine wave.

    min_rate (float):
        The max rate of the sine wave peaks.

    max_rate (float):
        The min rate of the sine wave peaks.

    even_spacing (True):
        If true, the rate sampling positions are evenly spread. If not, they
        are drawn from a uniform distribution along the length of the chrom.

    Returns:
    ---------
    pandas.DataFrame
        column 0: positions along the chromosome
        column 1: recombination rates
    """
    # wavelength
    sampspace_max = num_peaks * 2 * np.pi
    sampspace_min = 0

    # spacing between SNP markers
    if even_spacing:
        sampspace_x = np.linspace(sampspace_min, sampspace_max, num_pos)
    else:
        samp_points = np.hstack([
            np.random.uniform(sampspace_min, sampspace_max, num_pos - 2),
            [sampspace_min, sampspace_max]
        ])
        sampspace_x = np.sort(samp_points)

    # scale rates between min and max
    pos_rates = (-np.cos(sampspace_x) + 1) * ((max_rate - min_rate) / 2)
    pos_rates += min_rate
    pos = sampspace_x * (length / 2 / np.pi / num_peaks)

    # convert to a dataframe
    recomb_map = pd.DataFrame({
        "position": pos,
        "recomb_rate": pos_rates,
    })
    return recomb_map

  

def convert_intarr_to_bytearr(iarr):
    """
    An array of ints converted to bytes
    """
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
    arr[arr == b"01"] = b"M"
    arr[arr == b"10"] = b"M"
    arr[arr == b"02"] = b"R"
    arr[arr == b"20"] = b"R"
    arr[arr == b"03"] = b"W"
    arr[arr == b"30"] = b"W"
    arr[arr == b"12"] = b"S"
    arr[arr == b"21"] = b"S"
    arr[arr == b"13"] = b"Y"
    arr[arr == b"31"] = b"Y"
    arr[arr == b"23"] = b"K"
    arr[arr == b"32"] = b"K"

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



# def plot_test_values(self):

#     """
#     Returns a toyplot canvas
#     """
#     # canvas, axes = plot_test_values(self.tree)
#     if not self.counts.sum():
#         raise ipcoalError("No mutations generated. First call '.run()'")

#     # setup canvas
#     canvas = toyplot.Canvas(height=250, width=800)

#     ax0 = canvas.cartesian(
#         grid=(1, 3, 0))
#     ax1 = canvas.cartesian(
#         grid=(1, 3, 1),
#         xlabel="simulation index",
#         ylabel="migration intervals",
#         ymin=0,
#         ymax=self.tree.treenode.height)  # * 2 * self._Ne)
#     ax2 = canvas.cartesian(
#         grid=(1, 3, 2),
#         xlabel="proportion migrants",
#         # xlabel="N migrants (M)",
#         ylabel="frequency")

#     # advance colors for different edges starting from 1
#     colors = iter(toyplot.color.Palette())

#     # draw tree
#     self.tree.draw(
#         tree_style='c',
#         node_labels=self.tree.get_node_values("idx", 1, 1),
#         tip_labels=False,
#         axes=ax0,
#         node_sizes=16,
#         padding=50)
#     ax0.show = False

#     # iterate over edges
#     for tidx in range(self.aedges):
#         color = next(colors)

#         # get values for the first admixture edge
#         mtimes = self.test_values[tidx]["mtimes"]
#         mrates = self.test_values[tidx]["mrates"]
#         mt = mtimes[mtimes[:, 0].argsort()]
#         boundaries = np.column_stack((mt[:, 0], mt[:, 1]))

#         # plot
#         for idx in range(boundaries.shape[0]):
#             ax1.fill(
#                 # boundaries[idx],
#                 (boundaries[idx][0], boundaries[idx][0] + 0.1),
#                 (idx, idx),
#                 (idx + 0.5, idx + 0.5),
#                 along='y',
#                 color=color,
#                 opacity=0.5)

#         # migration rates/props
#         ax2.bars(
#             np.histogram(mrates, bins=20),
#             color=color,
#             opacity=0.5,
#         )

#     return canvas, (ax0, ax1, ax2)
