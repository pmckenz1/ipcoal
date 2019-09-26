#!/usr/bin/env python

import time
import datetime
import itertools
import numpy as np
import pandas as pd

from ipywidgets import IntProgress, HTML, Box
from IPython.display import display


class SimcatError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


def get_all_admix_edges(ttree, lower=0.25, upper=0.75, exclude_sisters=False):
    """
    Find all possible admixture edges on a tree. Edges are unidirectional, 
    so the source and dest need to overlap in time interval. To retrict 
    migration to occur away from nodes (these can be harder to detect when 
    validating methods) you can set upper and lower limits. For example, to 
    make all source migrations to occur at the midpoint of overlapping 
    intervals in which migration can occur you can set upper=.5, lower=.5.   
    """
    # bounds on edge overlaps
    if lower is None:
        lower = 0.0
    if upper is None:
        upper = 1.0

    ## for all nodes map the potential admixture interval
    for snode in ttree.treenode.traverse():
        if snode.is_root():
            snode.interval = (None, None)
        else:
            snode.interval = (snode.height, snode.up.height)

    ## for all nodes find overlapping intervals
    intervals = {}
    for snode in ttree.treenode.traverse():
        for dnode in ttree.treenode.traverse():
            if not any([snode.is_root(), dnode.is_root(), dnode == snode]):

                ## [option] skip sisters
                if (exclude_sisters) & (dnode.up == snode.up):
                    continue

                ## check for overlap
                smin, smax = snode.interval
                dmin, dmax = dnode.interval

                ## find if nodes have interval where admixture can occur
                low_bin = np.max([smin, dmin])
                top_bin = np.min([smax, dmax])              
                if top_bin > low_bin:

                    # restrict migration within bin to a smaller interval
                    length = top_bin - low_bin
                    low_limit = low_bin + (length * lower)
                    top_limit = low_bin + (length * upper)
                    intervals[(snode.idx, dnode.idx)] = (low_limit, top_limit)
    return intervals



def tile_reps(array, nreps):
    "used to fill labels in the simcat.Database for replicates"
    ts = array.size
    nr = nreps
    result = np.array(
        np.tile(array, nr)
        .reshape((nr, ts))
        .T.flatten())
    return result


# def progress_bar(njobs, nfinished, start, message=""):
#     "prints a progress bar"
#     ## measure progress
#     if njobs:
#         progress = 100 * (nfinished / njobs)
#     else:
#         progress = 100

#     ## build the bar
#     hashes = "#" * int(progress / 5.)
#     nohash = " " * int(20 - len(hashes))

#     ## get time stamp
#     elapsed = datetime.timedelta(seconds=int(time.time() - start))

#     ## print to stderr
#     args = [hashes + nohash, int(progress), elapsed, message]
#     print("\r[{}] {:>3}% | {} | {}".format(*args), end="")
#     sys.stderr.flush()



__INVARIANTS__ = """
AAAA AAAC AAAG AAAT  AACA AACC AACG AACT  AAGA AAGC AAGG AAGT  AATA AATC AATG AATT
ACAA ACAC ACAG ACAT  ACCA ACCC ACCG ACCT  ACGA ACGC ACGG ACGT  ACTA ACTC ACTG ACTT
AGAA AGAC AGAG AGAT  AGCA AGCC AGCG AGCT  AGGA AGGC AGGG AGGT  AGTA AGTC AGTG AGTT
ATAA ATAC ATAG ATAT  ATCA ATCC ATCG ATCT  ATGA ATGC ATGG ATGT  ATTA ATTC ATTG ATTT

CAAA CAAC CAAG CAAT  CACA CACC CACG CACT  CAGA CAGC CAGG CAGT  CATA CATC CATG CATT
CCAA CCAC CCAG CCAT  CCCA CCCC CCCG CCCT  CCGA CCGC CCGG CCGT  CCTA CCTC CCTG CCTT
CGAA CGAC CGAG CGAT  CGCA CGCC CGCG CGCT  CGGA CGGC CGGG CGGT  CGTA CGTC CGTG CGTT
CTAA CTAC CTAG CTAT  CTCA CTCC CTCG CTCT  CTGA CTGC CTGG CTGT  CTTA CTTC CTTG CTTT

GAAA GAAC GAAG GAAT  GACA GACC GACG GACT  GAGA GAGC GAGG GAGT  GATA GATC GATG GATT
GCAA GCAC GCAG GCAT  GCCA GCCC GCCG GCCT  GCGA GCGC GCGG GCGT  GCTA GCTC GCTG GCTT
GGAA GGAC GGAG GGAT  GGCA GGCC GGCG GGCT  GGGA GGGC GGGG GGGT  GGTA GGTC GGTG GGTT
GTAA GTAC GTAG GTAT  GTCA GTCC GTCG GTCT  GTGA GTGC GTGG GTGT  GTTA GTTC GTTG GTTT

TAAA TAAC TAAG TAAT  TACA TACC TACG TACT  TAGA TAGC TAGG TAGT  TATA TATC TATG TATT
TCAA TCAC TCAG TCAT  TCCA TCCC TCCG TCCT  TCGA TCGC TCGG TCGT  TCTA TCTC TCTG TCTT
TGAA TGAC TGAG TGAT  TGCA TGCC TGCG TGCT  TGGA TGGC TGGG TGGT  TGTA TGTC TGTG TGTT
TTAA TTAC TTAG TTAT  TTCA TTCC TTCG TTCT  TTGA TTGC TTGG TTGT  TTTA TTTC TTTG TTTT
"""
INVARIANTS = np.array(__INVARIANTS__.strip().split()).reshape(16, 16)
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


def abba_baba(counts):
    """
    Calculate ABBA/BABA statistic (D) as (ABBA - BABA) / (ABBA + BABA)
    """
    # store vals
    abbas = []
    babas = []
    dstats = []
    quartets = []
    reps = []
    
    # iterate over reps and quartets
    for rep in range(counts.shape[0]):
        
        # quartet iterator
        quarts = itertools.combinations(range(counts.shape[1]), 4)
    
        # iterate over each mat, quartet
        for matrix, qrt in zip(range(counts.shape[1]), quarts):
            count = counts[rep, matrix]

            abba = sum([count[i] for i in ABBA_IDX])
            abbas.append(abba)

            baba = sum([count[i] for i in BABA_IDX])
            babas.append(baba)

            dstat = abs(abba - baba) / (abba + baba)
            dstats.append(dstat)

            quartets.append(qrt)
            reps.append(rep)
    
    # convert to dataframe   
    df = pd.DataFrame({
        "ABBA": np.array(abbas, dtype=int),
        "BABA": np.array(babas, dtype=int),
        "D": dstats,
        "quartet": quartets,
        "reps": reps,
        }, 
        columns=["reps", "ABBA", "BABA", "D", "quartet"],
    )
    return df



class Progress(object):
    def __init__(self, njobs, message, children):

        # data
        self.njobs = njobs
        self.message = message
        self.start = time.time()

        # the progress bar 
        self.bar = IntProgress(
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
            children + [self.label, self.bar]
        ]
        self.widget = Box(
            children=children + [self.label, self.bar], 
            layout={
                "display": "flex",
                "flex_flow": "column",
                "height": "{}px".format(sum(heights) + 5),
                "margin": "5px 0px 5px 0px",
            })
        
    @property
    def printstr(self):
        elapsed = datetime.timedelta(seconds=int(time.time() - self.start))
        s1 = "<span style='font-size:14px; font-family:monospace'>"
        s2 = "</span>"
        inner = "{} | {:>3}% | {}".format(
            self.message, 
            int(100 * (self.bar.value / self.njobs)),
            elapsed,
        )

        return s1 + inner + s2

    def display(self):
        display(self.widget)
    
    def increment_all(self, value=1):
        self.bar.value += value
        if self.bar.value == self.njobs:
            self.bar.bar_style = "success"
        self.increment_time()
            
    def increment_time(self):
        self.label.value = self.printstr
