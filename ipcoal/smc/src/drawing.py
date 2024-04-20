#!/usr/bin/env python

"""Visualizations for the MS-SMC process

"""

from typing import Mapping, Sequence
from toytree import ToyTree
import toyplot
import numpy as np
from ipcoal.msc import get_genealogy_embedding_table
from ipcoal.smc.src.embedding import TreeEmbedding
from ipcoal.smc.src.ms_smc_tree_prob import get_prob_tree_unchanged_given_b_and_tr_from_arrays
from ipcoal.smc.src.ms_smc_topo_prob import get_prob_topo_unchanged_given_b_and_tr_from_arrays
from ipcoal.smc.src.ms_smc_rvs import (
    get_distribution_waiting_distance_to_recombination,
    get_distribution_waiting_distance_to_tree_change,
    get_distribution_waiting_distance_to_topo_change,
)


__all__ = ["plot_edge_probabilities", "plot_waiting_distance_distributions"]


def plot_edge_probabilities(
    species_tree: ToyTree,
    genealogy: ToyTree,
    imap: Mapping[str, Sequence[str]],
    branch: int,
    stack: int = 1,
    **kwargs,
) -> toyplot.canvas.Canvas:
    """Return a toyplot canvas with probabilities along an edge.

    """
    # setup the canvas and axes
    if not stack:
        canvas = toyplot.Canvas(
            height=kwargs.get("height", 250),
            width=kwargs.get("width", 800),
        )
        axstyle = dict(ymin=0, ymax=1, margin=65)
        ax0 = canvas.cartesian(grid=(1, 3, 0), label="Prob(no-change)", **axstyle)
        ax1 = canvas.cartesian(grid=(1, 3, 1), label="Prob(tree-change)", **axstyle)
        ax2 = canvas.cartesian(grid=(1, 3, 2), label="Prob(topo-change)", **axstyle)

    else:
        canvas = toyplot.Canvas(
            height=kwargs.get("height", 750),
            width=kwargs.get("width", 300),
        )
        axstyle = dict(ymin=0, ymax=1, margin=65)
        ax0 = canvas.cartesian(grid=(3, 1, 0), label="Prob(no-change)", **axstyle)
        ax1 = canvas.cartesian(grid=(3, 1, 1), label="Prob(tree-change)", **axstyle)
        ax2 = canvas.cartesian(grid=(3, 1, 2), label="Prob(topo-change)", **axstyle)

    # Select a branch to plot and get its relations
    branch = genealogy[branch]
    bidx = branch.idx
    sidx = branch.get_sisters()[0].idx
    pidx = branch.up.idx

    # Get genealogy embedding table
    etable = get_genealogy_embedding_table(species_tree, genealogy, imap, encode=True)
    btable = etable.loc[etable[bidx].astype(np.bool_)]

    emb, enc, barr, sarr, rarr = TreeEmbedding(
        species_tree, genealogy, imap, nproc=1).get_data()

    # Plot probabilities of change types over a single branch
    # Note these are 'unchange' probs and so we plot 1 - Prob here.
    times = np.linspace(branch.height, branch.up.height, 200, endpoint=False)
    pt_nochange_tree = [
        get_prob_tree_unchanged_given_b_and_tr_from_arrays(
            emb[0], enc[0], bidx, itime,
        ) for itime in times
    ]
    pt_nochange_topo = [
        get_prob_topo_unchanged_given_b_and_tr_from_arrays(
            emb[0], enc[0], bidx, sidx, pidx, itime,
        ) for itime in times
    ]

    # add line and fill for probabilities
    ax0.plot(times, pt_nochange_tree, stroke_width=5)
    ax0.fill(times, pt_nochange_tree, opacity=0.33)
    ax1.plot(times, 1 - np.array(pt_nochange_tree), stroke_width=5)
    ax1.fill(times, 1 - np.array(pt_nochange_tree), opacity=0.33)
    ax2.plot(times, 1 - np.array(pt_nochange_topo), stroke_width=5)
    ax2.fill(times, 1 - np.array(pt_nochange_topo), opacity=0.33)

    # add vertical lines at interval breaks
    style = {"stroke": "black", "stroke-width": 2, "stroke-dasharray": "4,2"}
    intervals = [btable.start.iloc[0]] + list(btable.stop - 0.001)
    for itime in intervals:
        iprob_tree = get_prob_tree_unchanged_given_b_and_tr_from_arrays(emb[0], enc[0], bidx, itime)
        iprob_topo = get_prob_topo_unchanged_given_b_and_tr_from_arrays(emb[0], enc[0], bidx, sidx, pidx, itime)
        ax0.plot([itime, itime], [0, iprob_tree], style=style)
        ax1.plot([itime, itime], [0, 1 - iprob_tree], style=style)
        ax2.plot([itime, itime], [0, 1 - iprob_topo], style=style)

    # style the axes
    for axis in (ax0, ax1, ax2):
        axis.x.ticks.show = axis.y.ticks.show = True
        axis.y.domain.show = False
        axis.x.ticks.near = axis.y.ticks.near = 7.5
        axis.x.ticks.far = axis.y.ticks.far = 0
        axis.x.ticks.labels.offset = axis.y.ticks.labels.offset = 15
        axis.x.label.text = f"Time of recombination on branch {bidx}"
        axis.y.label.text = "Probability"
        axis.x.label.offset = axis.y.label.offset = 35
        axis.x.spine.style['stroke-width'] = axis.y.spine.style['stroke-width'] = 1.5
        axis.x.ticks.style['stroke-width'] = axis.y.ticks.style['stroke-width'] = 1.5
        axis.label.offset = 20
        axis.x.ticks.locator = toyplot.locator.Explicit([btable.start.iloc[0]] + list(btable.stop))
    return canvas


def plot_waiting_distance_distributions(
    species_tree: ToyTree,
    genealogy: ToyTree,
    imap: Mapping[str, Sequence[str]],
    recombination_rate: float,
    **kwargs,
) -> toyplot.canvas.Canvas:
    """Return a toyplot canvas with waiting distance distributions.
    """
    canvas = toyplot.Canvas(
        height=kwargs.get("height", 300),
        width=kwargs.get("width", 450),
    )
    axes = canvas.cartesian(margin=65)
    axes.x.ticks.show = axes.y.ticks.show = True
    axes.y.domain.show = False
    axes.x.ticks.near = axes.y.ticks.near = 7.5
    axes.x.ticks.far = axes.y.ticks.far = 0
    axes.x.ticks.labels.offset = axes.y.ticks.labels.offset = 15
    axes.x.label.text = "Waiting distance to next event"
    axes.y.label.text = "Probability of event type"
    axes.x.label.offset = axes.y.label.offset = 35
    axes.x.spine.style['stroke-width'] = axes.y.spine.style['stroke-width'] = 1.5
    axes.x.ticks.style['stroke-width'] = axes.y.ticks.style['stroke-width'] = 1.5
    axes.label.offset = 20

    rv_recomb = get_distribution_waiting_distance_to_recombination(genealogy, recombination_rate)
    rv_tree = get_distribution_waiting_distance_to_tree_change(species_tree, genealogy, imap, recombination_rate)
    rv_topo = get_distribution_waiting_distance_to_topo_change(species_tree, genealogy, imap, recombination_rate)

    marks = []
    for dist in [rv_recomb, rv_tree, rv_topo]:
        xs_ = np.linspace(dist.ppf(0.025), dist.ppf(0.975), 100)
        ys_ = dist.pdf(xs_)
        marks.append(axes.plot(xs_, ys_, stroke_width=4))
        axes.fill(xs_, ys_, opacity=1 / 3)

    canvas.legend(
        entries=[
            ('no-change', marks[0]),
            ('tree-change', marks[1]),
            ('topo-change', marks[2]),
        ],
        bounds=("50%", "75%", "15%", "45%"),
    )
    return canvas


if __name__ == "__main__":

    import toytree
    from ipcoal.msc import get_test_data
    SPTREE, GTREE, IMAP = get_test_data()
    c0 = plot_edge_probabilities(SPTREE, GTREE, IMAP, branch=2)
    c1 = plot_waiting_distance_distributions(SPTREE, GTREE, IMAP, recombination_rate=2e-9)
    toytree.utils.show([c0, c1])
