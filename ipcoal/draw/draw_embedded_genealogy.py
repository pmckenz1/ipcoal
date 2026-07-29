#!/usr/bin/env python

"""Draw an embedded genealogy in a demographic model


Example
--------
>>> S, G, I = ipcoal.msc.get_test_data()
>>> ipcoal.draw.draw_embedded_genealogy(S, G, I)

"""

from typing import Dict, Sequence, Tuple, Union
from itertools import chain
import pandas as pd
import ipcoal
from toytree.color import ColorType
from toytree.core import ToyTree, Canvas, Cartesian, Mark
from toytree.drawing.src.draw_demography import EmbeddingPlot


def draw_embedded_genealogy(
    sptree: ToyTree,
    genealogy: ToyTree,
    imap: Dict[Union[str, int], Sequence[Union[str, int]]],
    container_width: int = 350,
    container_height: int = 300,
    container_blend: bool = False,
    container_fill: ColorType = "black",
    container_fill_opacity: float = 0.25,
    container_fill_opacity_alternate: bool = True,
    container_stroke: ColorType = "black",
    container_stroke_opacity: float = 1.0,
    container_stroke_width: float = 2,
    container_root_height: Union[int, bool] = True,
    container_interval_spacing: float = 1.5,
    container_interval_minwidth: float = 2,
    container_interval_maxwidth: float = 8,
    node_fill: ColorType = "black",
    node_fill_opacity: float = 1.0,
    node_stroke: ColorType = None,
    node_stroke_width: int = 1,
    node_size: int = 5,
    edge_stroke: ColorType = "black",
    edge_stroke_width: int = 2,
    edge_stroke_inherit: bool = True,
    edge_stroke_opacity: float = 1.0,
    edge_samples: int = 10,
    edge_variance: float = 0.0,
    tip_labels_size: int = 10,
    **kwargs,
) -> Tuple[Canvas, Cartesian, Mark]:
    """Return a toyplot drawing (c, a, m) of a genealogy embedding.

    """
    # check that inputs are matched
    assert sorted(sptree.get_tip_labels()) == sorted(imap), (
        "species tree tip names must all appear as keys in the imap dict.")
    g_tip_labels = sorted(chain(*imap.values()))
    assert sorted(genealogy.get_tip_labels()) == g_tip_labels, (
        "genealogy tip names must all appear as values in the imap dict.")

    emb = EmbeddingPlot(sptree, genealogy, imap, blend=container_blend)
    emb.draw(
        container_width,
        container_height,
        container_fill,
        container_fill_opacity,
        container_fill_opacity_alternate,
        container_stroke,
        container_stroke_opacity,
        container_stroke_width,
        container_root_height,
        node_fill,
        node_fill_opacity,
        node_stroke,
        node_stroke_width,
        node_size,
        edge_stroke,
        edge_stroke_width,
        edge_stroke_inherit,
        edge_stroke_opacity,
        edge_samples,
        edge_variance,
        tip_labels_size,
    )
    return emb.canvas, emb.axes

    # create a tmp demographic model requiring sptree to have Ne set.
    tmp = ipcoal.Model(
        sptree,
        nsamples={i: len(j) for (i, j) in imap.items()},
        Ne=kwargs.get("Ne"),
    )

    # rename alpha_ordered_names to match input genealogy/imap
    tmp.alpha_ordered_names = g_tip_labels

    # fill df
    tmp.df = pd.DataFrame({
        "locus": 0,
        "start": 0,
        "end": 1,
        "nbps": 1,
        "nsnps": 0,
        "tidx": 0,
        "genealogy": genealogy.write()
    }, index=[0])
    # return tmp.df
    canvas, axes = tmp.draw_demography(0)




    return canvas, axes


if __name__ == "__main__":
    S, G, I = ipcoal.msc.get_test_data()
    res = draw_embedded_genealogy(S, G, I)
    print(res)