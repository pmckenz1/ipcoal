#!/usr/bin/env python

"""Draw a sequence array.
"""

import numpy as np
import toyplot
from toytree.utils import ScrollableCanvas
from loguru import logger

# register logger to module
logger = logger.bind(name="ipcoal")


def draw_seqview(
    self,
    idx,
    start,
    end,
    width,
    height,
    show_text,
    scrollable,
    max_width,
    gaps=1.5,
    **kwargs):
    """Draws a sequence array as a colored toyplot table.

    """
    # bail out if no seqs array
    if not self.seqs.size:
        logger.warning("No sequences simulated.")
        return None, None

    # if SNPs then concatenate
    if self.seqs.ndim == 2:
        arr = self.seqs
    else:
        if not idx:
            arr = self.seqs[0]
        else:
            arr = self.seqs[idx]

    # enforce max_width
    end = end if end is not None else min(arr.shape[1], max_width)
    arr = arr[:, start:end]

    # auto set a good looking height and width based on arr dims
    margin = kwargs.get("margin", 50)
    if not height:
        height = 16 * arr.shape[0]
    if not width:
        width = max(16 * arr.shape[1], margin)
        width += width * .2

    # add margin space
    height += 100
    width += 100

    # build canvas and table
    if scrollable and (width > 500):
        canvas = ScrollableCanvas(width, height)
    else:
        canvas = toyplot.Canvas(width, height)
    table = canvas.table(
        rows=arr.shape[0],
        columns=arr.shape[1] + 1,
        bounds=(50, -50, 50, -50), #"10%", "90%"),
        **kwargs,
    )

    # style table cells
    colors = ['red', 'green', 'blue', 'orange', 'grey']
    bases = ["A", "C", "G", "T", "N"]
    for cidx in range(5):

        # select bases in 0-3 or 9
        if cidx == 4:
            tdx = np.where(arr[:, :] == 9)
        else:
            tdx = np.where(arr[:, :] == cidx)

        # color the cells
        table.cells.cell[tdx[0], tdx[1] + 1].style = {
            "fill": colors[cidx], "opacity": 0.5}

        # optionally overlay text
        if show_text:
            table.cells.cell[tdx[0], tdx[1] + 1].data = bases[cidx]
    table.cells.cell[:, 1:].lstyle = {"font-size": "8px"}

    # dividers
    table.body.gaps.columns[...] = gaps
    table.body.gaps.rows[...] = gaps

    # add taxon labels
    table.cells.cell[:, 0].data = self.alpha_ordered_names
    # table.cells.cell[:, 0].lstyle = {"text-anchor": "end", "font-size": "11px"}
    table.cells.cell[:, 0].width = 50
    return canvas, table
