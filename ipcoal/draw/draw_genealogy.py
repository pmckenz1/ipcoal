#!/usr/bin/env python

"""Draw a single genealogy with option to show substitutions.


TODO: we could alternatively simplify this by just extracting the
tree, and also extracting the substitution data from the ts, and
then drawing the substitutions as annotation marks on the tree. 
This would be more atomic in terms of toytree development.
"""

from typing import TypeVar, Optional
from loguru import logger
import toytree

logger = logger.bind(name="ipcoal")
Model = TypeVar("ipcoal.Model")


def draw_genealogy(
    model: Model, 
    idx: Optional[int]=None, 
    show_substitutions: bool=False, 
    **kwargs,
    ):
    """

    """
    # select which genealogy to draw
    idx = idx if idx else 0
    tree = toytree.tree(model.df.genealogy[idx])

    # optional: load as a tree sequence to extract substitution info.
    if show_substitutions:
        if idx not in model.ts_dict:
            logger.warning(
                "Can only show substitutions if ipcoal.Model object was "
                "initialized with the setting 'store_tree_sequences=True"
            )
        else:
            tseq = toytree.utils.toytree_sequence(
                model.ts_dict[0], name_dict=model.tipdict)
            canvas, axes, mark = tseq.draw_tree(
                idx=idx, tip_labels=True, **kwargs
            )
    else:
        canvas, axes, mark = tree.draw(ts='c', tip_labels=True, **kwargs)
    return canvas, axes, mark
