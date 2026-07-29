#!/usr/bin/env python

"""Draw a single genealogy with option to show substitutions.

TODO: we could alternatively simplify this by just extracting the
tree, and also extracting the substitution data from the ts, and
then drawing the substitutions as annotation marks on the tree.
This would be more atomic in terms of toytree development. It's just
more storing of data that is otherwise in ts generators.
"""

from typing import TypeVar, Optional
from loguru import logger
import toytree

logger = logger.bind(name="ipcoal")
Model = TypeVar("ipcoal.Model")


def draw_genealogy(
    model: Model,
    idx: Optional[int] = None,
    show_substitutions: bool = False,
    **kwargs,
):
    """Draw a single genealogy from the tree sequence.

    This uses the toytree.ToyTreeSequence object.
    """
    # select which genealogy to draw
    idx = idx if idx else 0

    # optional: load as a tree sequence to extract substitution info.
    if show_substitutions:
        # get which locus contains df index idx
        lidx = model.df.loc[idx, "locus"]

        # plot without substitutions and report warning
        if lidx not in model.ts_dict:
            tree = toytree.tree(model.df.genealogy[idx])
            canvas, axes, mark = tree.draw(ts='c', tip_labels=True, **kwargs)
            logger.warning(
                "Can only show substitutions if ipcoal.Model object has setting "
                "'store_tree_sequences=True' during or after init."
            )

        # create ToyTreeSequence and draw a tree from it.
        else:
            # load ts as a ToyTreeSequence object
            tseq = toytree.utils.toytree_sequence(
                model.ts_dict[lidx],
                name_dict=model.tipdict,
            )

            # extract tree at index tidx of locus lidx (model.df index idx)
            tidx = model.df.loc[idx, "tidx"]
            kwargs["tip_labels"] = kwargs.get("tip_labels", True)
            kwargs["scale_bar"] = kwargs.get("scale_bar", True)

            # call the ToyTreeSequence draw function to draw the tree
            # with mutations included. Note: this is where we could
            # alternatively extract mutation info and draw manually
            # on top of a normal tree plot for more flexibility.
            canvas, axes, mark = tseq.draw_tree(idx=tidx, **kwargs)

    else:
        tree = toytree.tree(model.df.genealogy[idx])
        canvas, axes, mark = tree.draw(ts='c', tip_labels=True, **kwargs)
    return canvas, axes, mark


if __name__ == "__main__":

    import ipcoal
    model = ipcoal.Model(Ne=100_000, nsamples=5)
    model.sim_loci(1, 10000)
    model.draw_genealogy(show_substitutions=True)
