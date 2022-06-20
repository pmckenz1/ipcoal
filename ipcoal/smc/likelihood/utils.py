#!/usr/bin/env python

"""Utility functions for SMC subpackage."""

from typing import Iterator, Tuple, Sequence
import toytree
import numpy as np
import ipcoal


def iter_spans_and_trees(model: ipcoal.Model) -> Iterator[Tuple[int, toytree.ToyTree]]:
    """Yield ToyTree objects parsed from newick strings."""
    for span, newick in model.df[["nbps", "genealogy"]].values:
        yield (span, toytree.tree(newick))


def iter_spans_and_topologies(model: ipcoal.Model) -> Iterator[int]:
    """Return an array of genealogical topology lengths.

    Parameters
    ----------
    model: ipcoal.Model
        An ipcoal.Model object that has called a simulate function
        such as `sim_trees`, `sim_loci`, etc. such that it has data
        in its `.df` attribute.
    """
    current = None
    interval = 0
    for span, gtree in iter_spans_and_trees(model):
        interval += span
        if current:
            new = gtree.get_topology_id(exclude_root=False)
            if current != new:
            # if current.distance.get_treedist_rf(gtree, False):
                yield interval, gtree
                # current = gtree
                current = new
                interval = 0
        else:
            # current = gtree
            current = gtree.get_topology_id(exclude_root=False)


def iter_unique_topologies_from_genealogies(
    genealogies: Sequence[toytree.ToyTree],
    #average_branch_lengths: bool=False,
    ) -> Sequence[toytree.ToyTree]:
    """Returns the genealogy at each topology change."""

    # initial tree
    current = genealogies[0]
    cidx = current.get_topology_id(exclude_root=False)
    tree_bunch = [current]

    # iterate over genealogies
    for gtree in genealogies[1:]:
        nidx = gtree.get_topology_id(exclude_root=False)
        if cidx != nidx:

            # optional: average branch lens across genealogies
            # deprecated for now, need to disallow negative blens.
            # if average_branch_lengths:
                # print(f'tree_bunch: {len(tree_bunch)}')
                # elens = {}
                # for nid in range(current.ntips, current.nnodes):
                #     hei = np.mean([g[nid].height for g in tree_bunch])
                #     elens[nid] = hei
                # print(elens)
                # current = current.set_node_data("height", mapping=elens)
        
            yield current
            current = gtree
            cidx = nidx
            tree_bunch = [gtree]
        else:
            tree_bunch.append(gtree)


def get_topology_interval_lengths(model: ipcoal.Model) -> np.ndarray:
    """Return an array of genealogical topology lengths.

    Parameters
    ----------
    model: ipcoal.Model
        An ipcoal.Model object that has called a simulate function
        such as `sim_trees`, `sim_loci`, etc. such that it has data
        in its `.df` attribute.
    """
    return np.array([i[0] for i in iter_spans_and_topologies(model)])


if __name__ == "__main__":

    MODEL = ipcoal.Model(Ne=100000, nsamples=5, seed_trees=333)
    MODEL.sim_trees(1, 80000)
    print(MODEL.df)

    for SPAN, GTREE in iter_spans_and_topologies(MODEL):
        print(SPAN)
        GTREE.treenode.draw_ascii()

    ivals = get_topology_interval_lengths(MODEL)
    print(ivals, ivals.sum())

    GENEALOGIES = toytree.mtree(MODEL.df.genealogy)
    for g in iter_unique_topologies_from_genealogies(GENEALOGIES, 0):
        print(g.get_topology_id(exclude_root=0), g.get_node_data("height"))
