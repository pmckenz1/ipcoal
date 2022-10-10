#!/usr/bin/env python

"""Simulate sequence data of len `nsites` on `nloci` unlinked genealogies.

"""

from typing import TypeVar
import numpy as np
import pandas as pd
import msprime as ms
from ipcoal.utils.utils import IpcoalError

Model = TypeVar("Model")


def sim_loci(model: Model, nloci: int=1, nsites: int=1, precision: int=14) -> None:
    """Simulate sequence data on unlinked genealogies. 

    See `ipcoal.Model.sim_loci` docstring.
    """
    # check conflicting args
    if model._recomb_is_map:
        if nsites:
            raise IpcoalError(
                "Both nsites and recomb_map cannot be used together since"
                "the recomb_map also specifies nsites. To use a recomb_map"
                "specify nsites=None.")
        nsites = model.recomb.sequence_length

    # clear any existing stored tree sequences
    model.ts_dict = {}

    # allow scientific notation, e.g., 1e6
    nsites = int(nsites)
    nloci = int(nloci)

    # multidimensional array of sequence arrays to fill
    aseqarr = np.zeros((nloci, nsites), dtype=np.uint8)
    seqarr = np.zeros((nloci, model.nstips, nsites), dtype=np.uint8)

    # a list to be concatenated into the final dataframe of genealogies
    datalist = []
    for lidx in range(nloci):
        msgen = model._get_tree_sequence_generator(nsites)
        tree_seq = next(msgen)
        breaks = [int(i) for i in tree_seq.breakpoints()]
        starts = breaks[0:len(breaks) - 1]
        ends = breaks[1:len(breaks)]
        lengths = [i - j for (i, j) in zip(ends, starts)]

        data = pd.DataFrame({
            "start": starts,
            "end": ends,
            "nbps": lengths,
            "nsnps": 0,
            "tidx": 0,
            "locus": lidx,
            "genealogy": "",
            },
            columns=[
                'locus', 'start', 'end', 'nbps',
                'nsnps', 'tidx', 'genealogy'
            ],
        )

        # mutate the tree sequence
        mutated_ts = ms.sim_mutations(
            tree_sequence=tree_seq,
            rate=model.mut,
            model=model.subst_model,
            random_seed=model.rng_muts.integers(2**31),
            discrete_genome=True,
        )
        # iterate over the index of the dataframe to store each genealogy
        for mstree in mutated_ts.trees():
            nwk = mstree.newick(node_labels=model.tipdict, precision=precision)
            data.loc[mstree.index, "genealogy"] = nwk
            data.loc[mstree.index, "tidx"] = mstree.index
            data.loc[mstree.index, "nsnps"] = sum(1 for i in mstree.sites())

        # get genotype array and count nsnps
        genos = mutated_ts.genotype_matrix(alleles=model._alleles)

        # get an ancestral array with same root frequencies
        aseqarr[lidx] = model.rng_muts.choice(
            range(len(model.subst_model.alleles)),
            size=nsites,
            replace=True,
            p=model.subst_model.root_distribution,
        )
        seqarr[lidx, :, :] = aseqarr[lidx].copy()

        # impute mutated genos into aseq at variant sites
        for var in mutated_ts.variants():
            pos = int(var.site.position)
            aseqarr[lidx, pos] = model._alleles.index(var.site.ancestral_state)
            seqarr[lidx, :, pos] = genos[var.index]

        # store the dataframe
        datalist.append(data)

        # store the tree_sequence
        if model.store_tree_sequences:
            model.ts_dict[lidx] = mutated_ts

    # concatenate all of the genetree dfs
    data = pd.concat(datalist)
    data = data.reset_index(drop=True)

    # store values to object
    model.df = data
    model.seqs = seqarr[:, model._reorder]
    model.ancestral_seq = aseqarr

    # reset random seeds
    model._reset_random_generators()


if __name__ == "__main__":

    import toytree
    import ipcoal

    TREE = toytree.rtree.unittree(ntips=6, treeheight=1e6)
    MODEL = ipcoal.Model(TREE, Ne=1e4)
    sim_loci(MODEL, 10, 10)
    print(MODEL.df)
