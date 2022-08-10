#!/usr/bin/env python

"""Simulate `nloci` unlinked genealogies.

"""

from typing import TypeVar, Optional
import numpy as np
import pandas as pd
import msprime as ms
# from ipcoal.utils.utils import IpcoalError

Model = TypeVar("Model")


def sim_snps(
    model: Model,
    nsnps: int=1,
    min_alleles: int=2,
    max_alleles: Optional[int]=None,
    min_mutations: int=1,
    max_mutations: Optional[int]=None,
    repeat_on_trees: bool=False,
    precision: int=14,
    # exclude_fixed: bool = False,       
    ) -> None:
    """Simulate a single SNP on each unlinked genealogy.

    See `ipcoal.Model.sim_snps` docstring.
    """
    # allow scientific notation, e.g., 1e6
    nsnps = int(nsnps)

    # get min and set max_mutations minimum to 1
    max_mutations = (max_mutations if max_mutations else 100000)
    max_alleles = (max_alleles if max_alleles else 100000)
    assert min_mutations > 0, "min_mutations must be >=1"
    assert max_alleles >= min_alleles, "max_alleles must be >= min_alleles"

    # get infinite-ish TreeSequence generator
    msgen = model._get_tree_sequence_generator(1, snp=True)

    # store results (nsnps, ntips); def. 1000 SNPs
    newicks = []
    snpidx = 0
    snparr = np.zeros((model.nstips, nsnps), dtype=np.uint8)
    ancarr = np.zeros(nsnps, np.uint8)

    # continue until we get nsnps
    while 1:

        # bail out if nsnps finished
        if snpidx == nsnps:
            break

        # get next tree from tree_sequence generator
        treeseq = next(msgen)

        # try to land a mutation
        mutated_ts = ms.sim_mutations(
            tree_sequence=treeseq,
            rate=model.mut,
            model=model.subst_model,
            random_seed=model.rng_muts.integers(2**31),
            discrete_genome=True,
        )

        # if repeat_on_trees then keep sim'n til we get a SNP
        if repeat_on_trees:
            while 1:
                mutated_ts = ms.sim_mutations(
                    tree_sequence=treeseq,
                    rate=model.mut,
                    model=model.subst_model,
                    random_seed=model.rng_muts.integers(2**31),
                    discrete_genome=True,
                )
                try:
                    variant = next(mutated_ts.variants())
                except StopIteration:
                    continue
                if not max_mutations >= len(variant.site.mutations) >= min_mutations:                        
                    continue
                if not max_alleles >= variant.num_alleles >= min_alleles:
                    continue
                break

        # otherwise simply require >1 mutation and >1 alleles
        else:
            try:
                variant = next(mutated_ts.variants())
            except StopIteration:
                continue
            # number of mutations (0, 1, or >1)
            if not max_mutations >= len(variant.site.mutations) >= min_mutations:
                continue
            # number of alleles 
            if not max_alleles >= variant.num_alleles >= min_alleles:
                continue

        # Store result and advance counter
        snparr[:, snpidx] = mutated_ts.genotype_matrix(alleles=model._alleles)
        ancarr[snpidx] = model._alleles.index(variant.site.ancestral_state)

        # store the newick string
        newicks.append(
            treeseq.first().newick(
                node_labels=model.tipdict,
                precision=precision,
            )
        )

        # store the tree_sequence
        if model.store_tree_sequences:
            model.ts_dict[snpidx] = mutated_ts

        # advance counter
        snpidx += 1

    # init dataframe
    model.df = pd.DataFrame({
        "start": 0,
        "end": 1,
        "genealogy": newicks,
        "nbps": 1,
        "nsnps": 1,
        "tidx": 0,
        "locus": range(nsnps),
        },
        columns=[
            'locus', 'start', 'end', 'nbps',
            'nsnps', 'tidx', 'genealogy',
        ],
    )

    # reorder rows to be alphanumeric sorted.
    model.seqs = snparr[model._reorder]
    model.ancestral_seq = ancarr

    # reset random seeds
    model._reset_random_generators()


if __name__ == "__main__":

    import toytree
    import ipcoal

    TREE = toytree.rtree.unittree(ntips=6, treeheight=1e6)
    MODEL = ipcoal.Model(TREE, Ne=1e4)
    sim_snps(MODEL, 10)
    print(MODEL.df)
