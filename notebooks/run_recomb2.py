


#!/usr/bin/env python




def get_gene_tree_accuracy(time: int, nloci: int, recomb: float, idx: int, nproc: int=10):
    """Returns the proportion of correct astral results from nreplicate analyses.
    
    This function infers species trees from simulated data. It first loads the 
    simulated sequence data and raxml trees from the previous 
    `nloci` from the full simulated dataset files from the largest global NLOCI.
    The Ne parameter is maintained constant using the global POPSIZE, but the
    edge lengths in units of generations are affected by `time`, thus
    the ratio of (tg) edge length / Ne can be varied by changing `time`.
    """
    # load the full largest NLOCI raxml inferred trees dataset
    params = f"th{int(time)}-nloci{NLOCI[-1]}-recomb{int(bool(recomb))}-rep{idx}"
    raxpath = TMPDIR / (params + "-raxml_trees.csv")       
    assert raxpath.exists(), f"{raxpath} not found."
    
    # pass inferred ML trees to ASTRAL to infer a species tree
    mtree = mtree = toytree.mtree(raxpath)
    ast_tree = ipcoal.phylo.infer_astral_tree(mtree, binary_path=ASTRAL)
        
    # does it match?
    ast_match = True if ast_tree.distance.get_treedist_rf(IMBTREE, True) == 0 else False

    # infer concatenation tree also
    concat_tree = ipcoal.phylo.infer_raxml_ng_tree(model, nthreads=nproc, binary_path=RAXML_NG)

    # does it match?
    concat_match = True if concat_tree.distance.get_treedist_rf(IMBTREE, True) == 0 else False

    # save result to a filename w/ param settings in WORKDIR
    result = f"ast{int(ast_match)}-concat{int(concat_match)}"
    outfile = (
        WORKDIR / path.name
        .replace("-sim_loci.csv", f"-{result}-raxml_trees.csv")
        .replace(f"-nloci{NLOCI[-1]}-", f"-nloci{nloci}-")
    )
    outfile.touch()