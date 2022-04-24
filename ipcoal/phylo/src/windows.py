


def infer_gene_tree_windows(
    self,
    window_size: Optional[int]=None,
    inference_method: str = 'raxml',
    inference_args: Dict[str,str] = None,
    diploid: bool = False,
    ) -> pd.DataFrame:
    """Infer gene trees in intervals using a phylogenetic tool.

    The method is applied to windows spanning every locus.
    If window_size is None then each locus is treated as an
    entire concatenated window.

    Parameters
    ----------
    window_size: int, None
        The size of non-overlapping windows to be applied across the
        sequence alignment to infer gene tree windows. If None then
        a single gene tree is inferred for the entire concatenated seq.
    diploid: bool
        Combine haploid samples into diploid genotype calls.
    method: str
        options include "iqtree", "raxml", "mrbayes".
    kwargs: dict
        a limited set of supported inference options. See docs.

    Returns
    -------
    pd.DataFrame: A DataFrame with window stats and newick trees.

    Example
    -------
    >>> sptree = toytree.rtree.unittree(ntips=10, treeheight=1e6)
    >>> model = ipcoal.Model(sptree, Ne=1e6, nsamples=2)
    >>> model.sim_loci(nloci=10, nsites=1000)
    >>> gene_trees = model.infer_gene_tree()
    """
    # bail out if the data is only unlinked SNPs
    if self.df.nbps.max() == 1:
        raise IpcoalError(
            "gene tree inference cannot be performed on individual SNPs\n"
            "perhaps you meant to run .sim_loci() instead of .sim_snps()."
            )
    # complain if no seq data exists
    if self.seqs is None:
        raise IpcoalError(
            "Cannot infer trees because no seq data exists. "
            "You likely called sim_trees() instead of sim_loci()."
        )

    # if window_size is None then use entire chrom
    if window_size is None:
        window_size = self.df.end.max()

    # create the results dataframe
    resdf = pd.DataFrame({
        "start": np.arange(0, self.df.end.max(), window_size),
        "end": np.arange(window_size, self.df.end.max() + window_size, window_size),
        "nbps": window_size,
        "nsnps": 0,
        "inferred_tree": np.nan,
    })

    # reshape seqs: (nloc, ntips, nsites) to (nwins, ntips, win_size)
    newseqs = np.zeros((resdf.shape[0], self.nstips, window_size), dtype=int)
    for idx in resdf.index:
        # TODO: HERE IT'S ONLY INFERRING AT LOC 0
        loc = self.seqs[0, :, resdf.start[idx]:resdf.end[idx]]
        newseqs[idx] = loc
        resdf.loc[idx, "nsnps"] = (np.any(loc != loc[0], axis=0).sum())

    # init the TreeInference object (similar to ipyrad inference code)
    tool = TreeInfer(
        self,
        inference_method=inference_method,
        inference_args=inference_args,
        diploid=diploid,
        alt_seqs=newseqs,
    )

    # iterate over nloci. This part could be easily parallelized...
    for idx in resdf.index:
        resdf.loc[idx, "inferred_tree"] = tool.run(idx)
    return resdf

def infer_gene_tree_windows(
    self,
    window_size: Optional[int]=None,
    inference_method: str = 'raxml',
    inference_args: Dict[str,str] = None,
    diploid: bool = False,
    ) -> pd.DataFrame:
    """Infer gene trees in intervals using a phylogenetic tool.

    The method is applied to windows spanning every locus.
    If window_size is None then each locus is treated as an
    entire concatenated window.

    Parameters
    ----------
    window_size: int, None
        The size of non-overlapping windows to be applied across the
        sequence alignment to infer gene tree windows. If None then
        a single gene tree is inferred for the entire concatenated seq.
    diploid: bool
        Combine haploid samples into diploid genotype calls.
    method: str
        options include "iqtree", "raxml", "mrbayes".
    kwargs: dict
        a limited set of supported inference options. See docs.

    Returns
    -------
    pd.DataFrame: A DataFrame with window stats and newick trees.

    Example
    -------
    >>> sptree = toytree.rtree.unittree(ntips=10, treeheight=1e6)
    >>> model = ipcoal.Model(sptree, Ne=1e6, nsamples=2)
    >>> model.sim_loci(nloci=10, nsites=1000)
    >>> gene_trees = model.infer_gene_tree()
    """
    # bail out if the data is only unlinked SNPs
    if self.df.nbps.max() == 1:
        raise IpcoalError(
            "gene tree inference cannot be performed on individual SNPs\n"
            "perhaps you meant to run .sim_loci() instead of .sim_snps()."
            )
    # complain if no seq data exists
    if self.seqs is None:
        raise IpcoalError(
            "Cannot infer trees because no seq data exists. "
            "You likely called sim_trees() instead of sim_loci()."
        )

    # if window_size is None then use entire chrom
    if window_size is None:
        window_size = self.df.end.max()

    # create the results dataframe
    resdf = pd.DataFrame({
        "start": np.arange(0, self.df.end.max(), window_size),
        "end": np.arange(window_size, self.df.end.max() + window_size, window_size),
        "nbps": window_size,
        "nsnps": 0,
        "inferred_tree": np.nan,
    })

    # reshape seqs: (nloc, ntips, nsites) to (nwins, ntips, win_size)
    newseqs = np.zeros((resdf.shape[0], self.nstips, window_size), dtype=int)
    for idx in resdf.index:
        # TODO: HERE IT'S ONLY INFERRING AT LOC 0
        loc = self.seqs[0, :, resdf.start[idx]:resdf.end[idx]]
        newseqs[idx] = loc
        resdf.loc[idx, "nsnps"] = (np.any(loc != loc[0], axis=0).sum())

    # init the TreeInference object (similar to ipyrad inference code)
    tool = TreeInfer(
        self,
        inference_method=inference_method,
        inference_args=inference_args,
        diploid=diploid,
        alt_seqs=newseqs,
    )

    # iterate over nloci. This part could be easily parallelized...
    for idx in resdf.index:
        resdf.loc[idx, "inferred_tree"] = tool.run(idx)
    return resdf

def infer_gene_trees(
    self,
    inference_method: str = 'raxml',
    inference_args: Optional[Dict[str,str]] = None,
    diploid: bool = False,
    ):
    """Infer gene trees at every discrete locus.

    Parameters
    ----------
    inferenc_method: str
        Select the tree inference method to run. Current options
        include "raxml", "iqtree", "mrbayes".
    inference_args: Dict
        A dict mapping inference arg names and values as strings.
        These will replace default options, or add additional
        args. Default raxml kwargs are:
        >>> raxml_kwargs = {
        >>>     "f": "d",
        >>>     "N": "10",
        >>>     "T": "4",
        >>>     "m": "GTRGAMMA",
        >>>     "w": tempfile.gettempdir()
        >>> }
    diploid: bool
        If True then pairs of genotypes are combined into diploid
        base calls if nsamples in each lineage is divisible by 2.
    """
    inference_args = inference_args if inference_args is not None else {}

    # bail out if the data is only unlinked SNPs
    if self.df.nbps.max() == 1:
        raise IpcoalError(
            "gene tree inference cannot be performed on individual SNPs\n"
            "perhaps you meant to run .sim_loci() instead of .sim_snps()."
            )

    # expand self.df to include an inferred_trees column
    self.df["inferred_tree"] = np.nan

    # init the TreeInference object (similar to ipyrad inference code)
    tool = TreeInfer(
        self,
        inference_method=inference_method,
        inference_args=inference_args,
        diploid=diploid,
    )

    # complain if no seq data exists
    if self.seqs is None:
        raise IpcoalError(
            "Cannot infer trees because no seq data exists. "
            "You likely called sim_trees() instead of sim_loci()."
        )

    # TODO; if sim_snps() infer one concatenated tree.
    # ...

    # iterate over nloci. This part could be easily parallelized...
    for lidx in range(self.seqs.shape[0]):

        # skip invariable loci
        if self.df.nsnps[self.df.locus == lidx].sum():
            # let low data fails return NaN
            try:
                tree = tool.run(lidx)
                # enter result
                self.df.loc[self.df.locus == lidx, "inferred_tree"] = tree

            # caught raxml exception (prob. low data)
            except IpcoalError as err:
                logger.error(err)
                raise err


