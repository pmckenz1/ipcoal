


def get_topo_waiting_distance_likelihood(
    species_tree: ToyTree,
    waiting_distances: np.ndarray,
    recombination_rate: float,
    ) -> float:
    """Return likelihood of species tree parameters given a 
    distribution of waiting distances.
    """


def fit_species_tree_parameters_from_topology_waiting_distances(
    model: ipcoal.Model,
    recombination_rate: float,
    ) -> Dict[str, float]:
    """Return a dict of species tree parameters and recomb rate.

    """
    # get topology at each breakpoint it changes in a tree sequence



    # find distances between topology changes
    # dist = ipcoal.smc.get_waiting_distance_to_topology_change_rv(SPTREE, GTREE, IMAP, RECOMB)


def get_tree_likelihood(params: np.ndarray, etable: np.ndarray, waiting_distances: np.ndarray) -> float:
    """...

    This function only estimates Ne and r values. It is much faster
    than methods that need to infer species_tree divergence times, 
    since 

    """
    # set the embedding table Ne values to the new proposed values
    # etable[]

    # 


def get_topology_data(model, ) -> pd.DataFrame:
    """..."""
    genealogies = []
    lengths = []





def decompose_distance_data(
    model: ipcoal.Model,
    nsites: int,
    ) -> Tuple[np.narray, List[pd.DataFrame]]:
    """

    This takes a tree sequence generated ...
    and decomposes it into three tables,
    one with the genealogy

    """
    imap = model.get_imap_dict()
    model.sim_trees(nloci=1, nsites=length)

    msgen = model._get_tree_sequence_generator(nsites)
    tree_seq = next(msgen)
    breaks = [int(i) for i in tree_seq.breakpoints()]
    starts = breaks[0:len(breaks) - 1]
    ends = breaks[1:len(breaks)]
    lengths = [i - j for (i, j) in zip(ends, starts)]    




tree0 = toytree.tree(MODEL.df.genealogy[0])
tree0_id = tree0.get_topology_id(exclude_root=False)
length = 0
for tidx in MODEL.df.index[1:]:
    row = MODEL.df.loc[tidx]
    tree1 = toytree.tree(row.genealogy)
    tree1_id = tree1.get_topology_id(exclude_root=False)
    if tree0_id == tree1_id:
        length += row.nbps
    else:
        # get embedding table for this genealogy
        tables.append(ipcoal.smc.get_genealogy_embedding_table(SPTREE, tree0, IMAP))
        lengths.append(length + row.nbps)
        length = 0
        tree0 = tree1

for i in zip(lengths, tables):
    print(i)
