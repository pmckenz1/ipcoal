## ipcoal
### Python package to interactively simulate genealogies and sequence data under the multispecies coalescent

Full documentation here: [https://eaton-lab.org/ipcoal/](https://eaton-lab.org/ipcoal/)

```python
import ipcoal
import toytree

# get a species tree 
tree = toytree.rtree.baltree(ntips=6, treeheight=1e6)

# init a simulator from the tree w/ additional parameters
model = ipcoal.Model(tree, Ne=1e6)

# simulate genealogies and sequences
model.sim_loci(nloci=5, nsites=1e5)

# access results in a dataframe
model.df

# infer gene trees for each locus, or in sliding windows
gtree = ipcoal.phylo.infer_raxml_ng_trees(model, idxs=[0, 1, 2, 3])
```


![https://raw.githubusercontent.com/eaton-lab/sptree-chapter/master/manuscript/figures/Fig1-revision.png](https://raw.githubusercontent.com/eaton-lab/sptree-chapter/master/manuscript/figures/Fig1-revision.png)