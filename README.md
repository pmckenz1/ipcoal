# ipcoal

Documentation under development: [https://ipcoal.readthedocs.io](https://ipcoal.readthedocs.io)

Python package for simulating genealogies and sequence data. 

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
model.infer_gene_trees()

# write sequences to a file
model.write_loci_to_phylip()
```

![https://raw.githubusercontent.com/eaton-lab/sptree-chapter/master/manuscript/figures/Fig1-revision.png](https://raw.githubusercontent.com/eaton-lab/sptree-chapter/master/manuscript/figures/Fig1-revision.png)