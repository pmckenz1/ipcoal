


Terminology and units
=====================
.. In general we try to follow the terminology used by `msprime <https://msprime.readthedocs.io>`__ when referring to simulation parameters such as the per-site mutation rate, or admixture proportions, so that further details can be sought from their documentation. However, we 


Terminology
-----------

**genealogy**: the *true ancestry* of a set of sampled genes at some location in the genome. In a simulation framework the genealogy is known, but in the real world it is an unobservable variable. In the presence of recombination, a single genome represents a mosaic of many ancestors, and consequently, the genealogical history for a set of sampled genomes differs spatially across the genome as different regions share different ancestors. The relationships among nearby (linked) genealogies which share *some* ancestors, versus unlinked genealogies that do not, is a source of phylogenetic information that can be easily explored in *ipcoal*. 


**gene tree**: the *inferred ancestry* of a set of sampled genes at some location in the genome based on a sequence alignment. In practice gene trees rarely match the true genealogy since there is often insufficient information (substitutions) within a small genomic region. Over deeper evolutionary time scales the size of a non-recombined locus becomes very small such that most loci represent substitutions evolved across several linked genealogies. 


**species tree**: a model describing the topology (ancestral relationships) and demographic parameters for a number of lineages. This includes divergence times, effective population sizes, and admixture events. 


**substitution model**: a parameterized Markov model for probabilistically modeling the evolution of sequences along the edges of a genealogical tree. This uses edge lengths described in units of generation times (g), a mutation rate in units of mutations per site per generation (u), and additional optional parameters of the Markov model (e.g., state frequencies, transition-transversion ratio). 


**loci**: a region of length "nsites" spanning some interval of a chromosome. The ancestry for some number of samples may represent multiple distinct genealogies across the length of a locus, if recombination has occurred within it. *ipcoal* includes functions to infer gene trees for individual loci, or in subwindows across
loci (e.g., sliding windows across a locus/chromosome), or to concatenate many loci to infer a concatenated gene tree.


**sites**: a single position in the genome, at which any individual might have an A, C, G, or T.



Units
-----

**genealogical branch lengths:** defined in number of generations.  


**gene tree branch lengths**:  Depends on the inference method. If using the default ML implementation in raxml then gene tree branch lengths are estimated parameters representing the expected number of substitutions per site.   


**species tree branch lengths**: defined in number of generations.  


**generation**: the length of time from birth until reproduction. Under the Wright-Fisher process, which the coalescent approximates, generations do not overlap.   


**coalescent units**: units of species tree branch lengths that describe the probability that n samples coalesce over a length of a branch. It is calculated as (time in generations) / 2Ne.  


**mutation rate**: the expected number of mutations per base per generation.   


**recombination rate**: the expected number of recombinations per base per generation.  


**admixture time**: a float between 0.0 and 1.0 defining the moment within the overlapping interval of two branches that an admixture event occurred. (For example, if admixture time=0.5, the event happens at the midpoint of the overlap between the two branches)


**admixture proportion**: the proportion of the source population that migrates to the destination branch.  

