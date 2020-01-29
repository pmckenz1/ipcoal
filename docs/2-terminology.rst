


Terminology and units
=====================
.. In general we try to follow the terminology used by `msprime <https://msprime.readthedocs.io>`__ when referring to simulation parameters such as the per-site mutation rate, or admixture proportions, so that further details can be sought from their documentation. However, we 


Terminology
-----------

**genealogy**: the *true ancestry* of a set of sampled genes at some location in the genome. In a simulation framework the genealogy is known, but in the real world it is an unobservable variable. In the presence of recombination, a single genome represents a mosaic of many ancestors, and consequently, the genealogical history for a set of sampled genomes differs spatially across the genome as different regions share different ancestors. The relationships among nearby (linked) genealogies which share *some* ancestors, versus unlinked genealogies that do not, is a source of phylogenetic information that can be easily explored in *ipcoal*. 


**gene tree**: the *inferred ancestry* of a set of sampled genes at some location in the genome based on a sequence alignment. In practice gene trees rarely match the true genealogy since there is often insufficient information (substitutions) within a small genomic region. Over deeper evolutionary time scales the size of a non-recombined locus becomes very small such that most loci represent substitutions evolved across several linked genealogies. 


**species tree**: a model describing the topology (ancestral relationships) and demographic parameters for a number of lineages. This includes divergence times, effective population sizes, and admixture events. 


**substitution model**: A parameterized Markov model for probabilistically modeling the evolution of sequences along the edges of a genealogical tree. This uses edge lengths described in units of generation times (g), a mutation rate in units of mutations per site per generation (u), and additional optional parameters of the Markov model (e.g., state frequencies, transition-transversion ratio). 


**loci**: pieces of chromosomes. Each locus is defined as just being an "nsites" length of chromosome, like what you'd expect from a sequenced read. This stretch can span multiple (unobserved) genealogies because recombination might happen within it, but it acts as the observed data for gene tree inference.


**sites**: a single position in the genome, at which any individual might have an A, C, G, or T.



Units
-----

**genealogical branch lengths:** defined in number of generations.


**gene tree branch lengths**: defined in inferred number of substitutions.


**species tree branch lengths**: defined in number of generations.


**generation**: the length of time from birth until reproduction. Under the Wright-Fisher process, which the coalescent approximates, generations do not overlap.


**coalescent units**: units of species tree branch lengths that incorporate both mutation rate and effective population size.


**mutation rate**: the number of mutations per base per generation. 


**recombination rate**: the number of recombinations expected per base per generation.


**admixture time**: a float between 0.0 and 1.0 defining the moment within the overlapping interval of two branches that an admixture event happens between them. (For example, if admixture time = 0.5, the event happens at the midpoint of the overlap between the two branches)


**admixture proportion**: the proportion of the source population that migrates to the the destination branch.

