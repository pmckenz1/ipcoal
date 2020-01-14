


Terminology and units
=====================


Terminology
-----------

**genealogy**: the *true ancestry* of a set of sampled genes at some location in the genome. In a simulation framework the genealogy is known, but in the real world it is an unobservable variable. In the presence of recombination, a single genome represents a mosaic of many ancestors, and consequently, the genealogical history for a set of sampled genomes differs spatially across the genome as different regions share different ancestors. The relationships among nearby (linked) genealogies which share *some* ancestors, versus unlinked genealogies that do not, is a source of phylogenetic information that can be easily explored in *ipcoal*. 


**gene tree**: the *inferred ancestry* of a set of sampled genes at some location in the genome based on a sequence alignment. In practice gene trees rarely match the true genealogy since there is often insufficient information (substitutions) within a small genomic region. Over deeper evolutionary time scales the size of a non-recombined locus becomes very small such that most loci represent substitutions evolved across several linked genealogies. 


**species tree**: a model describing the topology (ancestral relationships) and demographic parameters for a number of lineages. This includes divergence times, effective population sizes, and admixture events. 


**substitution model**: A parameterized Markov model for probabilistically modeling the evolution of sequences along the edges of a genealogical tree. This uses edge lengths described in units of generation times (g), a mutation rate in units of mutations per site per generation (u), and additional optional parameters of the Markov model (e.g., state frequencies, transition-transversion ratio). 


**loci**: ...



**sites**: ...



Units
-----

**genealogical branch lengths:**


**gene tree branch lengths**:


**species tree branch lengths**:


**generation**: 


**coalescent units**:


**mutation rate**:


**recombination rate**:


**admixture time**:


**admixture proportion**: 

