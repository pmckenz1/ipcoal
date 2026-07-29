---
section: terminology
---


## Scope
In general we try to follow the terminology used by 
[msprime](https://msprime.readthedocs.io) when referring to parameters that
will be used in setting up and implementing coalescent simulation, such as the
per-site per-generation mutation rate, admixture proportions, and diploid
effective population sizes. 


## Terminology
**locus**: a discrete sequence spanning 1 or more sites which may or may not 
include intra-locus recombination. Each locus is simulated independently in
*ipcoal*, as if each is a distinct chromosome.

**site**: a single position on a chromosome/locus. Each simulated site on a 
locus has a genealogical tree, and one genealogical tree can represent the
history for multiple linked sites that represent an interval on a locus
between recombination events. Note: A simulated site has a genealogical 
tree history whether or not mutations/sequences are simulated.

**genealogy**: the *true ancestry* of a set of sampled gene copies at some
location in the genome. In a simulation framework, the genealogy is known, 
but in the real world it is an unobservable variable that must be inferred
from sequence variation.

**gene tree**: the *inferred ancestry* of a set of sampled gene copies at some
location in the genome, based on a sequence alignment. In practice, gene trees
rarely match the true genealogy exactly since there is often insufficient 
information (substitutions) within a small genomic region.

**species tree**: a demographic model describing the topology (ancestral 
relationships) and demographic parameters (e.g., Ne, divergence times, and
admixture events) for one or more lineages. 

**substitution model**: a parameterized Markov model for the mutational process.
A mutation causes a change from one discrete state in the mutation model to
another. The probability of changes depend on the current state, the per-site
per-generation mutation rate ($\mu$), and the length of time over which a
substitution can occur, repesented by the edges on a genealogy in units of
generations. 


Units
-----

**genealogical branch lengths:** defined in number of generations.  

**gene tree branch lengths**: Depends on the inference method. If using the 
default ML implementation in raxml-ng then gene tree branch lengths are in 
units of expected number of substitutions per site.

**species tree branch lengths**: defined in number of generations. Sometimes
must be converted from coalescent units to generations, which requires only
dividing by 2N if generation times are constant.

**generation**: the length of time from birth until reproduction. Under the 
Wright-Fisher process, which the coalescent approximates, generations do not
overlap.   

**coalescent units**: units of species tree branch lengths that describe the 
probability that n samples coalesce over a length of time. It is calculated
as (time in generations) / 2Ne for diploid populations.

**mutation rate**: the expected number of mutations per site per generation.   

**recombination rate**: the expected number of recombinations per site per generation.  

**admixture time**: a time (absolute) or proportion along an edge (float between
0.0 and 1.0) defining the moment within an overlapping interval of two species
tree edges that an admixture event occurred. (For example, if admixture 
time=0.5, the event happens at the midpoint of the overlap between the two 
branches). 

**admixture proportion**: the proportion of the source population that migrates
to the destination population backwards in time (use visualization and 
debugging methods to ensure the model is doing what you expect).
