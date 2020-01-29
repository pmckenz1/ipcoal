

FAQs
====


Please contact us if you have questions about how to implement a complex model. We try to provide examples in the Cookbook section of many commonly implemented scenarios, and to answer here any other frequently asked questions. 


Mutation rate variation across the tree
---------------------------------------
You can only set one per-site per-generation mutation rate currently (e.g, mut=1e-8), however, you can emulate variation in the effective mutation rate (theta=4Neu) across edges of the tree by modifying Ne. 

.. code-block:: python

    # create a random tree
    tre = toytree.rtree.unittree(10, 1e6)

    # draw the tree to see numeric node numbers
    tre.draw(ts='p');

    # set Ne values to nodes based on their number
    ndict = {i: 1e4 for i in (1, 2, 3, 5, 7)}
    newtre = tre.set_node_values("Ne", default=1e5, values=ndict)

    # visualize tree with variable edge widths to 
    # verify that you set Ne how you want it.
    newtre.draw(ts='p')

    # create Model using tree with variable Ne (4Neu)
    model = ipcoal.Model(tree=newtre, Ne=None)


Generation time variation across the tree
-----------------------------------------
The edge lengths of you tree should be in units of generations, however, the species tree does not have to be ultrametric (all tips align at 0). If two sister lineages diverged and one is an annual and the other instantly became a biennial then all else being equal the annual's edge length should be 2X longer. See tips below for converting to generations from other time units.


Converting to generations from a time-scaled tree
-------------------------------------------------
To convert branch lengths from absolute time to generations, divide the length of each branch (in time) by the generation time for that branch.


Converting to generations from a coalescent-unit scaled tree
------------------------------------------------------------
...


How can I simulate a large locus or chromosome?
----------------------------------------------------
This is easy to do in *ipcoal*. 

.. code-block:: python

	# simulate a 10Mb chromosome with no recombination
	model = ipcoal.Model(tree, recomb=0)
	model.sim_loci(nloci=1, nsites=10e6)

	# simulate a 10Mb chromosome with recombination across it 
	model = ipcoal.Model(tree, recomb=1e-9)
	model.sim_loci(nloci=1, nsites=10e6)


Can I simulate variable recombination rate (hotspots, recomb map)?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Currently you cannot, but this would be easy to add, we simply haven't done it yet. Please raise a feature request.



Can I sample multiple individuals per population/species?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Yes, you can do this using the `samples` argument to the Model object. The samples from each tip in the species tree will be labeled as [tip-name]-0, [tip-name]-1, etc. 



