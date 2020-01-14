


Installation 
=============

Conda installation
------------------
*ipcoal* is available for Python 2.7 and 3.5 or greater. The simplest way to install it is with conda. Currently, we require a few different channels to pull in all dependencies, this will likely be simplified in the future. Please install with the following command:

.. code:: bash

	conda install ipcoal -c conda-forge -c bioconda -c eaton-lab



Developers welcomed
--------------------
We are interested in expanding the capabilities of *ipcoal*. If you are interested in joining as a developer please feel free to fork the repo from GitHub and join in the discussion on addressing tickets or developing new features. 


Dependencies
-------------
- numpy
- pandas
- numba
- msprime (conda-forge)
- toytree (eaton-lab)
- seq-gen (bioconda)
- raxml (bioconda)
- mrbayes (bioconda)