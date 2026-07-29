---
section: Getting Started
---

# Installation
`ipcoal` can be installed using pip or conda, either of which will pull in 
all required dependencies. We also provide instructions below for installing
from source (GitHub).


Conda install (recommended)
---------------------------
```bash
$ conda install ipcoal -c conda-forge
```

Pip install
-----------
```bash
$ pip install ipcoal
```

Dependencies
------------
Our goal is to maintain `ipcoal` as a small library that does not require
substantial dependencies outside of the standard Python scientific stack (i.e.,
numpy, scipy, and pandas).

    - python>=3.7
    - numpy
    - scipy
    - pandas
    - loguru
    - toytree
    - msprime


Optional dependencies
----------------------
`ipcoal` includes several submodules that provide wrappers around external
phylogenetic inference tools, such as raxml-ng, Astral, SNaQ, and others. 
These tools are optional. When you call the function it will raise an exception
if the required binary cannot be found in your path and will provide recommendations
for how to install it using conda.


Installing Development Versions
-------------------------------
```bash
$ conda install ipcoal -c conda-forge --only-deps
$ git clone https://github.com/eaton-lab/ipcoal.git
$ cd ipcoal
$ pip install -e . --no-deps
```

Building the documentation
---------------------------
```bash
$ conda install mkdocs-material mkdocstrings-python mkdocs-jupyter -c conda-forge
$ cd ipcoal/
$ mkdocs serve
```
