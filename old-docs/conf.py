# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'ipcoal'
copyright = '2019, Patrick McKenzie & Deren Eaton'
author = 'Patrick McKenzie & Deren Eaton'

# The short X.Y version
version = ''
release = ''
extensions = ['nbsphinx']
templates_path = ['_templates']
source_suffix = ".rst"
language = 'python'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', "**.ipynb_checkpoints"]
master_doc = "index"
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
