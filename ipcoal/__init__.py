#!/usr/bin/env python

"""
ipcoal: a minimalist framework for simulating genealogies on species
tree or networks and performing phylogenetic analyses.
"""

__version__ = "0.3.1"
__author__ = "Patrick McKenzie and Deren Eaton"

from ipcoal.Model import Model  # class clobbers module name on purpose
from ipcoal.utils import utils
from ipcoal.utils.logger_setup import set_loglevel
set_loglevel("WARNING")
