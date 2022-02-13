#!/usr/bin/env python

"""The ipcoal library: coalescent simulations for phylogenetics.

Summary
-------
The primary interface for `ipcoal` is the `ipcoal.Model` class 
which is used to setup, perform, and analyze simulations.

Example
-------
>>> import ipcoal
>>> model = ipcoal.Model(Ne=1000, nsamples=10)
>>> model.sim_snps(10)
>>> model.draw_seqview()
"""

__version__ = "0.4.0"
__author__ = "Patrick McKenzie and Deren Eaton"

from ipcoal.model import Model
from ipcoal.utils import utils

# start the logger at log_level WARNING
from ipcoal.utils.logger_setup import set_log_level
set_log_level("WARNING")
