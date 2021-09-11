#!/usr/bin/env python

"""
Summary
-------
The ipcoal module includes the ipcoal.model.Model class object
that is the primary interface by which users interact with the
ipcoal library.

Example
-------
>>> import ipcoal
>>> model = ipcoal.Model(Ne=1000, nsamples=10)
>>> model.sim_snps(10)
>>> model.draw_seqview()
"""

__version__ = "0.3.1"
__author__ = "Patrick McKenzie and Deren Eaton"

from ipcoal.model import Model
from ipcoal.utils import utils
from ipcoal.utils.logger_setup import set_log_level
set_log_level("WARNING")
