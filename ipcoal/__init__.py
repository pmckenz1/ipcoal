#!/usr/bin/env python

__version__ = "0.1.0"
__author__ = "Patrick McKenzie and Deren Eaton"

from .Model import Model
from . import utils


# option to JIT compile in fork-safe mode
__forksafe__ = False
