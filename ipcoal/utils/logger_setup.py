#!/usr/bin/env python

"""
Logging module primarily used for debugging by developers, but which 
can be turned on by users for more verbose output by calling:

ipcoal.set_loglevel("DEBUG")
"""

import sys
from loguru import logger
import ipcoal


LOGFORMAT = (
    "<level>{level: <7}</level> <white>|</white> "
    "<cyan>{file: <12}</cyan> <white>|</white> "
    "<level>{message}</level>"
)

def colorize():
    """check whether terminal/tty supports color."""
    try:
        import IPython
        tty1 = bool(IPython.get_ipython())
    except ImportError:
        tty1 = False
    tty2 = sys.stderr.isatty()
    return tty1 or tty2


LOGGERS = [0]
def set_log_level(log_level="INFO"):
    """Set the log level for loguru logger.

    This removes default loguru handler, but leaves any others, 
    and adds a new one that will filter to only print logs from 
    toytree modules, which should use `logger.bind(name='toytree')`.
    """
    for idx in LOGGERS:
        try:
            logger.remove(idx)
        except ValueError:
            pass
    idx = logger.add(
        sink=sys.stderr,
        level=log_level,
        colorize=colorize(),
        format=LOGFORMAT,
        filter=lambda x: x['extra'].get("name") == "ipcoal",
    )
    LOGGERS.append(idx)
    logger.enable("ipcoal")
    logger.bind(name="ipcoal").debug(
        f"ipcoal v.{ipcoal.__version__} logging enabled"
    )
