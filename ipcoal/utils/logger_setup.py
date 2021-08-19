#!/usr/bin/env python

"""
Logging module primarily used for debugging by developers, but which 
can be turned on by users for more verbose output by calling:

ipcoal.set_loglevel("DEBUG")
"""

import sys
from loguru import logger


LOGFORMAT = (
    "{time:hh:mm} <level>{level: <7}</level> <white>|</white> "
    "<cyan>{file: <12}</cyan> <white>|</white> "
    # "<cyan>{function: ^25}</cyan> <white>|</white> "
    "<level>{message}</level>"
)


def colorize():
    """
    check whether terminal/tty supports color
    """
    try:
        import IPython
        tty1 = bool(IPython.get_ipython())
    except ImportError:
        tty1 = False
    tty2 = sys.stderr.isatty()
    return tty1 or tty2


def set_loglevel(loglevel="DEBUG"):#, logfile=None):
    """
    Config and start the logger
    """
    config = {
        "handlers": [
            {
                "sink": sys.stderr, 
                "format": LOGFORMAT, 
                "level": loglevel,
                "enqueue": True,
                "colorize": colorize(),
                },
        ]
    }
    logger.configure(**config)
    logger.enable("ipcoal")
