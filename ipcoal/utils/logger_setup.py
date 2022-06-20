#!/usr/bin/env python

"""Logging module.

Logging module primarily used for debugging by developers, but which 
can be turned on by users for more verbose output by calling:

ipcoal.set_log_level("DEBUG")
"""

from typing import Optional
from pathlib import Path
import io
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
def set_log_level(
    log_level: str="INFO", 
    log_out: Optional[io.TextIOBase]=sys.stderr,
    log_file: Optional[Path]=None,
    ):
    """Set the log level for loguru logger.

    This removes default loguru handler, but leaves any others, 
    and adds a new one that will filter to only print logs from 
    ipcoal modules, which should use `logger.bind(name='ipcoal')`.

    Parameters
    ----------
    log_level: str
        Level of logging output: DEBUG, INFO, WARNING, ERROR.
    log_out: io.TextIOBase
        Option to log to stderr, stdout, or None.
    log_file: Path
        Option to log to a file, or None.
    """
    for idx in LOGGERS:
        try:
            logger.remove(idx)
        except ValueError:
            pass

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(exist_ok=True)
        log_file.touch(exist_ok=True)
        idx = logger.add(
            sink=log_file,
            level=log_level,
            colorize=False,
            format=LOGFORMAT,
            filter=lambda x: x['extra'].get('name') == "ipcoal",
            enqueue=True,
            rotation="50 MB",
            backtrace=True,
            diagnose=True,
        )
    if log_out:
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
