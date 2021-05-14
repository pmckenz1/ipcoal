#!/usr/bin/env python

"""
Run `pip install -e .` to install local git version.
"""

import os
import re
from setuptools import setup

# parse version from init.py
with open("ipcoal/__init__.py") as init:
    CUR_VERSION = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]",
        init.read(),
        re.M,
    ).group(1)


# nasty workaround for RTD low memory limits
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    install_requires = []
else:
    install_requires = [
        "numpy",
        "pandas",
        "toytree",
        "msprime",
        "numba",
        "scipy",
        # seq-gen (optional)
    ]


# setup installation
setup(
    name="ipcoal",
    packages=["ipcoal"],
    version=CUR_VERSION,
    author="Patrick McKenzie and Deren Eaton",
    author_email="p.mckenzie@columbia.edu",
    install_requires=install_requires,
    license='GPL',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',        
    ],
)
