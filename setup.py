#!/usr/bin/env python

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
        "numpy>=1.9",
        "pandas>=1.0",
        "toytree>=1.1.0",
        "msprime",
        "numba",
        "scipy>0.10",
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
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',                
    ],
)
