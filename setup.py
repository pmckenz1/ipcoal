#!/usr/bin/env python

"""
Run `pip install -e .` to install local git version.
"""

import os
import re
from setuptools import setup, find_packages

# parse version from init.py
with open("ipcoal/__init__.py", encoding="utf-8") as init:
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
        "loguru"
    ]


# setup installation
setup(
    name="ipcoal",
    packages=find_packages(),
    version=CUR_VERSION,
    author="Patrick McKenzie and Deren Eaton",
    author_email="p.mckenzie@columbia.edu",
    install_requires=install_requires,
    entry_points={'console_scripts': ['ms-smc-mcmc = ipcoal.smc.likelihood.mcmc:run']},    
    license='GPL',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',        
    ],
)
