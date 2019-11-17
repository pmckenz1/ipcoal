#!/usr/bin/env python

from setuptools import setup
setup(
    name="ipcoal",
    packages=["ipcoal"],
    version="0.0.3",
    author="Patrick McKenzie and Deren Eaton",
    author_email="p.mckenzie@columbia.edu",
    install_requires=[
        "scipy>0.10",
        "numpy>=1.9",
        "pandas>=0.16",
        "numba",
        "toytree>=1.0.3",
        "msprime",
        # "ipyparallel",
        # seq-gen (optional)
    ],
    license='GPL',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
)
