#!/usr/bin/env python

from setuptools import setup
setup(
    name="phymsim",
    packages=["phymsim"],
    version="0.0.1",
    author="Patrick McKenzie",
    author_email="p.mckenzie@columbia.edu",
    install_requires=[
        "scipy>0.10",
        "numpy>=1.9",
        "pandas>=0.16",
        "toytree",
        "toyplot",
    ],
    license='GPL',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
)
