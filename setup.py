#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

# Package meta-data.
NAME = 'pylfi'
DESCRIPTION = 'Likelihood-free inference with Python.'
URL = 'https://github.com/nicolossus/master-thesis'
EMAIL = 'nicolai.haug@fys.uio.no'
AUTHOR = 'Nicolai Haug'
REQUIRES_PYTHON = '>=3.8.0'
VERSION = '0.1.0'

about = {}
about['__version__'] = VERSION

setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(
        exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    setup_requires=["setuptools>=18.0"],
    install_requires=["numpy", "matplotlib", "scipy", "sklearn"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
