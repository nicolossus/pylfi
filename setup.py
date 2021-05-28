#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

# Package meta-data.
NAME = "pylfi"
DESCRIPTION = "Likelihood-free inference with Python."
URL = "https://github.com/nicolossus/neuromodels"
EMAIL = "prof.haug@gmail.com"
AUTHOR = "Nicolai Haug"
REQUIRES_PYTHON = '>=3.8.0'

REQUIRES_INSTALL = [
    "numpy",
    "matplotlib",
    "scipy",
    "sklearn",
]

REQUIRES_EXTRAS = {
    "dev": [
        "pytest",
        "pytest-cov",
        "flake8>=3.9.2",
        "isort",
        "twine",
    ],
}


with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

about = {}
with open(os.path.join(here, NAME, "__version__.py")) as f:
    exec(f.read(), about)

VERSION = about['__version__']


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=["tests", ]),
    python_requires=REQUIRES_PYTHON,
    install_requires=REQUIRES_INSTALL,
    extras_require=REQUIRES_EXTRAS,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
