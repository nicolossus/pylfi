# pyLFI

[![PyPI version](https://badge.fury.io/py/pylfi.svg)](https://badge.fury.io/py/pylfi)
[![Documentation Status](https://readthedocs.org/projects/pylfi/badge/?version=latest)](https://pylfi.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/nicolossus/pylfi/workflows/Tests/badge.svg?branch=main)](https://github.com/nicolossus/pylfi/actions)
[![GitHub license](https://img.shields.io/github/license/nicolossus/pylfi)](https://github.com/nicolossus/pylfi/blob/pylfi/LICENSE)

`pyLFI` is a Python toolbox for Bayesian parameter estimation in models with intractable likelihood functions. By using *Likelihood-Free Inference* (LFI) schemes, in particular *Approximate Bayesian Computation* (ABC), `pyLFI` estimates the posterior distributions over model parameters.

## Overview

`pyLFI` presently includes the following methods:

* Rejection ABC
* MCMC ABC
* Post-sampling regression adjustment.

`pyLFI` was created as a part of the author's [Master thesis](https://github.com/nicolossus/Master-thesis).

## Installation instructions

### Install with pip
`pyLFI` can be installed directly from [PyPI](https://pypi.org/project/pylfi/):

    $ pip install pylfi

## Requirements
* `Python` >= 3.8

## Documentation
Documentation can be found at [pylfi.readthedocs.io](https://pylfi.readthedocs.io/).

<!--## Getting started
Check out the [Examples gallery](https://pylfi.readthedocs.io/en/latest/auto_examples/index.html) in the documentation.
-->

## Automated build and test
The repository uses continuous integration (CI) workflows to build and test the project directly with GitHub Actions. Tests are provided in the [`tests`](tests) folder. Run tests locally with `pytest`:

    $ python -m pytest tests -v
