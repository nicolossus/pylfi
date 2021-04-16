.. _installation:

1. Installation
===============

pyLFI requires Python3.8

Install Package
~~~~~~~~~~~~~~~

Clone repository, ``cd`` into root directory and install with
::

   pip install .


Development Install
~~~~~~~~~~~~~~~~~~~

Clone repository, ``cd`` into root directory and install with
::

   pip install --editable .


Requirements
~~~~~~~~~~~~

Basic requirements are listed in ``environment.yml`` in the repository.

Environment
~~~~~~~~~~~

Install Anaconda, see `anaconda.com <https://www.anaconda.com/products/individual>`_.

To create environment
::

    conda env create --file environment.yml

To activate environment
::

    conda activate master

To deactivate environment
::

    conda deactivate

To remove environment
::

    conda remove --name master --all

To verify that the environment was removed
::

    conda info --envs
