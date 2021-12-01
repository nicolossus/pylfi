.. _pylfi:


=============
API Reference
=============

This reference gives details about the API of modules, classes and functions included in ``pyLFI``.

:mod:`pylfi.inferences`: Inference schemes
==========================================

.. automodule:: pylfi.inferences
    :no-members:
    :no-inherited-members:

Approximate Bayesian Computation
--------------------------------
.. currentmodule:: pylfi

.. autosummary::
   :nosignatures:
   :recursive:
   :toctree: generated/
   :template: class.rst

   inferences.ABCBase
   inferences.RejABC
   inferences.MCMCABC

:mod:`pylfi.priors`: Prior distributions
========================================

.. automodule:: pylfi.priors
    :no-members:
    :no-inherited-members:

.. currentmodule:: pylfi

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   priors.Prior

:mod:`pylfi.utils`: Utility functions
=====================================

.. automodule:: pylfi.utils
    :no-members:
    :no-inherited-members:

.. currentmodule:: pylfi

.. autosummary::
  :nosignatures:
  :recursive:
  :toctree: generated/

  utils.check_and_set_jobs
  utils.distribute_workload
  utils.generate_seed_sequence
  utils.advance_PRNG_state

:mod:`pylfi.journal`: Journal class
===================================

.. automodule:: pylfi.journal
    :no-members:
    :no-inherited-members:

.. currentmodule:: pylfi

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   journal.Journal
