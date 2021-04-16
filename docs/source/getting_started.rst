.. _gettingstarted:

2. Getting Started
==================

pyLFI is a Python toolbox for likelihood-free inference (LFI) for automated
parameter identification in mechanistic models with quantified uncertainty, and
is tailored towards computational neuroscience.

Introduction
~~~~~~~~~~~~

Mechanistic models in neuroscience aim to explain neural or behavioral phenomena
in terms of causal mechanisms, and candidate models are validated by investigating
whether proposed mechanisms can explain how experimental data manifests. A central
challenge in building a mechanistic model is to identify the parametrization of
the system which achieves an agreement between the model and experimental data.

Many mechanistic models are defined implicitly through simulators, i.e. a set of
dynamical equations, which can be run forward to generate data. Likelihoods can
be derived for purely statistical models, but are generally intractable or
computationally infeasible for simulation-based models. Hence are traditional
methods in the toolkit of statistical inference inaccessible for many mechanistic
models.

To overcome intractable likelihoods, a suite of methods that bypass the
evaluation of the likelihood function, called likelihood-free inference methods,
have been developed. These methods seek to directly estimate either the posterior or
the likelihood, and require only the ability to generate data from the simulator
to analyze the model in a fully Bayesian context.

Here, we explain how to use pyLFI to estimate the posterior distributions of
model parameters given some observed data. If you are new to parameter estimation
using Approximate Bayesian Computation (ABC), we recommend you to start with the
`The ABC of Approximate Bayesian Computation`_ section.

Moreover, we also provide an interactive notebook on Binder guiding through the
basics of ABC with pyLFI; without installing that on your machine. Please find
it `here <https://mybinder.org/v2/gh/eth-cscs/abcpy/master?filepath=examples>`_.

.. plot::

   import matplotlib.pyplot as plt
   import numpy as np
   plt.hist(np.random.randn(1000), 20)

The ABC of Approximate Bayesian Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As an example, if we have measurements of the height of a group of grown up humans and it is also known that a Gaussian
distribution is an appropriate probabilistic model for these kind of observations, then our observed dataset would be
measurement of heights and the probabilistic model would be Gaussian.

.. plot::
      :context: close-figs
      :format: doctest
      :include-source: False

      >>> import matplotlib.pyplot as plt
      >>> import numpy as np
      >>> x = np.linspace(0, 2*np, 100)
      >>> y = np.sin(x)
      >>> plt.plot(x, y)
      >>> plt.show()

..
  .. literalinclude:: ../../examples/extensions/models/gaussian_python/pmcabc_gaussian_model_simple.py
      :language: python
      :lines: 86-98, 103-105
      :dedent: 4
