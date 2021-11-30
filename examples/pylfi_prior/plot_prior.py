"""
=====================================================================
Using the pyLFI Prior class
=====================================================================
Example usage of the `pylfi.Prior` class.
"""

import matplotlib.pyplot as plt
import numpy as np
import pylfi

###############################################################################
# Initialize a Gaussian prior over the parameter :math:`\theta`. The first
# positional argument can be any `scipy.stats` distribution passed as `str`.
# Following positional and keyword arguments are distribution specific
# (see `scipy.stats` documentation). The `name` keyword argument is required
# and expects the name of the parameter passed as `str`. The optional `tex`
# keyword argument can be used to provide LaTeX typesetting for the parameter
# name, which is used as axis label in `pyLFI`'s plotting procedures if
# provided.
theta_prior = pylfi.Prior('norm',
                          loc=0,
                          scale=1,
                          name='theta',
                          tex=r'$\theta$'
                          )

###############################################################################
# Sampling from the prior is done through the `.rvs` method. The `size` keyword
# can be used to set the output size of the sample. The sampling procedures can
# also be seeded through the `seed` keyword argument.
theta_prior.rvs(size=10, seed=42)

###############################################################################
# The `~.plot_prior` method plots the prior pdf or pmf, depending on whether
# the distribution is continuous or discrete, respectively, evaluated at points
# :math:`x`.
x = np.linspace(-4, 4, 1000)
theta_prior.plot_prior(x)
