#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


class Prior:
    r"""Initialize a prior.

    In the Bayesian paradigm, all available information about an unknown
    parameter is incorporated in a prior probability distribution, which
    describes the range of possible parameter values.

    Parameters
    ----------
    distr_name : `str`
        Any distribution from `scipy.stats` as a string.
    params:
        Parameters of the prior distribution. Typically these would be
        ``shape`` parameters or ``loc`` and ``scale`` passed as positional
        arguments.
    name : `str`
        Name of the unknown parameter, which is used to keep track and access
        the parameter in the sampling algorithms. Default: ``None``.
    tex : `str`, optional
        LaTeX typesetting for the parameter name. ``pyLFI`` includes procedures
        for automatically plotting priors and posteriors, and will use the ``tex``
        name of the parameter as axis labels if provided. Default: ``None``.
    kwargs:
        kwargs are passed to the scipy distribution methods. Typically
        these would be ``loc`` and ``scale``.
    """

    def __init__(self, distr_name, *params, name=None, tex=None, **kwargs):

        if name is None:
            msg = ("'name' of random variate must be provided as str")
            raise ValueError(msg)
        if not isinstance(name, str):
            msg = ("'name' must be given as str")
            raise TypeError(msg)
        self._name = name

        if tex is not None:
            if not isinstance(tex, str):
                msg = ("'tex' must be given as a latex formatted str")
                raise TypeError(msg)
        self._tex = tex

        if not isinstance(distr_name, str):
            msg = ("'distr_name' must be given as str")
            raise TypeError(msg)

        self._distr_name = distr_name
        self.distr = getattr(stats.distributions, self._distr_name)
        self.params = params
        self.kwargs = kwargs

        self._rng = np.random.default_rng

    def rvs(self, size=None, seed=None):
        """Draw random variate.

        Parameters
        ----------
        size : {`int`, `tuple`}, optional
            Output size of a single random draw. Default: ``None``.
        seed : `int`, optional
            Seed for reproducibility.

        Returns
        -------
        rvs : `numpy.ndarray`
            Random variables.
        """

        rvs = self.distr.rvs(*self.params,
                             **self.kwargs,
                             size=size,
                             random_state=self._rng(seed=seed),
                             )
        return rvs

    def pdf(self, x):
        r"""Evaluate the probability density function (pdf).

        Method for continuous distributions.

        Parameters
        ----------
        x : :term:`array_like`
            Evaluation points.

        Returns
        -------
        pdf : `numpy.ndarray`
            pdf evaluated at ``x``.
        """

        pdf = self.distr.pdf(x, *self.params, **self.kwargs)
        return pdf

    def logpdf(self, x):
        r"""Evaluate the log of the probability density function (pdf).

        Method for continuous distributions.

        Parameters
        ----------
        x : :term:`array_like`
            Evaluation points.

        Returns
        -------
        logpdf : `numpy.ndarray`
            Log of pdf evaluated at ``x``.
        """

        logpdf = self.distr.logpdf(x, *self.params, **self.kwargs)
        return logpdf

    def pmf(self, x):
        r"""Evaluate the probability mass function (pmf).

        Method for discrete distributions.

        Parameters
        ----------
        x : :term:`array_like`
            Evaluation points.

        Returns
        -------
        pmf : `numpy.ndarray`
            pmf evaluated at ``x``.
        """

        pmf = self.distr.pmf(x, *self.params, **self.kwargs)
        return pmf

    @property
    def name(self):
        """Parameter name.

        Returns
        -------
        `str`
        """
        return self._name

    @property
    def distr_name(self):
        """
        Name of the `scipy.stats` distribution.

        Returns
        -------
        `str`
        """
        return self._distr_name

    @property
    def tex(self):
        """Parameter name with LaTeX typesetting.

        Returns
        -------
        `str`
        """
        return self._tex

    def plot_prior(
        self,
        x,
        color='C0',
        facecolor='lightblue',
        alpha=0.5,
        ax=None,
        **kwargs
    ):
        r"""Plot prior pdf or pmf evaluated at ``x``.

        Parameters
        ----------
        x : :term:`array_like`
            Evaluation points.
        color : `str`, optional
            Set the color of the line. Default: ``C0``.
        facecolor : `str`, optional
            Set the face color of area under the curve. Default: ``lightblue``.
        alpha : `float`, optional
            Set the alpha value used for blending the face color. Must be
            within the 0-1 range. Default: ``0.5``.
        ax : `matplotlib.axes.Axes`, optional
            Pre-existing axes for the plot. Otherwise, call
            `matplotlib.pyplot.gca` internally.
        kwargs:
            kwargs are passed to `matplotlib.pyplot.plot`.
        """

        if hasattr(self.distr, 'pdf'):
            pxf = self.pdf(x)
            y_handle = 'Density'
        elif hasattr(self.distr, 'pmf'):
            pxf = self.pmf(x)
            y_handle = 'Probability'
        else:
            msg = (f'{self.distr} does not have a pdf or pmf method.')
            raise AttributeError(msg)

        if self.tex is not None:
            x_handle = self.tex
        else:
            x_handle = self.name

        if ax is None:
            ax = plt.gca()

        ax.plot(x, pxf, color=color, **kwargs)
        ax.fill_between(x, pxf, facecolor=facecolor, alpha=alpha)
        ax.set(xlabel=x_handle, ylabel=y_handle)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Initialize a Gaussian prior
    theta_prior = Prior('norm',
                        loc=0,
                        scale=1,
                        name='theta',
                        tex=r'$\theta$'
                        )

    # Sample from prior
    theta_samples = theta_prior.rvs(size=10, seed=42)

    # Plot prior
    x = np.linspace(-4, 4, 1000)
    theta_prior.plot_prior(x)
    plt.show()
