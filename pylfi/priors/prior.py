#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


class Prior:
    r"""
    Initialize a Prior.

    Parameters
    ----------
    distr_name : str
        Any distribution from `scipy.stats` as a string.
    params:
        Parameters of the prior distribution. Typically these would be
        `shape` parameters or `loc` and `scale` passed as positional
        arguments.
    kwargs:
        kwargs are passed to the scipy distribution methods. Typically
        these would be `loc` and `scale`.

    Notes
    -----
    The parameters of the `scipy` distributions (typically `loc` and `scale`)
    must be given as positional arguments.

    Many algorithms (e.g. MCMC) also require a `pdf` method for the
    distribution. In general the definition of the distribution is a
    subset of `scipy.stats.rv_continuous`:
    Scipy distributions: https://docs.scipy.org/doc/scipy-0.19.0/reference/stats.html
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
        size : int, tuple or None, optional
            Output size of a single random draw.
        seed : int, optional
            Seed

        Returns
        -------
        rvs : ndarray
            Random variables
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
        x : array_like
            Quantiles

        Returns
        -------
        pdf : ndarray
            pdf evaluated at x
        """

        pdf = self.distr.pdf(x, *self.params, **self.kwargs)
        return pdf

    def logpdf(self, x):
        r"""Evaluate the log of the probability density function (pdf).

        Method for continuous distributions.

        Parameters
        ----------
        x : array_like
            Quantiles

        Returns
        -------
        pdf : ndarray
            Log of pdf evaluated at x
        """

        logpdf = self.distr.logpdf(x, *self.params, **self.kwargs)
        return logpdf

    def pmf(self, x):
        r"""Evaluate the probability mass function (pmf).

        Method for discrete distributions.

        Parameters
        ----------
        x : array_like
            Quantiles

        Returns
        -------
        pmf : ndarray
            pmf evaluated at x
        """

        pmf = self.distr.pmf(x, *self.params, **self.kwargs)
        return pmf

    @property
    def name(self):
        return self._name

    @property
    def distr_name(self):
        return self._distr_name

    @property
    def tex(self):
        return self._tex

    def plot_prior(
        self,
        x, ax=None,
        filename=None,
        figsize=(6, 4),
        dpi=100,
        **kwargs
    ):
        r"""Plot prior PDF evaluated at x.

        Parameters
        ----------
        x : array_like
            Quantiles
        ax : Axes, optional
            Axes object. Default is None.
        show : bool, optional
            Calls plt.show() if True. Default is True.
        filename : str, optional
            Saves the figure as filename if provided. Default is None.
        dpi : int, optional
            Set figure dpi, default=100.
        """
        # TODO: hasattr pdf or pmf
        pdf = self.pdf(x)

        if self.tex is not None:
            x_handle = self.tex
        else:
            x_handle = self.name

        if ax is None:
            fig, ax = plt.subplots(1, 1,
                                   figsize=figsize,
                                   dpi=dpi)
        ax.plot(x, pdf, **kwargs)
        ax.fill_between(x, pdf, alpha=0.5, facecolor='lightblue')
        ax.set_ylabel('Density')
        ax.set_xlabel(x_handle)
        if filename is not None:
            fig.savefig(filename)
        return ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dist = 'norm'
    theta = Prior(dist, loc=0, scale=1, name='theta')
    print(theta.rvs(1, seed=42))
    x = np.linspace(-0.1, 1.1, 1000)

    theta.plot_prior(x)

    #plt.plot(x, theta.logpdf(x))
    plt.show()
