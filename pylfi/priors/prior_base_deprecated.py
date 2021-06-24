#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from pylfi.utils import set_plot_style


class Prior(metaclass=ABCMeta):
    r"""Abstract base class defining handling of priors.

    The class wraps functionality contained in the `scipy.stats` module. In
    particular, the functionality for drawing random variables from probability
    distributions. For this purpose, `scipy.stats` has the following logic:

    >>> <dist_name>.rvs(<shape(s)>,
    ...                 loc=<param1>,
    ...                 scale=<param2>,
    ...                 size=(Nx, Ny),
    ...                 random_state=np.random.RandomState(seed=None)
                        )

    In general, one must provide shape parameters and, optionally, location
    and scale parameters to each call of a method of a distribution.

    The base class also implements methods for evaluating the pdf, plotting the
    prior and getting class attributes as properties.

    The constructor of a sub-class must accept the following arguments:

    Parameters
    ----------
    shape : array_like
        The shape parameter(s) for the distribution (see docstring of the
         instance object for more information)
    loc : array_like
        Location parameter
    scale: array_like
        Scale parameter
    name : str
        Name of random variate
    tex : raw str literal
        LaTeX formatted name of random variate given as `r"foo"`
    distr_name : str
        Name of distribution that exisist in `scipy.distribution`
    rng : Random number generator
        Defines the random number generator to be used.
    seed : {None, int}
        This parameter defines the object to use for drawing random
        variates. If seed is None the RandomState singleton is used.
        If seed is an int, a new RandomState instance is used, seeded
        with seed.
    """

    def __init__(self, shape, loc, scale, name, tex, distr_name):
        """Constructor that must be overwritten by the sub-class."""

        self.distr_name = distr_name
        self.distr = getattr(stats.distributions, self.distr_name)
        self.shape = shape
        self.loc = loc
        self.scale = scale

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

    @abstractmethod
    def rvs(self, size, rng, seed):
        r"""To be overwritten by sub-class; draw random variates from
        distribution.
        """

        raise NotImplementedError

    @ property
    def name(self):
        return self._name

    @ property
    def tex(self):
        return self._tex


class ContinuousPrior(Prior):
    r"""Base class for continuous priors"""

    def __init__(self, shape, loc, scale, name, tex, distr_name):
        """Constructor for continuous prior classes.

        Sub-classes must provide the following arguments:

        Parameters
        ----------
        shape : array_like
            The shape parameter(s) for the distribution (see docstring of the
             instance object for more information)
        loc : array_like
            Location parameter
        scale: array_like
            Scale parameter
        name : str
            Name of random variate
        tex : raw str literal
            LaTeX formatted name of random variate given as `r"foo"`
        distr_name : str
            Name of distribution that exisist in `scipy.distribution`
        rng : Random number generator
            Defines the random number generator to be used.
        seed : {None, int}
            This parameter defines the object to use for drawing random
            variates. If seed is None the RandomState singleton is used.
            If seed is an int, a new RandomState instance is used, seeded
            with seed.
        """

        super().__init__(
            shape=shape,
            loc=loc,
            scale=scale,
            name=name,
            tex=tex,
            distr_name=distr_name
        )

    def rvs(self, size=None, rng=np.random.RandomState, seed=None):
        r"""Draw random variates from distribution.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Defining number of random variates (default is None).

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given size.
        """

        rvs = self.distr.rvs(*self.shape,
                             loc=self.loc,
                             scale=self.scale,
                             size=size,
                             random_state=rng(seed=seed)
                             )
        return rvs

    def pdf(self, x):
        r"""Evaluate the probability density function (PDF).

        Parameters
        ----------
        x : array_like
            Quantiles

        Returns
        -------
        pdf : ndarray
            PDF evaluated at x
        """

        pdf = self.distr.pdf(x, *self.shape, loc=self.loc, scale=self.scale)
        return pdf

    def plot_prior(self, x, show=True, filename=None, dpi=100):
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

        pdf = self.pdf(x)
        set_plot_style()

        if self.tex is not None:
            x_handle = self.tex
        else:
            x_handle = self.name

        fig, ax = plt.subplots(1, 1,
                               figsize=(8, 6),
                               dpi=dpi)
        ax.plot(x, pdf, label=f'{self.__class__.__name__} pdf')
        ax.fill_between(x, pdf, alpha=0.5, facecolor='lightblue')
        ax.set_ylabel('Density')
        ax.set_xlabel(x_handle)
        ax.legend()
        if show:
            plt.show()
        if filename is not None:
            fig.savefig(filename)
        return ax


class DiscretePrior(Prior):
    r"""Base class for discrete priors"""

    def __init__(self, shape, loc, name, tex, distr_name):
        r"""Constructor for discrete prior classes.

        Sub-classes must provide the following arguments:

        Parameters
        ----------
        shape : array_like
            The shape parameter(s) for the distribution (see docstring of the
             instance object for more information)
        loc : array_like
            Location parameter
        name : str
            Name of random variate
        tex : raw str literal
            LaTeX formatted name of random variate given as `r"foo"`
        distr_name : str
            Name of distribution that exisist in `scipy.distribution`
        rng : Random number generator
            Defines the random number generator to be used.
        seed : {None, int}
            This parameter defines the object to use for drawing random
            variates. If seed is None the RandomState singleton is used.
            If seed is an int, a new RandomState instance is used, seeded
            with seed.
        """

        super().__init__(
            shape=shape,
            loc=loc,
            scale=None,
            name=name,
            tex=tex,
            distr_name=distr_name
        )

    def rvs(self, size=None, rng=np.random.RandomState, seed=None):
        r"""Draw random variates from distribution.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Defining number of random variates (default is None).

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given size.
        """

        rvs = self.distr.rvs(*self.shape,
                             loc=self.loc,
                             size=size,
                             random_state=rng(seed=seed)
                             )
        return rvs

    def pmf(self, x):
        r"""Evaluate the probability mass function (PMF).

        Parameters
        ----------
        x : array_like
            Quantiles

        Returns
        -------
        pmf : ndarray
            PMF evaluated at x
        """

        pmf = self.distr.pmf(x, *self.shape, loc=self.loc)
        return pmf

    def plot_prior(self, x, show=True, filename=None, dpi=100):
        r"""Plot prior PMF evaluated at x.

        Parameters
        ----------
        x : array_like
            Quantiles
        show : bool, optional, default True
            Calls plt.show() if True
        filename : str, optional, default None
            Saves the figure as filename if provided
        dpi : int, optional, default 100
            Set figure dpi
        """

        pmf = self.pmf(x)
        set_plot_style()

        if self.tex is not None:
            x_handle = self.tex
        else:
            x_handle = self.name

        fig, ax = plt.subplots(1, 1,
                               figsize=(8, 6),
                               dpi=dpi)
        ax.plot(x, pmf, 'o', ms=8, label=f'{self.__class__.__name__} PMF')
        ax.vlines(x, 0, pmf, colors='b', lw=3, alpha=0.5)
        ax.set_ylabel('Probability')
        ax.set_xlabel(x_handle)

        plt.legend()
        if show:
            plt.show()
        if filename is not None:
            fig.savefig(filename)
