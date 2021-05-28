#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as stats
from pylfi.priors import ContinuousPrior, DiscretePrior


# Continuous distributions
class Uniform(ContinuousPrior):
    r"""A uniform continuous random variable.

    In the standard form, the distribution is uniform on ``[0, 1]``. Using the
    parameters ``loc`` and ``scale``, one obtains the uniform distribution on
    ``[loc, loc + scale]``.

    Parameters
    ----------
    loc : array_like, optional
        Location parameter (default=0)
    scale : array_like, optional
        Scale parameter (default=1)
    name : str
        Name of random variate. Default is None (which will raise an ``Error``).
    tex : raw str literal, optional
        LaTeX formatted name of random variate given as ``r"foo"``. Default is
        None.
    rng : random number generator, optional
        Defines the random number generator to be used. Default is
        np.random.RandomState
    seed : {None, int}, optional
        This parameter defines the object to use for drawing random
        variates. If seed is None the RandomState singleton is used.
        If seed is an int, a new RandomState instance is used, seeded
        with seed. Default is None.

    Attributes
    ----------
    name : str
        Name of random variate.
    tex : str
        LaTeX formatted name of random variate.

    Notes
    -----
    The uniform distribution is a probability distribution that has constant
    probability. The distribution describes an experiment where there is an
    arbitrary outcome that lies between certain bounds. The bounds are defined
    by the parameters, :math:`a` and :math:`b`, which are the minimum and
    maximum values. :math:`a` is the location parameter and :math:`(b-a)` is
    the scale parameter.

    The probability density function for ``Uniform`` is:

    .. math::
        f(x) = \frac{1}{b - a}

    for :math:`a \leq x \leq b`, and

    .. math::
        f(x) = 0

    for :math:`x < a` or :math:`x > b`.

    Examples
    --------
    >>> import numpy as np
    >>> from pylfi.priors import Uniform

    Initialize prior distribution for random variate :math:`\theta`:

    >>> theta_prior = Uniform(loc=0, scale=1, name='theta', tex=r'$\theta$', seed=42)

    Draw from prior:

    >>> theta = theta_prior.rvs(size=10)

    Evaluate probability density function:

    >>> x = np.linspace(-1, 2, 1000)
    >>> pdf = theta_prior.pdf(x)

    Display the probability density function:

    >>> theta_prior.plot_prior(x)
    """

    def __init__(
        self,
        loc=0.0,
        scale=1.0,
        name=None,
        tex=None
    ):
        super().__init__(
            shape=(),
            loc=loc,
            scale=scale,
            name=name,
            tex=tex,
            distr_name='uniform',
        )


class Normal(ContinuousPrior):
    r"""A normal continuous random variable.

    The location (``loc``) keyword specifies the mean. The scale (``scale``)
    keyword specifies the standard deviation.

    Parameters
    ----------
    loc : array_like, optional
        Location parameter (default=0)
    scale : array_like, optional
        Scale parameter (default=1)
    name : str
        Name of random variate. Default is None (which will raise an ``Error``).
    tex : raw str literal, optional
        LaTeX formatted name of random variate given as ``r"foo"``. Default is
        None.
    rng : Random number generator, optional
        Defines the random number generator to be used. Default is
        np.random.RandomState
    seed : {None, int}, optional
        This parameter defines the object to use for drawing random
        variates. If seed is None the RandomState singleton is used.
        If seed is an int, a new RandomState instance is used, seeded
        with seed. Default is None.

    Attributes
    ----------
    name : str
        Name of random variate.
    tex : str
        LaTeX formatted name of random variate.

    Notes
    -----
    The normal distribution is a continuous probability distribution that is
    symmetrical at the center, i.e. around the mean. Normal distribution is
    the proper term for a probability bell curve.

    The probability density function for ``Normal`` is:

    .. math::
        f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{1}{2} \left(\frac{x - \mu}{\sigma}\right)^2 \right)

    for a real number :math:`x`, where the mean :math:`\mu` is the location parameter
    and the standard deviation :math:`\sigma` the scale parameter.

    The case where :math:`\mu=0` and :math:`\sigma=1` is called the standard
    normal distribution.

    Examples
    --------
    >>> import numpy as np
    >>> from pylfi.priors import Normal

    Initialize prior distribution for random variate :math:`\theta`:

    >>> theta_prior = Normal(loc=0, scale=1, name='theta', tex=r'$\theta$', seed=42)

    Draw from prior:

    >>> theta = theta_prior.rvs(size=10)

    Evaluate probability density function:

    >>> x = np.linspace(-2, 2, 1000)
    >>> pdf = theta_prior.pdf(x)

    Display the probability density function:

    >>> theta_prior.plot_prior(x)
    """

    def __init__(
        self,
        loc=0.0,
        scale=1.0,
        name=None,
        tex=None
    ):
        super().__init__(
            shape=(),
            loc=loc,
            scale=scale,
            name=name,
            tex=tex,
            distr_name='norm'
        )


class Beta(ContinuousPrior):
    r"""A beta continuous random variable.

    ``Beta`` takes ``a`` and ``b`` as shape parameters.

    Parameters
    ----------
    a : float
        Shape parameter
    b : float
        Shape parameter
    loc : array_like, optional
        Location parameter (default=0)
    scale : array_like, optional
        Scale parameter (default=1)
    name : str
        Name of random variate. Default is None (which will raise an ``Error``).
    tex : raw str literal, optional
        LaTeX formatted name of random variate given as ``r"foo"``. Default is
        None.
    rng : Random number generator, optional
        Defines the random number generator to be used. Default is
        np.random.RandomState
    seed : {None, int}, optional
        This parameter defines the object to use for drawing random
        variates. If seed is None the RandomState singleton is used.
        If seed is an int, a new RandomState instance is used, seeded
        with seed. Default is None.

    Attributes
    ----------
    name : str
        Name of random variate.
    tex : str
        LaTeX formatted name of random variate.

    Notes
    -----
    The beta distribution is a family of continuous probability distributions
    set on the interval [0, 1] parameterized by two positive shape parameters,
    denoted by :math:`a` and :math:`b`, that appear as exponents of the random
    variable and control the shape of the distribution.

    The probability density function for ``Beta`` is:

    .. math::
        f(x, a, b) = \frac{\Gamma(a+b) x^{a-1} (1-x)^{b-1}}
                          {\Gamma(a) \Gamma(b)}

    for :math:`0 <= x <= 1`, :math:`a > 0`, :math:`b > 0`, where
    :math:`\Gamma` is the gamma function (`scipy.special.gamma`).

    Typically we define the general form of a distribution in terms of location
    and scale parameters. The beta is different in that we define the general
    distribution in terms of the lower and upper bounds. However, the location
    and scale parameters can be defined in terms of the lower and upper limits
    as follows:

    THIS NEEDS CORRECTION

    .. math::
        location = a

    .. math::
        scale = b - a

    The case where :math:`a = 0` and :math:`b = 1` is called the standard beta
    distribution.


    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from pylfi.priors import Beta

    Initialize prior distribution for random variate :math:`\theta`:

    >>> theta_prior = Beta(0, 1, loc=0, scale=1, name='theta', tex=r'$\theta$', seed=42)

    Draw from prior:

    >>> theta = theta_prior.rvs(size=10)

    Evaluate probability density function:

    >>> x = np.linspace(0, 1, 1000)
    >>> pdf = theta_prior.pdf(x)

    Display the probability density function:

    >>> theta_prior.plot_prior(x)

    Explore different shapes:

    >>> shapes = [[0.5, 5, 1, 2, 2], [0.5, 1, 3, 2, 5]]
    >>> x = np.linspace(0, 1, 1000)
    >>> for i in range(5):
    ...     a = shapes[0][i]
    ...     b = shapes[1][i]
    ...     theta_prior = Beta(a, b, loc=0, scale=1, name='theta')
    ...     plt.plot(x, theta_prior.pdf(x), label=f'{a=}, {b=}')
    >>> plt.legend()
    >>> plt.show()
    """

    def __init__(
        self,
        a,
        b,
        loc=0.0,
        scale=1.0,
        name=None,
        tex=None
    ):
        super().__init__(
            shape=(a, b),
            loc=loc,
            scale=scale,
            name=name,
            tex=tex,
            distr_name='beta'
        )


class LogNormal(ContinuousPrior):
    r"""A lognormal continuous random variable.

    `LogNormal` takes ``s`` as a shape parameter.

    Parameters
    ----------
    s : float
        Shape parameter
    loc : array_like, optional
        Location parameter (default=0)
    scale : array_like, optional
        Scale parameter (default=1)
    name : str
        Name of random variate. Default is None (which will raise an ``Error``).
    tex : raw str literal, optional
        LaTeX formatted name of random variate given as ``r"foo"``. Default is
        None.
    rng : Random number generator, optional
        Defines the random number generator to be used. Default is
        np.random.RandomState
    seed : {None, int}, optional
        This parameter defines the object to use for drawing random
        variates. If seed is None the RandomState singleton is used.
        If seed is an int, a new RandomState instance is used, seeded
        with seed. Default is None.

    Attributes
    ----------
    name : str
        Name of random variate.
    tex : str
        LaTeX formatted name of random variate.

    Notes
    -----
    The lognormal distribution is a continuous probability distribution of a
    random variable whose logarithm is normally distributed. Meaning, a random
    variable :math:`X` is lognormally distributed if :math:`Y=\ln(X)` is normally
    distributed.

    The probability density function for ``LogNormal`` is:

    .. math::
        f(x, s) = \frac{1}{(x - \theta) s \sqrt{2\pi}}
                  \exp\left(-\frac{\log^2\left(\frac{x-\theta}{m}\right)}{2s^2}\right)

    for :math:`x > \theta`, :math:`m, s > 0`. Here, :math:`s` is the shape
    parameter (and the standard deviation of the log of the distribution),
    :math:`\theta` is the location parameter and :matm:`m` is the scale parameter.

    If :math:`x=\theta`, then :math:`f(x)=0`. The case where :math:`\theta=0`
    and :math:`m=1` is called the standard lognormal distribution.

    The probability density is defined in the standardized form. To shift
    and/or scale the distribution use the ``loc`` and ``scale`` parameters.

    A common parametrization for a lognormal random variable ``Y`` is in
    terms of the mean, ``mu``, and standard deviation, ``sigma``, of the
    unique normally distributed random variable ``X`` such that exp(X) = Y.
    This parametrization corresponds to setting ``s = sigma`` and ``scale =
    exp(mu)``.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html#scipy.stats.lognorm
    .. [2] https://www.itl.nist.gov/div898/handbook/eda/section3/eda3669.htm
    .. [3] https://en.wikipedia.org/wiki/Log-normal_distribution
    """

    def __init__(
        self,
        s,
        loc=0.0,
        scale=1.0,
        name=None,
        tex=None
    ):
        super().__init__(
            shape=(s,),
            loc=loc,
            scale=scale,
            name=name,
            tex=tex,
            distr_name='lognorm'
        )


class Exponential(ContinuousPrior):
    r"""An exponential continuous random variable.

    Parameters
    ----------
    loc : array_like, optional
        Location parameter (default=0)
    scale : array_like, optional
        Scale parameter (default=1)
    name : str
        Name of random variate. Default is None (which will raise an ``Error``).
    tex : raw str literal, optional
        LaTeX formatted name of random variate given as ``r"foo"``. Default is
        None.
    rng : random number generator, optional
        Defines the random number generator to be used. Default is
        np.random.RandomState
    seed : {None, int}, optional
        This parameter defines the object to use for drawing random
        variates. If seed is None the RandomState singleton is used.
        If seed is an int, a new RandomState instance is used, seeded
        with seed. Default is None.

    Attributes
    ----------
    name : str
        Name of random variate.
    tex : str
        LaTeX formatted name of random variate.

    Notes
    -----
    The probability density function for ``Exponential`` is:

    .. math::
        f(x) = \exp(-x)

    for :math:`x \ge 0`.

    The probability density is defined in the standardized form. To shift
    and/or scale the distribution use the ``loc`` and ``scale`` parameters.

    A common parameterization for ``Exponential`` is in terms of the
    rate parameter ``lambda``, such that ``pdf = lambda * exp(-lambda * x)``.
    This parameterization corresponds to using ``scale = 1 / lambda``.
    """

    def __init__(
        self,
        loc=0.0,
        scale=1.0,
        name=None,
        tex=None
    ):
        super().__init__(
            shape=(),
            loc=loc,
            scale=scale,
            name=name,
            tex=tex,
            distr_name='expon'
        )


class Gamma(ContinuousPrior):
    r"""A gamma continuous random variable.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html#scipy.stats.gamma
    """

    def __init__(
        self,
        a,
        loc=0.0,
        scale=1.0,
        name=None,
        tex=None
    ):
        super().__init__(
            shape=(a,),
            loc=loc,
            scale=scale,
            name=name,
            tex=tex,
            distr_name='gamma'
        )


class InvGamma(ContinuousPrior):
    r"""An inverted gamma continuous random variable.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgamma.html
    """

    def __init__(
        self,
        a,
        loc=0.0,
        scale=1.0,
        name=None,
        tex=None
    ):
        super().__init__(
            shape=(a,),
            loc=loc,
            scale=scale,
            name=name,
            tex=tex,
            distr_name='invgamma'
        )


# Discrete distributions
class Randint(DiscretePrior):
    r"""A uniform discrete random variable.

    Parameters
    ----------
    loc : array_like, optional
        Location parameter (default=0)
    name : str
        Name of random variate. Default is None (which will raise an ``Error``).
    tex : raw str literal, optional
        LaTeX formatted name of random variate given as ``r"foo"``. Default is
        None.
    rng : Random number generator, optional
        Defines the random number generator to be used. Default is
        np.random.RandomState
    seed : {None, int}, optional
        This parameter defines the object to use for drawing random
        variates. If seed is None the RandomState singleton is used.
        If seed is an int, a new RandomState instance is used, seeded
        with seed. Default is None.

    Attributes
    ----------
    name : str
        Name of random variate.
    tex : str
        LaTeX formatted name of random variate.

    Notes
    -----
    The probability mass function for ``Randint`` is:

    .. math::
        f(k) = \frac{1}{high - low}

    for ``k = low, ..., high - 1``.

    ``Randint`` takes ``low`` and ``high`` as shape parameters.

    The probability mass function is defined in the standardized form.
    To shift distribution use the ``loc`` parameter.

    References
    ----------
    """

    def __init__(
        self,
        low,
        high,
        loc=0.0,
        name=None,
        tex=None
    ):
        super().__init__(
            shape=(low, high),
            loc=loc,
            name=name,
            tex=tex,
            distr_name='randint'
        )


class Binomial(DiscretePrior):
    r"""A binomial discrete random variable.

    Parameters
    ----------
    n : int
        Shape parameter; number of trials
    p : float
        Shape parameter; probability of single success
    loc : array_like, optional
        Location parameter (default=0)
    name : str
        Name of random variate. Default is None (which will raise an ``Error``).
    tex : raw str literal, optional
        LaTeX formatted name of random variate given as ``r"foo"``. Default is
        None.
    rng : Random number generator, optional
        Defines the random number generator to be used. Default is
        np.random.RandomState
    seed : {None, int}, optional
        This parameter defines the object to use for drawing random
        variates. If seed is None the RandomState singleton is used.
        If seed is an int, a new RandomState instance is used, seeded
        with seed. Default is None.

    Attributes
    ----------
    name : str
        Name of random variate.
    tex : str
        LaTeX formatted name of random variate.

    Notes
    -----
    The probability mass function for ``Binomial`` is:

    .. math::
       f(k) = \binom{n}{k} p^k (1-p)^{n-k}

    for ``k`` in ``{0, 1,..., n}``, :math:`0 \leq p \leq 1`

    ``Binomial`` takes ``n`` and ``p`` as shape parameters, where
    ``p`` is the probability of a single success and ``1 - p`` is
    the probability of a single failure.

    The probability mass function is defined in the “standardized” form.
    To shift distribution use the ``loc`` parameter.
    """

    def __init__(
        self,
        n,
        p,
        loc=0.0,
        name=None,
        tex=None
    ):
        super().__init__(
            shape=(n, p),
            loc=loc,
            name=name,
            tex=tex,
            distr_name='binom'
        )


class NegativeBinomial(DiscretePrior):
    r"""A negative binomial discrete random variable.
    """

    def __init__(
        self,
        n,
        p,
        loc=0.0,
        name=None,
        tex=None
    ):
        super().__init__(
            shape=(n, p),
            loc=loc,
            name=name,
            tex=tex,
            distr_name='nbinom'
        )


class Poisson(DiscretePrior):
    r"""A Poisson discrete random variable.

    Parameters
    ----------
    mu : float
        Shape parameter; the average number of events in the given time interval
    loc : array_like, optional
        Location parameter (default=0)
    name : str
        Name of random variate. Default is None (which will raise an ``Error``).
    tex : raw str literal, optional
        LaTeX formatted name of random variate given as ``r"foo"``. Default is
        None.
    rng : Random number generator, optional
        Defines the random number generator to be used. Default is
        np.random.RandomState
    seed : {None, int}, optional
        This parameter defines the object to use for drawing random
        variates. If seed is None the RandomState singleton is used.
        If seed is an int, a new RandomState instance is used, seeded
        with seed. Default is None.

    Attributes
    ----------
    name : str
        Name of random variate.
    tex : str
        LaTeX formatted name of random variate.

    Notes
    -----
    The Poisson distribution is a discrete probability distribution that
    expresses the probability of a given number of events occuring in a fixed
    interval of time or space if these events occur with a known constant mean
    rate and independently of the time since the last event.[1]

    The probability mass function for ``Poisson`` is:

    .. math::
        f(k) = \exp(-\mu) \frac{\mu^k}{k!}

    for :math:`k \ge 0`.

    ``Poisson`` takes :math:`\mu` as shape parameter. When ``mu = 0`` then at
    quantile ``k = 0``, ``pmf`` method returns ``1.0``.

    The probability mass function is defined in the “standardized” form.
    To shift distribution use the ``loc`` parameter.
    """

    def __init__(
        self,
        mu,
        loc=0.0,
        name=None,
        tex=None
    ):
        super().__init__(
            shape=(mu,),
            loc=loc,
            name=name,
            tex=tex,
            distr_name='poisson'
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(0, 10, 1000)
    theta_prior = LogNormal(s=0.1, loc=0, scale=1, name='theta')
    theta_prior.plot_prior(x)

    x = np.arange(0, 11)
    theta_prior = NegativeBinomial(n=10, p=0.5, loc=0, name='theta')
    # theta_prior.plot_prior(x)

    '''
    shapes = [[0.5, 5, 1, 2, 2], [0.5, 1, 3, 2, 5]]
    x = np.linspace(0, 1, 1000)

    for i in range(5):
        a = shapes[0][i]
        b = shapes[1][i]
        theta_prior = Beta(a, b, loc=0, scale=1, name='theta')
        plt.plot(x, theta_prior.pdf(x), label=f'{a=}, {b=}')
    plt.legend()
    plt.show()
    '''

    '''
    rv = Exponential(name='gbar_K', tex=r'$\bar{g}_K$', seed=42)
    print(rv.rvs(10))
    x = np.linspace(-2, 2, 1000)
    # rv.plot_prior(x)

    x = np.linspace(-2, 2, 1000)

    rv = Normal(loc=0, scale=0.5, name='rv', seed=42)
    print(rv.rvs(size=2))
    # rv.plot_prior(x)

    #

    rv = Uniform(loc=0, scale=1, name='rv', tex=r'rv', seed=42)
    print(rv.rvs())
    x = np.linspace(-1, 2, 1000)
    rv.plot_prior(x)

    a = b = 1
    rv = Beta(a, b, name='rv')
    print(rv.rvs())
    # rv.plot_prior(x)

    n, p = 10, 0.5
    x = np.arange(0, n + 1)
    rv = Binomial(n, p, name='rv')
    print(rv.rvs(10))
    # rv.plot_prior(x)
    '''
