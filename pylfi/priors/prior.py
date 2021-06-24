import numpy as np
import scipy.stats as stats


class Prior:

    def __init__(self, distr_name, *params, name=None, tex=None, **kwargs):
        """
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
        The parameters of the `scipy` distributions (typically `loc` and `scale`) must be
        given as positional arguments.
        Many algorithms (e.g. SMC) also require a `pdf` method for the distribution. In
        general the definition of the distribution is a subset of
        `scipy.stats.rv_continuous`.
        Scipy distributions: https://docs.scipy.org/doc/scipy-0.19.0/reference/stats.html
        """

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

        pdf = self.distr.logpdf(x, *self.params, **self.kwargs)
        return pdf

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

    @ property
    def name(self):
        return self._name

    @ property
    def distr_name(self):
        return self._distr_name

    @ property
    def tex(self):
        return self._tex

    def plot_prior(self, x):
        # hasattr pdf or pmf
        pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dist = 'norm'
    theta = Prior(dist, loc=0, scale=1, name='theta')
    print(theta.rvs(1, seed=42))
    x = np.linspace(0, 1, 1000)

    plt.plot(x, theta.logpdf(x))
    plt.show()
