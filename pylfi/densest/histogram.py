#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from pylfi.utils import check_1D_data, check_number_of_entries, iqr
from scipy import optimize, special


def histogram(data, bins=10, density=True, **kwargs):
    """
    """
    bins = calculate_bins(data, bins=bins)
    return np.histogram(data, bins=bins, density=density, **kwargs)


def calculate_bins(data, bins):
    """
    """
    if isinstance(bins, str):
        data = np.asarray(data).ravel()

        if bins == "sqrt":
            bins = square_root_rule(data)
        elif bins == "sturges":
            bins = sturges_rule(data)
        elif bins == "scott":
            bins = scotts_rule(data)
        elif bins == "freedman":
            bins = freedman_diaconis_rule(data)
        elif bins == "knuth":
            bins = knuths_rule(data)
        else:
            msg = (f"Unrecognized bin rule: '{bins}'. Supported rules are: "
                   "'sqrt', 'sturges', 'scott', 'freedman', 'knuth'")
            raise ValueError(msg)

    return bins


def square_root_rule(data):
    """
    Calculate number of histogram bins using the Square-root rule.
    """
    data = np.asarray(data)
    check_1D_data(data)
    n = data.size
    return int(np.ceil(np.sqrt(n)))


def sturges_rule(data):
    """
    Calculate number of histogram bins using Sturges' rule.
    """
    data = np.asarray(data)
    check_1D_data(data)
    check_number_of_entries(data, n_entries=1)
    n = data.size
    return int(np.ceil(np.log2(n)) + 1)


def scotts_rule(data):
    """
    Calculate the optimal histogram bin width using Scott's rule.

    References
    ----------
    Scott, David W. (1979).
    "On optimal and data-based histograms".
    Biometricka 66 (3): 605-610
    """
    data = np.asarray(data)
    check_1D_data(data)
    check_number_of_entries(data, n_entries=1)

    n = data.size
    dmin, dmax = data.min(), data.max()
    h = 3.49 * np.std(data) * n**(-1 / 3)
    k = int(np.ceil((dmax - dmin) / h))
    bins = dmin + h * np.arange(k + 1)

    return bins


def freedman_diaconis_rule(data):
    """
    Calculate the optimal histogram bin width using the Freedman-Diaconis rule.

    References
    ----------
    D. Freedman & P. Diaconis. (1981).
    "On the histogram as a density estimator: L2 theory".
    Probability Theory and Related Fields 57 (4): 453-476
    """
    data = np.asarray(data)
    check_1D_data(data)
    check_number_of_entries(data, n_entries=3)

    n = data.size
    dmin, dmax = data.min(), data.max()
    h = 2 * iqr(data) * n**(-1 / 3)

    if h == 0:
        bins = scotts_rule(data)
    else:
        k = int(np.ceil((dmax - dmin) / h))
        bins = dmin + h * np.arange(k + 1)

    return bins


def knuths_rule(data):
    """
    Knuth’s rule chooses a constant bin size which minimizes the error of the
    histogram’s approximation to the data

    Based on the implementation in astropy:
    https://docs.astropy.org/en/stable/_modules/astropy/stats/histogram.html#knuth_bin_width

    References
    ----------
    Knuth, K.H. (2006).
    "Optimal Data-Based Binning for Histograms".
    arXiv:0605197
    """
    data = np.array(data, copy=True)
    check_1D_data(data)
    check_number_of_entries(data, n_entries=3)

    n = data.size
    data.sort()

    def knuth_func(M):
        """Evaluate the negative Knuth likelihood function => smaller values optimal"""
        M = int(M)

        if M <= 0:
            return np.inf

        bins = np.linspace(data[0], data[-1], int(M) + 1)
        nk, bins = np.histogram(data, bins)

        return -(n * np.log(M) +
                 special.gammaln(0.5 * M) -
                 M * special.gammaln(0.5) -
                 special.gammaln(n + 0.5 * M) +
                 np.sum(special.gammaln(nk + 0.5)))

    bins0 = freedman_diaconis_rule(data)
    M = optimize.fmin(knuth_func, len(bins0), disp=False)[0]
    bins = np.linspace(data[0], data[-1], int(M) + 1)

    return bins


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    def make_data(N):
        rng = np.random.RandomState(1)
        x = rng.randn(N)
        x[int(0.3 * N):] += 5
        return x

    def complicated_data():
        # generate some complicated data
        rng = np.random.RandomState(1)
        x = np.concatenate([-5 + 1.8 * rng.standard_cauchy(500),
                            -4 + 0.8 * rng.standard_cauchy(2000),
                            -1 + 0.3 * rng.standard_cauchy(500),
                            2 + 0.8 * rng.standard_cauchy(1000),
                            4 + 1.5 * rng.standard_cauchy(1000)])

        # truncate to a reasonable range
        x = x[(x > -15) & (x < 15)]
        return x

    data = make_data(1000)
    #data = complicated_data()

    '''
    counts, bins = histogram(data, bins='knuth')
    fig, ax = plt.subplots(1, 2)
    ax[0].hist(bins[:-1], bins=bins, weights=counts)
    ax[1].hist(data, bins=bins, density=True)
    plt.show()
    '''

    rules = [30, 'sqrt', 'sturges', 'scott', 'freedman', 'knuth']

    fig = plt.figure(figsize=(10, 6), tight_layout=True, dpi=120)
    gs = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)

    for i, rule in enumerate(rules):
        ax = fig.add_subplot(gs[i])
        counts, bins = histogram(data, bins=rule)
        ax.hist(bins[:-1], bins=bins, weights=counts, histtype='bar',
                edgecolor=None, color='steelblue', alpha=0.9, label=f"bins = {rule}")
        #ax.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
        ax.set_xlabel("$x$")
        ax.set_ylabel("Density")
        ax.legend(loc="upper left")

    plt.show()
