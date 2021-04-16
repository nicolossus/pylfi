#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from pylfi.densest import histogram


def histplot(data, ax=None, bins=10, **kwargs):
    """
    Histogram plot

    Parameters
    ----------
    data : array_like
        data
    ax : Axes, optional
        Axes object. Default is None.

    Returns
    -------
    ax : Axes
        The new Axes object
    """
    if ax is None:
        ax = plt.gca()
    counts, bins = histogram(data, bins=bins)
    ax.hist(bins[:-1], bins=bins, weights=counts, histtype='bar',
            edgecolor=None, color='steelblue', alpha=0.9, **kwargs)


def rugplot(data, ax=None, **kwargs):
    """
    Rug plot

    """
    if ax is None:
        ax = plt.gca()
    ax.plot(data, np.full_like(data, -0.01), '|k', markeredgewidth=1, **kwargs)


def kdeplot(x, density, ax=None, fill=False, **kwargs):
    """
    KDE plot


    """
    if ax is None:
        ax = plt.gca()

    if fill:
        ax.fill_between(x, density, alpha=0.5, **kwargs)
    else:
        ax.plot(x, density, **kwargs)


if __name__ == "__main__":
    from matplotlib import gridspec

    def make_data(N, f=0.3, rseed=1):
        rand = np.random.RandomState(rseed)
        x = rand.randn(N)
        x[int(f * N):] += 5
        return x

    data = make_data(500)

    '''
    fig, ax = plt.subplots(1, 2)
    histplot(data, ax[0])
    rugplot(data, ax[0])
    histplot(data, ax[1], bins='freedman')
    rugplot(data, ax[1])
    plt.show()
    '''

    rules = [30, 'sqrt', 'sturges', 'scott', 'freedman', 'knuth']

    fig = plt.figure(figsize=(10, 6), tight_layout=True, dpi=120)
    gs = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)

    for i, rule in enumerate(rules):
        ax = fig.add_subplot(gs[i])
        histplot(data, ax, bins=rule, label=f"bins = {rule}")
        rugplot(data, ax)
        ax.set_xlabel("$x$")
        ax.set_ylabel("Density")
        ax.legend(loc="upper left")

    plt.show()
