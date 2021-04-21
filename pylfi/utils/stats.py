#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def sigmoid(x):
    """Numerical stable sigmoid function"""
    if x < 0:
        a = np.exp(x)
        return a / (1 + a)
    else:
        return 1 / (1 + np.exp(-x))


def iqr(data, norm=False):
    """Calculate inter-quartile range (IQR) of the given data. Returns
    normalized IQR(x) if keyword norm=True.
    """
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if norm:
        normalize = 1.349  # normalize = norm.ppf(.75) - norm.ppf(.25)
        iqr /= normalize
    return iqr


def KL(P, Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001

    # You may want to instead make copies to avoid changing the np arrays.
    P = P + epsilon
    Q = Q + epsilon

    divergence = np.sum(P * np.log(P / Q))
    return divergence


def kl_div(d1, d2):
    """
    Calculate Kullback-Leibler divergence of d1 and d2, which are assumed to
    values of two different density functions at the given positions xs.
    """
    x1 = np.sort(d1)
    x2 = np.sort(d2)
    xs = np.sort(np.concatenate([x1, x2]))
    with np.errstate(divide='ignore', invalid='ignore'):
        kl = d1 * (np.log(d1) - np.log(d2))
    # small numbers in p1 or p2 can cause NaN/-inf, etc.
    kl[~np.isfinite(kl)] = 0
    return np.trapz(kl, x=xs)  # integrate curve


def covmatrix():
    pass


if __name__ == "__main__":
    rng = np.random.RandomState(42)
    data = rng.randn(100)
    print(iqr(data))
    print(iqr(data, norm=True))
