#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# https://github.com/eth-cscs/abcpy/blob/master/abcpy/utils.py
# https://github.com/eth-cscs/abcpy/blob/master/abcpy/distances.py


def euclidean(s1, s2):
    """Euclidean distance.

    Calculates the distance between two sets of summary statistics by pairwise
    computing the Euclidean distance and then averaging.

    Parameters
    ----------
    s1: array_like
        First set of summary statistics.
    s2: array_like
        Second set of summary statistics.

    Returns
    -------
    numpy.float
        The distance between the summary statistic sets.
    """

    if isinstance(s1, (int, float)):
        s1 = [s1]
    if isinstance(s2, (int, float)):
        s2 = [s2]

    s1 = np.asarray(s1)
    s2 = np.asarray(s2)

    if s1.ndim == 1:
        s1 = s1.reshape(-1, 1)

    if s2.ndim == 1:
        s2 = s2.reshape(-1, 1)

    if s1.shape != s2.shape:
        msg = ("The observed and simulated sets of summary statistics must "
               "have equal shape, i.e. the same number of summary statistics."
               "\nDebug tip: Double-check that the passed 'observation' is "
               "a set of summary statistics and not the raw observed data.")
        raise RuntimeError(msg)

    dist = np.linalg.norm(s1 - s2, ord=2, axis=1)

    return dist.mean()


class DistanceMetrics:

    def __init__(self):
        pass

    @staticmethod
    def euclidean(sim_data, obs_data):
        return np.sqrt(np.sum((sim_data - obs_data) * (sim_data - obs_data)))


if __name__ == "__main__":
    s1 = 0.5
    s2 = [1., 1., 0.5, 0.8]

    if isinstance(s1, (int, float)):
        s1 = [s1]
    if isinstance(s2, (int, float)):
        s2 = [s2]

    s1 = np.asarray(s1)
    s2 = np.asarray(s2)

    if s1.ndim == 1:
        s1 = s1.reshape(-1, 1)

    if s2.ndim == 1:
        s2 = s2.reshape(-1, 1)

    print(s1.shape)
    print(s2.shape)

    dist = np.linalg.norm(s1 - s2, ord=2, axis=1)
    print(dist)
    #print(euclidean(s1, s2))
    #print(euclidean(1, np.inf))
    #dist = euclidean(1, np.inf)
