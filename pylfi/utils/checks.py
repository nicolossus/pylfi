#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

VALID_DISTANCES = ['l1', 'l2', 'mse']
VALID_KERNELS = ['auto', 'gaussian', 'tophat',
                 'epanechnikov', 'exponential', 'linear', 'cosine']
VALID_BANDWIDTHS = ['auto', 'scott', 'silverman']
VALID_BANDWIDTHS2 = ['scott', 'silverman']


def check_distance_str(distance):
    """Check if distance function str is valid"""
    if not distance in VALID_DISTANCES:
        msg = (f"distance function str must be one of {VALID_DISTANCES}")
        raise ValueError(msg)


def check_kernel(kernel):
    """Check for valid kernel(s)"""
    msg1 = ("Kernel must be specified as a str and kernels as sequence of str")

    msg2 = (f"Unrecognized kernel(s): '{kernel}'. Choose among 'auto', "
            "'gaussian', 'tophat', 'epanechnikov', 'exponential', "
            "'linear', 'cosine'")

    if not isinstance(kernel, (str, list, np.ndarray)):
        raise TypeError(msg1)

    if isinstance(kernel, str):
        if kernel not in VALID_KERNELS:
            raise ValueError(msg2)

    elif isinstance(kernel, (list, np.ndarray)):
        if not any(e in kernel for e in VALID_KERNELS):
            raise ValueError(msg2)


def check_bandwidth(bandwidth):
    """Check for valid bandwidth(s)"""
    if not isinstance(bandwidth, (int, str, float, list, np.ndarray)):
        msg = ("Bandwidth must be specified as a number, sequence or rule")
        raise TypeError(msg)

    if isinstance(bandwidth, (list, np.ndarray)):
        if not all(isinstance(e, (str, int, float)) for e in bandwidth):
            msg = ("Elements in sequence must be a number or rule")
            raise ValueError(msg)

        for e in (e for e in bandwidth if isinstance(e, str)):
            if e not in VALID_BANDWIDTHS2:
                msg = (f"Unrecognized bandwidth rule: '{e}'. Supported "
                       "rules for sequences are: 'scott', 'silverman'")
                raise ValueError(msg)

        for e in (e for e in bandwidth if isinstance(e, (int, float))):
            if e <= 0:
                msg = ("Bandwidth must be positive")
                raise ValueError(msg)

    if isinstance(bandwidth, (int, float)):
        if bandwidth <= 0:
            msg = ("Bandwidth must be positive")
            raise ValueError(msg)

    if isinstance(bandwidth, str):
        if bandwidth not in VALID_BANDWIDTHS:
            msg = (f"Unrecognized bandwidth rule: '{bandwidth}'. Supported "
                   "rules are: 'auto', 'scott', 'silverman'")
            raise ValueError(msg)


def check_1D_data(data):
    """Check that data is one-dimensional"""
    if data.ndim != 1:
        msg = ("data must be one-dimensional")
        raise ValueError(msg)


def check_number_of_entries(data, n_entries=1):
    """Check that data has more than specified number of entries"""
    if not data.size > n_entries:
        msg = (f"Data should have more than {n_entries} entries")
        raise ValueError(msg)


if __name__ == "__main__":
    import scipy.stats as stats

    bandwidth = [2, 3]

    print(np.isscalar('a'))

    data = np.array([1, 2])
    check_number_of_entries(data)

    check_bandwidth(bandwidth)

    #kernel = ['gaussian', 'tophat']
    kernel = 'auto'
    check_kernel(kernel)

    '''
    ## kernel sampling
    rseed = 42
    rng = np.random.RandomState(rseed)
    n_samples = 1000
    h = 0.1

    data = rng.randn(n_samples)

    i = rng.randint(0, len(data), size=n_samples)
    points = data[i]
    samples = rng.normal(loc=points, scale=h)

    data_mean = np.mean(data, axis=0)
    kde_mean = np.mean(samples, axis=0)

    data_std = np.std(data, axis=0, ddof=1)
    std = np.std(data, axis=0, ddof=1)
    '''

    '''
    print(f"{data=}")
    print(f"{i=}")
    print(f"{points=}")
    print(f"{samples=}")
    '''

    '''
    print(f"{data_mean=}")
    print(f"{mean=}")
    print(f"{data_std=}")
    print(f"{std=}")
    print(np.std(samples))
    '''
