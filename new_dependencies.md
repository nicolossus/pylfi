## Pathos

    $ pip install pathos

Pathos, a python parallel processing library from caltech.

Unlike Python's default multiprocessing library, pathos provides a more flexible
parallel map which can apply almost any type of function --- including lambda
functions, nested functions, and class methods --- and can easily handle
functions with multiple arguments.

In author’s own words:

"Pathos is a framework for heterogenous computing.
It primarily provides the communication mechanisms for configuring
and launching parallel computations across heterogenous resources"

multiprocess.Pool is a fork of multiprocessing.Pool, with the only difference
being that multiprocessing uses pickle and multiprocess uses dill

The preferred interface is pathos.pools.ProcessPool

The pathos-wrapped pool is pathos.pools.ProcessPool (and the old interface
provides it at pathos.multiprocessing.Pool).


## Colorlog

    $ pip install colorlog

colorful progress bar

## tqdm

progress bar


## Notes

* Check that posterior != prior with KL div
* Prior as elfi



https://stackoverflow.com/questions/12915177/same-output-in-different-workers-in-multiprocessing

import numpy as np
from multiprocessing import Pool

entropy = 42
seed_sequence = np.random.SeedSequence(entropy)

number_processes = 5

seeds = seed_sequence.spawn(number_processes)

def good_practice(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0,10,size=10)

pool = Pool(number_processes)


print(pool.map(good_practice, seeds))

https://numpy.org/doc/stable/reference/random/parallel.html
https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.SeedSequence.html
