#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pylfi
import scipy.stats as stats
from pylfi.inferences import ABCBase
from pylfi.utils import (advance_PRNG_state, check_and_set_jobs,
                         distribute_workload, generate_seed_sequence,
                         setup_logger)


class MCMCABC(ABCBase):

    def __init__(self, observation, simulator, stat_calc, priors, distance='l2', seed=None):

        super().__init__(
            observation=observation,
            simulator=simulator,
            statistics_calculator=stat_calc,
            priors=priors,
            distance_metric=distance,
            seed=seed
        )

    def sample(self):
        pass

    def sample(
        self,
        n_samples,
        epsilon=None,
        scaling=0.5,
        burn=100,
        tune=True,
        n_tune=500,
        tune_interval=50,
        n_jobs=-1,
        log=False
    ):
        _inference_scheme = "MCMC-ABC"
        self._epsilon = epsilon
        self._sigma = scaling
        self._rng = np.random.default_rng
        self._prior_logpdfs = [prior.logpdf for prior in self._priors]

        self._uniform_distr = stats.uniform(loc=0, scale=1)

        samples = self._sample(n_samples, self._seed)


def metropolis(n_samples, params0, target, sigma_proposals, warmup=0, seed=0):
    """
    ELFI

    Sample the target with a Metropolis Markov Chain Monte Carlo using Gaussian proposals.

    Parameters
    ----------
    n_samples : int
        The number of requested samples.
    params0 : np.array
        Initial values for each sampled parameter.
    target : function
        The target log density to sample (possibly unnormalized).
    sigma_proposals : np.array
        Standard deviations for Gaussian proposals of each parameter.
    warmup : int
        Number of warmup samples.
    seed : int, optional
        Seed for pseudo-random number generator.
    Returns
    -------
    samples : np.array
    """
    random_state = np.random.RandomState(seed)

    samples = np.empty((n_samples + warmup + 1, ) + params0.shape)
    samples[0, :] = params0
    target_current = target(params0)
    if np.isinf(target_current):
        raise ValueError(
            "Metropolis: Bad initialization point {},logpdf -> -inf.".format(params0))

    n_accepted = 0

    for ii in range(1, n_samples + warmup + 1):
        samples[ii, :] = samples[ii - 1, :] + \
            sigma_proposals * random_state.randn(*params0.shape)
        target_prev = target_current
        target_current = target(samples[ii, :])
        if ((np.exp(target_current - target_prev) < random_state.rand())
            or np.isinf(target_current)
                or np.isnan(target_current)):  # reject proposal
            samples[ii, :] = samples[ii - 1, :]
            target_current = target_prev
        else:
            n_accepted += 1

    logger.info(
        "{}: Total acceptance ratio: {:.3f}".format(__name__,
                                                    float(n_accepted) / (n_samples + warmup)))

    return samples[(1 + warmup):, :]


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import seaborn as sns
    from arviz import autocorr

    # global variables
    groundtruth = 2.0  # true variance
    N = 1000           # number of observations

    # observed data
    likelihood = stats.norm(loc=0, scale=np.sqrt(groundtruth))
    obs_data = likelihood.rvs(size=N)

    def summary_calculator(data):
        return np.var(data)

    def simulator(theta, N=1000):
        """Simulator model, returns summary statistic"""
        model = stats.norm(loc=0, scale=np.sqrt(theta))
        sim = model.rvs(size=N)
        return sim

    # prior (conjugate)
    alpha = 60         # prior hyperparameter (inverse gamma distribution)
    beta = 130         # prior hyperparameter (inverse gamma distribution)
    theta = pylfi.Prior('invgamma', alpha, loc=0, scale=beta, name='theta')
    priors = [theta]

    # initialize sampler
    sampler = MCMCABC(obs_data, simulator, summary_calculator,
                      priors, distance='l2', seed=42)

    # inference config
    n_samples = 1000
    epsilon = 0.5

    # run inference
    journal = sampler.sample(n_samples, epsilon=epsilon,
                             scaling=0.5, n_jobs=-1, log=False)

    samples = np.concatenate(journal, axis=0)

    # print(autocorr(samples))

    fig, ax = plt.subplots()
    # ax.plot(autocorr(samples))

    sns.distplot(samples)
    plt.show()
