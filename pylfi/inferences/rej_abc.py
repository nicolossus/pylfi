#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from multiprocessing import RLock

import numpy as np
from pathos.pools import ProcessPool
from pylfi.inferences import ABCBase
from pylfi.utils import (advance_PRNG_state, check_and_set_jobs,
                         distribute_workload, generate_seed_sequence,
                         setup_logger)
from tqdm.auto import tqdm


class RejABC(ABCBase):

    def __init__(self, observation, simulator, priors, distance='l2',
                 rng=np.random.RandomState, seed=None):

        super().__init__(
            observation=observation,
            simulator=simulator,
            priors=priors,
            distance=distance,
            rng=rng,
            seed=seed
        )

        #self._n_sims = 0

    def sample(self, n_samples, epsilon=None, n_jobs=-1, log=False):
        """
        Due to multiprocessing, estimation time (iteration per loop, total
        time, etc.) could be unstable, but the progress bar works perfectly.

        A good choice for the number of jobs is the number of cores or processors on your computer.
        If your processor supports hyperthreading, you can select an even higher number of jobs.
        The number of jobs is set to the number of cores found in the system by default.
        """

        _inference_scheme = "Rejection ABC"
        self._epsilon = epsilon

        if log:
            self.logger = setup_logger(self.__class__.__name__)
            self.logger.info(f"Run {_inference_scheme} sampler.")
            n_jobs = check_and_set_jobs(n_jobs, self.logger)
        else:
            n_jobs = check_and_set_jobs(n_jobs)

        seeds = generate_seed_sequence(self._seed, n_jobs)
        tasks = distribute_workload(n_samples, n_jobs)

        if log:
            tqdm.set_lock(RLock())  # for managing output contention
            with ProcessPool(n_jobs) as pool:
                samples, distances, sum_stats, epsilons, n_sims = zip(*pool.map(
                    self._sample_with_log,
                    tasks,
                    range(n_jobs),
                    seeds,
                    initializer=tqdm.set_lock)
                )
        else:
            with ProcessPool(n_jobs) as pool:
                samples, distances, sum_stats, epsilons, n_sims = zip(*pool.map(
                    self._sample,
                    tasks,
                    seeds)
                )

        samples = np.concatenate(samples, axis=0)
        distances = np.concatenate(distances, axis=0)
        sum_stats = np.concatenate(sum_stats, axis=0)
        epsilons = np.concatenate(epsilons, axis=0)
        n_sims = np.sum(n_sims)

        # return results
        return samples, distances, sum_stats, epsilons, n_sims

    def _sample(self, n_samples, seed):
        """Sample n_samples from posterior."""

        self._n_sims = 0
        samples = []
        distances = []
        sum_stats = []
        epsilons = []

        for _ in range(n_samples):
            sample, distance, sim, epsilon = self._draw_posterior_sample(seed)
            samples.append(sample)
            distances.append(distance)
            sum_stats.append(sim)
            epsilons.append(epsilon)

        return samples, distances, sum_stats, epsilons, self._n_sims

    def _sample_with_log(self, n_samples, position, seed):
        """Sample n_samples from posterior with progress bar."""

        self._n_sims = 0
        samples = []
        distances = []
        sum_stats = []
        epsilons = []

        t_range = tqdm(range(n_samples),
                       desc=f"[Sampling progress] CPU {position+1}",
                       position=position,
                       leave=True,
                       colour='green')
        for _ in t_range:
            sample, distance, sim, epsilon = self._draw_posterior_sample(seed)
            samples.append(sample)
            distances.append(distance)
            sum_stats.append(sim)
            epsilons.append(epsilon)

        t_range.clear()

        return samples, distances, sum_stats, epsilons, self._n_sims

    def _draw_posterior_sample(self, seed):
        """Rejection ABC algorithm."""
        sample = None
        while sample is None:
            next_gen = advance_PRNG_state(seed, self._n_sims)
            thetas = [prior.rvs(seed=next_gen) for prior in self._priors]
            sim = self._simulator(*thetas)
            self._n_sims += 1
            distance = self._distance(self._obs, sim)
            if distance <= self._epsilon:
                sample = thetas

        return sample, distance, sim, self._epsilon


if __name__ == "__main__":
    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np
    import pylfi
    import scipy.stats as stats
    import seaborn as sns

    # Task: infer variance parameter in zero-centered Gaussian model
    #
    groundtruth = 2.0  # true variance
    observation = groundtruth

    def summary_statistic(data):
        return np.var(data)

    def simulator(theta, seed=42, N=10000):
        """Simulator model, returns summary statistic"""
        model = stats.norm(loc=0, scale=np.sqrt(theta))
        sim = model.rvs(size=N, random_state=np.random.default_rng(seed))
        sim_sumstat = summary_statistic(sim)
        return sim_sumstat

    # prior (conjugate)
    alpha = 60         # prior hyperparameter (inverse gamma distribution)
    beta = 130         # prior hyperparameter (inverse gamma distribution)
    theta = pylfi.Prior('invgamma', alpha, loc=0, scale=beta, name='theta')
    priors = [theta]

    # initialize sampler
    sampler = RejABC(observation, simulator, priors, distance='l2', seed=42)

    # inference config
    n_samples = 1000
    epsilon = 0.2

    # run inference
    '''
    results = sampler.sample(n_samples, epsilon=epsilon, n_jobs=2, log=False)

    print("1 PARAMETER")
    print(len(results))
    print(results)

    print()

    print(results[0][0])
    '''

    samples, distances, sum_stats, epsilons, n_sims = sampler.sample(
        n_samples, epsilon=epsilon, n_jobs=4, log=True)

    print(len(samples))
    # print(samples)
    print(n_sims)

    # 1 worker [Finished in 16.568s]
    # 2 workers [Finished in 9.154s]
    # 3 workers [Finished in 6.616s]
    # 4 workers [Finished in 5.525s]
    # 5 workers [Finished in 4.881s]
    # 6 workers [Finished in 4.552s]
    # 7 workers [Finished in 4.17s]
    # 8 workers [Finished in 3.924s]

    '''
    # 2 parameter inference
    #
    N = 1000
    mu_true = 163
    sigma_true = 15
    true_parameter_values = [mu_true, sigma_true]
    likelihood = stats.norm(loc=mu_true, scale=sigma_true)
    likelihood = Normal(loc=mu_true, scale=sigma_true, name="likelihood")
    data = likelihood.rvs(size=N, seed=42)
    obs = np.array([np.mean(data), np.std(data)])

    def gaussian_model(mu, sigma, n_samples=1000):
        """Simulator model, returns summary statistic"""
        sim = stats.norm(loc=mu, scale=sigma).rvs(size=n_samples)
        sumstat = np.array([np.mean(sim), np.std(sim)])
        #sumstat = np.mean(sim)
        return sumstat

    # priors
    mu = Normal(loc=165, scale=2, name="mu", tex="$\mu$")
    sigma = Normal(loc=17, scale=4, name="sigma", tex="$\sigma$")
    priors = [mu, sigma]

    # initialize sampler
    sampler = RejABC(obs, gaussian_model, priors, distance='l2', seed=42)

    # inference config
    n_samples = 10
    epsilon = 1.0

    samples = sampler.sample(n_samples, epsilon=epsilon, n_jobs=-1, log=True)

    print("2 PARAMETERS")
    print(len(samples))
    print(samples)
    '''
