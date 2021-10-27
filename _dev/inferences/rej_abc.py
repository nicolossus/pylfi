#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from multiprocessing import RLock

import numpy as np
from pathos.pools import ProcessPool
from pylfi.inferences import ABCBase
from pylfi.journal import Journal
from pylfi.utils import (advance_PRNG_state, check_and_set_jobs,
                         distribute_workload, generate_seed_sequence,
                         setup_logger)
from tqdm.auto import tqdm


class RejABC(ABCBase):

    def __init__(self, observation, simulator, statistics_calculator, priors, distance_metric='l2', seed=None):

        super().__init__(
            observation=observation,
            simulator=simulator,
            statistics_calculator=statistics_calculator,
            priors=priors,
            distance_metric=distance_metric,
            seed=seed
        )

    def sample(self, n_samples, epsilon=None, quantile=None, n_tune=500, n_jobs=-1, log=False):
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

        if quantile is not None:
            tasks = distribute_workload(n_samples, n_jobs)

            distances_tune = self._pilot_study(n_tune, seeds[0])
            # print(distances_tune)
            #distances_tune = np.concatenate(distances_tune, axis=0)
            self._epsilon = np.quantile(np.array(distances_tune), quantile)
            # print(self._epsilon)

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

        journal = Journal()
        journal._write_to_journal(
            observation=self._obs_data,
            simulator=self._simulator,
            stat_calc=self._stat_calc,
            priors=self._priors,
            distance_metric=self._distance_metric,
            inference_scheme=_inference_scheme,
            n_samples=n_samples,
            n_simulations=n_sims,
            posterior_samples=samples,
            summary_stats=sum_stats,
            distances=distances,
            epsilons=epsilons,
            log=log)

        # return results
        # return samples, distances, sum_stats, epsilons, n_sims
        return journal

    def _pilot_study(self, n_tune, seed):
        """Set threshold"""

        distances = []

        for i in range(n_tune):

            next_gen = advance_PRNG_state(seed, i)
            thetas = [prior.rvs(seed=next_gen) for prior in self._priors]
            sim = self._simulator(*thetas)
            if isinstance(sim, tuple):
                sim_sumstat = self._stat_calc(*sim)
            else:
                sim_sumstat = self._stat_calc(sim)
            distance = self._distance_metric(self._obs_sumstat, sim_sumstat)
            distances.append(distance)

        return distances

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
            if isinstance(sim, tuple):
                sim_sumstat = self._stat_calc(*sim)
            else:
                sim_sumstat = self._stat_calc(sim)
            self._n_sims += 1
            distance = self._distance_metric(self._obs_sumstat, sim_sumstat)
            if distance <= self._epsilon:
                sample = thetas

        return sample, distance, sim_sumstat, self._epsilon


if __name__ == "__main__":
    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np
    import pylfi
    import scipy.stats as stats
    import seaborn as sns

    # Task: infer variance parameter in zero-centered Gaussian model
    #
    '''
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
    epsilon = 0.5

    # run inference
    journal = sampler.sample(n_samples, epsilon=epsilon, n_jobs=-1, log=False)

    # print(journal.results_frame())

    posterior_df = journal.posterior_frame()
    posterior_dict = journal.posterior_dict()
    idata = az.convert_to_inference_data(posterior_dict)
    print(idata)
    # , var_names=["mu", "theta"], coords=coords, rope=(-1, 1))
    az.plot_trace(idata)
    # az.plot_posterior(idata)
    plt.show()

    # print(posterior_dict)
    #sns.displot(posterior_df, kind="kde")
    # plt.show()
    '''

    # 2 parameter inference
    #
    N = 1000
    mu_true = 163
    sigma_true = 15
    true_parameter_values = [mu_true, sigma_true]
    likelihood = stats.norm(loc=mu_true, scale=sigma_true)

    obs_data = likelihood.rvs(size=N)

    # simulator model
    def gaussian_model(mu, sigma, n_samples=1000):
        """Simulator model"""
        sim = stats.norm(loc=mu, scale=sigma).rvs(size=n_samples)
        return sim

    # summary stats
    def summary_calculator(data):
        """returns summary statistic(s)"""
        sumstat = np.array([np.mean(data), np.std(data)])
        #sumstat = np.mean(sim)
        return sumstat

    # priors
    #mu = pylfi.Prior('norm', loc=165, scale=2, name='mu', tex='$\mu$')
    #sigma = pylfi.Prior('norm', loc=17, scale=4, name='sigma', tex='$\sigma$')
    mu = pylfi.Prior('uniform', loc=160, scale=10, name='mu')
    sigma = pylfi.Prior('uniform', loc=10, scale=10, name='sigma')
    priors = [mu, sigma]

    # initialize sampler
    sampler = RejABC(obs_data,
                     gaussian_model,
                     summary_calculator,
                     priors,
                     distance_metric='l2',
                     seed=42
                     )

    # inference config
    n_samples = 1000
    epsilon = 1.0
    quantile = 0.01

    # run inference
    journal = sampler.sample(n_samples,
                             epsilon=epsilon,
                             quantile=quantile,
                             n_jobs=-1,
                             log=False
                             )

    # print(journal.results_frame())
    journal.plot_trace()
    journal.plot_posterior()
    journal.plot_pair(var_names=["mu", "sigma"])
    plt.show()
