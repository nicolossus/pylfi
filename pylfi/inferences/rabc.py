#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
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

VALID_STAT_SCALES = ["sd", "mad"]


class PilotStudyMissing(Exception):
    """Failed attempt at accessing pilot study.

    A call to the pilot_study method must be carried out first.
    """
    pass


class SamplingNotPerformed(Exception):
    """Failed attempt at accessing posterior samples.

    A call to the sample method must be carried out first.
    """
    pass


class RABC:

    def __init__(
        self,
        observation,
        simulator,
        stat_calc,
        priors,
        log=False
    ):

        self._obs_data = observation
        self._stat_calc = stat_calc
        self._simulator = simulator
        self._priors = priors
        self._log = log

        if isinstance(self._obs_data, tuple):
            self._obs_sumstat = self._stat_calc(*self._obs_data)
        else:
            self._obs_sumstat = self._stat_calc(self._obs_data)

        _inference_scheme = "Rejection ABC"

        if self._log:
            self.logger = setup_logger(self.__class__.__name__)
            self.logger.info(f"Initialize {_inference_scheme} sampler.")

        self._done_pilot_study = False
        self._done_sampling = False

    def _distance(self, s_sim, s_obs, weight=1., scale=1.):
        """
        weighted euclidean distance

        s_sim: simulated summary statistics
        s_obs: observed summary statistics
        weight: importance weight(s)
        scale: scaling weight(s)
        """
        if isinstance(s_sim, (int, float)):
            s_sim = [s_sim]
        if isinstance(s_obs, (int, float)):
            s_obs = [s_obs]

        s_sim = np.asarray(s_sim)
        s_obs = np.asarray(s_obs)

        q = weight * (s_sim - s_obs) / scale
        dist = np.linalg.norm(q, ord=2)
        return dist

    def _batches(self, n_samples, n_jobs, seed):
        """
        Divide and conquer
        """
        if self._log:
            n_jobs = check_and_set_jobs(n_jobs, self.logger)
        else:
            n_jobs = check_and_set_jobs(n_jobs)

        seeds = generate_seed_sequence(seed, n_jobs)
        tasks = distribute_workload(n_samples, n_jobs)
        return n_jobs, tasks, seeds

    def pilot_study(
        self,
        n_sim=500,
        quantile=None,
        stat_scale=None,  # accept sd, mad
        stat_weight=1.,
        n_jobs=-1,
        seed=None,
    ):
        """
        Pilot study to set threshold and optionally summary statistics scale.

        Set scale and epsilon  (add bool for if weights also?)
        """

        if self._log:
            msg = "Run pilot study to estimate:\n"
            msg += "* epsilon as the p-quantile of the distances"
            # self.logger.info("Run pilot study to estimate:\n")
            # self.logger.info("* epsilon as the p-quantile of the distances.")
            if stat_scale is not None:
                msg += "\n* summary statistics scale from the prior "
                msg += "predictive distribution"
                # "predictive distribution"
                # msg = ("* summary statistics scale from the prior "
                #       "predictive distribution")
            self.logger.info(msg)

        if quantile is None:
            msg = ("quantile must be passed. The pilot study sets the "
                   "accept/reject threshold as the provided p-quantile of the "
                   "distances.")
            raise ValueError(msg)

        if stat_scale is not None:
            if stat_scale not in VALID_STAT_SCALES:
                msg = ("scale can be set as either sd (standard deviation) or "
                       "mad (median absolute deviation). If None, it defaults "
                       "to 1.")
                raise ValueError(msg)

        self._quantile = quantile
        n_jobs, tasks, seeds = self._batches(n_sim, n_jobs, seed)

        if self._log:
            tqdm.set_lock(RLock())  # for managing output contention
            initializer = tqdm.set_lock
        else:
            initializer = None

        with ProcessPool(n_jobs) as pool:
            results = pool.map(self._pilot_study,
                               tasks,
                               range(n_jobs),
                               seeds,
                               initializer=initializer
                               )

        sum_stats = np.concatenate(results, axis=0)

        if stat_scale is None:
            self._stat_scale = 1.
        elif stat_scale == "sd":
            self._stat_scale = sum_stats.std(axis=0)
        elif stat_scale == "mad":
            self._stat_scale = np.median(np.absolute(
                sum_stats - np.median(sum_stats, axis=0)), axis=0)
        else:
            msg = ("scale can be set as either sd (standard deviation) or "
                   "mad (median absolute deviation). If None, defaults to 1.")
            raise ValueError(msg)

        distances = []
        for sum_stat in sum_stats:
            distance = self._distance(sum_stat,
                                      self._obs_sumstat,
                                      weight=stat_weight,
                                      scale=self._stat_scale
                                      )
            distances.append(distance)

        self._epsilon = np.quantile(np.array(distances), self._quantile)
        self._done_pilot_study = True

    def _pilot_study(self, n_sims, position, seed):

        if self._log:
            t_range = tqdm(range(n_sims),
                           desc=f"[Simulation progress] CPU {position+1}",
                           position=position,
                           leave=True,
                           colour='green')
        else:
            t_range = range(n_sims)

        sum_stats = []

        for i in t_range:
            next_gen = advance_PRNG_state(seed, i)
            thetas = [prior.rvs(seed=next_gen) for prior in self._priors]
            sim = self._simulator(*thetas)
            if isinstance(sim, tuple):
                sim_sumstat = self._stat_calc(*sim)
            else:
                sim_sumstat = self._stat_calc(sim)
            sum_stats.append(sim_sumstat)

        if self._log:
            t_range.clear()

        return sum_stats

    def sample(
        self,
        n_samples,
        epsilon=None,
        stat_weight=1.,
        stat_scale=1.,
        use_pilot=False,
        n_jobs=-1,
        seed=None,
        return_journal=False
    ):
        """
        Due to multiprocessing, estimation time (iteration per loop, total
        time, etc.) could be unstable, but the progress bar works perfectly.

        A good choice for the number of jobs is the number of cores or processors on your computer.
        If your processor supports hyperthreading, you can select an even higher number of jobs.
        The number of jobs is set to the number of cores found in the system by default.
        """

        if self._log:
            self.logger.info("Run rejection sampler.")

        if use_pilot:
            if not self._done_pilot_study:
                msg = ("In order to use tuning from pilot study, the "
                       "pilot_study method must be run in advance.")
                raise PilotStudyMissing(msg)
        else:
            if epsilon is None:
                msg = ("epsilon must be passed.")
                raise ValueError(msg)
            self._epsilon = epsilon
            self._quantile = None
            self._stat_scale = stat_scale

        self._stat_weight = stat_weight

        n_jobs, tasks, seeds = self._batches(n_samples, n_jobs, seed)

        if self._log:
            tqdm.set_lock(RLock())  # for managing output contention
            initializer = tqdm.set_lock
        else:
            initializer = None

        with ProcessPool(n_jobs) as pool:
            # '''
            r0, r1, r2, r3 = zip(
                *pool.map(
                    self._sample,
                    tasks,
                    range(n_jobs),
                    seeds,
                    initializer=initializer
                )
            )

        self._original_samples = np.concatenate(r0, axis=0)
        self._samples = copy.deepcopy(self._original_samples)
        self._distances = np.concatenate(r1, axis=0)
        self._sum_stats = np.concatenate(r2, axis=0)
        self._n_sims = np.sum(r3)

        self._done_sampling = True

        '''

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
        '''
        # return samples
        if return_journal:
            return self.journal()

    def _sample(self, n_samples, position, seed):
        """Sample n_samples from posterior."""

        if self._log:
            t_range = tqdm(range(n_samples),
                           desc=f"[Sampling progress] CPU {position+1}",
                           position=position,
                           leave=True,
                           colour='green')
        else:
            t_range = range(n_samples)

        self._n_sims = 0
        samples = []
        distances = []
        sum_stats = []

        for _ in t_range:
            sample, distance, sim = self._draw_posterior_sample(seed)
            samples.append(sample)
            distances.append(distance)
            sum_stats.append(sim)

        if self._log:
            t_range.clear()

        return [samples, distances, sum_stats, self._n_sims]

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

            distance = self._distance(sim_sumstat,
                                      self._obs_sumstat,
                                      weight=self._stat_weight,
                                      scale=self._stat_scale
                                      )
            if distance <= self._epsilon:
                sample = thetas

        return sample, distance, sim_sumstat

    def reg_adjust(
        self,
        method="loclinear",
        standardize=True,
        kernel='epkov',
        lmbda=1.,
        return_journal=False
    ):
        """
        method: linear, loclinear
        """

        if not self._done_sampling:
            msg = ("In order to perform regression adjustment, the "
                   "sample method must be run in advance.")
            raise SamplingNotPerformed(msg)

        self._kernel = kernel
        self._lmbda = lmbda

        if self._log:
            self.logger.info(f"Perform {method} regression adjustment.")

        X = copy.deepcopy(self._sum_stats)
        thetas = copy.deepcopy(self._original_samples)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # augment with column of ones
        X = np.c_[np.ones(X.shape[0]), X]

        if standardize:
            X_mean = np.mean(X[:, 1:], axis=0)
            X_sd = np.std(X[:, 1:], axis=0)
            X_stand = (X[:, 1:] - X_mean[np.newaxis, :]) / X_sd[np.newaxis, :]
            X = np.c_[np.ones(X.shape[0]), X_stand]

            thetas_mean = np.mean(thetas)
            thetas_sd = np.std(thetas)
            thetas = (thetas - thetas_mean) / thetas_sd

        if method == "linear":
            self._lra(X, thetas)
        elif method == "loclinear":
            self._loclra(X, thetas)
        elif method == "ridge":
            self._ridge(X, thetas)
        elif method == "locridge":
            self._locridge(X, thetas)
        else:
            raise ValueError("Unrecognized regression method.")

        if return_journal:
            return self.journal()

    def _lra(self, X, y):
        """linear regression adjustment"""

        # pinv = pseudo-inverse
        coef = np.linalg.pinv(X.T @ X)  @ X.T @ y
        alpha = coef[0]
        beta = coef[1:]

        correction = (self._sum_stats - self._obs_sumstat) @ beta
        self._samples = self._original_samples - correction

    def _ridge(self, X, y):
        xTx = X.T @ X
        lmb_eye = self._lmbda * np.identity(xTx.shape[0])
        # pinv = pseudo-inverse
        coef = np.linalg.pinv(xTx + lmb_eye) @ X.T @ y

        alpha = coef[0]
        beta = coef[1:]

        correction = (self._sum_stats - self._obs_sumstat) @ beta
        self._samples = self._original_samples - correction

    def _locridge(self, X, y):

        if self._kernel == "gaussian":
            weights = np.array([self._gaussian_kernel(d, self._epsilon)
                                for d in self._distances])
        elif self._kernel == "epkov":
            weights = np.array([self._epkov_kernel(d, self._epsilon)
                                for d in self._distances])
        else:
            raise ValueError(f'Unrecognized kernel')

        # scale kernel values so that the sum of weights is 1
        weights /= np.sum(weights)

        W = np.diag(weights)

        self._lmbda = 1.0
        xTWx = X.T @ W @ X
        lmb_eye = self._lmbda * np.identity(xTWx.shape[0])
        # pinv = pseudo-inverse
        coef = np.linalg.pinv(xTWx + lmb_eye) @ X.T @ W @ y

        alpha = coef[0]
        beta = coef[1:]

        correction = (self._sum_stats - self._obs_sumstat) @ beta
        self._samples = self._original_samples - correction

    def _loclra(self, X, y):
        """local linear regression adjustment"""

        if self._kernel == "gaussian":
            weights = np.array([self._gaussian_kernel(d, self._epsilon)
                                for d in self._distances])
        elif self._kernel == "epkov":
            weights = np.array([self._epkov_kernel(d, self._epsilon)
                                for d in self._distances])
        else:
            raise ValueError(f'Unrecognized kernel')

        # scale kernel values so that the sum of weights is 1
        weights /= np.sum(weights)

        W = np.diag(weights)

        # pinv = pseudo-inverse
        coef = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
        alpha = coef[0]
        beta = coef[1:]

        correction = (self._sum_stats - self._obs_sumstat) @ beta
        self._samples = self._original_samples - correction

    def _gaussian_kernel(self, d, h):
        return 1 / (h * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (d * d) / (h * h))

    def _epkov_kernel(self, d, h):
        return 0.75 / h * (1.0 - (d * d) / (h * h)) * (d < h)

    def journal(self):
        """
        Journal
        """
        if not self._done_sampling:
            msg = ("In order to access the journal, the "
                   "sample method must be run in advance.")
            raise SamplingNotPerformed(msg)
        return self._samples


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pylfi
    import scipy.stats as stats
    import seaborn as sns

    N = 1000
    mu_true = 163
    sigma_true = 15
    true_parameter_values = [mu_true, sigma_true]
    #likelihood = stats.norm(loc=mu_true, scale=sigma_true)
    likelihood = pylfi.Prior('norm',
                             loc=mu_true,
                             scale=sigma_true,
                             name='likelihood'
                             )

    obs_data = likelihood.rvs(size=N, seed=30)

    # simulator model
    def gaussian_model(mu, sigma, seed=43, n_samples=1000):
        """Simulator model"""
        #sim = stats.norm(loc=mu, scale=sigma).rvs(size=n_samples)
        model = pylfi.Prior('norm', loc=mu, scale=sigma, name='model')
        sim = model.rvs(size=n_samples, seed=seed)

        return sim

    # summary stats
    def summary_calculator(data):
        """returns summary statistic(s)"""
        sumstat = np.array([np.mean(data), np.std(data)])
        # sumstat = np.mean(sim)
        return sumstat

    s_obs = summary_calculator(obs_data)
    # print(f"{s_obs=}")
    # priors
    mu = pylfi.Prior('norm', loc=165, scale=2, name='mu', tex='$\mu$')
    sigma = pylfi.Prior('norm', loc=17, scale=4, name='sigma', tex='$\sigma$')
    #mu = pylfi.Prior('uniform', loc=160, scale=10, name='mu')
    #sigma = pylfi.Prior('uniform', loc=10, scale=10, name='sigma')
    priors = [mu, sigma]

    # initialize sampler
    sampler = RABC(obs_data,
                   gaussian_model,
                   summary_calculator,
                   priors,
                   log=True
                   )

    sampler.pilot_study(3000,
                        quantile=1.0,
                        stat_scale="mad",
                        n_jobs=4,
                        seed=4
                        )

    samples = sampler.sample(200,
                             use_pilot=True,
                             n_jobs=4,
                             seed=42,
                             return_journal=True
                             )
    '''
    sampler.sample(200,
                   use_pilot=True,
                   n_jobs=4,
                   seed=42,
                   return_journal=False
                   )
    '''

    print(np.mean(samples[:, 0]))
    print(np.mean(samples[:, 1]))

    samples = sampler.reg_adjust(method="linear",
                                 kernel="epkov",
                                 standardize=False,
                                 return_journal=True
                                 )

    print(np.mean(samples[:, 0]))
    print(np.mean(samples[:, 1]))

    samples = sampler.reg_adjust(method="loclinear",
                                 kernel="epkov",
                                 standardize=False,
                                 return_journal=True
                                 )
    #samples = sampler.journal()
    #print(samples[:, 0])

    print(np.mean(samples[:, 0]))
    print(np.mean(samples[:, 1]))

    samples = sampler.reg_adjust(method="loclinear",
                                 kernel="gaussian",
                                 standardize=False,
                                 return_journal=True
                                 )

    print(np.mean(samples[:, 0]))
    print(np.mean(samples[:, 1]))

    samples = sampler.reg_adjust(method="ridge",
                                 kernel="gaussian",
                                 standardize=False,
                                 return_journal=True
                                 )

    print(np.mean(samples[:, 0]))
    print(np.mean(samples[:, 1]))

    samples = sampler.reg_adjust(method="locridge",
                                 kernel="epkov",
                                 standardize=False,
                                 return_journal=True
                                 )
    #samples = sampler.journal()
    #print(samples[:, 0])

    print(np.mean(samples[:, 0]))
    print(np.mean(samples[:, 1]))

    samples = sampler.reg_adjust(method="locridge",
                                 kernel="gaussian",
                                 lmbda=1.,
                                 standardize=False,
                                 return_journal=True
                                 )

    print(np.mean(samples[:, 0]), np.median(samples[:, 0]))
    print(np.mean(samples[:, 1]))

    samples = sampler.reg_adjust(method="locridge",
                                 kernel="gaussian",
                                 lmbda=1000.,
                                 standardize=False,
                                 return_journal=True
                                 )

    print(np.mean(samples[:, 0]))
    print(np.mean(samples[:, 1]))

    '''
    fig, axes = plt.subplots(nrows=1, ncols=2)
    sns.kdeplot(samples[:, 0], ax=axes[0])
    sns.kdeplot(samples[:, 1], ax=axes[1])
    axes[0].set(xlabel='mu')
    axes[1].set(xlabel='sigma')
    plt.show()
    '''

    '''
    fig, axes = plt.subplots(nrows=1, ncols=2)
    sns.kdeplot(samples[:, 0], ax=axes[0])
    sns.kdeplot(samples[:, 1], ax=axes[1])
    axes[0].set(xlabel='mu')
    axes[1].set(xlabel='sigma')
    # plt.show()

    # adjustment

    sampler.post_adjust(method="linear")
    samples = sampler.journal()
    print(np.mean(samples[:, 0]))
    print(np.mean(samples[:, 1]))

    fig, axes = plt.subplots(nrows=1, ncols=2)
    sns.kdeplot(samples[:, 0], ax=axes[0])
    sns.kdeplot(samples[:, 1], ax=axes[1])
    axes[0].set(xlabel='mu')
    axes[1].set(xlabel='sigma')

    plt.show()
    '''

    '''
    loclinear :
    163.40171339954006
    13.81441439963763

    linear :
    163.40171339954006
    13.814414399637629

    plain :

    '''
