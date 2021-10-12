#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import multiprocessing
from multiprocessing import Lock, RLock

import numpy as np
from pathos.pools import ProcessPool
from pylfi.inferences import ABCBase, PilotStudyMissing, SamplingNotPerformed
from pylfi.journal import Journal
from pylfi.utils import advance_PRNG_state
from tqdm.auto import tqdm


class RejABC(ABCBase):
    """Class implementing the rejection ABC algorithm.
    """

    def __init__(
        self,
        observation,
        simulator,
        stat_calc,
        priors,
        log=True
    ):
        super().__init__(
            observation=observation,
            simulator=simulator,
            stat_calc=stat_calc,
            priors=priors,
            inference_scheme="Rejection ABC",
            log=log
        )

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

        self._n_samples = n_samples
        self._stat_weight = stat_weight

        _, n_jobs, tasks, seeds = self.batches(n_samples, n_jobs, seed)

        if self._log:
            # for managing output contention
            initializer = tqdm.set_lock(RLock(),)
            initargs = (tqdm.get_lock(),)
        else:
            initializer = None
            initargs = None

        with ProcessPool(n_jobs) as pool:
            r0, r1, r2, r3 = zip(
                *pool.map(
                    self._sample,
                    tasks,
                    range(n_jobs),
                    seeds,
                    initializer=initializer,
                    initargs=initargs
                )
            )

        self._original_samples = np.concatenate(r0, axis=0)
        self._samples = copy.deepcopy(self._original_samples)
        self._distances = np.concatenate(r1, axis=0)
        self._sum_stats = np.concatenate(r2, axis=0)
        self._n_sims = np.sum(r3)

        self._done_sampling = True

        if return_journal:
            return self.journal()

    def _sample(self, n_samples, position, seed):
        """Sample n_samples from posterior."""

        if self._log:
            t_range = tqdm(range(n_samples),
                           desc=f"[Sampling progress] CPU {position+1}",
                           position=position,
                           leave=False,
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
            # Advance PRNG state
            next_gen = advance_PRNG_state(seed, self._n_sims)
            # Draw proposal parameters from priors
            thetas = [prior.rvs(seed=next_gen) for prior in self._priors]
            # Simulator call to generate simulated data
            sim = self._simulator(*thetas)
            # Calculate summary statistics of simulated data
            if isinstance(sim, tuple):
                sim_sumstat = self._stat_calc(*sim)
            else:
                sim_sumstat = self._stat_calc(sim)
            # Increase simulations counter
            self._n_sims += 1
            # Compute distance between summary statistics
            distance = self.distance(sim_sumstat,
                                     self._obs_sumstat,
                                     weight=self._stat_weight,
                                     scale=self._stat_scale
                                     )
            # ABC reject/accept step
            if distance <= self._epsilon:
                sample = thetas

        return sample, distance, sim_sumstat

    def journal(self):
        """
        Create and return an instance of Journal class.

        Returns
        -------

        """
        if not self._done_sampling:
            msg = ("In order to access the journal, the "
                   "sample method must be run in advance.")
            raise SamplingNotPerformed(msg)

        if self._log:
            self.logger.info(f"Write results to journal.")

        accept_ratio = self._n_samples / self._n_sims
        journal = Journal()
        journal._write_to_journal(
            inference_scheme=self._inference_scheme,
            observation=self._obs_data,
            simulator=self._simulator,
            stat_calc=self._stat_calc,
            priors=self._priors,
            n_samples=self._n_samples,
            n_chains=1,
            n_sims=self._n_sims,
            samples=self._samples,
            accept_ratio=accept_ratio,
            epsilon=self._epsilon,
            quantile=self._quantile
        )

        return journal


if __name__ == "__main__":
    import arviz as az
    import matplotlib.pyplot as plt
    import pylfi
    import scipy.stats as stats
    import seaborn as sns

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
    sampler = RejABC(obs_data,
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

    journal = sampler.sample(200,
                             use_pilot=True,
                             n_jobs=2,
                             seed=42,
                             return_journal=True
                             )

    df = journal.df
    # print(df)
    #idata = journal.idata
    # print(idata)
    # print(idata["posterior"])
    #sns.kdeplot(data=df, x="mu", y="sigma", fill=True)
    #sns.pairplot(df, kind="kde")
    sns.jointplot(
        data=df,
        x="mu",
        y="sigma",
        kind="kde",
        fill=True
    )
    # az.plot_trace(idata)
    plt.ticklabel_format(useOffset=False)

    journal = sampler.reg_adjust(method="loclinear",
                                 kernel="epkov",
                                 standardize=False,
                                 return_journal=True
                                 )

    df = journal.df
    sns.jointplot(
        data=df,
        x="mu",
        y="sigma",
        kind="kde",
        fill=True
    )
    plt.ticklabel_format(useOffset=False)
    plt.show()
    # print(df)
    #idata = journal.idata
    # print(idata)
    # print(idata["posterior"])
    #print(idata.posterior.stack(sample=("chain", "draw")))
    # az.plot_trace(idata)
